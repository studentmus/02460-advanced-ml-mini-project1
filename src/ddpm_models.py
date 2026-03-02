import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1), torch.nn.LogSigmoid()),
            torch.nn.Sequential(torch.nn.MaxPool2d(2), torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1), torch.nn.LogSigmoid()),
            torch.nn.Sequential(torch.nn.MaxPool2d(2), torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1), torch.nn.LogSigmoid()),
            torch.nn.Sequential(torch.nn.MaxPool2d(2, stride=2, padding=1), torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1), torch.nn.LogSigmoid()),
            torch.nn.Sequential(torch.nn.MaxPool2d(2), torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1), torch.nn.LogSigmoid()),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.ConvTranspose2d(chs[4], chs[3], 3, 2, 1, 1), torch.nn.LogSigmoid()),
            torch.nn.Sequential(torch.nn.ConvTranspose2d(chs[3]*2, chs[2], 3, 2, 1, 0), torch.nn.LogSigmoid()),
            torch.nn.Sequential(torch.nn.ConvTranspose2d(chs[2]*2, chs[1], 3, 2, 1, 1), torch.nn.LogSigmoid()),
            torch.nn.Sequential(torch.nn.ConvTranspose2d(chs[1]*2, chs[0], 3, 2, 1, 1), torch.nn.LogSigmoid()),
            torch.nn.Sequential(torch.nn.Conv2d(chs[0]*2, chs[0], 3, padding=1), torch.nn.LogSigmoid(), torch.nn.Conv2d(chs[0], 1, 3, padding=1)),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)
        signal = torch.cat((x2, tt), dim=-3)
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1: signals.append(signal)
        for i, tconv in enumerate(self._tconvs):
            if i > 0: signal = torch.cat((signal, signals[-i]), dim=-3)
            signal = tconv(signal)
        return torch.reshape(signal, (*signal.shape[:-3], -1))

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=1000):
        super(DDPM, self).__init__()
        self.network = network
        self.T = T

        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1. - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)

    def negative_elbo(self, x):
        t = torch.randint(0, self.T, (x.shape[0],), device=x.device)
        epsilon = torch.randn_like(x)
        a_bar = self.alpha_cumprod[t].view(-1, 1)
        x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * epsilon
        epsilon_theta = self.network(x_t, t.float().view(-1, 1) / self.T)
        return F.mse_loss(epsilon_theta, epsilon, reduction='none').sum(dim=-1)

    @torch.no_grad()
    def sample(self, shape):
        x_t = torch.randn(shape, device=self.beta.device)

        for t_idx in reversed(range(self.T)):
            t_tensor = torch.full((shape[0], 1), t_idx, device=x_t.device, dtype=torch.float)
            epsilon_theta = self.network(x_t, t_tensor / self.T)

            alpha_t = self.alpha[t_idx]
            alpha_bar_t = self.alpha_cumprod[t_idx]
            beta_t = self.beta[t_idx]

            mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon_theta)

            if t_idx > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        return x_t

    def loss(self, x):
        return self.negative_elbo(x).mean()


class BetaVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(BetaVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + beta * kl_divergence) / x.size(0)


class LatentMLP(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z, t):
        zt = torch.cat([z, t], dim=-1)
        return self.net(zt)