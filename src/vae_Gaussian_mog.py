import os
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoGPrior(nn.Module):
    def __init__(self, M, K):
        super().__init__()
        self.M = M
        self.K = K
        self.mu = nn.Parameter(torch.randn(K, M) * 2.0, requires_grad=False)
        self.log_std = nn.Parameter(torch.zeros(K, M) - 0.5, requires_grad=False)
        self.logits = nn.Parameter(torch.zeros(K), requires_grad=False)

    def forward(self):
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(self.mu, torch.exp(self.log_std)), 1)
        return td.MixtureSameFamily(mix, comp)


class GaussPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.M = 2
        self.K = 1

        self.mu = nn.Parameter(torch.zeros(self.K, self.M), requires_grad=False)
        self.log_std = nn.Parameter(torch.zeros(self.K, self.M), requires_grad=False)
        self.logits = nn.Parameter(torch.zeros(self.K), requires_grad=False)

    def forward(self):
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(self.mu, torch.exp(self.log_std)), 1)
        return td.MixtureSameFamily(mix, comp)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)

        log_std = torch.clamp(log_std, min=-10.0, max=10.0)
        return td.Independent(td.Normal(mean, torch.exp(log_std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        params = self.decoder_net(z)               # (B, 2, 28, 28)
        mu, log_var = torch.chunk(params, 2, dim=1)

        sigma = torch.exp(0.5 * log_var)
        return td.Independent(td.Normal(loc=mu, scale=sigma), 3)

class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        pz = self.prior()

        log_px = self.decoder(z).log_prob(x)
        kl = q.log_prob(z) - pz.log_prob(z)

        return torch.mean(log_px - kl)

    def forward(self, x):
        return -self.elbo(x)


def train(model, optimizer, loader, epochs):
    model.train()
    total_steps = len(loader) * epochs
#    progress_bar = tqdm(range(total_steps), desc="Training")
    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

#            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch + 1}/{epochs}")
#            progress_bar.update()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(loader):.4f}")


def plot_mog_contours(prior, lim=6, grid=300, levels=15):
    prior.eval()

    xs = torch.linspace(-lim, lim, grid, device=device)
    ys = torch.linspace(-lim, lim, grid, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    with torch.no_grad():
        logp = prior().log_prob(pts).reshape(grid, grid)
        mu = prior.mu.detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    cs = plt.contour(X.cpu().numpy(), Y.cpu().numpy(), logp.cpu().numpy(), levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)

    plt.scatter(mu[:, 0], mu[:, 1], marker="x", s=80)
    plt.title("MoG Prior Contours in Latent Space (log p(z))")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.grid(True)
    plt.show()


def plot_mog_contours(prior, model, loader, lim=14, grid=300, levels=15,
                      max_batch=32, use_mean=False, alpha=0.6, s=8):
    prior.eval()
    model.eval()

    xs = torch.linspace(-lim, lim, grid, device=device)
    ys = torch.linspace(-lim, lim, grid, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    with torch.no_grad():
        logp = prior().log_prob(pts).reshape(grid, grid)
        mu = prior.mu.detach().cpu().numpy()

    zs = []
    ys_lab = []
    collected = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            q = model.encoder(x)

            if use_mean:
                z = q.base_dist.loc
            else:
                z = q.rsample()

            zs.append(z.detach().cpu())
            ys_lab.append(y.detach().cpu())
            collected += x.size(0)
            if max_batch is not None and i >= max_batch:
                break

    Z = torch.cat(zs, dim=0).numpy()
    Ylab = torch.cat(ys_lab, dim=0).numpy()

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    cs = plt.contour(X.cpu().numpy(), Y.cpu().numpy(), logp.cpu().numpy(), levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)

    sc = plt.scatter(Z[:, 0], Z[:, 1], c=Ylab, cmap="tab10", s=s, alpha=alpha)
    plt.colorbar(sc, ticks=list(range(10)), label="Digit label")

    plt.scatter(mu[:, 0], mu[:, 1], marker="x", s=80)

    plt.title("MoG/Gauss Prior Contours (log p(z)) + Encoded Data (colored by class)")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.grid(True)
    plt.show()


def show_reconstructions(model, loader, n=10, thr=0.5):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)

    with torch.no_grad():
        z = model.encoder(x).rsample()
        xhat = model.decoder(z).mean

    x = x.cpu()
    xhat = xhat.cpu()

    plt.figure(figsize=(2 * n, 3))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")

        plt.subplot(2, n, n + i + 1)
        plt.imshow(xhat[i], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")

    plt.suptitle("Top: Original | Bottom: Reconstruction (thresholded)")
    plt.show()

def evaluate_elbo(model, device):
    model.eval()
    total_elbo = 0

    data_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                    batch_size=32 * 313, shuffle=True)
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            total_elbo += model.elbo(x).item()

    return total_elbo



if __name__ == "__main__":
    threshold = 0.5
    transform = transforms.ToTensor()

    train_loader = DataLoader(
        datasets.MNIST("data/", train=True, download=True, transform=transform),
        batch_size=256, shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST("data/", train=False, download=True, transform=transform),
        batch_size=256, shuffle=True
    )

    elbos = []

    for i in range(1):
        M = 2
        K = 10

        prior = MoGPrior(M, K).to(device)  # use this code if u want prior

        # prior = GaussPrior().to(device)  # use this if you want standard gaussian N(0,I)

        encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, M * 2),
        )

        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2*784),
            nn.Unflatten(-1, (2, 28, 28))
        )

        encoder = GaussianEncoder(encoder_net)
        decoder = GaussianDecoder(decoder_net)
        model = VAE(prior, decoder, encoder).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train(model, optimizer, train_loader, epochs=100)

        elbo = evaluate_elbo(model, device)
        elbos.append(elbo)
        print(f"ELBO of the training {i}th is {elbo}")
        print("-------------------------------------------------------------------------------------------------")

        if 10 - i == 1:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/VAE_MoG_Gaussian_likelihood.pt")

    elbos = np.array(elbos)

    mean = elbos.mean()
    std = elbos.std(ddof=1)  # sample standard deviation
    print(f"The mean and standard deviation of elbos among {10}: {mean} and {std}")

    # plot_mog_contours(prior, model, test_loader, use_mean=False)
    # show_reconstructions(model, test_loader, n=10, thr=0.5)
