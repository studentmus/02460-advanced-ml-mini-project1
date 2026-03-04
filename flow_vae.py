import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ===============================
# Setup
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

M = 2
BATCH_SIZE = 256
N_RUNS = 10
EPOCHS = 50
THRESHOLD = 0.5

os.makedirs("models", exist_ok=True)
os.makedirs("figs", exist_ok=True)

# ===============================
# Dataset
# ===============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > THRESHOLD).float().squeeze())
])

train_loader = DataLoader(
    datasets.MNIST("data/", train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST("data/", train=False, download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ===============================
# Flow Components
# ===============================

class CouplingLayer(nn.Module):

    def __init__(self, dim, mask):
        super().__init__()

        self.register_buffer("mask", mask)

        self.scale_net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
            nn.Tanh()
        )

        self.shift_net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, z):

        z_masked = z * self.mask

        s = self.scale_net(z_masked) * (1 - self.mask)
        t = self.shift_net(z_masked) * (1 - self.mask)

        z_out = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)

        log_det = (s * (1 - self.mask)).sum(dim=-1)

        return z_out, log_det


class FlowPrior(nn.Module):

    def __init__(self, dim=2, n_layers=6):

        super().__init__()

        self.dim = dim

        masks = []

        for i in range(n_layers):

            m = torch.zeros(dim)
            m[:dim // 2] = 1.0

            if i % 2 == 1:
                m = 1 - m

            masks.append(m)

        self.layers = nn.ModuleList(
            [CouplingLayer(dim, m) for m in masks]
        )

        self.base = td.Independent(
            td.Normal(torch.zeros(dim), torch.ones(dim)), 1
        )

    def log_prob(self, z):

        log_det_total = torch.zeros(z.shape[0], device=z.device)

        u = z

        for layer in reversed(self.layers):

            mask = layer.mask.to(z.device)

            u_masked = u * mask

            s = layer.scale_net(u_masked) * (1 - mask)
            t = layer.shift_net(u_masked) * (1 - mask)

            u = u_masked + (1 - mask) * ((u - t) * torch.exp(-s))

            log_det_total -= (s * (1 - mask)).sum(dim=-1)

        return self.base.log_prob(u) + log_det_total

    def sample(self, n):

        u = self.base.sample((n,)).to(next(self.parameters()).device)

        z = u

        for layer in self.layers:

            mask = layer.mask

            z_masked = z * mask

            s = layer.scale_net(z_masked) * (1 - mask)
            t = layer.shift_net(z_masked) * (1 - mask)

            z = z_masked + (1 - mask) * (z * torch.exp(s) + t)

        return z


# ===============================
# Encoder / Decoder
# ===============================

class GaussianEncoder(nn.Module):

    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):

        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)

        log_std = torch.clamp(log_std, -10.0, 10.0)

        return td.Independent(
            td.Normal(mean, torch.exp(log_std)), 1
        )


class BernoulliDecoder(nn.Module):

    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):

        logits = self.decoder_net(z)

        return td.Independent(
            td.Bernoulli(logits=logits), 2
        )


# ===============================
# VAE
# ===============================

class VAE(nn.Module):

    def __init__(self, prior, decoder, encoder):

        super().__init__()

        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo_loss(self, x, beta=1.0):

        q = self.encoder(x)

        z = q.rsample()

        log_px = self.decoder(z).log_prob(x)

        kl = q.log_prob(z) - self.prior.log_prob(z)

        elbo = log_px - beta * kl

        loss = -torch.mean(elbo)

        return loss, torch.mean(elbo)


# ===============================
# Plotting
# ===============================

def plot_flow_contours(prior, model, loader, lim=20):

    prior.eval()
    model.eval()

    xs = torch.linspace(-lim, lim, 200)
    ys = torch.linspace(-lim, lim, 200)

    X, Y = torch.meshgrid(xs, ys, indexing="xy")

    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

    with torch.no_grad():
        logp = prior.log_prob(pts).reshape(200, 200).cpu()

    zs, labs = [], []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            zs.append(model.encoder(x).base_dist.loc.cpu())
            labs.append(y)

            if len(torch.cat(zs)) >= 3000:
                break

    Z = torch.cat(zs)[:3000].numpy()
    L = torch.cat(labs)[:3000].numpy()

    plt.figure(figsize=(7,6))

    plt.contour(X.numpy(), Y.numpy(), logp.numpy(), levels=15)

    plt.scatter(Z[:,0], Z[:,1], c=L, cmap="tab10", s=10, alpha=0.6)

    plt.title("Flow Prior & Aggregate Posterior")

    plt.savefig("figs/flow_contour.png")


def show_reconstructions(model, loader, n=10):

    model.eval()

    x, _ = next(iter(loader))

    x = x[:n].to(device)

    with torch.no_grad():

        z = model.encoder(x).rsample()

        x_hat = (model.decoder(z).mean >= 0.5).float().cpu()

    plt.figure(figsize=(2*n,3))

    for i in range(n):

        plt.subplot(2,n,i+1)
        plt.imshow(x[i].cpu(), cmap="gray")
        plt.axis("off")

        plt.subplot(2,n,n+i+1)
        plt.imshow(x_hat[i], cmap="gray")
        plt.axis("off")

    plt.savefig("figs/flow_reconstruction.png")


# ===============================
# Training
# ===============================

def train():

    scores = []

    for run in range(N_RUNS):

        print(f"Run {run+1}/{N_RUNS}")

        fp = FlowPrior(dim=M, n_layers=6).to(device)

        enc = GaussianEncoder(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(784,512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512, M*2)
            )
        ).to(device)

        dec = BernoulliDecoder(
            nn.Sequential(
                nn.Linear(M,512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512,784),
                nn.Unflatten(-1,(28,28))
            )
        ).to(device)

        model = VAE(fp, dec, enc).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()

        for epoch in range(EPOCHS):

            beta = min(1.0, (epoch+1)/15)

            for x,_ in train_loader:

                x = x.to(device)

                optimizer.zero_grad()

                loss,_ = model.elbo_loss(x, beta)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

                optimizer.step()

        # evaluation

        model.eval()

        run_elbos = []

        with torch.no_grad():

            for x,_ in test_loader:

                x = x.to(device)

                _,elbo = model.elbo_loss(x)

                run_elbos.append(elbo.item())

        score = np.mean(run_elbos)

        scores.append(score)

        print("Test ELBO:", score)

    print("\nFinal ELBO:", np.mean(scores), "±", np.std(scores))

    torch.save(model.state_dict(), "models/VAE_FlowPrior.pt")

    plot_flow_contours(fp, model, test_loader)

    show_reconstructions(model, test_loader)


# ===============================
# Main
# ===============================

if __name__ == "__main__":
    train()