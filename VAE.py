import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class BasePrior(nn.Module):

    def log_prob(self, z):
        raise NotImplementedError

    def sample(self, n):
        raise NotImplementedError

class GaussianPrior(BasePrior):

    def __init__(self, dim):
        super().__init__()
        self.base = td.Independent(
            td.Normal(torch.zeros(dim), torch.ones(dim)), 1
        )

    def log_prob(self, z):
        return self.base.log_prob(z)

    def sample(self, n):
        return self.base.sample((n,))

class MoGPrior(BasePrior):

    def __init__(self, M, K):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(K, M) * 2.0)
        self.log_std = nn.Parameter(torch.zeros(K, M) - 0.5)
        self.logits = nn.Parameter(torch.zeros(K))

    def _dist(self):
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(
            td.Normal(self.mu, torch.exp(self.log_std)), 1
        )
        return td.MixtureSameFamily(mix, comp)

    def log_prob(self, z):
        return self._dist().log_prob(z)

    def sample(self, n):
        return self._dist().sample((n,))

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

class VAE(nn.Module):

    def __init__(self, prior, encoder, decoder):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, x):

        q = self.encoder(x)

        z = q.rsample()

        log_px = self.decoder(z).log_prob(x)

        kl = q.log_prob(z) - self.prior.log_prob(z)

        return torch.mean(log_px - kl)

    def forward(self, x):
        return -self.elbo(x)

def build_prior(name, dim, k=10):

    if name == "gaussian":
        return GaussianPrior(dim)

    elif name == "mog":
        return MoGPrior(dim, k)

    elif name == "flow":
        return FlowPrior(dim)

    else:
        raise ValueError("Unknown prior")

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        decoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

def main():

    import argparse
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()

    parser.add_argument("--prior",
                        type=str,
                        default="gaussian",
                        choices=["gaussian", "mog", "flow"])

    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--runs", type=int, default=10)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running VAE with prior:", args.prior)
    print("Device:", device)

    # -------------------------
    # Dataset
    # -------------------------

    threshold = 0.5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > threshold).float().squeeze())
    ])

    train_loader = DataLoader(
        datasets.MNIST("data/", train=True, download=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST("data/", train=False, download=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=False
    )

    # -------------------------
    # Networks
    # -------------------------

    M = args.latent_dim

    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M * 2)
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(decoder_net)

    # -------------------------
    # Training runs
    # -------------------------

    elbos = []

    for run in range(args.runs):

        print(f"\nRun {run+1}/{args.runs}")

        # ---- build prior ----
        if args.prior == "gaussian":
            prior = GaussianPrior(M)

        elif args.prior == "mog":
            prior = MoGPrior(M, args.k)

        elif args.prior == "flow":
            prior = FlowPrior(M)

        prior = prior.to(device)

        # ---- build model ----
        model = VAE(prior, decoder, encoder).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # ---- training ----
        model.train()

        for epoch in range(args.epochs):

            total_loss = 0

            for x, _ in train_loader:

                x = x.to(device)

                optimizer.zero_grad()

                loss = model(x)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}")

        # -------------------------
        # Evaluate ELBO
        # -------------------------

        model.eval()

        run_elbos = []

        with torch.no_grad():

            for x, _ in test_loader:

                x = x.to(device)

                elbo = model.elbo(x)

                run_elbos.append(elbo.item())

        score = np.mean(run_elbos)

        print("Test ELBO:", score)

        elbos.append(score)

    elbos = np.array(elbos)

    print("\n===================================")
    print("Final Results")
    print("Mean ELBO:", elbos.mean())
    print("Std ELBO :", elbos.std(ddof=1))
    print("===================================")

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), f"models/VAE_{args.prior}.pt")

if __name__ == "__main__":
    main()

