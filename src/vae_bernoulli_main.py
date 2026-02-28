import torch
import torch.nn as nn
import numpy as np
import torch.distributions as td
import torch.utils.data
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


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
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def get_aggregate_posterior_samples(model, device, max_batches=None):
    model.eval()
    zs = []

    data_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                               transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(
                                                   lambda x: (thresshold < x).float().squeeze())])),
                                batch_size=32, shuffle=True)

    with torch.no_grad():
        for i, x in enumerate(data_loader):
            x = x[0].to(device)
            q = model.encoder(x)
            z = q.rsample()
            zs.append(z.cpu())

            if max_batches is not None and i >= max_batches:
                break

    return torch.cat(zs, dim=0)

def get_prior_samples(model, n_samples = 32 * 313):
    with torch.no_grad():
        z = model.prior().sample((n_samples,))
    return z.cpu()

def compute_moments(z):
    mean = z.mean(dim=0)
    cov = torch.cov(z.T)
    return mean, cov

def evaluate_elbo(model, device):
    model.eval()
    total_elbo = 0

    data_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                    batch_size=32 * 313, shuffle=True)
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            total_elbo += model.elbo(x).item()

    return total_elbo

def plot_latent_with_labels(model, data_loader, device, filename="latent_contour.png", max_batches=None):
    model.eval()

    zs = []
    ys = []

    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)

            q = model.encoder(x)
            z = q.rsample()          # use mean instead of rsample() for cleaner structure

            zs.append(z.cpu())
            ys.append(y)

            if max_batches is not None and i >= max_batches:
                break

    Z = torch.cat(zs, dim=0)        # (N, 2)
    Y = torch.cat(ys, dim=0)        # (N,)

    assert Z.shape[1] == 2, "Latent dimension must be 2 for this plot."

    Z_np = Z.numpy()
    Y_np = Y.numpy()

    # -------- Plot --------
    plt.figure(figsize=(8, 8), dpi=600)

    x = np.linspace(-4, 4, 1000)
    y = np.linspace(-4, 4, 1000)
    X, Y_grid = np.meshgrid(x, y)

    # Standard Gaussian density
    # Change this to match your prior distribution
    ####################################################################################################################
    Z_density = stats.multivariate_normal(mean = [0, 0], cov = [1, 1]).pdf(np.dstack((X, Y_grid)))
    plt.contour(X, Y_grid, Z_density, levels=10, colors="black", linewidths = 0.5, linestyles="dashed")
    ####################################################################################################################

    # Scatter points colored by digit label
    for digit in range(10):
        idx = (Y_np == digit)
        plt.scatter(Z_np[idx, 0], Z_np[idx, 1],
                    s=10, alpha=0.6, linewidth=0.3, edgecolors="black", label=f"digit={digit}")

    plt.legend(markerscale=2, fontsize=8)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent Space with Prior Contours")
    plt.axis("equal")
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument("--no-trains", type=int, default = 10, metavar="N", help="Number of training you want to runs, default is %(default)")

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float())])),
                                        batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float())])),
                                        batch_size=args.batch_size, shuffle=True)

    # Choose mode to run
    if args.mode == 'train':
        no_trains = args.no_trains
        elbos = []

        # Define prior distribution
        M = args.latent_dim
        prior = GaussianPrior(M)

        for i in range(no_trains):

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
                nn.Linear(512, 784),
                nn.Unflatten(-1, (28, 28))
            )

            # Define VAE model
            decoder = BernoulliDecoder(decoder_net)
            encoder = GaussianEncoder(encoder_net)
            model = VAE(prior, decoder, encoder).to(device)

            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Train model
            train(model, optimizer, mnist_train_loader, args.epochs, args.device)
            elbo = evaluate_elbo(model, args.device)
            elbos.append(elbo)
            print(f"ELBO of the training {i}th is {elbo}")
            print("-------------------------------------------------------------------------------------------------")

            # Save model
            if (no_trains - i) == 1:
                torch.save(model.state_dict(), args.model)
        elbos = np.array(elbos)

        mean = elbos.mean()
        std = elbos.std(ddof=1)  # sample standard deviation
        print(f"The mean and standard deviation of elbos among {no_trains}: {mean} and {std}")

    elif args.mode == 'sample':
        # Define prior distribution
        M = args.latent_dim
        prior = GaussianPrior(M)

        # Define encoder and decoder networks
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
            nn.Linear(512, 784),
            nn.Unflatten(-1, (28, 28))
        )

        # Define VAE model
        decoder = BernoulliDecoder(decoder_net)
        encoder = GaussianEncoder(encoder_net)
        model = VAE(prior, decoder, encoder).to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)
            plot_latent_with_labels(model, mnist_test_loader, args.device, max_batches=32)
