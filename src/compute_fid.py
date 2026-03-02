import argparse
import time

import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.fid import compute_fid
# from vae_bernoulli_main import VAE, BernoulliDecoder, GaussianPrior, GaussianEncoder
from vae_bernoulli_mog import MoGPrior, VAE, BernoulliDecoder, GaussianEncoder


def main():
    parser = argparse.ArgumentParser(description="Evaluate FID for Generative Models")
    parser.add_argument("--model", type=str, choices=["ddpm", "latent_ddpm", "vae"], required=True,
                        help="Which model to evaluate")
    parser.add_argument("--beta", type=float, default=1e-6,
                        help="Beta value (only used if model is latent_ddpm or vae)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to generate for FID calculation (default: 1000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Evaluating Model: {args.model}")

    # 1. Load Real Images
    # The classifier expects shapes (N, 1, 28, 28) and values in [-1, 1]

    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
    #                                 transforms.Lambda(lambda x: (x - 0.5) * 2.0),
    #                                 transforms.Lambda(lambda x: x.flatten())]
    #                                )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # We load num_samples of real images to compare against
    dataloader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)

    # 2. Generate Fake Images based on the chosen model
    generated_images = None

    with torch.no_grad():

        # M = 2
        # prior = GaussianPrior(M)
        #
        # # Define encoder and decoder networks
        # encoder_net = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(784, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, M * 2),
        # )
        #
        # decoder_net = nn.Sequential(
        #     nn.Linear(M, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 784),
        #     nn.Unflatten(-1, (28, 28))
        # )
        #
        # # Define VAE model
        # decoder = BernoulliDecoder(decoder_net)
        # encoder = GaussianEncoder(encoder_net)
        # model = VAE(prior, decoder, encoder).to(device)
        #
        # model.load_state_dict(torch.load("model_base_2.pt", map_location=torch.device(device)))
        #
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # start_time = time.time()
        # z = model.prior.forward()
        # z_samples = z.sample((args.num_samples,))
        # dist = model.decoder(z_samples)
        # generated_original = dist.mean.unsqueeze(1)
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # end_time = time.time()
        #
        # total_time = end_time - start_time
        # samples_per_second = args.num_samples / total_time
        #
        # print(f"Sampling finished in {total_time:.2f} seconds.")
        # print(f"Speed: {samples_per_second:.2f} samples/second.")
        # generated_images = generated_original * 2 - 1
        # save_image(generated_images[:10].view(10, 1, 28, 28), "sample_std_gauss_fid.png")

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
            nn.Linear(512, 784),
            nn.Unflatten(-1, (28, 28))
        )

        encoder = GaussianEncoder(encoder_net)
        decoder = BernoulliDecoder(decoder_net)
        model = VAE(prior, decoder, encoder).to(device)
        model.load_state_dict(torch.load("models/VAE_MoG.pt", map_location=torch.device(device)))

        fid_scores = []
        total_times = []
        samples_per_seconds = []
        for i in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            z = model.prior.forward()
            z_samples = z.sample((args.num_samples,))
            dist = model.decoder(z_samples)
            generated_original = dist.mean.unsqueeze(1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            total_time = end_time - start_time
            total_times.append(total_time)
            samples_per_second = args.num_samples / total_time
            samples_per_seconds.append(samples_per_second)

            # print(f"Sampling finished in {total_time:.2f} seconds.")
            # print(f"Speed: {samples_per_second:.2f} samples/second.")
            generated_images = generated_original * 2 - 1
            save_image(generated_images[:10].view(10, 1, 28, 28), "sample_mog_fid.png")

    # Ensure generated images are clamped to [-1, 1] just in case
    # generated_images = torch.clamp(generated_images, -1.0, 1.0)

            # 3. Compute FID
            # print("Computing Fréchet Inception Distance...")

            fid_score = compute_fid(
                x_real=real_images,
                x_gen=generated_images,
                device=str(device),
                classifier_ckpt="utils/mnist_classifier.pth"
            )
            fid_scores.append(fid_score)

            # print("-" * 30)
            # print(f"Model: {args.model.upper()}")
            # if args.model in ["latent_ddpm", "vae"]:
            #     print(f"Beta: {args.beta}")
            # print(f"FID Score: {fid_score:.4f}")
            # print("-" * 30)

    fid_scores = np.array(fid_scores)
    total_times = np.array(total_times)
    samples_per_seconds = np.array(samples_per_seconds)

    mean = fid_scores.mean()
    std = fid_scores.std(ddof=1)  # sample standard deviation
    print(f"The mean and standard deviation of fid score among {10}: {mean} and {std}")

    mean = total_times.mean()
    std = total_times.std(ddof=1)  # sample standard deviation
    print(f"The mean and standard deviation of total times score among {10}: {mean} and {std}")

    mean = samples_per_seconds.mean()
    std = samples_per_seconds.std(ddof=1)  # sample standard deviation
    print(f"The mean and standard deviation of samples per second score among {10}: {mean} and {std}")
if __name__ == "__main__":
    main()