import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ddpm_models import Unet, DDPM, BetaVAE, LatentMLP
from utils.fid import compute_fid

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
        if args.model == "ddpm":
            unet = Unet().to(device)
            ddpm = DDPM(network=unet, T=1000).to(device)
            ddpm.load_state_dict(torch.load("weights/trained_ddpm_mnist.pth", map_location=device))
            ddpm.eval()
            
            print(f"Generating {args.num_samples} samples from standard DDPM...")
            samples_flat = ddpm.sample((args.num_samples, 784))
            generated_images = samples_flat.view(-1, 1, 28, 28)

        elif args.model == "latent_ddpm":
            latent_dim = 20
            b_vae = BetaVAE(latent_dim=latent_dim).to(device)
            latent_net = LatentMLP(latent_dim=latent_dim).to(device)
            latent_ddpm = DDPM(network=latent_net, T=1000).to(device)
            
            b_vae.load_state_dict(torch.load(f"weights/b_vae_beta_{args.beta}.pth", map_location=device))
            latent_ddpm.load_state_dict(torch.load(f"weights/latent_ddpm_beta_{args.beta}.pth", map_location=device))
            b_vae.eval()
            latent_ddpm.eval()

            print(f"Generating {args.num_samples} samples from Latent DDPM (beta={args.beta})...")
            sampled_z = latent_ddpm.sample((args.num_samples, latent_dim))
            samples_flat = b_vae.decode(sampled_z)
            generated_images = samples_flat.view(-1, 1, 28, 28)

        elif args.model == "vae":
            latent_dim = 20
            b_vae = BetaVAE(latent_dim=latent_dim).to(device)
            b_vae.load_state_dict(torch.load(f"weights/b_vae_beta_{args.beta}.pth", map_location=device))
            b_vae.eval()

            print(f"Generating {args.num_samples} samples directly from VAE prior (beta={args.beta})...")
            # Sample z directly from a standard normal prior N(0, I)
            sampled_z = torch.randn(args.num_samples, latent_dim).to(device)
            samples_flat = b_vae.decode(sampled_z)
            generated_images = samples_flat.view(-1, 1, 28, 28)

    # Ensure generated images are clamped to [-1, 1] just in case
    generated_images = torch.clamp(generated_images, -1.0, 1.0)

    # 3. Compute FID
    print("Computing Fréchet Inception Distance...")

    fid_score = compute_fid(
        x_real=real_images, 
        x_gen=generated_images, 
        device=str(device), 
        classifier_ckpt="utils/mnist_classifier.pth"
    )

    print("-" * 30)
    print(f"Model: {args.model.upper()}")
    if args.model in ["latent_ddpm", "vae"]:
        print(f"Beta: {args.beta}")
    print(f"FID Score: {fid_score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()