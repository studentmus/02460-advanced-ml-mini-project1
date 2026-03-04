import os
import time
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
    parser.add_argument("--num_samples", type=int, default=2000, 
                        help="Number of samples to generate for FID calculation (default: 2000)")
    parser.add_argument("--weights_dir", type=str, default="weights", 
                        help="Directory containing the model weights")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Evaluating Model: {args.model}")
    print(f"Loading weights from directory: {args.weights_dir}")

    # 1. Load Real Images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)

    # Variables to track time
    start_time = 0
    end_time = 0

    # 2. Generate Fake Images based on the chosen model
    generated_images = None

    with torch.no_grad():
        if args.model == "ddpm":
            unet = Unet().to(device)
            ddpm = DDPM(network=unet, T=1000).to(device)
            
            ddpm_path = os.path.join(args.weights_dir, "trained_ddpm_mnist_100epochs.pth")
            ddpm.load_state_dict(torch.load(ddpm_path, map_location=device, weights_only=True))
            ddpm.eval()
            
            print(f"Generating {args.num_samples} samples from standard DDPM...")
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start_time = time.time()
            
            samples_flat = ddpm.sample((args.num_samples, 784))
            generated_images = samples_flat.view(-1, 1, 28, 28)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end_time = time.time()

        elif args.model == "latent_ddpm":
            latent_dim = 20
            b_vae = BetaVAE(latent_dim=latent_dim).to(device)
            latent_net = LatentMLP(latent_dim=latent_dim).to(device)
            latent_ddpm = DDPM(network=latent_net, T=1000).to(device)
            
            vae_path = os.path.join(args.weights_dir, f"b_vae_beta_{args.beta}.pth")
            latent_ddpm_path = os.path.join(args.weights_dir, f"latent_ddpm_beta_{args.beta}.pth")
            
            b_vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
            latent_ddpm.load_state_dict(torch.load(latent_ddpm_path, map_location=device, weights_only=True))
            b_vae.eval()
            latent_ddpm.eval()

            print(f"Generating {args.num_samples} samples from Latent DDPM (beta={args.beta})...")
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start_time = time.time()
            
            sampled_z = latent_ddpm.sample((args.num_samples, latent_dim))
            samples_flat = b_vae.decode(sampled_z)
            generated_images = samples_flat.view(-1, 1, 28, 28)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end_time = time.time()

        elif args.model == "vae":
            latent_dim = 20
            b_vae = BetaVAE(latent_dim=latent_dim).to(device)
            
            vae_path = os.path.join(args.weights_dir, f"b_vae_beta_{args.beta}.pth")
            
            b_vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
            b_vae.eval()

            print(f"Generating {args.num_samples} samples directly from VAE prior (beta={args.beta})...")
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start_time = time.time()
            
            sampled_z = torch.randn(args.num_samples, latent_dim).to(device)
            samples_flat = b_vae.decode(sampled_z)
            generated_images = samples_flat.view(-1, 1, 28, 28)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end_time = time.time()

    # Calculate sampling timings
    sampling_time = end_time - start_time
    samples_per_second = args.num_samples / sampling_time

    # Ensure generated images are clamped to [-1, 1]
    generated_images = torch.clamp(generated_images, -1.0, 1.0)

    # 3. Compute FID
    print("Computing Fréchet Inception Distance...")


    fid_score = compute_fid(
        x_real=real_images, 
        x_gen=generated_images, 
        device=str(device), 
        classifier_ckpt="utils/mnist_classifier.pth"
    )

    print("-" * 40)
    print(f"Model: {args.model.upper()}")
    if args.model in ["latent_ddpm", "vae"]:
        print(f"Beta: {args.beta}")
    print(f"FID Score: {fid_score:.4f}")
    print(f"Total Sampling Time: {sampling_time:.4f} seconds")
    print(f"Speed: {samples_per_second:.2f} samples/second")
    print("-" * 40)

if __name__ == "__main__":
    main()