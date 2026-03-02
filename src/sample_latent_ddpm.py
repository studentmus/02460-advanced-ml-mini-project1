import time
import argparse
import torch
import matplotlib.pyplot as plt
from ddpm_models import BetaVAE, LatentMLP, DDPM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=1e-6, help="Beta value of the model to load")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of images to generate (default: 4)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Loading models for Beta: {args.beta}")

    latent_dim = 20
    b_vae = BetaVAE(latent_dim=latent_dim).to(device)
    latent_net = LatentMLP(latent_dim=latent_dim).to(device)
    latent_ddpm = DDPM(network=latent_net, T=1000).to(device)

    # Load weights
    try:
        b_vae.load_state_dict(torch.load(f"weights/b_vae_beta_{args.beta}.pth", map_location=device))
        latent_ddpm.load_state_dict(torch.load(f"weights/latent_ddpm_beta_{args.beta}.pth", map_location=device))
    except FileNotFoundError:
        print("Weights not found. Please run train_latent_ddpm.py first with the matching --beta argument.")
        return

    b_vae.eval()
    latent_ddpm.eval()

    num_samples = args.num_samples
    
    print(f"Sampling {num_samples} images from Latent DDPM... (timing started)")
    
    # 1. Synchronize and start timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        # Sample z from DDPM 
        sampled_z = latent_ddpm.sample((num_samples, latent_dim))
        
        # Decode z into images using VAE
        generated_images = b_vae.decode(sampled_z)

    # 2. Synchronize and stop timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    
    print(f"Sampling finished in {total_time:.4f} seconds.")
    print(f"Speed: {num_samples / total_time:.2f} samples/second.")

    imgs = generated_images.cpu().view(-1, 28, 28).numpy()
    imgs = (imgs + 1.0) / 2.0 # Scale back to [0, 1] for plotting

    # Dynamically adjust figure size based on the number of samples
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i].imshow(imgs[i].clip(0, 1), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Latent Sample {i+1}")
        
    plt.suptitle(f"Latent DDPM (Beta={args.beta})")
    plt.tight_layout()
    
    filename = f"latent_ddpm_samples_beta_{args.beta}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved generated images to {filename}")

if __name__ == "__main__":
    main()