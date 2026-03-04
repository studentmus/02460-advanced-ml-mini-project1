import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ddpm_models import BetaVAE, LatentMLP, DDPM

def main():
    parser = argparse.ArgumentParser(description="Plot Latent Space Distributions")
    parser.add_argument("--beta", type=float, default=1e-6, help="Beta value of the model to load")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches for the scatter plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Loading models for Beta: {args.beta}")

    # MUST be 2 for this specific contour visualization!
    latent_dim = 2 
    
    b_vae = BetaVAE(latent_dim=latent_dim).to(device)
    latent_net = LatentMLP(latent_dim=latent_dim).to(device)
    latent_ddpm = DDPM(network=latent_net, T=1000).to(device)

    # Load weights
    try:
        b_vae.load_state_dict(torch.load(f"weights_100vae50ddpm/b_vae_beta_{args.beta}.pth", map_location=device))
        latent_ddpm.load_state_dict(torch.load(f"weights_100vae50ddpm/latent_ddpm_beta_{args.beta}.pth", map_location=device))
    except FileNotFoundError:
        print("Weights not found. Please run your training script first.")
        return

    b_vae.eval()
    latent_ddpm.eval()

    # Load MNIST Dataset to get the Aggregate Posterior
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)), 
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    print("Collecting aggregate posterior samples...")
    zs = []
    ys = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            mu, _ = b_vae.encode(x) # Grab the deterministic mean to avoid noise puddle
            
            zs.append(mu.cpu())
            ys.append(y)

            if i >= args.num_batches:
                break
                
        print("Sampling from Latent DDPM...")
        # Sample enough points to make a smooth KDE plot
        ddpm_tensor = latent_ddpm.sample((2000, latent_dim))
        ddpm_samples = ddpm_tensor.cpu().numpy()

    Z = torch.cat(zs, dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy()

    # -------- Plotting --------
    print("Generating plot...")
    plt.figure(figsize=(9, 8), dpi=300)

    # 1. Standard Gaussian density (The Beta-VAE Prior)
    x_grid = np.linspace(-4, 4, 1000)
    y_grid = np.linspace(-4, 4, 1000)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    
    Z_density = stats.multivariate_normal(mean=[0, 0], cov=[1, 1]).pdf(np.dstack((X_mesh, Y_mesh)))
    plt.contour(X_mesh, Y_mesh, Z_density, levels=10, colors="black", linewidths=0.8, linestyles="dashed")

    # 2. Scatter points colored by digit label (The Aggregate Posterior)
    for digit in range(10):
        idx = (Y == digit)
        plt.scatter(Z[idx, 0], Z[idx, 1],
                    s=15, alpha=0.6, linewidth=0.3, edgecolors="black", label=f"Digit {digit}")

    # 3. Latent DDPM Distribution (The Learned Prior)
    sns.kdeplot(x=ddpm_samples[:, 0], y=ddpm_samples[:, 1], 
                levels=8, color="red", linewidths=2, linestyles="solid")

    # Legend Cleanup
    plt.plot([], [], color='black', linestyle='dashed', linewidth=1.5, label='Standard Prior')
    plt.plot([], [], color='red', linestyle='solid', linewidth=2.5, label='Latent DDPM')
    plt.legend(markerscale=1.5, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlabel(r"$z_0$")
    plt.ylabel(r"$z_1$")
    plt.title(f"Latent Space Distributions (Beta={args.beta})")
    plt.axis("equal")

    # --- CROPPING THE PLOT ---
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    plt.tight_layout() 

    # Save logic
    output_dir = "plots_100vae50ddpm"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"latent_distributions_beta_{args.beta}.png")
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Success! Saved latent space plot to {filename}")

if __name__ == "__main__":
    main()