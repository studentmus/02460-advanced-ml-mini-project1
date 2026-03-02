import torch
import matplotlib
matplotlib.use('Agg') # Safe for SSH!
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Imports your specific model classes
from ddpm_models import BetaVAE, LatentMLP, DDPM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    beta = 1e-6
    latent_dim = 20 # Matches your BetaVAE default
    num_samples = 2000

    # 1. LOAD YOUR TRAINED MODELS
    print(f"Loading Beta-VAE and Latent DDPM (beta={beta})...")
    vae = BetaVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(f"weights/b_vae_beta_{beta}.pth", map_location=device))
    vae.eval()

    latent_net = LatentMLP(latent_dim=latent_dim).to(device)
    latent_ddpm = DDPM(network=latent_net, T=1000).to(device)
    latent_ddpm.load_state_dict(torch.load(f"weights/latent_ddpm_beta_{beta}.pth", map_location=device))
    latent_ddpm.eval()

    # 2. SETUP STANDARD MNIST DATALOADER
    # Flattens to 784, matching your BetaVAE input_dim
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1)) 
    ])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 3. EXTRACT AGGREGATE POSTERIOR q(z|x)
    print("Extracting Aggregate Posterior from Beta-VAE...")
    z_posterior = []
    collected = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            # Uses your specific encode and reparameterize methods
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu, logvar)
            
            z_posterior.append(z.cpu())
            collected += x.size(0)
            if collected >= num_samples:
                break
                
    z_posterior = torch.cat(z_posterior, dim=0)[:num_samples].numpy()

    # 4. GENERATE FROM LATENT DDPM p_ddpm(z)
    print("Generating samples from Latent DDPM...")
    with torch.no_grad():
        # CRITICAL FIX: Your DDPM sample method takes a 'shape' tuple!
        z_ddpm = latent_ddpm.sample(shape=(num_samples, latent_dim)).cpu().numpy()

    # 5. SAMPLE FROM PRIOR p(z)
    print("Sampling from Beta-VAE Prior...")
    z_prior = torch.randn(num_samples, latent_dim).numpy()

    # 6. APPLY PCA (Squash 20D to 2D)
    print(f"Applying PCA to squash {latent_dim}D down to 2D...")
    pca = PCA(n_components=2)
    # Fit PCA ONLY on the posterior to define the coordinate space properly
    post_2d = pca.fit_transform(z_posterior)
    ddpm_2d = pca.transform(z_ddpm)
    prior_2d = pca.transform(z_prior)

    # 7. CREATE THE PLOT
    print("Plotting distributions...")
    plt.figure(figsize=(9, 7))
    
    # Plot Prior (Gray)
    plt.scatter(prior_2d[:, 0], prior_2d[:, 1], alpha=0.3, label='$\\beta$-VAE Prior $p(z)$', color='gray', s=10)
    
    # Plot Aggregate Posterior (Blue)
    plt.scatter(post_2d[:, 0], post_2d[:, 1], alpha=0.4, label='Aggregate Posterior $q(z|x)$', color='blue', s=10)
    
    # Plot DDPM Generated Latents (Orange)
    plt.scatter(ddpm_2d[:, 0], ddpm_2d[:, 1], alpha=0.5, label='Latent DDPM $p_{ddpm}(z)$', color='darkorange', s=10)

    # Aesthetics
    plt.title(f"Distribution Comparison ($\\beta={beta}$)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"beta_vae_vs_ddpm_beta_{beta}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Done! Check your folder for {filename}")

if __name__ == "__main__":
    main()