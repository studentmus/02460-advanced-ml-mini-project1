import os
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ddpm_models import BetaVAE, LatentMLP, DDPM
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Train Latent DDPM")
    parser.add_argument("--beta", type=float, default=1e-6, help="Beta value for VAE")
    parser.add_argument("--vae_epochs", type=int, default=100, help="Epochs to train VAE")
    parser.add_argument("--ddpm_epochs", type=int, default=50, help="Epochs to train Latent DDPM")
    args = parser.parse_args()

    os.makedirs("weights_100vae50ddpm", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Beta: {args.beta}")

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)), 
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataloader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)

    latent_dim = 2
    b_vae = BetaVAE(latent_dim=latent_dim).to(device)
    latent_net = LatentMLP(latent_dim=latent_dim).to(device)
    latent_ddpm = DDPM(network=latent_net, T=1000).to(device)

    # --- PHASE 1: Train Beta-VAE ---
    print(f"\n--- Training Beta-VAE (beta={args.beta}) ---")
    optimizer_vae = optim.Adam(b_vae.parameters(), lr=1e-3)
    b_vae.train()
    
    for epoch in range(args.vae_epochs):
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{args.vae_epochs}")
        
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            optimizer_vae.zero_grad()
            recon_x, mu, logvar = b_vae(batch_x)
            
            loss = b_vae.loss_function(recon_x, batch_x, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer_vae.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"VAE Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")

    # --- PHASE 2: Train Latent DDPM ---
    print("\n--- Training Latent DDPM ---")
    b_vae.eval() # Freeze VAE
    latent_ddpm.train()
    optimizer_ddpm = optim.Adam(latent_ddpm.parameters(), lr=1e-3)
    
    for epoch in range(args.ddpm_epochs):
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"DDPM Epoch {epoch+1}/{args.ddpm_epochs}")
        
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            with torch.no_grad():
                mu, logvar = b_vae.encode(batch_x)
                z = b_vae.reparameterize(mu, logvar)
            
            optimizer_ddpm.zero_grad()
            loss = latent_ddpm.loss(z)
            loss.backward()
            optimizer_ddpm.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"DDPM Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")

    # Save weights with beta value in filename
    torch.save(b_vae.state_dict(), f"weights_100vae50ddpm/b_vae_beta_{args.beta}.pth")
    torch.save(latent_ddpm.state_dict(), f"weights_100vae50ddpm/latent_ddpm_beta_{args.beta}.pth")
    print("Training complete and weights saved!")

if __name__ == "__main__":
    main()


