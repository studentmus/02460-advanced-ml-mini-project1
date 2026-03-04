import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from ddpm_models import Unet, DDPM

def main():
    os.makedirs("weights", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    print("Loading Standard MNIST dataset...")
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist_train, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    unet = Unet().to(device)
    ddpm = DDPM(network=unet, T=1000).to(device)

    optimizer = optim.Adam(ddpm.parameters(), lr=2e-4)
    epochs = 100 
    
    ddpm.train()
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, _ in progress_bar:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            loss = ddpm.loss(batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")

    save_path = f"weights/trained_ddpm_mnist_{epochs}epochs.pth"
    torch.save(ddpm.state_dict(), save_path)
    print(f"Training complete! Model weights saved to {save_path}")

if __name__ == "__main__":
    main()