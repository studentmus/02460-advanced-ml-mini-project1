import os
import time
import argparse
import torch
import matplotlib.pyplot as plt
from ddpm_models import Unet, DDPM

def main():
    parser = argparse.ArgumentParser(description="Sample from trained DDPM")
    parser.add_argument("--num_samples", type=int, default=4, 
                        help="Number of images to generate (default: 4 for the project requirement)")
    parser.add_argument("--weights_dir", type=str, default="weights", 
                        help="Directory containing the model weights")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Looking for weights in directory: {args.weights_dir}")

    unet = Unet().to(device)
    ddpm = DDPM(network=unet, T=1000).to(device)

    weights_path = os.path.join(args.weights_dir, "trained_ddpm_mnist_100epochs.pth")
    
    try:
        ddpm.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {weights_path}. Please run train_ddpm.py first.")
        return

    ddpm.eval()

    num_samples = args.num_samples
    sample_shape = (num_samples, 784)

    print(f"Generating {num_samples} samples... (this will take a moment)")
    
    # Synchronize and start timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        samples = ddpm.sample(sample_shape)
    
    # Synchronize and stop timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    samples_per_second = num_samples / total_time
    
    print(f"Sampling finished in {total_time:.2f} seconds.")
    print(f"Speed: {samples_per_second:.2f} samples/second.")

    samples_img = samples.cpu().view(-1, 28, 28).numpy()
    
    # Normalizing back to [0, 1] for matplotlib plotting
    samples_img = (samples_img + 1.0) / 2.0

    # Dynamically scale the plot width based on num_samples
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    
    # Handle the edge case if you only request 1 sample (axes becomes a single object, not an array)
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i].imshow(samples_img[i].clip(0, 1), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Sample {i+1}")

    plt.suptitle("Trained DDPM Samples")
    plt.tight_layout()
    
    filename = f"ddpm_samples_100epochs.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved generated images to {filename}")
    
if __name__ == "__main__":
    main()