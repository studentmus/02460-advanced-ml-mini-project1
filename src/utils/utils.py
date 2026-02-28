import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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