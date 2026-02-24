# 02460 Advanced ML – Mini-project 1

Deep generative models on (binarized) MNIST:
- VAEs with Gaussian, MoG and flow-based priors
- DDPM and latent DDPM
- FID evaluation using provided `fid.py` and `mnist_classifier.pth`

Structure:
- `src/vae/` – VAE models, priors, training scripts
- `src/ddpm/` – DDPM, latent DDPM, training & sampling
- `src/utils/` – FID, data loading, common helpers
- `notebooks/` – quick experiments & visualization
- `experiments/` – saved configs, logs, metrics
- `reports/` – LaTeX report (`template.tex`, figures, tables)
