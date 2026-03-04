# 02460 Advanced ML – Mini-project 1

Deep generative models on (binarized) MNIST:
- VAEs with Gaussian, MoG and flow-based priors
- DDPM and latent DDPM
- FID evaluation using provided `fid.py` and `mnist_classifier.pth`

Structure:
- `src/vae/` – VAE and DDPM models
- `src/utils/` – FID, data loading, common helpers
- `weights/` – saved model checkpoints
- `samples/` - saved model samples (contour plots, reconstructed samples, etc.)
- `reports/` – LaTeX report (`template.tex`, figures, tables)

# How to run the code:
First, install all the required packages by:

```pip install -r requirements.txt```

## VAE model

### Basic usage

```bash
python src/VAE.py
```

This trains a VAE with the **default Gaussian prior**.

---

### Choosing the Prior
Standard Gaussian Prior

```bash
python src/VAE.py --prior gaussian
```

---
Mixture of Gaussians Prior

```bash
python src/VAE.py --prior mog
```

Optional number of mixture components:

```bash
python src/VAE.py --prior mog --k 10
```

---
Flow-Based Prior

```bash
python src/VAE.py --prior flow
```

---

### Training Parameters

Example:

```bash
python src/VAE.py \
    --prior flow \
    --latent_dim 2 \
    --epochs 100 \
    --batch_size 256 \
    --runs 10 \
    --samples 10
```

| Argument | Description | Default |
|--------|--------|--------|
| `--prior` | Prior distribution (`gaussian`, `mog`, `flow`) | `gaussian` |
| `--latent_dim` | Latent space dimension | `2` |
| `--k` | Number of Gaussians (for MoG prior) | `10` |
| `--epochs` | Training epochs | `100` |
| `--batch_size` | Batch size | `256` |
| `--runs` | Number of independent training runs | `10` |
| `--samples` | Number of reconstruction samples shown | `10` |


## DDPM


This project contains two scripts for training and sampling a **Denoising Diffusion Probabilistic Model (DDPM)** on the MNIST dataset.

Files:

```
train_ddpm_unet.py
sample_ddpm_unet.py
```

The model uses a **U-Net architecture** and trains on flattened MNIST images.

---

### Training the DDPM

Run the training script:

```bash
python src/train_ddpm_unet.py
```
Training uses:

- **batch size:** 128  
- **optimizer:** Adam  
- **learning rate:** 2e-4  
- **diffusion steps:** 1000

The trained model weights will be saved to:

```
weights/trained_ddpm_mnist_100epochs.pth
```

---

### Custom Training Settings
Arguments:

| Argument | Description | Default |
|--------|--------|--------|
| `--epochs` | Number of training epochs | 100 |
| `--weights_dir` | Directory where model weights are saved | `weights` |

---

### Sampling Images

After training finishes, generate samples using:

```bash
python src/sample_ddpm_unet.py
```

This script:

- loads the trained DDPM weights
- generates images via the reverse diffusion process
- saves generated samples to an image file

The sampling script loads weights from:

```
weights/trained_ddpm_mnist_100epochs.pth
```

and produces output:

```
ddpm_samples_100epochs.png
```

Arguments:

| Argument | Description | Default |
|--------|--------|--------|
| `--num_samples` | Number of generated images | 4 |
| `--weights_dir` | Directory containing trained weights | `weights` |

---

### Example Workflow

Train the model:

```bash
python src/train_ddpm_unet.py
```

Generate samples:

```bash
python src/sample_ddpm_unet.py --num_samples 8
```

This produces an output file:

```
ddpm_samples_100epochs.png
```

containing the generated MNIST digits.

---

## Latent DDPM with Beta-VAE

Training happens in **two stages**:

1. Train a **Beta-VAE**
2. Train a **DDPM in the latent space**

The sampling script then:

1. Samples latent vectors using the trained DDPM
2. Decodes them using the trained Beta-VAE

---

### Training the Latent DDPM

Run:

```bash
python src/train_latent_ddpm.py
```

This will:

1. Train a **Beta-VAE**
2. Freeze the VAE encoder
3. Train a **DDPM on latent variables**

The script automatically downloads MNIST and saves model weights.

Training pipeline implemented in the script .

---

### Training Arguments

| Argument | Description | Default |
|--------|--------|--------|
| `--beta` | Beta value for the Beta-VAE | `1e-6` |
| `--vae_epochs` | Number of epochs for VAE training | `100` |
| `--ddpm_epochs` | Number of epochs for latent DDPM training | `50` |
| `--weights_dir` | Directory to save model weights | `weights` |

---

### Saved Models

After training, two models are saved:

```
weights/
├── b_vae_beta_<beta>.pth
└── latent_ddpm_beta_<beta>.pth
```

Example:

```
weights/b_vae_beta_1e-06.pth
weights/latent_ddpm_beta_1e-06.pth
```

---

### Sampling Images from Latent DDPM

After training finishes, generate samples using:

```bash
python src/sample_latent_ddpm.py
```
Arguments:

| Argument | Description | Default |
|--------|--------|--------|
| `--beta` | Beta value used during training | `1e-6` |
| `--num_samples` | Number of generated images | `4` |
| `--weights_dir` | Directory containing trained weights | `weights` |
| `--output_dir` | Directory to save generated images | `samples` |

# Compute FID score
Run
```bash
python src/utils/calculate_fid_speed.py
```
to compute the fid scores of three models: DDPM, DDPM with $\beta-$VAE, and $VAE$.

**Arguments:**

| Argument | Type | Description | Default |
|--------|--------|--------|--------|
| `--model` | `str` | Generative model to evaluate. Options: `ddpm`, `latent_ddpm`, `vae`. | **Required** |
| `--beta` | `float` | Beta value used for the Beta-VAE model. Only used when `model` is `latent_ddpm` or `vae`. | `1e-6` |
| `--num_samples` | `int` | Number of generated samples used to compute the FID score. | `2000` |
| `--weights_dir` | `str` | Directory containing the trained model weights. | `weights` |

Note that you should have the file ``fid.py`` and the model checkpoint ``mnist_classifier.pth`` in the same 
folder as ``calculate_fid_speed.py``. Otherwise, the code will fail to run.