# Harappan Seal Restoration - Diffusion-Based Image Restoration

This project implements a comprehensive framework for image restoration using multiple deep learning architectures. It includes 7 state-of-the-art models trained on both grayscale and RGB images with polynomial fracture mask generation to simulate real damage patterns.

## Features

- **7 Model Architectures**: Baseline U-Net, Deep Residual U-Net, Context Encoder, Pix2Pix, RFA-Net, Partial Convolution U-Net, and Guided DDPM
- **Dual Color Modes**: Train and evaluate models on both grayscale and RGB images
- **Advanced Mask Generation**: Polynomial fracture masks that simulate realistic damage patterns
- **Comprehensive Metrics**: PSNR, SSIM, and LPIPS evaluation
- **Automatic Training & Evaluation**: End-to-end pipeline with visualization and reporting
- **Mixed Precision Training**: Uses AMP for faster training on GPU

## Requirements

- Python 3.8+
- PyTorch with CUDA support (GPU recommended)
- See `install_packages()` in the script for all dependencies

## Setup

### 1. Configure Paths

Edit the `Config` class in `ddpm.py` to point to your data:

```python
TRAIN_IMG_DIR = "./data/train"  # Path to your training images
BASE_OUTPUT = "./output"         # Path for outputs (checkpoints, results, figures)
```

Create the required directory structure:

```bash
mkdir -p ./data/train
mkdir -p ./output
```

### 2. Prepare Data

Place your training images in `./data/train/`. Supported formats: `.png`, `.jpg`, `.jpeg`

Images should be organized as a flat directory (no subdirectories).

### 3. Run

```bash
python ddpm.py
```

The script will automatically:

- Install required packages
- Train all 7 models on both grayscale and RGB modes (200 epochs each)
- Generate evaluation metrics and visualizations
- Save checkpoints and results to `./output/`

## Configuration

Key parameters in the `Config` class:

- `COLOR_MODE`: "GRAY" or "RGB"
- `EPOCHS`: Number of training epochs (default: 200)
- `BATCH_SIZE`: Batch size (default: 8)
- `IMG_SIZE`: Image resolution (default: 256)
- `LR`: Learning rate (default: 2e-4)
- `MASK_CACHE_SIZE`: Pre-generated mask cache size (default: 1024)

## Output Structure

```
./output/
├── GRAY/
│   ├── checkpoints/        # Trained model weights
│   ├── results/            # CSV metrics files
│   ├── figures/            # Comparison and detail figures
│   └── training_curves/    # Loss and metric curves
├── RGB/
│   └── (same structure)
└── combined_figures/       # Cross-mode comparison visualizations
```

## Models

| Model               | Architecture                           | Key Feature                 |
| ------------------- | -------------------------------------- | --------------------------- |
| Baseline U-Net      | Symmetric encoder-decoder              | Simple and efficient        |
| Deep Residual U-Net | U-Net + Residual blocks + SE attention | Strong feature learning     |
| Context Encoder     | Large-kernel encoder-decoder           | Competitive restoration      |
| Pix2Pix Generator   | U-Net generator + Patch discriminator  | Adversarial training        |
| RFA-Net             | Multi-dilation + Feature fusion        | Receptive field aggregation |
| PConv U-Net         | Partial convolutions                   | Mask-aware convolutions     |
| Guided DDPM         | Diffusion model with conditioning      | Generative approach         |

## Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity): Higher is better (0-1)
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better

## Outputs Generated

- **Comparison figures**: Side-by-side restoration results for all models
- **Detailed analysis**: Per-sample restoration with error maps
- **Bar charts**: Model performance comparison
- **Metrics tables**: Quantitative results in CSV and PNG formats
- **Training curves**: Loss, PSNR, SSIM, LPIPS progression
- **Mask examples**: Visualization of generated fracture patterns

## Notes

- First run will download pre-trained VGG16 for perceptual loss
- GPU with 8GB+ VRAM recommended
- Training time varies by hardware and dataset size
- Checkpoints are saved only when validation PSNR improves
- **Dataset Note:** The Harappan seals image dataset used in this study is not publicly distributed.
