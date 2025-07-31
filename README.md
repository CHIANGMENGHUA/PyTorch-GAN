# MNIST GAN - Generative Adversarial Network

A PyTorch implementation of a Generative Adversarial Network (GAN) for generating handwritten digits using the MNIST dataset.

## Overview

This project implements a basic GAN architecture to generate realistic handwritten digits. The model consists of two neural networks competing against each other:
- **Generator**: Creates fake images from random noise
- **Discriminator**: Distinguishes between real and fake images

## Features

- CUDA GPU support for accelerated training
- Visualization of generated samples during training
- Configurable hyperparameters
- Progress tracking with loss monitoring

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- tqdm
- CUDA-compatible GPU (recommended)

## Installation

```bash
pip install torch torchvision matplotlib tqdm
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook GAN.ipynb
```

2. Run all cells to:
   - Check CUDA availability
   - Load and preprocess MNIST data
   - Define Generator and Discriminator architectures
   - Train the GAN model
   - Visualize generated samples

## Model Architecture

### Generator
- Input: 64-dimensional noise vector
- Hidden layers: 128 → 256 → 512 → 1024 neurons
- Output: 784-dimensional vector (28×28 flattened image)
- Activation: ReLU + BatchNorm, final Sigmoid

### Discriminator
- Input: 784-dimensional flattened image
- Hidden layers: 1024 → 512 → 256 neurons
- Output: Single value (real/fake probability)
- Activation: LeakyReLU

## Training Configuration

- **Epochs**: 500
- **Batch Size**: 128
- **Learning Rate**: 0.00001
- **Noise Dimension**: 64
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: Adam

## Results

The model generates increasingly realistic handwritten digits as training progresses. Sample outputs are displayed every 300 steps during training.

## File Structure

```
.
├── GAN.ipynb              # Main notebook with implementation
├── MNIST/                 # MNIST dataset (auto-downloaded)
│   └── raw/              # Raw MNIST data files
├── .jupyter/             # Jupyter configuration
└── README.md             # This file
```

## GPU Requirements

This implementation is optimized for CUDA-enabled GPUs. The notebook includes GPU detection and will automatically use available hardware acceleration.
