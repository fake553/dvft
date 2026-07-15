# DVFT: Reusing Frozen Vision Transformers via Cross-Scale Feature Fusion

Official implementation of the paper "Reusing Frozen Vision Transformers via Cross-Scale Feature Fusion" (Scientific Reports, 2026).

## Overview

DVFT (Dual-ViT Fusion Transformer) is a lightweight framework that fuses two frozen Vision Transformers (ViT-Base and ViT-Large) of different scales through cross-attention alignment, without updating backbone parameters.

Two variants are provided:
- **DVFT-Anchor**: Uses pretrained CLS tokens as intrinsic anchors for bidirectional cross-scale calibration.
- **DVFT-Query**: Uses learnable query tokens as an extrinsic bottleneck for alternating cross-scale extraction.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DVFT.git
cd DVFT

# Create conda environment
conda create -n dvft python=3.11 -y
conda activate dvft

# Install dependencies
pip install -r requirements.txt
```

---

# Data Preparation

## ImageNet-1K
Download ImageNet-1K and organize as:

```
data/imagenet/
├── train/
│   ├── n01440764/
│   │   └── *.JPEG
│   └── ...
└── val/
    ├── n01440764/
    │   └── *.JPEG
    └── ...
```

## CIFAR-100
Will be automatically downloaded by torchvision.

## COCO / Flickr30k (for retrieval tasks)
Download the datasets and place them under `data/coco/` and `data/flickr30k/` respectively. Refer to the original dataset websites for download instructions.

---

# Training

## Train DVFT-Anchor on ImageNet-1K
```bash
python scripts/train_classification.py \
    --model dvft_anchor \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --batch_size 32 \
    --epochs 20 \
    --save_path checkpoints/dvft_anchor.pth
```

## Train DVFT-Query on ImageNet-1K
```bash
python scripts/train_classification.py \
    --model dvft_query \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --batch_size 32 \
    --epochs 20 \
    --save_path checkpoints/dvft_query.pth
```

## Train Baseline: Single Backbone + 4L Transformer
```bash
python scripts/train_classification.py \
    --model single_adapter \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --save_path checkpoints/adapter.pth
```

## Train Baseline: Dual-Backbone + 4L Transformer Fusion
```bash
python scripts/train_classification.py \
    --model transformer_fusion \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --save_path checkpoints/transformer_fusion.pth
```

## Train on CIFAR-100
```bash
python scripts/train_classification.py \
    --model dvft_anchor \
    --dataset cifar100 \
    --batch_size 32 \
    --epochs 20
```

---

# Evaluation
```bash
python scripts/evaluate.py \
    --model dvft_anchor \
    --checkpoint checkpoints/dvft_anchor.pth \
    --dataset imagenet \
    --data_dir /path/to/imagenet
```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21367363.svg)](https://doi.org/10.5281/zenodo.21367363)