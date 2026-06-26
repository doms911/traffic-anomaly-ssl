# Anomaly Detection in Traffic Scene Videos using Self-Supervised Learning

Implementation for the master's thesis at FER Zagreb, 2026.

## Overview

Three-phase pipeline for anomaly detection in traffic scenes:

1. **Self-supervised pretraining** — ResNet50 backbone pretrained with MoCo v2 on unlabeled Cityscapes-Sequence frames (standard and temporal variants)
2. **Segmentation fine-tuning** — DeepLabV3+ fine-tuned on annotated Cityscapes (19 classes)
3. **Anomaly inference** — Post-hoc scoring (MSP, MaxLogit, Energy) on SMIYC benchmark

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.9+
- PyTorch 2.0+
- lightly
- segmentation-models-pytorch
- h5py

## Datasets

| Dataset | Purpose | Path |
|---|---|---|
| Cityscapes | Segmentation fine-tuning | `datasets/cityscapes/` |
| Cityscapes-Sequence | SSL pretraining | `datasets/cityscapes_sequence/` |
| SMIYC (RoadAnomaly21, RoadObstacle21) | Evaluation | `datasets/smiyc/` |

## Usage

### 1. SSL Pretraining

```bash
# Standard variant
python scripts/train_moco.py --mode image --epochs 100 --batch-size 256

# Temporal variant
python scripts/train_moco.py --mode temporal --max-frame-gap 15 --epochs 100 --batch-size 256
```

### 2. Segmentation Fine-tuning

```bash
# Baseline (ImageNet init)
python scripts/train_segmentation.py --backbone imagenet --epochs 200

# MoCo standard
python scripts/train_segmentation.py --backbone checkpoints/moco_standard/last.pth --epochs 200

# MoCo temporal
python scripts/train_segmentation.py --backbone checkpoints/moco_temporal/last.pth --epochs 200
```

### 3. Anomaly Inference & Evaluation

```bash
# Generate SMIYC submissions
python scripts/generate_smiyc_submission.py --checkpoint checkpoints/finetune_moco_temporal/best.pth

# Local validation evaluation
python scripts/evaluate_anomaly.py --checkpoint checkpoints/finetune_moco_temporal/best.pth --split val
```

## Results

### Cityscapes Validation mIoU

| Model | mIoU |
|---|---|
| Baseline (ImageNet) | 76.91% |
| MoCo standard | 76.81% |
| MoCo temporal | 76.57% |

### SMIYC Test Set (Energy scoring)

| Model | RA21 AUPRC | RO21 AUPRC | RO21 sIoU |
|---|---|---|---|
| Baseline | 34.97% | 6.85% | 8.79% |
| MoCo standard | 30.86% | 4.44% | 10.72% |
| MoCo temporal | 28.45% | 7.34% | 18.77% |

## Project Structure

```
traffic-anomaly-ssl/
├── src/
│   ├── anomaly/             # Anomaly scoring functions (MSP, MaxLogit, Energy)
│   ├── datasets/            # Cityscapes SSL + segmentation datasets
│   ├── evaluation/          # Evaluation metrics (AUPRC, AUROC, FPR95)
│   ├── models/              # MoCo v2, DeepLabV3+
│   ├── training/            # Training loops
│   ├── utils/               # Utilities
│   └── paths.py             # Path configuration
├── scripts/                 # Training and inference scripts
├── configs/                 # Configuration files
├── logs/                    # TensorBoard logs
├── figures/                 # Generated visualizations
├── anomalies/               # SMIYC submission files
├── requirements.txt
└── README.md
```

## Hardware

Trained on NVIDIA A100 40GB (HPC SRCE).

## Citation

```
@mastersthesis{barukcic2026anomaly,
  author = {Barukčić, Dominik},
  title  = {Anomaly Detection in Traffic Scene Videos using Self-Supervised Learning},
  school = {Faculty of Electrical Engineering and Computing, University of Zagreb},
  year   = {2026}
}
```

## Related

- **Thesis source:** [master-thesis-traffic-anomaly-ssl](https://github.com/doms911/master-thesis-traffic-anomaly-ssl)