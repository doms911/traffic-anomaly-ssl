"""Sanity check for SMIYCDataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.datasets.smiyc import SMIYCDataset


def inspect(subset: str):
    print(f"\n=== {subset} ===")
    ds = SMIYCDataset(subset=subset) # type: ignore
    print(f"Number of samples: {len(ds)}")

    image, label = ds[0]
    print(f"Image shape: {image.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Label shape: {label.shape}")
    print(f"Unique label values: {torch.unique(label).tolist()}")

    normal = (label == 0).sum().item()
    anomaly = (label == 1).sum().item()
    ignore = (label == 255).sum().item()
    total = label.numel()
    print(f"Pixels: normal={normal:,} ({100*normal/total:.1f}%), "
          f"anomaly={anomaly:,} ({100*anomaly/total:.1f}%), "
          f"ignore={ignore:,} ({100*ignore/total:.1f}%)")


def main():
    inspect("RoadAnomaly21")
    inspect("RoadObstacle21")


if __name__ == "__main__":
    main()