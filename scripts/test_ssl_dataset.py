"""Sanity check for CityscapesSSL dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.datasets.cityscapes_ssl import CityscapesSSL
from src.datasets.ssl_transforms import build_moco_v2_transforms
from src.paths import CITYSCAPES_ROOT


def inspect(mode: str):
    print(f"\n=== Mode: {mode} ===")
    transform = build_moco_v2_transforms(crop_size=224)
    ds = CityscapesSSL(
        root=str(CITYSCAPES_ROOT),
        split="train",
        mode=mode,
        transform=transform,
    )
    print(f"Number of samples: {len(ds)}")

    view1, view2 = ds[0]
    print(f"View1 shape: {view1.shape}, dtype: {view1.dtype}")
    print(f"View2 shape: {view2.shape}, dtype: {view2.dtype}")
    print(f"View1 range: [{view1.min():.3f}, {view1.max():.3f}]")

    # Check views are different (different augmentations / different frames)
    diff = (view1 - view2).abs().mean().item()
    print(f"Mean abs diff between views: {diff:.3f}")
    print("(Should be > 0 — views must differ)")

    # Batch test
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    v1, v2 = next(iter(loader))
    print(f"\nBatch shapes: {v1.shape}, {v2.shape}")


def main():
    inspect("image")
    inspect("temporal")


if __name__ == "__main__":
    main()