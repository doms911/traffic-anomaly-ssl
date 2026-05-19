"""Sanity check for CityscapesSegmentation dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.cityscapes import CityscapesSegmentation, NUM_CLASSES, IGNORE_INDEX
import numpy as np


def main():
    root = "/lustre/home/dbarukci/datasets/cityscapes"

    for split in ["train", "val"]:
        print(f"\n=== Split: {split} ===")
        ds = CityscapesSegmentation(root=root, split=split, transform=None)
        print(f"Number of samples: {len(ds)}")

        img, label = ds[0]
        label_np = np.array(label)

        print(f"Image size (PIL): {img.size}")        # (W, H)
        print(f"Label shape (np): {label_np.shape}")  # (H, W)
        print(f"Unique label values: {np.unique(label_np)}")
        print(f"Valid classes count: {(label_np < NUM_CLASSES).sum()}")
        print(f"Ignore pixels count: {(label_np == IGNORE_INDEX).sum()}")


if __name__ == "__main__":
    main()