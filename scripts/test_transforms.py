"""Sanity check for SegmentationTransforms."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.datasets.cityscapes import CityscapesSegmentation
from src.datasets.transforms import SegmentationTransforms
from src.paths import CITYSCAPES_ROOT


def main():
    transforms = SegmentationTransforms(split="train", crop_size=(768, 768))
    ds = CityscapesSegmentation(
        root=str(CITYSCAPES_ROOT), split="train", transform=transforms
    )

    image, mask = ds[0]
    print(f"Image tensor shape: {image.shape}")     # (3, 768, 768)
    print(f"Image dtype: {image.dtype}")             # torch.float32
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mask tensor shape: {mask.shape}")        # (768, 768)
    print(f"Mask dtype: {mask.dtype}")               # torch.int64 (long)
    print(f"Unique mask values: {torch.unique(mask).tolist()}")

    # Provjera batch loadera
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    images, masks = next(iter(loader))
    print(f"\nBatch images shape: {images.shape}")   # (4, 3, 768, 768)
    print(f"Batch masks shape: {masks.shape}")       # (4, 768, 768)


if __name__ == "__main__":
    main()