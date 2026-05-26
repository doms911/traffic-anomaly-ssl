"""SegmentMeIfYouCan dataset wrapper (RoadAnomaly21 + RoadObstacle21).

Only the public validation subset is supported (test labels are hidden).

Label format: 0=normal, 1=anomaly, 255=ignore.
"""
from pathlib import Path
from typing import Tuple, Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.paths import DATA_ROOT


SMIYC_ROOT = DATA_ROOT / "smiyc"

SUBSET_DIRS = {
    "RoadAnomaly21": "dataset_AnomalyTrack",
    "RoadObstacle21": "dataset_ObstacleTrack",
}


class SMIYCDataset(Dataset):
    """
    SegmentMeIfYouCan validation dataset wrapper.

    Only loads images whose names start with 'validation' (i.e., those with
    publicly available semantic masks). Test images have hidden labels.

    Returns (image_tensor, label_tensor) where:
      - image_tensor: (3, H, W) normalized float32
      - label_tensor: (H, W) int64 with {0=normal, 1=anomaly, 255=ignore}
    """

    def __init__(
        self,
        subset: Literal["RoadAnomaly21", "RoadObstacle21"] = "RoadAnomaly21",
        normalize: bool = True,
    ):
        self.subset = subset
        self.normalize = normalize

        if subset not in SUBSET_DIRS:
            raise ValueError(f"Unknown subset: {subset}")

        root = SMIYC_ROOT / SUBSET_DIRS[subset]
        self.img_dir = root / "images"
        self.lbl_dir = root / "labels_masks"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.img_dir}")

        # Match images and masks by the "validation*" prefix
        self.samples = []
        for img_path in sorted(self.img_dir.iterdir()):
            stem = img_path.stem  # e.g. 'validation0000' or 'validation_13'
            if not stem.startswith("validation"):
                continue  # skip non-validation (test) images
            mask_path = self.lbl_dir / f"{stem}_labels_semantic.png"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No (image, mask) pairs found in {self.img_dir}. "
                f"Did you extract the dataset properly?"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask_np = np.array(Image.open(mask_path), dtype=np.int64)

        image_tensor = TF.to_tensor(image)
        if self.normalize:
            image_tensor = TF.normalize(
                image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD
            )

        label_tensor = torch.from_numpy(mask_np)
        return image_tensor, label_tensor