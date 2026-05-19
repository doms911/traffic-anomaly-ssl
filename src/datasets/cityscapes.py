"""
Cityscapes dataset for semantic segmentation.
Maps 35 original label IDs to 19 train IDs used in standard evaluation.
"""
import os
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# Standard Cityscapes label ID -> train ID mapping
# 19 valid classes + 255 (ignore)
CITYSCAPES_LABEL_TO_TRAIN_ID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,    # road
    8: 1,    # sidewalk
    9: 255, 10: 255,
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    14: 255, 15: 255, 16: 255,
    17: 5,   # pole
    18: 255,
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255, 30: 255,
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: 255,
}

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

NUM_CLASSES = 19
IGNORE_INDEX = 255


def label_to_train_id(label: np.ndarray) -> np.ndarray:
    """Map raw Cityscapes label IDs to 19-class train IDs."""
    output = np.full_like(label, IGNORE_INDEX, dtype=np.uint8)
    for label_id, train_id in CITYSCAPES_LABEL_TO_TRAIN_ID.items():
        output[label == label_id] = train_id
    return output


class CityscapesSegmentation(Dataset):
    """
    Cityscapes dataset for semantic segmentation.

    Args:
        root: Path to Cityscapes root (containing leftImg8bit/ and gtFine/).
        split: One of {'train', 'val', 'test'}.
        transform: Callable that takes (image_pil, mask_pil) and returns (image_tensor, mask_tensor).
                   If None, returns PIL images directly.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.images_dir = self.root / "leftImg8bit" / split
        self.labels_dir = self.root / "gtFine" / split

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Collect (image_path, label_path) pairs across all city subfolders
        self.samples = []
        for city_dir in sorted(self.images_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = img_path.name.replace("_leftImg8bit.png", "")
                label_path = self.labels_dir / city_dir.name / f"{stem}_gtFine_labelIds.png"
                if label_path.exists():
                    self.samples.append((img_path, label_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        label = np.array(Image.open(label_path), dtype=np.uint8)
        label = label_to_train_id(label)
        label = Image.fromarray(label)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label