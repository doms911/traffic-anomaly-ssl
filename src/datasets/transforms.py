"""Synchronized augmentations for semantic segmentation (image + mask)."""
import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image


# ImageNet statistike (standardno za pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SegmentationTransforms:
    """
    Synchronized augmentations for (image, mask) pairs.

    Train: random crop, horizontal flip, color jitter on image only, normalize.
    Val: full image, normalize only.
    """

    def __init__(
        self,
        split: str = "train",
        crop_size: Tuple[int, int] = (768, 768),
        scale_range: Tuple[float, float] = (0.5, 2.0),
        horizontal_flip_prob: float = 0.5,
        color_jitter: dict = None,
        ignore_index: int = 255,
    ):
        self.split = split
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter = color_jitter or {
            "brightness": 0.4, "contrast": 0.4, "saturation": 0.4
        }
        self.ignore_index = ignore_index

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split == "train":
            image, mask = self._train_transforms(image, mask)
        else:
            image, mask = self._val_transforms(image, mask)

        # Image: ToTensor + Normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Mask: long tensor of class indices
        mask = torch.as_tensor(
            list(mask.getdata()), dtype=torch.long
        ).reshape(mask.size[1], mask.size[0])

        return image, mask

    def _train_transforms(self, image, mask):
        # Random scale
        scale = random.uniform(*self.scale_range)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Pad if smaller than crop
        pad_w = max(self.crop_size[1] - new_w, 0)
        pad_h = max(self.crop_size[0] - new_h, 0)
        if pad_w > 0 or pad_h > 0:
            image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
            mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=self.ignore_index)

        # Random crop
        w, h = image.size
        x = random.randint(0, w - self.crop_size[1])
        y = random.randint(0, h - self.crop_size[0])
        image = image.crop((x, y, x + self.crop_size[1], y + self.crop_size[0]))
        mask = mask.crop((x, y, x + self.crop_size[1], y + self.crop_size[0]))

        # Random horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Color jitter (image only)
        image = TF.adjust_brightness(image, 1 + random.uniform(-self.color_jitter["brightness"], self.color_jitter["brightness"]))
        image = TF.adjust_contrast(image, 1 + random.uniform(-self.color_jitter["contrast"], self.color_jitter["contrast"]))
        image = TF.adjust_saturation(image, 1 + random.uniform(-self.color_jitter["saturation"], self.color_jitter["saturation"]))

        return image, mask

    def _val_transforms(self, image, mask):
        # Val: no augmentations, just convert to tensor
        return image, mask