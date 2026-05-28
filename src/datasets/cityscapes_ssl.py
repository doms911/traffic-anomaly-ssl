"""Cityscapes-Sequence dataset for SSL pretraining.

Supports two modes:
- 'image': two augmentations of the same frame (standard MoCo v2)
- 'temporal': two frames from the same 30-frame snippet (your contribution)
"""
import random
import re
from pathlib import Path
from typing import Tuple, Literal, List

import torch
from torch.utils.data import Dataset
from PIL import Image


# Parses Cityscapes filenames: {city}_{snippet_id}_{frame_id}_leftImg8bit.png
FILENAME_PATTERN = re.compile(r"^(.+)_(\d{6})_(\d{6})_leftImg8bit\.(?:png|jpg)$")


class CityscapesSSL(Dataset):
    """
    Cityscapes-Sequence dataset for self-supervised contrastive learning.

    Args:
        root: Path to Cityscapes root (must contain leftImg8bit_sequence/).
        split: 'train' (only split used for SSL).
        mode: 'image' (two augs of same frame) or 'temporal' (two frames from same snippet).
        transform: Callable applied to each image individually. Should return a tensor.
        max_frame_gap: For temporal mode, max distance between paired frames (in frames).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: Literal["image", "temporal"] = "image",
        transform=None,
        max_frame_gap: int = 15,
    ):
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        self.max_frame_gap = max_frame_gap

        resized = self.root / "leftImg8bit_sequence_resized" / split
        original = self.root / "leftImg8bit_sequence" / split
        seq_dir = resized if resized.exists() else original
        print(f"[CityscapesSSL] Using {seq_dir}")
        if not seq_dir.exists():
            raise FileNotFoundError(f"Sequence dir not found: {seq_dir}")

        # Collect all frames, grouped by snippet
        # snippets: dict[(city, snippet_id)] -> sorted list of frame paths
        self.snippets: dict = {}
        for city_dir in sorted(seq_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in city_dir.iterdir():
                m = FILENAME_PATTERN.match(img_path.name)
                if m is None:
                    continue
                city, snippet_id, frame_id = m.groups()
                key = (city, snippet_id)
                self.snippets.setdefault(key, []).append(img_path)

        # Sort each snippet by frame_id
        for key in self.snippets:
            self.snippets[key].sort()

        # For mode='image', index = single frame
        # For mode='temporal', index = snippet (we pick two frames at __getitem__)
        if mode == "image":
            self.samples: List = [
                f for frames in self.snippets.values() for f in frames
            ]
        elif mode == "temporal":
            # Index by frame (like image mode), but remember which snippet each frame
            # belongs to, so we can pick a temporal partner from the same snippet.
            self.samples = []  # list of (frame_path, snippet_key)
            for key, frames in self.snippets.items():
                if len(frames) < 2:
                    continue
                for f in frames:
                    self.samples.append((f, key))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {seq_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "image":
            img_path = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            view1 = self.transform(image) if self.transform else image
            view2 = self.transform(image) if self.transform else image
            return view1, view2

        else:  # temporal
            anchor_path, snippet_key = self.samples[idx]
            frames = self.snippets[snippet_key]
            n = len(frames)

            # Find anchor's position in the snippet
            anchor_idx = frames.index(anchor_path)

            # Pick a partner within max_frame_gap, different from anchor
            low = max(0, anchor_idx - self.max_frame_gap)
            high = min(n - 1, anchor_idx + self.max_frame_gap)
            candidates = list(range(low, high + 1))
            candidates.remove(anchor_idx)
            partner_idx = random.choice(candidates)

            img1 = Image.open(anchor_path).convert("RGB")
            img2 = Image.open(frames[partner_idx]).convert("RGB")

            view1 = self.transform(img1) if self.transform else img1
            view2 = self.transform(img2) if self.transform else img2
            return view1, view2