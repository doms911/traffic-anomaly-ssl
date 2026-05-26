"""End-to-end SegmentMeIfYouCan evaluation pipeline."""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.smiyc import SMIYCDataset
from src.anomaly.scoring import compute_anomaly_score
from src.evaluation.metrics import compute_anomaly_metrics


@torch.no_grad()
def evaluate_smiyc(
    model: nn.Module,
    subset: str = "RoadAnomaly21",
    scoring_method: str = "energy",
    device: Optional[torch.device] = None,
    num_workers: int = 2,
    verbose: bool = True,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    ds = SMIYCDataset(subset=subset) # type: ignore
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    all_scores, all_labels = [], []
    iterator = tqdm(loader, desc=f"Eval {subset} ({scoring_method})") if verbose else loader

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)

        # Pad to multiple of 16 (SMP encoder requirement)
        _, _, h, w = images.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            images_padded = F.pad(images, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            images_padded = images

        logits = model(images_padded)

        # Crop logits back to original size
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :h, :w]

        scores = compute_anomaly_score(logits, method=scoring_method)
        all_scores.append(scores.cpu().numpy().flatten())
        all_labels.append(labels.numpy().flatten())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    metrics = compute_anomaly_metrics(all_scores, all_labels, ignore_value=255)

    if verbose:
        print(f"\n{subset} | {scoring_method}: "
              f"AUPRC={metrics['auprc']*100:.2f}%, "
              f"AUROC={metrics['auroc']*100:.2f}%, "
              f"FPR95={metrics['fpr95']*100:.2f}%")
    return metrics