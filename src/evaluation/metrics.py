"""Evaluation metrics for semantic segmentation."""
import torch
import numpy as np


class ConfusionMatrix:
    """
    Accumulates a confusion matrix across batches for mIoU computation.

    Standard approach: maintain (num_classes, num_classes) integer matrix,
    update with each batch, compute mIoU at the end.
    """
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        
    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix.

        Args:
            preds: (B, H, W) predicted class indices (argmax of logits)
            targets: (B, H, W) ground truth class indices
        """
        preds = preds.flatten()
        targets = targets.flatten()
        
        # mask out ignore pixels
        valid = (targets != self.ignore_index) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]
        
        # Bincount trick - combine pred and target into single index
        indices = self.num_classes * targets + preds
        bincount = torch.bincount(indices, minlength=self.num_classes**2)
        self.matrix += bincount.view(self.num_classes, self.num_classes).cpu()
        
    def compute_iou(self) -> torch.Tensor:
        """
        Compute per-class IoU.

        Returns:
            iou: (num_classes,) tensor of IoU per class.
        """
        matrix = self.matrix.float()
        intersection = torch.diag(matrix)
        union = matrix.sum(dim=0) + matrix.sum(dim=1) - intersection
        return intersection / torch.clamp(union, min=1)
    
    def compute_miou(self) -> float:
        """
        Compute mean IoU across classes (excluding ignored classes).

        Returns:
            miou: scalar mean IoU.
        """
        iou = self.compute_iou()
        valid_classes = self.matrix.sum(dim=1) > 0  # only consider classes present in GT
        if valid_classes.sum() == 0:
            return 0.0  # no valid classes, return 0 to avoid NaN
        return iou[valid_classes].mean().item()
    
    def reset(self) -> None:
        """Reset confusion matrix to zeros."""
        self.matrix.zero_()