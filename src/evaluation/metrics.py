"""Evaluation metrics for semantic segmentation."""
import torch
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score


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
        
        
# ----- Anomaly detection metrics (AUPRC, AUROC, FPR95) -----        
def compute_anomaly_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    ignore_value: int = 255,
) -> dict:
    """
    Compute pixel-level anomaly detection metrics.

    Args:
        scores: (N,) flattened anomaly scores (higher = more anomalous).
        labels: (N,) flattened binary labels (0 = normal, 1 = anomaly, 255 = ignore).
        ignore_value: label value to exclude from evaluation.

    Returns:
        dict with keys: 'auprc', 'auroc', 'fpr95', 'num_anomaly', 'num_normal'.
    """
    # Filter out ignore pixels
    valid = labels != ignore_value
    scores = scores[valid]
    labels = labels[valid]

    # Ensure binary labels
    labels = (labels == 1).astype(np.int32)

    num_anomaly = int(labels.sum())
    num_normal = int(len(labels) - num_anomaly)

    if num_anomaly == 0 or num_normal == 0:
        return {
            "auprc": 0.0, "auroc": 0.0, "fpr95": 1.0,
            "num_anomaly": num_anomaly, "num_normal": num_normal,
        }

    # AUPRC (Average Precision)
    auprc = average_precision_score(labels, scores)

    # AUROC
    auroc = roc_auc_score(labels, scores)

    # FPR at 95% TPR
    fpr95 = compute_fpr_at_tpr(scores, labels, target_tpr=0.95)

    return {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "fpr95": float(fpr95),
        "num_anomaly": num_anomaly,
        "num_normal": num_normal,
    }


def compute_fpr_at_tpr(
    scores: np.ndarray, labels: np.ndarray, target_tpr: float = 0.95
) -> float:
    """
    Compute False Positive Rate at a given True Positive Rate.

    FPR95 is standard metric for OOD/anomaly detection:
    "When the model catches 95% of anomalies, what fraction of
    normal pixels does it incorrectly flag?"

    Lower is better. Random predictor gives FPR95 = 0.95.
    """
    # Sort by score descending (highest anomaly score first)
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    # Cumulative true positives and false positives
    tp = np.cumsum(labels_sorted == 1)
    fp = np.cumsum(labels_sorted == 0)

    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()

    tpr = tp / num_pos
    fpr = fp / num_neg

    # Find first threshold where TPR >= target
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(fpr):
        return 1.0
    return float(fpr[idx])