"""
Anomaly scoring functions for semantic segmentation.

All functions take segmentation logits (B, C, H, W) and return
anomaly scores (B, H, W), where higher score = more anomalous.
"""
import torch
import torch.nn.functional as F


def msp_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Maximum Softmax Probability (MSP) anomaly score.

    Hendrycks & Gimpel, 2017.
    Anomaly = low maximum softmax probability.
    Score = -max(softmax(logits)).

    Args:
        logits: (B, C, H, W) raw logits from segmentation model.

    Returns:
        scores: (B, H, W) higher = more anomalous.
    """
    probs = F.softmax(logits, dim=1)
    max_prob, _ = probs.max(dim=1)
    return -max_prob


def max_logit_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Maximum Logit anomaly score.

    Hendrycks et al., 2022.
    Anomaly = low maximum raw logit (before softmax).
    Score = -max(logits).

    Logits are unnormalized: softmax destroys magnitude info,
    raw logits preserve it. Anomalous pixels tend to have lower
    logits across all classes.

    Args:
        logits: (B, C, H, W) raw logits.

    Returns:
        scores: (B, H, W) higher = more anomalous.
    """
    max_logit, _ = logits.max(dim=1)
    return -max_logit


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Energy-based anomaly score.

    Liu et al., 2020.
    Score = -T * logsumexp(logits / T) per pixel.

    Theoretically grounded in energy-based models. Aggregates info
    across ALL classes (not just max), so more robust than MSP/MaxLogit.

    Args:
        logits: (B, C, H, W) raw logits.
        temperature: temperature for softmax-like scaling.

    Returns:
        scores: (B, H, W) higher = more anomalous.
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


SCORING_FUNCTIONS = {
    "msp": msp_score,
    "max_logit": max_logit_score,
    "energy": energy_score,
}


def compute_anomaly_score(
    logits: torch.Tensor, method: str = "energy", **kwargs
) -> torch.Tensor:
    """Dispatcher for anomaly scoring."""
    if method not in SCORING_FUNCTIONS:
        raise ValueError(
            f"Unknown scoring method: {method}. "
            f"Choose from {list(SCORING_FUNCTIONS.keys())}."
        )
    return SCORING_FUNCTIONS[method](logits, **kwargs)