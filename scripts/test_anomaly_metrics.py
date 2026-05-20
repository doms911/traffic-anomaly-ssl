"""Sanity check for anomaly metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.evaluation.metrics import compute_anomaly_metrics


def main():
    np.random.seed(42)

    # Simulate perfect classifier
    n = 10000
    labels = np.concatenate([np.zeros(9000), np.ones(1000)]).astype(np.int32)
    perfect_scores = np.concatenate([
        np.random.normal(-2, 0.5, 9000),  # normal: low scores
        np.random.normal(2, 0.5, 1000),    # anomaly: high scores
    ])

    metrics = compute_anomaly_metrics(perfect_scores, labels)
    print("=== Perfect classifier (anomalies have higher scores) ===")
    print(f"AUPRC: {metrics['auprc']:.4f}    (expected: close to 1.0)")
    print(f"AUROC: {metrics['auroc']:.4f}    (expected: close to 1.0)")
    print(f"FPR95: {metrics['fpr95']:.4f}    (expected: close to 0.0)")

    # Random classifier
    random_scores = np.random.randn(n)
    metrics = compute_anomaly_metrics(random_scores, labels)
    print("\n=== Random classifier ===")
    print(f"AUPRC: {metrics['auprc']:.4f}    (expected: ~0.1 = ratio of anomalies)")
    print(f"AUROC: {metrics['auroc']:.4f}    (expected: ~0.5)")
    print(f"FPR95: {metrics['fpr95']:.4f}    (expected: ~0.95)")


if __name__ == "__main__":
    main()