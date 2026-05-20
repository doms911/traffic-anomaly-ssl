"""Sanity check for anomaly scoring functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.anomaly.scoring import msp_score, max_logit_score, energy_score


def main():
    # Simulate model output: 2 images, 19 classes, 768x768
    logits = torch.randn(2, 19, 768, 768)

    msp = msp_score(logits)
    ml = max_logit_score(logits)
    en = energy_score(logits)

    print(f"Logits shape: {logits.shape}")
    print(f"\nMSP scores:")
    print(f"  Shape: {msp.shape}")
    print(f"  Range: [{msp.min():.3f}, {msp.max():.3f}]")
    print(f"  Mean: {msp.mean():.3f}")

    print(f"\nMax Logit scores:")
    print(f"  Shape: {ml.shape}")
    print(f"  Range: [{ml.min():.3f}, {ml.max():.3f}]")
    print(f"  Mean: {ml.mean():.3f}")

    print(f"\nEnergy scores:")
    print(f"  Shape: {en.shape}")
    print(f"  Range: [{en.min():.3f}, {en.max():.3f}]")
    print(f"  Mean: {en.mean():.3f}")

    # Test "confident" vs "uncertain" predictions
    print("\n=== Confident vs uncertain predictions ===")
    # Confident: one class has very high logit
    confident_logits = torch.zeros(1, 19, 1, 1)
    confident_logits[0, 5, 0, 0] = 10.0  # high logit for class 5

    # Uncertain: all classes have similar low logits
    uncertain_logits = torch.zeros(1, 19, 1, 1) + 0.1

    print(f"Confident: MSP={msp_score(confident_logits).item():.3f}, "
          f"MaxLogit={max_logit_score(confident_logits).item():.3f}, "
          f"Energy={energy_score(confident_logits).item():.3f}")
    print(f"Uncertain: MSP={msp_score(uncertain_logits).item():.3f}, "
          f"MaxLogit={max_logit_score(uncertain_logits).item():.3f}, "
          f"Energy={energy_score(uncertain_logits).item():.3f}")
    print("Uncertain should have HIGHER score (more anomalous).")


if __name__ == "__main__":
    main()