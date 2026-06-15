"""Evaluate baseline model on SegmentMeIfYouCan validation sets."""
import argparse
import sys
from pathlib import Path
from tabnanny import check
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.models.segmentation import build_deeplabv3plus
from src.evaluation.eval_smiyc import evaluate_smiyc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/baseline_imagenet/best.pth",
    )
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    
    model = build_deeplabv3plus(
        encoder_name="resnet50",
        encoder_weights=None,
        num_classes=19,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Epoch {ckpt.get('epoch', '?')}, mIoU = {ckpt.get('miou', '?'):.4f}")

    results = {}
    for subset in ["RoadAnomaly21", "RoadObstacle21"]:
        for method in ["msp", "max_logit", "energy"]:
            key = f"{subset}_{method}"
            print(f"\n{'='*55}\n{key}\n{'='*55}")
            results[key] = evaluate_smiyc(model, subset=subset, scoring_method=method)

    print("\n" + "="*65)
    print(f"{'Subset':<20} {'Method':<12} {'AUPRC':>8} {'AUROC':>8} {'FPR95':>8}")
    print("="*65)
    for subset in ["RoadAnomaly21", "RoadObstacle21"]:
        for method in ["msp", "max_logit", "energy"]:
            m = results[f"{subset}_{method}"]
            print(f"{subset:<20} {method:<12} "
                  f"{m['auprc']*100:>7.2f}% "
                  f"{m['auroc']*100:>7.2f}% "
                  f"{m['fpr95']*100:>7.2f}%")


if __name__ == "__main__":
    main()