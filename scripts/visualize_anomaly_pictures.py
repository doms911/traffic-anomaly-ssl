"""Visualize anomaly scoring methods on custom anomaly images.

Input:  anomalies/anomaly_01.jpg ... anomaly_06.jpg
Output: anomalies/results/anomaly_XX_scores.png

Each output figure shows:
    Original | MSP score | MaxLogit score | Energy score
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from src.models.segmentation import build_deeplabv3plus
from src.anomaly.scoring import compute_anomaly_score
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD


CHECKPOINT = "checkpoints/baseline_imagenet/best.pth"
ANOMALIES_DIR = Path("anomalies")
RESULTS_DIR = ANOMALIES_DIR / "results_baseline_imagenet"
SCORING_METHODS = ["msp", "max_logit", "energy"]
METHOD_LABELS = {
    "msp": "MSP",
    "max_logit": "Max Logit",
    "energy": "Energy",
}


def load_model(checkpoint_path, device):
    model = build_deeplabv3plus(
        encoder_name="resnet50", encoder_weights=None, num_classes=19
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def get_scores(model, img_pil, device):
    """Returns dict of {method: (H,W) numpy score array}."""
    w, h = img_pil.size
    t = TF.to_tensor(img_pil)
    t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    t = t.unsqueeze(0).to(device)

    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")

    logits = model(t)
    if pad_h or pad_w:
        logits = logits[:, :, :h, :w]

    scores = {}
    for method in SCORING_METHODS:
        s = compute_anomaly_score(logits, method=method)
        scores[method] = s[0].cpu().numpy()
    return scores


def visualize(img_path, scores, out_path):
    img_np = np.array(Image.open(img_path).convert("RGB"))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Original
    axes[0].imshow(img_np)
    axes[0].set_title("Originalna slika", fontsize=11)
    axes[0].axis("off")

    # Score maps
    for i, method in enumerate(SCORING_METHODS):
        ax = axes[i + 1]
        score = scores[method]
        im = ax.imshow(score, cmap="jet")
        ax.set_title(f"Anomaly score — {METHOD_LABELS[method]}", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    stem = img_path.stem
    fig.suptitle(f"{stem}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = load_model(CHECKPOINT, device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Pronađi sve anomaly_XX slike (jpg ili png)
    images = sorted([
        p for p in ANOMALIES_DIR.iterdir()
        if p.stem.startswith("anomaly_") and p.suffix in {".jpg", ".jpeg", ".png"}
    ])

    if not images:
        print(f"Nema slika u {ANOMALIES_DIR}/ koje počinju s 'anomaly_'")
        return

    print(f"Pronađeno {len(images)} slika\n")

    for img_path in images:
        print(f"Processing: {img_path.name}")
        img_pil = Image.open(img_path).convert("RGB")
        scores = get_scores(model, img_pil, device)
        out_path = RESULTS_DIR / f"{img_path.stem}_scores.png"
        visualize(img_path, scores, out_path)

    print(f"\nSvi rezultati spremljeni u: {RESULTS_DIR}")


if __name__ == "__main__":
    main()