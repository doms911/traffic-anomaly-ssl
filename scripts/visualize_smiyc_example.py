"""Visualize one random SMIYC example: image | GT mask | anomaly heatmap | overlay."""
import argparse
import random
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
from src.paths import DATA_ROOT


SMIYC_ROOT = DATA_ROOT / "smiyc"

SUBSETS = {
    "RoadAnomaly21": SMIYC_ROOT / "dataset_AnomalyTrack",
    "RoadObstacle21": SMIYC_ROOT / "dataset_ObstacleTrack",
}


def load_model(checkpoint_path, device):
    model = build_deeplabv3plus(
        encoder_name="resnet50", encoder_weights=None, num_classes=19,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def get_anomaly_score(model, img_pil, scoring_method, device):
    """Returns (H, W) anomaly score numpy array."""
    w, h = img_pil.size
    tensor = TF.to_tensor(img_pil)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = tensor.unsqueeze(0).to(device)

    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    logits = model(tensor)
    if pad_h > 0 or pad_w > 0:
        logits = logits[:, :, :h, :w]
    scores = compute_anomaly_score(logits, method=scoring_method)
    return scores[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method_name", type=str, default="model",
                        help="Used in plot title and output filename")
    parser.add_argument("--subset", choices=list(SUBSETS), default="RoadAnomaly21")
    parser.add_argument("--scoring", choices=["msp", "max_logit", "energy"], default="energy")
    parser.add_argument("--image_name", type=str, default=None,
                        help="Specific image stem (e.g. 'validation0003'). If None, pick random.")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Find validation images (have public masks)
    root = SUBSETS[args.subset]
    img_dir = root / "images"
    mask_dir = root / "labels_masks"

    candidates = [p for p in img_dir.iterdir()
                  if p.stem.startswith("validation") and
                  (mask_dir / f"{p.stem}_labels_semantic.png").exists()]

    if args.image_name:
        img_path = next((p for p in candidates if p.stem == args.image_name), None)
        if img_path is None:
            raise FileNotFoundError(f"Image {args.image_name} not found in {args.subset}")
    else:
        img_path = random.choice(candidates)

    mask_path = mask_dir / f"{img_path.stem}_labels_semantic.png"
    print(f"Image: {img_path.name}")

    # Load
    image_pil = Image.open(img_path).convert("RGB")
    mask_np = np.array(Image.open(mask_path))  # 0=normal, 1=anomaly, 255=ignore
    image_np = np.array(image_pil)

    # Run model
    scores = get_anomaly_score(model, image_pil, args.scoring, device)

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # GT mask: 0=normal (black), 1=anomaly (red), 255=ignore (gray)
    gt_vis = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    gt_vis[mask_np == 1] = [255, 0, 0]
    gt_vis[mask_np == 255] = [128, 128, 128]
    axes[1].imshow(gt_vis)
    axes[1].set_title("Ground truth (red=anomaly)")
    axes[1].axis("off")

    im = axes[2].imshow(scores, cmap="jet")
    axes[2].set_title(f"Anomaly score ({args.scoring})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Overlay: image + heatmap with alpha
    axes[3].imshow(image_np)
    axes[3].imshow(scores, cmap="jet", alpha=0.5)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.suptitle(f"{args.method_name} | {args.subset} | {img_path.stem}", fontsize=14)
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method_name}_{args.subset}_{img_path.stem}_{args.scoring}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()