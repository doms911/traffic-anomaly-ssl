"""Generate SMIYC benchmark submission files (.hdf5 per image).

Output structure:
    anomaly_p/
        <method_name>/
            AnomalyTrack-all/
                airplane0000.hdf5
                ...
            ObstacleTrack-all/
                curvy-street_carton_1.hdf5
                ...
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.models.segmentation import build_deeplabv3plus
from src.anomaly.scoring import compute_anomaly_score
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.paths import DATA_ROOT


SMIYC_ROOT = DATA_ROOT / "smiyc"

SUBSETS = {
    "AnomalyTrack-all":  SMIYC_ROOT / "dataset_AnomalyTrack" / "images",
    "ObstacleTrack-all": SMIYC_ROOT / "dataset_ObstacleTrack" / "images",
}


def load_model(checkpoint_path: str, device):
    model = build_deeplabv3plus(
        encoder_name="resnet50",
        encoder_weights=None,
        num_classes=19,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def preprocess(img_path: Path, device):
    """Load image, normalize, pad to multiple of 16 for SMP encoder."""
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = tensor.unsqueeze(0).to(device)

    # Pad to multiple of 16
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w, pad_h, pad_w


@torch.no_grad()
def process_one(model, img_path: Path, scoring_method: str, device):
    """Run model on image, return anomaly score map at original resolution."""
    tensor, h, w, pad_h, pad_w = preprocess(img_path, device)
    logits = model(tensor)
    if pad_h > 0 or pad_w > 0:
        logits = logits[:, :, :h, :w]
    scores = compute_anomaly_score(logits, method=scoring_method)
    return scores[0].cpu().numpy().astype(np.float32)  # (H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to segmentation model checkpoint")
    parser.add_argument("--method_name", type=str, required=True,
                        help="Submission method name (e.g. 'MoCoTemporal')")
    parser.add_argument("--scoring", type=str, default="energy",
                        choices=["msp", "max_logit", "energy"])
    parser.add_argument("--output_dir", type=str, default="submissions",
                        help="Where to place anomaly_p/<method_name>/...")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    out_root = Path(args.output_dir) / "anomaly_p" / args.method_name
    out_root.mkdir(parents=True, exist_ok=True)

    for subset_name, img_dir in SUBSETS.items():
        out_subset = out_root / subset_name
        out_subset.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(p for p in img_dir.iterdir()
                           if p.suffix in {".jpg", ".jpeg", ".png", ".webp"})
        print(f"\n=== {subset_name}: {len(img_paths)} images ===")

        for img_path in tqdm(img_paths, desc=subset_name):
            scores = process_one(model, img_path, args.scoring, device)
            # Save as HDF5 with key 'anomaly_p' (SMIYC convention)
            out_path = out_subset / f"{img_path.stem}.hdf5"
            with h5py.File(out_path, "w") as f:
                f.create_dataset("value", data=scores, compression="gzip")

    print(f"\nDone. Output in: {out_root}")
    print(f"To submit: zip {args.output_dir}/anomaly_p folder and upload to Codabench.")


if __name__ == "__main__":
    main()