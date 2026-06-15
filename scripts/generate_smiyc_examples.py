"""Generate SMIYC example figures for thesis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.paths import DATA_ROOT

SMIYC_ROOT = DATA_ROOT / "smiyc"

SUBSETS = {
    "RoadAnomaly21": {
        "img_dir":  SMIYC_ROOT / "dataset_AnomalyTrack" / "images",
        "mask_dir": SMIYC_ROOT / "dataset_AnomalyTrack" / "labels_masks",
        "mask_suffix": "_labels_semantic.png",
        "out_file": "figures/smiyc_anomaly_examples.png",
        "title": "RoadAnomaly21 — primjeri validacijskih slika",
    },
    "RoadObstacle21": {
        "img_dir":  SMIYC_ROOT / "dataset_ObstacleTrack" / "images",
        "mask_dir": SMIYC_ROOT / "dataset_ObstacleTrack" / "labels_masks",
        "mask_suffix": "_labels_semantic.png",
        "out_file": "figures/smiyc_obstacle_examples.png",
        "title": "RoadObstacle21 — primjeri validacijskih slika",
    },
}

NUM_EXAMPLES = 3  # koliko primjera prikazati


def load_mask_vis(mask_path):
    """Pretvori masku u RGB vizualizaciju: crvena=anomalija, siva=ignore."""
    mask = np.array(Image.open(mask_path))
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask == 0]   = [50, 50, 50]      # normalno — tamno siva
    vis[mask == 1]   = [220, 50, 50]     # anomalija — crvena
    vis[mask == 255] = [150, 150, 150]   # ignore — svijetlo siva
    return vis


def main():
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    for subset_name, cfg in SUBSETS.items():
        img_dir  = cfg["img_dir"]
        mask_dir = cfg["mask_dir"]
        suffix   = cfg["mask_suffix"]

        # Uzmi samo validation slike (imaju javne maske)
        candidates = sorted([
            p for p in img_dir.iterdir()
            if p.stem.startswith("validation") and
               (mask_dir / f"{p.stem}{suffix}").exists()
        ])

        if not candidates:
            print(f"No validation images found for {subset_name}")
            continue

        # Odaberi NUM_EXAMPLES ravnomjerno raspoređenih
        step = max(1, len(candidates) // NUM_EXAMPLES)
        selected = candidates[::step][:NUM_EXAMPLES]

        fig, axes = plt.subplots(NUM_EXAMPLES, 2,
                                 figsize=(10, 3.5 * NUM_EXAMPLES))
        if NUM_EXAMPLES == 1:
            axes = [axes]

        fig.suptitle(cfg["title"], fontsize=13, y=1.01)

        for i, img_path in enumerate(selected):
            mask_path = mask_dir / f"{img_path.stem}{suffix}"

            img  = np.array(Image.open(img_path).convert("RGB"))
            mask = load_mask_vis(mask_path)

            axes[i][0].imshow(img)
            axes[i][0].set_title(f"Ulazna slika ({img_path.stem})",
                                 fontsize=9)
            axes[i][0].axis("off")

            axes[i][1].imshow(mask)
            axes[i][1].set_title("Anotacija (crvena = anomalija)",
                                 fontsize=9)
            axes[i][1].axis("off")

        plt.tight_layout()
        plt.savefig(cfg["out_file"], dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {cfg['out_file']}")


if __name__ == "__main__":
    main()