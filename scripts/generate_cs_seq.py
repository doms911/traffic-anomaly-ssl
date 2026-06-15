"""Generate Cityscapes-Sequence figure for thesis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

from src.paths import CITYSCAPES_ROOT

# Koristimo resized verziju (brže učitavanje)
SEQ_DIR = CITYSCAPES_ROOT / "leftImg8bit_sequence" / "train"

def main():
    # Pronađi jedan grad i jedan snippet
    cities = sorted(SEQ_DIR.iterdir())
    city_dir = cities[0]  # npr. aachen
    
    # Uzmi prvih 5 uzastopnih frameova iz jednog snippeta
    all_frames = sorted(city_dir.glob("*.png"))
    
    # Pronađi snippet (frameovi koji dijele isti prefix bez zadnjeg broja)
    # Cityscapes naming: {city}_{seq:06d}_{frame:06d}_leftImg8bit.jpg
    # Grupiramo po seq broju
    snippets = {}
    for f in all_frames:
        parts = f.stem.split("_")
        seq_id = parts[1]  # npr. 000000
        if seq_id not in snippets:
            snippets[seq_id] = []
        snippets[seq_id].append(f)
    
    # Uzmi snippet s dovoljno frameova
    seq_id = next(k for k, v in snippets.items() if len(v) >= 6)
    frames = sorted(snippets[seq_id])[:6]  # prvih 6 frameova
    
    # Plot
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    
    for i, (ax, frame_path) in enumerate(zip(axes, frames)):
        img = Image.open(frame_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"t = {i}", fontsize=10)
        ax.axis("off")
        
        # Označi srednji frame (annotated frame) drugačije
        if i == 2:
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
                spine.set_visible(True)
    
    plt.suptitle("Primjer isječka iz Cityscapes-Sequence", 
                 fontsize=11, y=1.02)
    plt.tight_layout()
    
    out_path = Path("figures/cityscapes_sequence.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()