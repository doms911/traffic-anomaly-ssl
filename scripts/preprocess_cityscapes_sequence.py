"""Pre-resize Cityscapes-Sequence frames to 512x256 for faster SSL training.

Reads 2048x1024 PNGs from leftImg8bit_sequence/, writes 512x256 JPGs to
leftImg8bit_sequence_resized/. Uses multiprocessing for speed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

from src.paths import CITYSCAPES_ROOT


SRC_DIR = CITYSCAPES_ROOT / "leftImg8bit_sequence"
DST_DIR = CITYSCAPES_ROOT / "leftImg8bit_sequence_resized"
TARGET_SIZE = (512, 256)  # (W, H) — 4x downsample from 2048x1024


def resize_one(args):
    src_path, dst_path = args
    if dst_path.exists():
        return  # idempotent
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    # Save as JPEG quality 90 — small files, fast read, lossless enough for SSL
    img.save(dst_path, "JPEG", quality=90)


def main():
    # Collect all PNG paths
    tasks = []
    for split_dir in sorted(SRC_DIR.iterdir()):
        if not split_dir.is_dir():
            continue
        for city_dir in sorted(split_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in city_dir.iterdir():
                if img_path.suffix != ".png":
                    continue
                # mirror structure under DST_DIR with .jpg extension
                rel = img_path.relative_to(SRC_DIR)
                dst_path = (DST_DIR / rel).with_suffix(".jpg")
                tasks.append((img_path, dst_path))

    print(f"Found {len(tasks)} frames to resize.")
    print(f"Source: {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} JPEG q90")

    # Process in parallel
    with Pool(processes=32) as pool:
        list(tqdm(
            pool.imap_unordered(resize_one, tasks, chunksize=50),
            total=len(tasks),
            desc="Resizing",
        ))

    print("Done.")


if __name__ == "__main__":
    main()