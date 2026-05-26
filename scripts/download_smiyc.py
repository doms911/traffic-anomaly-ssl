"""Download SegmentMeIfYouCan (RoadAnomaly21, RoadObstacle21) from HuggingFace."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.paths import DATA_ROOT


def main():
    smiyc_root = DATA_ROOT / "smiyc"
    smiyc_root.mkdir(parents=True, exist_ok=True)

    print("Downloading SegmentMeIfYouCan (RoadAnomaly21 + RoadObstacle21)...")
    ds = load_dataset(
        "kumuji/roadanomaly21_roadobstacle21",
        cache_dir=str(smiyc_root),
    )

    print(f"\nDataset structure: {ds}")
    print(f"Available splits: {list(ds.keys())}")

    # Inspect first sample of each split
    for split_name in ds.keys():
        split = ds[split_name]
        print(f"\n=== Split: {split_name} ===")
        print(f"Number of samples: {len(split)}")
        print(f"Features: {split.features}")
        sample = split[0]
        print(f"First sample keys: {list(sample.keys())}")


if __name__ == "__main__":
    main()