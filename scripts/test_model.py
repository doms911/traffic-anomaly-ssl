"""Sanity check for DeepLabV3+ model wrapper."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.segmentation import build_deeplabv3plus, count_parameters


def main():
    # Build model with ImageNet pretrained backbone
    model = build_deeplabv3plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        num_classes=19,
    )

    print(f"Model: DeepLabV3+ with ResNet50 (ImageNet pretrained)")
    print(f"Total parameters: {count_parameters(model):,}")

    # Forward pass on dummy input
    model.eval()
    dummy_input = torch.randn(2, 3, 768, 768)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Provjeri shape: trebao bi biti (B, num_classes, H, W)
    assert output.shape == (2, 19, 768, 768), f"Unexpected output shape: {output.shape}"
    print("\n✓ Forward pass radi.")

    # Test na GPU ako je dostupan
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\n✓ GPU forward pass radi. Output device: {output.device}")


if __name__ == "__main__":
    main()