from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def build_deeplabv3plus(
    encoder_name: str = "resnet50",
    encoder_weights: Optional[str] = "imagenet",
    num_classes: int = 19,
    ssl_checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Build DeepLabV3+ model with specified encoder.

    Args:
        encoder_name: SMP encoder name, e.g. 'resnet50'.
        encoder_weights: 'imagenet', None (random init), or 'ssl' if loading custom weights.
                         If 'ssl', pass ssl_checkpoint_path.
        num_classes: Number of segmentation classes.
        ssl_checkpoint_path: Path to SSL pretrained encoder state_dict (used only if
                             encoder_weights is None or 'ssl').

    Returns:
        DeepLabV3+ model.
    """
    # if we load SSL weights, we first create a model without pretrained weights
    init_weights = encoder_weights if encoder_weights == "imagenet" else None
    
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=init_weights,
        in_channels=3,
        classes=num_classes,
    )
    
    if ssl_checkpoint_path is not None:
        load_ssl_encoder_weights(model, ssl_checkpoint_path)
    
    return model

def load_ssl_encoder_weights(model: nn.Module, checkpoint_path: str) -> None:
    """
    Load SSL pretrained backbone weights into model.encoder.

    Expects a Lightly MoCo checkpoint with keys like 'backbone.encoder.X'.
    Skips momentum encoder and projection heads (SSL-only components).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # We only want the QUERY backbone: keys starting with 'backbone.encoder.'
    # Everything else (backbone_momentum, projection_head*) is SSL-only, skip it.
    cleaned = {}
    prefix = "backbone.encoder."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]   # 'backbone.encoder.conv1.weight' -> 'conv1.weight'
            cleaned[new_key] = v

    if len(cleaned) == 0:
        raise RuntimeError(
            f"No keys with prefix '{prefix}' found in checkpoint. "
            f"Available prefixes: {set('.'.join(k.split('.')[:2]) for k in state_dict)}"
        )

   # SMP's ResNetEncoder.load_state_dict returns None, so verify manually
    encoder_keys = set(model.encoder.state_dict().keys())
    cleaned_keys = set(cleaned.keys())

    matched = cleaned_keys & encoder_keys
    missing = encoder_keys - cleaned_keys
    unexpected = cleaned_keys - encoder_keys

    model.encoder.load_state_dict(cleaned, strict=False)

    print(f"Loaded SSL encoder weights from {checkpoint_path}")
    print(f"    Matched keys: {len(matched)} / {len(encoder_keys)}")
    print(f"    Missing (in SMP, not in ckpt): {len(missing)}")
    print(f"    Unexpected (in ckpt, not in SMP): {len(unexpected)}")
    if missing:
        print(f"    First missing: {sorted(missing)[:3]}")
    if unexpected:
        print(f"    First unexpected: {sorted(unexpected)[:3]}")
        
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)