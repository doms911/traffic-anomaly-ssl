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
    Load SSL pretrained weights into model.encoder.

    Handles common SSL checkpoint formats (MoCo's encoder_q prefix, etc.).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Different SSL frameworks have different checkpoint structures.
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint  # assume checkpoint is directly the state_dict
    
    cleaned = {}
    prefixes_to_strip = ["module.encoder_q.", "encoder_q.", 
                         "module.encoder.", "encoder.",
                         "module.backbone.", "backbone.",
                         "module."]  # common patterns
    
    for k, v in state_dict.items():
        new_key = k
        for prefix in prefixes_to_strip:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
            
        # ignore fc/projection head (used only in SSL phase)
        if new_key.startswith("fc.") or new_key.startswith("projector."):
            continue
        
        cleaned[new_key] = v
        
    missing, unexpected = model.encoder.load_state_dict(cleaned, strict=False)
    print(f"Loaded SSL encoder weights from {checkpoint_path}")
    print(f"    Missing keys (not loaded): {missing}")
    print(f"    Unexpected keys (ignored): {unexpected}")
    if len(missing) > 0:
        print(f"    First missing: {missing[:3]}")
    if len(unexpected) > 0:
        print(f"    First unexpected: {unexpected[:3]}")
        
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)