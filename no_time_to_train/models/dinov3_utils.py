"""
Utility functions for loading and using DINOv3 models.
This module provides a unified interface for DINOv3 models similar to DINOv2.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add dinov3 to the path if not already there
dinov3_path = Path(__file__).parent.parent.parent / "dinov3"
if str(dinov3_path) not in sys.path:
    sys.path.insert(0, str(dinov3_path))

import dinov3.dinov3.models.vision_transformer as dinov3_vit


def load_pretrained_weights(model, pretrained_weights, checkpoint_key="teacher"):
    """
    Load pretrained weights into a DINOv3 model.
    
    Args:
        model: The DINOv3 model to load weights into
        pretrained_weights: Path to the checkpoint file or URL
        checkpoint_key: Key in the checkpoint dict (default: "teacher")
    """
    if pretrained_weights is None or pretrained_weights == "":
        print("No pretrained weights provided, using random initialization")
        model.init_weights()
        return model
    
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    
    # Handle different checkpoint formats
    if checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Remove "backbone." prefix if present
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    
    # Load the state dict
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {pretrained_weights}")
    if msg.missing_keys:
        print(f"Missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"Unexpected keys: {msg.unexpected_keys}")
    
    return model


def get_dinov3_model(model_name, **kwargs):
    """
    Get a DINOv3 model by name.
    
    Args:
        model_name: Name of the model architecture (e.g., "vit_small", "vit_base", "vit_large")
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        A DINOv3 vision transformer model
    """
    if hasattr(dinov3_vit, model_name):
        return dinov3_vit.__dict__[model_name](**kwargs)
    else:
        raise ValueError(f"Unknown DINOv3 model: {model_name}")


# Predefined configurations for common DINOv3 models
DINOV3_CONFIGS = {
    "dinov3_small": {
        "model_size": "vit_small",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 4.0,
        "qkv_bias": True,
        "layerscale_init": 1.0e-05,
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
        "ffn_bias": True,
        "proj_bias": True,
        "n_storage_tokens": 4,
        "mask_k_bias": True,
        "feat_dim": 384,
    },
    "dinov3_base": {
        "model_size": "vit_base",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_ratio": 4.0,
        "qkv_bias": True,
        "layerscale_init": 1.0e-05,
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
        "ffn_bias": True,
        "proj_bias": True,
        "n_storage_tokens": 4,
        "mask_k_bias": True,
        "feat_dim": 768,
    },
    "dinov3_large": {
        "model_size": "vit_large",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_ratio": 4.0,
        "qkv_bias": True,
        "layerscale_init": 1.0e-05,
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
        "ffn_bias": True,
        "proj_bias": True,
        "n_storage_tokens": 4,
        "mask_k_bias": True,
        "feat_dim": 1024,
    },
    "dinov3_giant": {
        "model_size": "vit_giant2",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "ffn_ratio": 4.0,
        "qkv_bias": True,
        "layerscale_init": 1.0e-05,
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
        "ffn_bias": True,
        "proj_bias": True,
        "n_storage_tokens": 4,
        "mask_k_bias": True,
        "feat_dim": 1536,
    },
}
