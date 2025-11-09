# DINOv3 Integration Guide

This document provides information on how to use DINOv3 models in the no-time-to-train framework.

## Overview

DINOv3 is the next generation of self-supervised vision transformers from Meta AI. This integration allows you to use DINOv3 models as feature encoders alongside the existing DINOv2 support.

## Supported Models

The following DINOv3 models are supported:

- **dinov3_small**: ViT-S/16 (384-dim embeddings)
- **dinov3_base**: ViT-B/16 (768-dim embeddings)
- **dinov3_large**: ViT-L/16 (1024-dim embeddings)
- **dinov3_giant**: ViT-G/16 (1536-dim embeddings)

## Installation

Make sure you have the DINOv3 repository in your workspace:

```bash
# The dinov3 directory should be at the root of the no-time-to-train repository
ls dinov3/
```

## Usage

### 1. Configuration File

Update your configuration file to use a DINOv3 model. Here's an example:

```yaml
model:
  class_path: no_time_to_train.pl_wrapper.sam2matcher_pl.Sam2MatcherLightningModel
  init_args:
    model_cfg:
      name: "matching_baseline"
      encoder_cfg:
        name: "dinov3_large"  # Options: dinov3_small, dinov3_base, dinov3_large, dinov3_giant
        img_size: 224
        patch_size: 16
      encoder_ckpt_path: "./checkpoints/dinov3/dinov3_vitl16_pretrain.pth"
```

### 2. Downloading Pretrained Weights

DINOv3 pretrained weights can be downloaded from Meta AI's repository:

```bash
# Create checkpoints directory
mkdir -p checkpoints/dinov3

# Download DINOv3 Large model (example)
# Visit https://github.com/facebookresearch/dinov3 for official download links
wget -O checkpoints/dinov3/dinov3_vitl16_pretrain.pth <DOWNLOAD_URL>
```

### 3. Using DINOv3 in Code

The integration is transparent - simply specify the DINOv3 model in your configuration:

```python
from no_time_to_train.models.Sam2MatchingBaseline import Sam2MatchingBaseline

model = Sam2MatchingBaseline(
    sam2_cfg_file="sam2_hiera_t.yaml",
    sam2_ckpt_path="./checkpoints/sam2_hiera_tiny.pt",
    sam2_amg_cfg={...},
    encoder_cfg={
        "name": "dinov3_large",
        "img_size": 224,
        "patch_size": 16
    },
    encoder_ckpt_path="./checkpoints/dinov3/dinov3_vitl16_pretrain.pth",
    memory_bank_cfg={...}
)
```

## Configuration Parameters

### Encoder Configuration

- `name`: Model variant (required)
  - `dinov3_small`, `dinov3_base`, `dinov3_large`, `dinov3_giant`
  
- `img_size`: Input image size (default: 224)
  - Recommended: 224 for standard models
  
- `patch_size`: Patch size for the vision transformer (default: 16)
  - Standard value: 16

### Example Configurations

#### DINOv3 Small (Fastest)
```yaml
encoder_cfg:
  name: "dinov3_small"
  img_size: 224
  patch_size: 16
encoder_ckpt_path: "./checkpoints/dinov3/dinov3_vits16_pretrain.pth"
```

#### DINOv3 Large (Recommended Balance)
```yaml
encoder_cfg:
  name: "dinov3_large"
  img_size: 224
  patch_size: 16
encoder_ckpt_path: "./checkpoints/dinov3/dinov3_vitl16_pretrain.pth"
```

#### DINOv3 Giant (Best Performance)
```yaml
encoder_cfg:
  name: "dinov3_giant"
  img_size: 224
  patch_size: 16
encoder_ckpt_path: "./checkpoints/dinov3/dinov3_vitg16_pretrain.pth"
```

## Differences from DINOv2

1. **Architecture**: DINOv3 uses updated attention mechanisms and normalization
2. **Storage Tokens**: DINOv3 uses `n_storage_tokens` instead of `num_register_tokens`
3. **Output Format**: DINOv3 may use different keys in the output dictionary
4. **Training**: DINOv3 was trained on larger datasets with improved objectives

## Testing

A sample configuration file is provided at:
```
no_time_to_train/pl_configs/matching_baseline_dinov3_test.yaml
```

Run it with:
```bash
python run_lightening.py --config no_time_to_train/pl_configs/matching_baseline_dinov3_test.yaml
```

## Compatibility

The DINOv3 integration is compatible with:
- ✅ Sam2MatchingBaseline
- ✅ Sam2MatchingBaseline_noAMG
- ✅ All existing datasets and configurations

Simply change the `encoder_cfg.name` from `dinov2_*` to `dinov3_*` to switch models.

## Performance Considerations

- **Memory**: DINOv3 models may require more GPU memory than DINOv2
- **Speed**: DINOv3 Large is comparable to DINOv2 Large in inference speed
- **Accuracy**: DINOv3 typically provides better feature representations

## Troubleshooting

### Issue: "Unknown DINOv3 model"
**Solution**: Check that the model name is correctly spelled (e.g., `dinov3_large`, not `dinov3_Large`)

### Issue: "Module 'dinov3' has no attribute..."
**Solution**: Ensure the dinov3 directory is properly installed and accessible

### Issue: Checkpoint loading errors
**Solution**: Verify the checkpoint path and ensure it matches the model architecture

## Additional Resources

- DINOv3 Paper: Check Meta AI Research publications
- Official Repository: https://github.com/facebookresearch/dinov3
- DINOv2 for comparison: https://github.com/facebookresearch/dinov2

## Contributing

To add support for additional DINOv3 variants:

1. Update `DINOV3_CONFIGS` in `no_time_to_train/models/dinov3_utils.py`
2. Add model-specific parameters
3. Test with the provided configuration templates
