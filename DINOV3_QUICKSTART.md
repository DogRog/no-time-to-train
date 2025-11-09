# Quick Start: Using DINOv3 in no-time-to-train

This guide will help you quickly get started with DINOv3 models in the no-time-to-train framework.

## Step 1: Verify Setup

Ensure you have the DINOv3 repository in your workspace:

```bash
ls dinov3/dinov3/models/
# Should show: __init__.py  convnext.py  vision_transformer.py
```

## Step 2: Download Pretrained Weights

Create a checkpoints directory and download DINOv3 weights:

```bash
mkdir -p checkpoints/dinov3
# Download your preferred model weights and place them in checkpoints/dinov3/
```

## Step 3: Update Configuration

Create or modify a YAML configuration file:

```yaml
model:
  class_path: no_time_to_train.pl_wrapper.sam2matcher_pl.Sam2MatcherLightningModel
  init_args:
    model_cfg:
      encoder_cfg:
        name: "dinov3_large"  # Choose: dinov3_small, dinov3_base, dinov3_large, dinov3_giant
        img_size: 224
        patch_size: 16
      encoder_ckpt_path: "./checkpoints/dinov3/dinov3_vitl16_pretrain.pth"
```

## Step 4: Run Your Experiment

```bash
python run_lightening.py --config your_config.yaml
```

## Example Configurations

### Small Model (Fast)
```yaml
encoder_cfg:
  name: "dinov3_small"
  img_size: 224
  patch_size: 16
encoder_ckpt_path: "./checkpoints/dinov3/dinov3_vits16_pretrain.pth"
```

### Large Model (Recommended)
```yaml
encoder_cfg:
  name: "dinov3_large"
  img_size: 224
  patch_size: 16
encoder_ckpt_path: "./checkpoints/dinov3/dinov3_vitl16_pretrain.pth"
```

### Giant Model (Best Performance)
```yaml
encoder_cfg:
  name: "dinov3_giant"
  img_size: 224
  patch_size: 16
encoder_ckpt_path: "./checkpoints/dinov3/dinov3_vitg16_pretrain.pth"
```

## Testing the Integration

Use the provided test configuration:

```bash
python run_lightening.py --config no_time_to_train/pl_configs/matching_baseline_dinov3_test.yaml
```

## Switching from DINOv2 to DINOv3

If you have an existing configuration using DINOv2, just change two lines:

**Before (DINOv2):**
```yaml
encoder_cfg:
  name: "dinov2_large"
encoder_ckpt_path: "./checkpoints/dinov2/dinov2_vitl14_pretrain.pth"
```

**After (DINOv3):**
```yaml
encoder_cfg:
  name: "dinov3_large"
encoder_ckpt_path: "./checkpoints/dinov3/dinov3_vitl16_pretrain.pth"
```

## Troubleshooting

### Issue: Import Error
```
ModuleNotFoundError: No module named 'dinov3'
```
**Solution:** Ensure the `dinov3` directory is at the root of your workspace.

### Issue: Checkpoint Loading Error
```
RuntimeError: Error loading checkpoint
```
**Solution:** Verify the checkpoint path and ensure it matches the model architecture.

### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Try a smaller model (e.g., `dinov3_small`) or reduce batch size.

## Next Steps

- Read the full documentation: `DINOV3_INTEGRATION.md`
- Check implementation details: `DINOV3_IMPLEMENTATION_SUMMARY.md`
- Experiment with different model sizes
- Compare DINOv2 vs DINOv3 performance on your task

## Support

For questions or issues:
1. Check the documentation files
2. Review the example configuration files
3. Examine the model code in `no_time_to_train/models/dinov3_utils.py`
