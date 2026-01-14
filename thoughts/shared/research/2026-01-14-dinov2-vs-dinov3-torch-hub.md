---
date: 2026-01-14T16:51:00+08:00
researcher: Claude
git_commit: c326dcc3b95f11d7c171322aa78becb0431db59c
branch: fix_dinov2_torch_hub
repository: fix_dinov2_torch_hub
topic: "DINOv2 vs DINOv3 Torch Hub Integration Differences"
tags: [research, codebase, dinov2, dinov3, torch-hub, segmentation]
status: complete
last_updated: 2026-01-14
last_updated_by: Claude
---

# Research: DINOv2 vs DINOv3 Torch Hub Integration Differences

**Date**: 2026-01-14T16:51:00+08:00
**Researcher**: Claude
**Git Commit**: c326dcc3b95f11d7c171322aa78becb0431db59c
**Branch**: fix_dinov2_torch_hub
**Repository**: fix_dinov2_torch_hub

## Research Question

When using DINOv3 segmentation code with `model.dino_hub=dinov3_vitl16`, everything works. However, when switching to `model.dino_hub=dinov2_vitl14`, the training fails. What are the differences between DINOv2 and DINOv3 models and why does the failure occur?

## Summary

The primary bug causing the DINOv2 failure is a **patch size incompatibility**:
- **DINOv3** uses `patch_size=16`, so image size 512 is valid (512/16=32 patches)
- **DINOv2** uses `patch_size=14`, so image size 512 is invalid (512/14=36.57, not an integer)

The error occurs at `dinov2/layers/patch_embed.py:72`:
```
AssertionError: Input image height 512 is not a multiple of patch height 14
```

## Detailed Findings

### Model Architecture Differences

| Feature | DINOv2 (`dinov2_vitl14`) | DINOv3 (`dinov3_vitl16`) |
|---------|--------------------------|--------------------------|
| **Model Class** | `DinoVisionTransformer` | `DinoVisionTransformer` |
| **patch_size** | 14 | 16 |
| **n_storage_tokens / num_register_tokens** | 0 | 4 |
| **Total prefix tokens** | 1 (CLS only) | 5 (1 CLS + 4 storage) |
| **embed_dim** | 1024 | 1024 |
| **Number of blocks** | 24 | 24 |
| **Valid image sizes** | Multiples of 14 (448, 518, 560, 644, 700, ...) | Multiples of 16 (512, 640, 768, 896, ...) |

### Torch Hub Loading Mechanism

**File**: `dinov3/eval/setup.py:34-54`

The `load_model_and_context()` function determines which repository to load from based on the model name:

```python
def load_model_and_context(model_config: ModelConfig, output_dir: str):
    if model_config.dino_hub is not None:
        if "dinov3" in model_config.dino_hub:
            repo = "dinov3"
        elif "dinov2" in model_config.dino_hub:
            repo = "dinov2"
        else:
            raise ValueError
        model = torch.hub.load(f"facebookresearch/{repo}", model_config.dino_hub)
```

### get_intermediate_layers Output

Both models' `get_intermediate_layers()` method returns **only spatial tokens** (no CLS or register/storage tokens in the features tensor):

**DINOv3** (224x224 input):
- feats shape: `[1, 196, 1024]` (196 = 14*14 spatial patches from 224/16)
- cls_token shape: `[1, 1024]`

**DINOv2** (224x224 input):
- feats shape: `[1, 256, 1024]` (256 = 16*16 spatial patches from 224/14)
- cls_token shape: `[1, 1024]`

### DINOv3 Adapter Code

**File**: `dinov3/eval/segmentation/models/backbone/dinov3_adapter.py:305-485`

The `DINOv3_Adapter` class wraps the backbone for segmentation. Key observations:

1. **Lines 331-335**: Reads `patch_size` from the backbone dynamically:
   ```python
   self.patch_size = self.backbone.patch_size
   print("patch_size", self.patch_size)
   ```

2. **Lines 422-426**: Calls `get_intermediate_layers()` on the backbone:
   ```python
   all_layers = self.backbone.get_intermediate_layers(
       x, n=self.interaction_indexes, return_class_token=True
   )
   ```

3. **Lines 432-441**: Contains potentially confusing dead code that slices assuming 5 prefix tokens:
   ```python
   cls, x = (
       x[:, :1,],
       x[:, 5:,],
   )
   ```
   However, this code is superseded by the loop at lines 444-456 where `x` is reassigned from `all_layers[i]`.

### Configuration Issue

**File**: `dinov3/eval/segmentation/configs/config-ade20k-m2f-training.yaml`

The config uses fixed image sizes:
```yaml
transforms:
  train:
    img_size: 512
    crop_size: [512, 512]
  eval:
    img_size: 896
    crop_size: 896
```

These sizes (512, 896) are:
- **Valid for DINOv3** (patch_size=16): 512/16=32, 896/16=56
- **Invalid for DINOv2** (patch_size=14): 512/14=36.57, 896/14=64

### Error Trace

```
File "dinov3/eval/segmentation/models/backbone/dinov3_adapter.py", line 424, in forward
    all_layers = self.backbone.get_intermediate_layers(
File "dinov2/models/vision_transformer.py", line 276
    x = self.prepare_tokens_with_masks(x)
File "dinov2/models/vision_transformer.py", line 218
    x = self.patch_embed(x)
File "dinov2/layers/patch_embed.py", line 72
    assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
AssertionError: Input image height 512 is not a multiple of patch height 14
```

## Code References

- `dinov3/eval/setup.py:34-54` - Model loading with hub detection
- `dinov3/eval/setup.py:17-24` - ModelConfig dataclass with dino_hub field
- `dinov3/eval/segmentation/models/backbone/dinov3_adapter.py:305-485` - DINOv3_Adapter class
- `dinov3/eval/segmentation/configs/config-ade20k-m2f-training.yaml:41-56` - Transform config with fixed sizes
- `hubconf.py:6-24` - Hub entry points (dinov3 models only)

## Architecture Documentation

### Token Naming Conventions

| Concept | DINOv2 Term | DINOv3 Term |
|---------|-------------|-------------|
| Extra tokens (besides CLS) | `register_tokens` | `storage_tokens` |
| Count attribute | `num_register_tokens` | `n_storage_tokens` |

### Compatibility Layer

**File**: `dinov3/eval/text/build_dinotxt.py:29-38`

There's a `rename_register_token()` function that converts between naming conventions:
```python
# "This allows us to load OSS DINOv2 models from pretrained weights using DINOv3 ViT"
# Converts register_tokens (dinov2) to storage_tokens (dinov3)
```

## Potential Fixes

To support DINOv2 in the segmentation pipeline, the code needs to:

1. **Dynamically adjust image sizes** based on `backbone.patch_size`
2. **Update config files** to use DINOv2-compatible sizes (multiples of 14) or make sizes configurable per model
3. **Add validation** in the adapter to check image size compatibility before forwarding

### Valid Image Sizes for DINOv2 (patch_size=14)

| Size | Patches |
|------|---------|
| 448 | 32x32 |
| 518 | 37x37 |
| 560 | 40x40 |
| 644 | 46x46 |
| 700 | 50x50 |
| 784 | 56x56 |
| 868 | 62x62 |

## Open Questions

1. Should the codebase automatically adjust image sizes based on patch_size?
2. Should there be separate config files for DINOv2 and DINOv3?
3. Are there other incompatibilities (e.g., in the adapter's `interaction_indexes`)?
