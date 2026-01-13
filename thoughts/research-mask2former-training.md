# Research: Adding Mask2Former Training Support to DINOv3

**Date**: 2026-01-13
**Researcher**: Claude
**Git Commit**: 54694f7627fd815f62a5dcc82944ffa6153bbb76
**Branch**: main
**Repository**: dinov3

## Research Question

How to add Mask2Former training support to the DINOv3 segmentation module, given that the repo currently only offers inference for M2F and training for linear head.

## Summary

The DINOv3 segmentation module has a complete Mask2Former inference pipeline but lacks training support. The user has written `mask_classification_loss.py` which provides the Hungarian matching and loss computation. This document maps out the existing architecture and identifies what components exist vs what gaps need to be filled.

## Detailed Findings

### 1. Current Training Pipeline (Linear Head Only)

**Location**: `dinov3/eval/segmentation/train.py`

The current training pipeline is explicitly restricted to linear head:

```python
# train.py:150
assert config.decoder_head.type == "linear", "Only linear head is supported for training"
```

**Training Loop Structure** (`train.py:100-143`):
1. Load batch: `batch_img, (_, gt) = batch` - expects `(image, (index, label))` tuple
2. Forward pass with autocast
3. Squeeze ground truth to long tensor: `gt = torch.squeeze(gt).long()`
4. Interpolate prediction to match GT size if needed
5. Compute loss: `loss = criterion(pred, gt)` - expects dense predictions
6. Backward pass with gradient scaling

**Key Issue**: The `train_step` function expects:
- `pred`: Tensor `[B, num_classes, H, W]` (dense per-pixel logits)
- `gt`: Tensor `[B, H, W]` (semantic labels per pixel)

Mask2Former outputs a dictionary with query-based predictions, incompatible with this interface.

### 2. Mask2Former Head Output Format

**Location**: `dinov3/eval/segmentation/models/heads/mask2former_head.py`

The M2F head returns a dictionary (not a tensor):

```python
# Output from forward():
{
    "pred_logits": Tensor[B, 100, num_classes+1],  # Class predictions per query
    "pred_masks": Tensor[B, 100, H, W],            # Binary mask per query
    "aux_outputs": List[dict]                       # 9 intermediate layer predictions
}
```

**Architecture Details**:
- 100 learnable queries (`mask2former_transformer_decoder.py:352-354`)
- 9 decoder layers with auxiliary outputs for deep supervision
- Pixel decoder: 6 transformer encoder layers with deformable attention
- Transformer decoder: Multi-scale masked attention

### 3. User's MaskClassificationLoss Implementation

**Location**: `dinov3/eval/segmentation/mask_classification_loss.py`

The user has implemented a complete loss class extending HuggingFace's `Mask2FormerLoss`:

```python
class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,           # Points sampled for mask loss
        oversample_ratio: float,   # Oversampling ratio
        importance_sample_ratio: float,
        mask_coefficient: float,   # Weight for mask loss
        dice_coefficient: float,   # Weight for dice loss
        class_coefficient: float,  # Weight for classification loss
        num_labels: int,           # Number of classes
        no_object_coefficient: float,  # Weight for "no object" class
    ):
```

**Forward signature** (`mask_classification_loss.py:55-76`):
```python
def forward(
    self,
    masks_queries_logits: torch.Tensor,  # [B, Q, H, W]
    targets: List[dict],                  # List of {"masks": Tensor, "labels": Tensor}
    class_queries_logits: Optional[torch.Tensor] = None,  # [B, Q, C+1]
):
```

**Expected Target Format**:
```python
targets = [
    {"masks": Tensor[N_i, H, W], "labels": Tensor[N_i]},  # Per-image instance masks
    ...
]
```

Where `N_i` is the number of instances in image `i`.

**Loss Components** (`loss_total` method at line 99-120):
- `loss_mask_*`: Binary mask loss (weighted by `mask_coefficient`)
- `loss_dice_*`: Dice loss (weighted by `dice_coefficient`)
- `loss_cross_entropy_*`: Classification loss (weighted by `class_coefficient`)

### 4. Model Building Differences

**Location**: `dinov3/eval/segmentation/models/__init__.py`

| Aspect | Linear Head | Mask2Former |
|--------|-------------|-------------|
| Backbone wrapper | `ModelWithIntermediateLayers` | `DINOv3_Adapter` |
| Features format | List of tensors | Dict `{"1": ..., "4": ...}` |
| Output format | Tensor `[B,C,H,W]` | Dict with pred_masks, pred_logits |
| Backbone frozen | Yes (line 115) | Yes (adapter line 326) |
| Feature layers | `LAST` (default) | `FOUR_EVEN_INTERVALS` |

### 5. Config Structure

**Linear Training Config** (`configs/config-ade20k-linear-training.yaml`):
- Has `scheduler`, `optimizer`, `train` sections
- `decoder_head.type: "linear"`
- `decoder_head.backbone_out_layers: LAST`
- `transforms.train` with augmentations

**M2F Inference Config** (`configs/config-ade20k-m2f-inference.yaml`):
- No training-related sections
- `decoder_head.type: "m2f"`
- `decoder_head.backbone_out_layers: FOUR_EVEN_INTERVALS`
- Only `transforms.eval` section

**Missing for M2F Training**:
- No `config-ade20k-m2f-training.yaml` exists
- Config dataclass `DecoderConfig` already has `hidden_dim` for M2F (line 65)

### 6. Data Pipeline Gap

**Current Transform Output** (`transforms.py:391-440`):
```python
# Returns: (image_tensor, label_tensor)
# Where label_tensor is [H, W] with class IDs per pixel (semantic segmentation)
```

**Required for M2F Training**:
```python
# Returns: (image_tensor, target_dict)
# Where target_dict = {"masks": [N, H, W], "labels": [N]}
# N = number of unique instances/classes in the image
```

**Conversion Needed**: Semantic mask `[H, W]` → Instance format `{"masks": [N, H, W], "labels": [N]}`

For semantic segmentation with M2F, each unique class becomes an "instance":
```python
def semantic_to_instance_targets(semantic_mask, num_classes):
    masks = []
    labels = []
    for class_id in range(num_classes):
        if class_id == ignore_label:
            continue
        mask = (semantic_mask == class_id)
        if mask.any():
            masks.append(mask)
            labels.append(class_id)
    return {"masks": torch.stack(masks), "labels": torch.tensor(labels)}
```

### 7. Inference Post-Processing (Reference)

**Location**: `dinov3/eval/segmentation/inference.py:81-85`

Shows how M2F outputs are converted back to semantic segmentation:
```python
if decoder_head_type == "m2f":
    mask_pred, mask_cls = pred["pred_masks"], pred["pred_logits"]
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # Remove "no object"
    mask_pred = mask_pred.sigmoid()
    pred = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
```

This einsum aggregates query predictions into per-class predictions.

## Architecture Documentation

### Current Module Structure

```
dinov3/eval/segmentation/
├── run.py              # Entry point, routes to train/test
├── train.py            # Training loop (linear only)
├── eval.py             # Evaluation with sliding window
├── inference.py        # Inference with M2F post-processing
├── loss.py             # CrossEntropy/Dice for linear head
├── mask_classification_loss.py  # User's M2F loss (NEW)
├── config.py           # Config dataclasses
├── transforms.py       # Data transforms (semantic format)
├── models/
│   ├── __init__.py     # build_segmentation_decoder factory
│   ├── heads/
│   │   ├── linear_head.py      # Simple 1x1 conv decoder
│   │   └── mask2former_head.py # Query-based decoder
│   └── backbone/
│       └── dinov3_adapter.py   # Multi-scale feature adapter
└── configs/
    ├── config-ade20k-linear-training.yaml
    └── config-ade20k-m2f-inference.yaml
```

### Data Flow Comparison

**Linear Training**:
```
Image → Backbone → Features[List] → LinearHead → Logits[B,C,H,W]
                                                       ↓
GT[B,H,W] ─────────────────────────────→ CrossEntropyLoss
```

**M2F Training (Required)**:
```
Image → DINOv3_Adapter → Features[Dict] → M2FHead → {pred_masks, pred_logits, aux_outputs}
                                                              ↓
GT[List[dict]] ─────────────────────────────→ MaskClassificationLoss
    ↑                                                (with Hungarian matching)
    │
semantic_to_instance_targets()
```

## Code References

| Component | File | Line |
|-----------|------|------|
| Training restriction | `train.py` | 150 |
| Linear train_step | `train.py` | 100-143 |
| M2F head forward | `mask2former_head.py` | 79-81 |
| M2F output structure | `mask2former_transformer_decoder.py` | 435-442 |
| User's loss class | `mask_classification_loss.py` | 22-121 |
| Hungarian matcher | `mask_classification_loss.py` | 47-52 |
| Target format | `mask_classification_loss.py` | 61-64 |
| M2F inference post-proc | `inference.py` | 81-85 |
| Model factory | `models/__init__.py` | 76-139 |
| Config dataclass | `config.py` | 109-127 |

## Gap Analysis

### What Exists

1. **Mask2Former head** - Complete implementation with pixel decoder and transformer decoder
2. **MaskClassificationLoss** - User-provided, extends HuggingFace implementation
3. **Model building** - Factory supports both linear and m2f decoder types
4. **Inference pipeline** - Full support for M2F with post-processing
5. **Config infrastructure** - Dataclasses support M2F parameters (hidden_dim, etc.)

### What's Missing

1. **M2F Training Config** - Need `config-ade20k-m2f-training.yaml`
2. **Target Conversion Transform** - Convert semantic masks to instance format
3. **M2F Train Step** - New function handling dict outputs and aux losses
4. **Training Loop Integration** - Remove linear-only assertion, add M2F branch
5. **Collate Function** - Handle variable-length instance lists per image

### Key Integration Points

1. **train.py:150** - Remove/modify assertion to allow M2F
2. **train.py:100-143** - Add M2F-specific train_step or modify existing
3. **train.py:251-253** - Replace `MultiSegmentationLoss` with `MaskClassificationLoss` for M2F
4. **transforms.py** - Add `SemanticToInstanceTargets` transform
5. **config.py** - Add M2F-specific training parameters (loss coefficients)

## Appendix: MaskClassificationLoss Parameters

The user's loss class requires these hyperparameters:

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `num_points` | 12544 | Points sampled for mask loss |
| `oversample_ratio` | 3.0 | Oversampling ratio for point sampling |
| `importance_sample_ratio` | 0.75 | Ratio of important points |
| `mask_coefficient` | 5.0 | Weight for binary mask loss |
| `dice_coefficient` | 5.0 | Weight for dice loss |
| `class_coefficient` | 2.0 | Weight for classification loss |
| `num_labels` | 150 | Number of classes (ADE20K) |
| `no_object_coefficient` | 0.1 | Weight for "no object" class |

These values are from the original Mask2Former paper for semantic segmentation.
