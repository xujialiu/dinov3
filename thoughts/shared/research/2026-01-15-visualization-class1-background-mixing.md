---
date: 2026-01-15T11:36:51+08:00
researcher: Claude
git_commit: b58c304b71400d0c9adbffba8d83f39e15bbf510
branch: add_visualization
repository: dinov3/add_visualiztion
topic: "Visualization Bug - Class 1 Mixed with Background After reduce_zero_label Fix"
tags: [research, codebase, segmentation, reduce_zero_label, visualization, bug-analysis]
status: complete
last_updated: 2026-01-15
last_updated_by: Claude
---

# Research: Visualization Bug - Class 1 Mixed with Background

**Date**: 2026-01-15T11:36:51+08:00
**Researcher**: Claude
**Git Commit**: b58c304b71400d0c9adbffba8d83f39e15bbf510
**Branch**: add_visualization
**Repository**: dinov3/add_visualiztion

## Research Question

After fixing the `reduce_zero_label` bug, visualization images show classes [1, 2, 3] as expected, but class 1 appears mixed with the background. Why is this happening and how can it be resolved?

## Summary

**Root Cause Identified**: When `reduce_zero_label=True` with `num_classes=3`, the model has NO output channel for background. During inference, the model must predict something for EVERY pixel (including background pixels). Since the model only knows classes [0, 1, 2], background pixels get predicted as one of these - typically class 0, which becomes class 1 after the `restore_original_labels(+1)` transformation.

**This is expected behavior** given the current architecture, not a bug. The previous fix correctly added the `+1` restoration, but this reveals a fundamental limitation: the model cannot distinguish background when trained with `reduce_zero_label=True`.

## Detailed Findings

### 1. The Data Flow with `reduce_zero_label=True`

| Stage | Label Values | Notes |
|-------|--------------|-------|
| **Original Dataset** | [0, 1, 2, 3] | 0=background, 1/2/3=foreground classes |
| **Training Transform** | [255, 0, 1, 2] | `ReduceZeroLabel` shifts labels, 0→ignored |
| **Model Output Channels** | 3 channels | Only [0, 1, 2] - NO background channel |
| **Model Predictions** | [0, 1, 2] | For ALL pixels including background |
| **After restore_original_labels** | [1, 2, 3] | +1 to all predictions |

### 2. Why Background Pixels Become Class 1

```
Background pixel flow:
┌─────────────────────────────────────────────────────────────────────────┐
│ Original GT: class 0 (background)                                       │
│      ↓                                                                  │
│ Training: Ignored (255) - model NEVER learns what background looks like │
│      ↓                                                                  │
│ Inference: Model must output SOMETHING for this pixel                   │
│      ↓                                                                  │
│ Model's guess: Usually class 0 (most common/default output)             │
│      ↓                                                                  │
│ After restore_original_labels(+1): Class 0 → Class 1                    │
│      ↓                                                                  │
│ Visualization shows: Class 1 (INCORRECT - should be background)         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. Model Architecture Limitation

From `dinov3/eval/segmentation/inference.py:136-139`:
```python
mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # Remove no-object class
mask_pred = mask_pred.sigmoid()
crop_pred = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
```

- Mask2Former outputs `num_classes + 1` classes (3 semantic + 1 no-object)
- The "no-object" class is removed during inference via `[..., :-1]`
- Final output has exactly `num_classes=3` channels
- There is NO channel representing "this pixel is background"

### 4. The Visualization Code is Correct

From `dinov3/eval/segmentation/eval.py:94-101`:
```python
if sample_idx < num_visualizations and vis_dir is not None and distributed.get_rank() == 0:
    pred_mask = aggregated_preds[0, 0]  # [H, W] with class indices
    if reduce_zero_label:
        pred_mask = restore_original_labels(pred_mask)  # +1 to all
    vis_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}.png")
    save_visualization(pred_mask, vis_path)
```

The fix from the previous plan is **correctly implemented**. The issue is that:
- `aggregated_preds` contains [0, 1, 2] for ALL pixels (including background)
- `restore_original_labels(+1)` correctly maps to [1, 2, 3]
- But background pixels were predicted as class 0 → now show as class 1

### 5. Comparison: What Happens with `reduce_zero_label=False`

| Setting | Model Output | Background Handling | Visualization |
|---------|--------------|--------------------|----|
| `reduce_zero_label=True` | 3 classes [0,1,2] | No background class | Background shows as class 1 |
| `reduce_zero_label=False` | 4 classes [0,1,2,3] | Class 0 IS background | Background shows as class 0 |

When `reduce_zero_label=False` with `num_classes=4`:
- Model learns all 4 classes including background (class 0)
- Model can correctly predict "this pixel is background"
- Visualization correctly shows background as class 0

## Code References

- `dinov3/eval/segmentation/eval.py:94-101` - Visualization with restore_original_labels
- `dinov3/eval/segmentation/metrics.py:76-89` - restore_original_labels function
- `dinov3/eval/segmentation/inference.py:136-139` - Model output post-processing
- `dinov3/eval/segmentation/transforms.py:104-113` - ReduceZeroLabel transform
- `dinov3/eval/segmentation/config.py:115` - reduce_zero_label config

## Architecture Documentation

### Why This Behavior Exists

The `reduce_zero_label` feature was designed for ADE20K dataset where:
- Label 0 represents "unlabeled/background" pixels
- These pixels should be **excluded from evaluation metrics**
- The model doesn't need to distinguish them - metrics simply ignore them

This works correctly for **metrics calculation** but creates ambiguity in **visualization**:
- Metrics: Background pixels (GT=0) are ignored, so model's prediction doesn't matter
- Visualization: We want to see what the model predicted, but it has no background concept

### Fundamental Trade-off

| Approach | Pros | Cons |
|----------|------|------|
| **reduce_zero_label=True** | Model focuses on foreground classes, smaller output layer | Cannot visualize background correctly |
| **reduce_zero_label=False** | Can distinguish background | Model must learn an extra class |

## Potential Solutions

### Solution 1: Use GT to Mask Background in Visualization

Modify visualization to use GT labels to mask out background pixels:

```python
# In eval.py visualization section
if sample_idx < num_visualizations and vis_dir is not None and distributed.get_rank() == 0:
    pred_mask = aggregated_preds[0, 0]  # [H, W]
    if reduce_zero_label:
        pred_mask = restore_original_labels(pred_mask)
        # Mask background pixels using GT
        pred_mask[gt[0] == 0] = 0  # Set background to class 0
    vis_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}.png")
    save_visualization(pred_mask, vis_path)
```

**Pros**: Visualization shows correct background
**Cons**: Requires GT during visualization (may not be available during pure inference)

### Solution 2: Use `reduce_zero_label=False` for Datasets with Background

For datasets where visualizing background is important:
- Set `reduce_zero_label=False`
- Set `num_classes=4` (including background)
- Model learns to predict background as class 0

**Pros**: Architecturally clean, no post-processing needed
**Cons**: Model must learn an extra class

### Solution 3: Threshold-Based Background Detection

Use prediction confidence to detect "uncertain" pixels as background:

```python
# Alternative approach using confidence
pred_probs = aggregated_preds / len(batch_img)  # Before argmax
max_prob = pred_probs.max(dim=1)[0]
pred_mask = pred_probs.argmax(dim=1)
pred_mask[max_prob < threshold] = 0  # Low confidence → background
```

**Pros**: Doesn't require GT
**Cons**: Threshold selection is arbitrary, may misclassify valid predictions

### Solution 4: Save Separate Visualization Types

Save multiple visualizations:
1. Raw predictions (current behavior)
2. Predictions masked with GT
3. GT labels for comparison

**Pros**: Comprehensive visualization
**Cons**: More disk space, more complex code

## Historical Context (from thoughts/)

From `thoughts/shared/research/2026-01-15-reduce-zero-label-bug-analysis.md`:
- Previous analysis correctly identified hardcoded `reduce_zero_label=True` bug
- Previous analysis correctly identified missing `restore_original_labels` call
- However, did not anticipate the "background has no class" issue

From `thoughts/shared/plans/2026-01-15-reduce-zero-label-bug-fixes.md`:
- Phase 1: Added `restore_original_labels` function ✓
- Phase 2: Threaded `reduce_zero_label` parameter ✓
- Phase 3: Created tests ✓
- **Missing**: Background handling in visualization

## Open Questions

1. **What is the expected visualization behavior?**
   - Should background pixels show as class 0?
   - Or should they be masked/transparent?
   - Or should we overlay predictions on the original image?

2. **Is GT available during inference?**
   - If yes, Solution 1 (GT masking) is viable
   - If no, need Solution 2 or 3

3. **Should we change the model architecture?**
   - Always include a background class regardless of `reduce_zero_label`?
   - This would require retraining existing models

## Recommendations

### Immediate Fix (Solution 1)
If GT is available during visualization, add GT-based background masking:

```python
# In eval.py, line 97-98
if reduce_zero_label:
    pred_mask = restore_original_labels(pred_mask)
    pred_mask[gt[0] == 0] = 0  # Restore background from GT
```

### Long-term Recommendation
For future training runs where background visualization is important:
- Use `reduce_zero_label=False` with `num_classes` including background
- This allows the model to learn and predict background explicitly

## Verification Command

After implementing fix, verify with:
```bash
CUDA_VISIBLE_DEVICES=0 python dinov3/eval/segmentation/run.py \
    config=dinov3/eval/segmentation/configs/config-ade20k-m2f-training.yaml \
    output_dir=./output/m2f_reduce_zero_label_T \
    model.dino_hub=dinov3_vitl16 \
    datasets.root=../semantic_retina_vessel_segmentation \
    bs=3 \
    n_gpus=1 \
    eval.num_visualizations=3 \
    decoder_head.num_classes=3 \
    eval.reduce_zero_label=true

# Check visualization outputs - background should now be class 0, foreground [1,2,3]
```

## Related Research

- `thoughts/shared/research/2026-01-15-reduce-zero-label-bug-analysis.md` - Original bug analysis
- `thoughts/shared/plans/2026-01-15-reduce-zero-label-bug-fixes.md` - Implementation plan
