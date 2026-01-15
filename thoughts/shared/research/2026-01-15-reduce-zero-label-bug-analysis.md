---
date: 2026-01-15T10:30:00+08:00
researcher: Claude
git_commit: e1f9462aa66f1242e45df8a140fb1972d0333cc5
branch: add_visualization
repository: dinov3/add_visualiztion
topic: "reduce_zero_label Bug Analysis - Visualization and Metrics"
tags: [research, codebase, segmentation, reduce_zero_label, bug-analysis]
status: complete
last_updated: 2026-01-15
last_updated_by: Claude
---

# Research: reduce_zero_label Bug Analysis

**Date**: 2026-01-15T10:30:00+08:00
**Researcher**: Claude
**Git Commit**: e1f9462aa66f1242e45df8a140fb1972d0333cc5
**Branch**: add_visualization
**Repository**: dinov3/add_visualiztion

## Research Question

When using `eval.reduce_zero_label=true` with `decoder_head.num_classes=3`, the visualization only shows 3 classes and the class mapping appears wrong: [0,1,2,3] -> [0,null,1,2]. Is there a bug in visualization, loss calculation, or metrics?

## Summary

**Two critical bugs were identified:**

1. **BUG #1 (eval.py:103)**: `reduce_zero_label=True` is HARDCODED in the metrics calculation, ignoring the config value entirely.

2. **BUG #2 (eval.py:92-97)**: Visualization saves raw model predictions without reversing the class index mapping when `reduce_zero_label=True`.

**Loss calculation is CORRECT** - the training pipeline properly uses `config.eval.reduce_zero_label` for both training transforms and loss computation.

## Detailed Findings

### 1. The `reduce_zero_label` Label Transformation

The transformation is implemented in `metrics.py:68-73`:

```python
def preprocess_nonzero_labels(label, ignore_index=255):
    label_new = label.clone()
    label_new[label_new == ignore_index] += 1  # 255 -> 256
    label_new -= 1                              # All values -1
    label_new[label_new == -1] = ignore_index   # -1 (was 0) -> 255
    return label_new
```

**Effect on labels:**
| Original Label | After Transformation | Meaning |
|----------------|---------------------|---------|
| 0 | 255 (ignored) | Background excluded from metrics/loss |
| 1 | 0 | Becomes class 0 |
| 2 | 1 | Becomes class 1 |
| 3 | 2 | Becomes class 2 |
| 255 | 255 | Stays ignored |

### 2. Training Flow (CORRECT)

The training pipeline correctly uses the config value:

1. **train_m2f.py:262** - Passes `reduce_zero_label=config.eval.reduce_zero_label` to transforms
2. **transforms.py:473-474** - Applies `ReduceZeroLabel()` transform if enabled
3. **transforms.py:494-498** - `SemanticToM2FTargets` converts to M2F format using reduced labels
4. **mask_classification_loss.py** - Loss is computed on reduced labels [0,1,2]

When `reduce_zero_label=True` and `num_classes=3`:
- Model learns to predict classes [0,1,2]
- These correspond to original classes [1,2,3]
- Original class 0 is excluded from training

### 3. Evaluation Flow (BUG #1 - HARDCODED VALUE)

In `eval.py:99-104`:

```python
intersect_and_union = calculate_intersect_and_union(
    aggregated_preds[0],
    gt,
    num_classes=num_classes,
    reduce_zero_label=True,  # <-- HARDCODED! Should use config value
)
```

**Impact**: Even if user sets `eval.reduce_zero_label=false`, the evaluation will ALWAYS apply the label transformation to ground truth. This breaks the assumption that `reduce_zero_label=false` means direct comparison.

The `evaluate_segmentation_model` function signature doesn't even accept a `reduce_zero_label` parameter to pass from the config.

### 4. Visualization Flow (BUG #2 - NO REVERSE MAPPING)

In `eval.py:92-97`:

```python
if sample_idx < num_visualizations and vis_dir is not None and distributed.get_rank() == 0:
    pred_mask = aggregated_preds[0, 0]  # [H, W] with class indices
    vis_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}.png")
    save_visualization(pred_mask, vis_path)
```

**Problem**: The raw model predictions (classes [0,1,2]) are saved directly without reversing the mapping.

**Expected behavior when `reduce_zero_label=True`**:
- Model predicts class 0 → should save as original class 1
- Model predicts class 1 → should save as original class 2
- Model predicts class 2 → should save as original class 3

**Actual behavior**:
- Model predicts class 0 → saves as class 0 (WRONG - should be 1)
- Model predicts class 1 → saves as class 1 (WRONG - should be 2)
- Model predicts class 2 → saves as class 2 (WRONG - should be 3)

### 5. Linear Head Training (Same Pattern)

`train.py:175` also uses `reduce_zero_label=config.eval.reduce_zero_label` for training transforms, so the same behavior applies.

### 6. Class Mapping Summary

When user sets `eval.reduce_zero_label=true` and `decoder_head.num_classes=3`:

| Data Stage | Class Values | Correct? |
|------------|--------------|----------|
| Original GT mask | [0,1,2,3] | - |
| After training transform | [255,0,1,2] | YES |
| M2F binary masks | masks for [0,1,2] | YES |
| Loss computation | classes [0,1,2] | YES |
| Model predictions | classes [0,1,2] | YES |
| Metrics GT transform | [255,0,1,2] (hardcoded) | BUG |
| Visualization output | classes [0,1,2] | BUG (should be [1,2,3]) |

## Code References

- `dinov3/eval/segmentation/eval.py:103` - Hardcoded `reduce_zero_label=True`
- `dinov3/eval/segmentation/eval.py:92-97` - Visualization without reverse mapping
- `dinov3/eval/segmentation/eval.py:45-57` - `evaluate_segmentation_model` function signature (missing reduce_zero_label param)
- `dinov3/eval/segmentation/metrics.py:68-73` - `preprocess_nonzero_labels` transformation
- `dinov3/eval/segmentation/metrics.py:76-98` - `calculate_intersect_and_union` function
- `dinov3/eval/segmentation/transforms.py:104-113` - `ReduceZeroLabel` transform class
- `dinov3/eval/segmentation/transforms.py:391-439` - `SemanticToM2FTargets` transform
- `dinov3/eval/segmentation/train_m2f.py:262` - Training uses config value correctly
- `dinov3/eval/segmentation/train.py:175` - Linear training uses config value correctly
- `dinov3/eval/segmentation/config.py:115` - Config default `reduce_zero_label: True`

## Architecture Documentation

### Data Flow Diagram

```
Training:
  Raw GT [0,1,2,3]
       │
       ▼ (ReduceZeroLabel transform, if config.eval.reduce_zero_label=true)
  Reduced GT [255,0,1,2]
       │
       ▼ (SemanticToM2FTargets)
  Binary masks for classes [0,1,2]
       │
       ▼ (MaskClassificationLoss)
  Model learns to predict [0,1,2]

Validation:
  Raw GT [0,1,2,3]
       │
       ├──▶ Model predictions [0,1,2]
       │           │
       │           ├──▶ save_visualization() → saves [0,1,2] (BUG: should reverse to [1,2,3])
       │           │
       │           └──▶ calculate_intersect_and_union(reduce_zero_label=True) ← HARDCODED
       │                        │
       │                        ▼
       │              Transforms GT: [255,0,1,2]
       │                        │
       │                        ▼
       │              Compares pred [0,1,2] with GT [0,1,2] → metrics (correct comparison)
       │
       └──▶ No transform applied to GT during validation dataloader
            (val_transforms doesn't include ReduceZeroLabel)
```

### Key Insight

The metrics calculation itself is **mathematically correct** when `reduce_zero_label=True` because:
- Model outputs [0,1,2]
- GT is transformed to [0,1,2] (from original [1,2,3])
- Comparison is valid

**However**, the hardcoded value means:
- Users cannot disable this behavior via config
- `eval.reduce_zero_label=false` is silently ignored during evaluation

## Historical Context (from thoughts/)

The `notes.md` file documents the expected behavior of `reduce_zero_label`:

> Despite being named `eval.reduce_zero_label`, this setting affects **training loss** as well, not just evaluation metrics.

The notes correctly describe the transformation but don't mention the hardcoded bug in `eval.py:103`.

## Open Questions

1. **Why was `reduce_zero_label=True` hardcoded?** - Possibly leftover from ADE20K-specific code that assumed this would always be true.

2. **Should visualization show original or reduced class indices?** - Depends on use case. Showing original indices makes visual comparison with GT easier.

3. **Are there other places where config values are ignored?** - A systematic audit of all config parameter usages would be valuable.

## Recommendations

### Fix #1: Make `reduce_zero_label` configurable in evaluation

```python
# eval.py - evaluate_segmentation_model signature should include:
def evaluate_segmentation_model(
    ...
    reduce_zero_label: bool = True,  # Add parameter
):
    ...
    intersect_and_union = calculate_intersect_and_union(
        aggregated_preds[0],
        gt,
        num_classes=num_classes,
        reduce_zero_label=reduce_zero_label,  # Use parameter
    )
```

### Fix #2: Add reverse mapping to visualization

```python
# eval.py - save_visualization or caller should reverse mapping
if sample_idx < num_visualizations and vis_dir is not None:
    pred_mask = aggregated_preds[0, 0]  # [H, W]
    if reduce_zero_label:
        # Reverse the mapping: add 1 to get original class indices
        pred_mask = pred_mask + 1
    vis_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}.png")
    save_visualization(pred_mask, vis_path)
```

### Fix #3: Pass `reduce_zero_label` through the call chain

Update all callers of `evaluate_segmentation_model`:
- `train_m2f.py:validate_m2f`
- `train.py:validate`
- `eval.py:test_segmentation`

To pass `config.eval.reduce_zero_label` to the evaluation function.
