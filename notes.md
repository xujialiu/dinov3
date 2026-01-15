# DINOv3 Notes

## Segmentation: `num_classes` and `reduce_zero_label`

When configuring segmentation training/evaluation, these two parameters work together:

### `reduce_zero_label=False` (default for most custom datasets)
- Label 0 is treated as a real class (e.g., background)
- `num_classes` should equal the total number of unique labels in your mask
- Example: mask with labels [0, 1, 2, 3] → `num_classes=4`

### `reduce_zero_label=True` (ADE20K convention)
- Label 0 is ignored during evaluation (set to ignore_index=255)
- Labels are shifted down by 1: label 1→0, label 2→1, label 3→2
- `num_classes` should be (total unique labels - 1)
- Example: mask with labels [0, 1, 2, 3] → `num_classes=3`

### Important: `eval.reduce_zero_label` affects BOTH training and evaluation

Despite being named `eval.reduce_zero_label`, this setting affects **training loss** as well, not just evaluation metrics.

The transform is applied during data loading in both training loops:
- `train_m2f.py:257`: `reduce_zero_label=config.eval.reduce_zero_label`
- `train.py:175`: `reduce_zero_label=config.eval.reduce_zero_label`

The `ReduceZeroLabel` transform (`transforms.py:104-113`) remaps labels before they reach the loss function:
```python
def preprocess_nonzero_labels(label, ignore_index=255):
    label_new = label.clone()
    label_new[label_new == ignore_index] += 1  # Protect existing 255
    label_new -= 1                              # Shift ALL labels down
    label_new[label_new == -1] = ignore_index   # Class 0 → 255 (ignored)
    return label_new
```

**In summary, `eval.reduce_zero_label=True` causes:**
1. Training data transforms → class 0 excluded from loss
2. Validation data transforms → class 0 excluded from metrics
3. Metric calculation → uses remapped labels

The config naming is misleading - it should probably be at a higher level.

| Original mask label | After `reduce_zero_label=True` | Model predicts |
|---------------------|-------------------------------|----------------|
| 0 (background)      | → 255 (ignored)               | - |
| 1                   | → 0                           | class 0 |
| 2                   | → 1                           | class 1 |
| 3                   | → 2                           | class 2 |

### Configuration examples

**Include background in evaluation:**
```bash
decoder_head.num_classes=4 \
eval.reduce_zero_label=False
```

**Ignore background in evaluation:**
```bash
decoder_head.num_classes=3 \
eval.reduce_zero_label=True
```

### Troubleshooting

If you see `NaN` for a class in mIoU output, it usually means:
1. Mismatch between `num_classes` and `reduce_zero_label` settings
2. That class has no ground truth samples in the validation set

### Why `reduce_zero_label` Exists (Historical Context)

This feature was designed for **ADE20K dataset** where label 0 has a special meaning:

| Dataset Type | Label 0 Meaning | Use |
|--------------|-----------------|-----|
| **ADE20K** | "Unlabeled/void" - pixels annotators couldn't categorize | `reduce_zero_label=True` |
| **Most custom datasets** | "Background" - a real semantic class | `reduce_zero_label=False` |

**Key insight**: In ADE20K, label 0 is NOT "background" - it's "we don't know what this is."

### Visualization Implication

When `reduce_zero_label=True`:
- Model has **NO output channel for background/label-0**
- During inference, model must predict something for ALL pixels
- Background pixels get predicted as class 0, 1, or 2 (model's best guess)
- After `restore_original_labels(+1)`, these show as class 1, 2, or 3
- **Result**: Background appears "mixed" with foreground classes in visualization

This is **expected behavior**, not a bug. The model was never trained to recognize background.

### How to Choose the Right Setting

```
Is your label 0 a real class you want to segment?
│
├─ YES (e.g., "background" class) ──→ reduce_zero_label=False, num_classes=N
│
└─ NO (e.g., "ignore/void" pixels) ──→ reduce_zero_label=True, num_classes=N-1
```

**Example: Retina vessel segmentation with labels [0, 1, 2, 3]**

| If label 0 means... | Setting | num_classes |
|---------------------|---------|-------------|
| Background (real class) | `reduce_zero_label=False` | 4 |
| Ignore these pixels | `reduce_zero_label=True` | 3 |

---

## M2F Training: Evaluation Code Execution Flow

During Mask2Former training, evaluation is triggered periodically based on `eval.eval_interval`.

### 1. Training Loop Triggers Evaluation
**File:** `dinov3/eval/segmentation/train_m2f.py:381-395`
```python
# Periodic validation
if global_step % config.eval.eval_interval == 0:
    dist.barrier()
    is_better, best_metric_values_dict = validate_m2f(...)
```

### 2. `validate_m2f()` Function
**File:** `dinov3/eval/segmentation/train_m2f.py:105-137`

Wrapper that calls the core evaluation:
```python
new_metric_values_dict = evaluate_segmentation_model(
    segmentation_model,
    val_dataloader,
    device,
    eval_res,
    eval_stride,
    decoder_head_type="m2f",
    num_classes=num_classes,
    autocast_dtype=autocast_dtype,
    max_samples=max_val_samples,
)
```

### 3. Core Evaluation Logic
**File:** `dinov3/eval/segmentation/eval.py:29-83`

The `evaluate_segmentation_model()` function:
1. **Line 41:** Sets model to eval mode
2. **Line 45-71:** Iterates over validation samples
3. **Line 52-63:** Runs sliding window inference via `make_inference()`
4. **Line 65-70:** Calculates intersection and union per sample
5. **Line 77-80:** Aggregates metrics (mIoU, dice, fscore)

### Key Files Summary

| File | Function | Line |
|------|----------|------|
| `train_m2f.py` | Training loop check | 381 |
| `train_m2f.py` | `validate_m2f()` | 105-137 |
| `eval.py` | `evaluate_segmentation_model()` | 29-83 |
| `inference.py` | `make_inference()` | sliding window |
| `metrics.py` | `calculate_intersect_and_union()` | per-sample |
| `metrics.py` | `calculate_segmentation_metrics()` | final aggregation |
