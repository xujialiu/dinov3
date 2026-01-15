# Fix `reduce_zero_label` Bugs in Evaluation and Visualization

## Overview

This plan addresses two critical bugs identified in the segmentation evaluation pipeline:

1. **BUG #1**: `reduce_zero_label=True` is hardcoded at `eval.py:103`, ignoring the config value
2. **BUG #2**: Visualizations save raw model predictions without reversing the class index mapping

## Current State Analysis

### Code Flow
- Training correctly uses `config.eval.reduce_zero_label` for transforms (`train_m2f.py:262`, `train.py:175`)
- Evaluation IGNORES the config and hardcodes `reduce_zero_label=True` at `eval.py:103`
- Visualization saves predictions in reduced label space (0,1,2) instead of original space (1,2,3)

### Files Affected
| File | Purpose |
|------|---------|
| `dinov3/eval/segmentation/eval.py` | Main fix location - hardcoded value and visualization |
| `dinov3/eval/segmentation/train.py` | Thread parameter through `validate()` |
| `dinov3/eval/segmentation/train_m2f.py` | Thread parameter through `validate_m2f()` |
| `dinov3/eval/segmentation/metrics.py` | Add reverse mapping helper (new function) |

## Desired End State

After implementation:
1. `config.eval.reduce_zero_label` is respected throughout evaluation
2. Visualizations show original class indices (1,2,3,...) when `reduce_zero_label=True`
3. Automated tests verify both behaviors

### How to Verify
```bash
# Run automated tests
PYTHONPATH=. pytest dinov3/eval/segmentation/tests/ -v

# Manual verification (optional)
# Train with reduce_zero_label=True, check visualization outputs show classes 1,2,3 not 0,1,2
```

## What We're NOT Doing

- NOT changing the training pipeline (already correct)
- NOT modifying the loss calculation (already correct)
- NOT changing the `preprocess_nonzero_labels` function
- NOT adding color-mapped visualizations (keeping grayscale)

## Implementation Approach

Thread `reduce_zero_label` parameter through the evaluation call chain and add reverse mapping for visualization. Create comprehensive automated tests.

---

## Phase 1: Add Reverse Mapping Function

### Overview
Add a helper function in `metrics.py` to reverse the label reduction mapping.

### Changes Required:

#### 1. Add `restore_original_labels` function
**File**: `dinov3/eval/segmentation/metrics.py`
**Location**: After `preprocess_nonzero_labels` (line 73)

```python
def restore_original_labels(reduced_labels: torch.Tensor) -> torch.Tensor:
    """Reverse the label reduction mapping.

    When reduce_zero_label=True, model predicts classes [0, 1, 2, ...N-1]
    which correspond to original classes [1, 2, 3, ...N].
    This function restores predictions to the original label space.

    Args:
        reduced_labels: Tensor with reduced class indices (0-indexed)

    Returns:
        Tensor with original class indices (1-indexed)
    """
    return reduced_labels + 1
```

### Success Criteria:

#### Automated Verification:
- [x] Unit test passes: `PYTHONPATH=. pytest dinov3/eval/segmentation/tests/test_metrics.py::test_restore_original_labels -v`
- [x] Function correctly reverses [0,1,2] -> [1,2,3]

---

## Phase 2: Thread `reduce_zero_label` Through Evaluation

### Overview
Add `reduce_zero_label` parameter to `evaluate_segmentation_model` and all callers.

### Changes Required:

#### 1. Update `evaluate_segmentation_model` signature
**File**: `dinov3/eval/segmentation/eval.py`
**Location**: Line 45-58

```python
def evaluate_segmentation_model(
    segmentation_model: torch.nn.Module,
    test_dataloader,
    device,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    autocast_dtype,
    max_samples: int = 0,
    num_visualizations: int = 0,
    output_dir: str | None = None,
    global_step: int = 0,
    reduce_zero_label: bool = True,  # NEW PARAMETER
):
```

#### 2. Use parameter in metrics calculation
**File**: `dinov3/eval/segmentation/eval.py`
**Location**: Line 99-104

Replace:
```python
intersect_and_union = calculate_intersect_and_union(
    aggregated_preds[0],
    gt,
    num_classes=num_classes,
    reduce_zero_label=True,  # HARDCODED - BUG
)
```

With:
```python
intersect_and_union = calculate_intersect_and_union(
    aggregated_preds[0],
    gt,
    num_classes=num_classes,
    reduce_zero_label=reduce_zero_label,  # USE PARAMETER
)
```

#### 3. Add reverse mapping in visualization
**File**: `dinov3/eval/segmentation/eval.py`
**Location**: Line 92-97

Add import at top of file:
```python
from dinov3.eval.segmentation.metrics import (
    calculate_intersect_and_union,
    calculate_segmentation_metrics,
    restore_original_labels,  # NEW IMPORT
)
```

Replace visualization code:
```python
# Save visualization for first num_visualizations samples (only on rank 0)
if sample_idx < num_visualizations and vis_dir is not None and distributed.get_rank() == 0:
    pred_mask = aggregated_preds[0, 0]  # [H, W] with class indices
    if reduce_zero_label:
        pred_mask = restore_original_labels(pred_mask)
    vis_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}.png")
    save_visualization(pred_mask, vis_path)
    logger.info(f"Saved visualization to {vis_path}")
```

#### 4. Update `validate` function in train.py
**File**: `dinov3/eval/segmentation/train.py`
**Location**: Line 67-98

Add parameter to function signature:
```python
def validate(
    segmentation_model: torch.nn.Module,
    val_dataloader,
    device,
    autocast_dtype,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    global_step,
    metric_to_save,
    current_best_metric_to_save_value,
    reduce_zero_label: bool = True,  # NEW PARAMETER
):
    new_metric_values_dict = evaluate_segmentation_model(
        segmentation_model,
        val_dataloader,
        device,
        eval_res,
        eval_stride,
        decoder_head_type,
        num_classes,
        autocast_dtype,
        reduce_zero_label=reduce_zero_label,  # PASS THROUGH
    )
```

#### 5. Update `validate` call site in train.py
**File**: `dinov3/eval/segmentation/train.py`
**Location**: Around line 286-298 (the call to `validate`)

Add `reduce_zero_label=config.eval.reduce_zero_label` to the call.

#### 6. Update `validate_m2f` function in train_m2f.py
**File**: `dinov3/eval/segmentation/train_m2f.py`
**Location**: Line 105-142

Add parameter to function signature:
```python
def validate_m2f(
    segmentation_model: torch.nn.Module,
    val_dataloader,
    device,
    autocast_dtype,
    eval_res,
    eval_stride,
    num_classes,
    global_step,
    metric_to_save,
    current_best_metric_to_save_value,
    max_val_samples: int = 0,
    num_visualizations: int = 0,
    output_dir: str | None = None,
    reduce_zero_label: bool = True,  # NEW PARAMETER
):
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
        num_visualizations=num_visualizations,
        output_dir=output_dir,
        global_step=global_step,
        reduce_zero_label=reduce_zero_label,  # PASS THROUGH
    )
```

#### 7. Update `validate_m2f` call site in train_m2f.py
**File**: `dinov3/eval/segmentation/train_m2f.py`
**Location**: Around line 388-402 (the call to `validate_m2f`)

Add `reduce_zero_label=config.eval.reduce_zero_label` to the call.

#### 8. Update `test_segmentation` function
**File**: `dinov3/eval/segmentation/eval.py`
**Location**: Line 175-184

Add `reduce_zero_label` parameter to the `evaluate_segmentation_model` call:
```python
return evaluate_segmentation_model(
    segmentation_model=segmentation_model,
    test_dataloader=test_dataloader,
    device=device,
    eval_res=eval_res,
    eval_stride=eval_stride,
    decoder_head_type=config.decoder_head.type,
    num_classes=config.decoder_head.num_classes,
    autocast_dtype=config.model_dtype.autocast_dtype,
    reduce_zero_label=config.eval.reduce_zero_label,  # ADD THIS
)
```

### Success Criteria:

#### Automated Verification:
- [x] All unit tests pass: `PYTHONPATH=. pytest dinov3/eval/segmentation/tests/ -v`
- [x] Type checking passes (if configured): `mypy dinov3/eval/segmentation/eval.py` (no new errors from Phase 2 changes; pre-existing errors unrelated to this plan)
- [x] No import errors: `python -c "from dinov3.eval.segmentation.eval import evaluate_segmentation_model"`

---

## Phase 3: Create Automated Tests

### Overview
Create comprehensive test suite for `reduce_zero_label` functionality.

### Changes Required:

#### 1. Create test directory structure
**Directory**: `dinov3/eval/segmentation/tests/`

```bash
mkdir -p dinov3/eval/segmentation/tests
touch dinov3/eval/segmentation/tests/__init__.py
```

#### 2. Create test file for metrics
**File**: `dinov3/eval/segmentation/tests/test_metrics.py`

```python
"""Tests for segmentation metrics with reduce_zero_label handling."""

import pytest
import torch

from dinov3.eval.segmentation.metrics import (
    preprocess_nonzero_labels,
    restore_original_labels,
    calculate_intersect_and_union,
)


class TestPreprocessNonzeroLabels:
    """Tests for the label reduction transformation."""

    def test_basic_transformation(self):
        """Test that labels are shifted correctly."""
        # Original labels: [0, 1, 2, 3]
        labels = torch.tensor([[0, 1], [2, 3]])
        result = preprocess_nonzero_labels(labels, ignore_index=255)

        # Expected: [255, 0, 1, 2] (0 becomes ignored, others shift down)
        expected = torch.tensor([[255, 0], [1, 2]])
        assert torch.equal(result, expected)

    def test_ignore_index_preserved(self):
        """Test that existing ignore_index values stay as ignore_index."""
        labels = torch.tensor([[255, 1], [2, 255]])
        result = preprocess_nonzero_labels(labels, ignore_index=255)

        # 255 should remain 255, 1->0, 2->1
        expected = torch.tensor([[255, 0], [1, 255]])
        assert torch.equal(result, expected)

    def test_all_zeros_become_ignored(self):
        """Test that all-zero labels all become ignore_index."""
        labels = torch.tensor([[0, 0], [0, 0]])
        result = preprocess_nonzero_labels(labels, ignore_index=255)

        expected = torch.tensor([[255, 255], [255, 255]])
        assert torch.equal(result, expected)

    def test_does_not_modify_input(self):
        """Test that input tensor is not modified in place."""
        labels = torch.tensor([[0, 1], [2, 3]])
        original = labels.clone()
        _ = preprocess_nonzero_labels(labels, ignore_index=255)

        assert torch.equal(labels, original)


class TestRestoreOriginalLabels:
    """Tests for the reverse label mapping."""

    def test_basic_restoration(self):
        """Test that reduced labels are restored correctly."""
        # Reduced labels: [0, 1, 2] (from model predictions)
        reduced = torch.tensor([[0, 1], [2, 0]])
        result = restore_original_labels(reduced)

        # Expected: [1, 2, 3] (add 1 to all)
        expected = torch.tensor([[1, 2], [3, 1]])
        assert torch.equal(result, expected)

    def test_single_class(self):
        """Test restoration with single class predictions."""
        reduced = torch.tensor([[0, 0], [0, 0]])
        result = restore_original_labels(reduced)

        expected = torch.tensor([[1, 1], [1, 1]])
        assert torch.equal(result, expected)


class TestCalculateIntersectAndUnion:
    """Tests for the intersection and union calculation."""

    def test_perfect_prediction_no_reduce(self):
        """Test with perfect predictions and reduce_zero_label=False."""
        pred = torch.tensor([[0, 1], [2, 0]]).float()
        label = torch.tensor([[0, 1], [2, 0]])

        result = calculate_intersect_and_union(
            pred, label, num_classes=3, reduce_zero_label=False
        )

        # area_intersect should equal area_label for perfect prediction
        area_intersect, area_union, area_pred, area_label = result
        assert torch.equal(area_intersect, area_label)

    def test_with_reduce_zero_label_true(self):
        """Test that reduce_zero_label transforms GT correctly."""
        # Predictions in reduced space: [0, 1, 2]
        pred = torch.tensor([[0, 1], [2, 0]]).float()
        # GT in original space: [1, 2, 3, 1]
        label = torch.tensor([[1, 2], [3, 1]])

        result = calculate_intersect_and_union(
            pred, label, num_classes=3, reduce_zero_label=True
        )

        # After GT transformation: [0, 1, 2, 0] -> should match predictions
        area_intersect, area_union, area_pred, area_label = result
        assert torch.equal(area_intersect, area_label)

    def test_with_reduce_zero_label_original_zero_ignored(self):
        """Test that original class 0 is ignored when reduce_zero_label=True."""
        pred = torch.tensor([[0, 0], [0, 0]]).float()
        # GT with original class 0 (should be ignored)
        label = torch.tensor([[0, 0], [0, 0]])

        result = calculate_intersect_and_union(
            pred, label, num_classes=3, reduce_zero_label=True
        )

        # All GT pixels become ignore_index, so all areas should be 0
        area_intersect, area_union, area_pred, area_label = result
        assert area_label.sum() == 0

    def test_ignore_index_excluded(self):
        """Test that ignore_index pixels are excluded from calculation."""
        pred = torch.tensor([[0, 1], [2, 0]]).float()
        label = torch.tensor([[0, 255], [2, 255]])  # Two pixels ignored

        result = calculate_intersect_and_union(
            pred, label, num_classes=3, reduce_zero_label=False, ignore_index=255
        )

        # Only 2 valid pixels (label 0 and label 2)
        area_intersect, area_union, area_pred, area_label = result
        assert area_label.sum() == 2  # Two valid pixels


class TestEndToEndLabelFlow:
    """Integration tests for the full label transformation flow."""

    def test_train_eval_consistency_reduce_true(self):
        """Test that training and evaluation use consistent label spaces.

        When reduce_zero_label=True:
        - Training: GT [1,2,3] -> reduced [0,1,2], model learns [0,1,2]
        - Eval: Model predicts [0,1,2], GT [1,2,3] -> reduced [0,1,2], compare
        - Visualization: Predictions [0,1,2] -> restored [1,2,3]
        """
        # Simulate model predictions (in reduced space)
        model_pred = torch.tensor([[0, 1], [2, 0]]).float()

        # Simulate original GT labels
        original_gt = torch.tensor([[1, 2], [3, 1]])

        # Evaluation with reduce_zero_label=True should work correctly
        result = calculate_intersect_and_union(
            model_pred, original_gt, num_classes=3, reduce_zero_label=True
        )
        area_intersect, _, _, area_label = result

        # Perfect match expected
        assert torch.equal(area_intersect, area_label)

        # Visualization should restore to original space
        vis_pred = restore_original_labels(model_pred.long())
        assert torch.equal(vis_pred, original_gt)

    def test_train_eval_consistency_reduce_false(self):
        """Test consistency when reduce_zero_label=False.

        When reduce_zero_label=False:
        - Training: GT [0,1,2,3] used directly, model learns [0,1,2,3]
        - Eval: Model predicts [0,1,2,3], GT [0,1,2,3], direct compare
        - Visualization: Predictions used directly
        """
        model_pred = torch.tensor([[0, 1], [2, 3]]).float()
        original_gt = torch.tensor([[0, 1], [2, 3]])

        result = calculate_intersect_and_union(
            model_pred, original_gt, num_classes=4, reduce_zero_label=False
        )
        area_intersect, _, _, area_label = result

        # Perfect match expected
        assert torch.equal(area_intersect, area_label)
```

#### 3. Create test file for visualization
**File**: `dinov3/eval/segmentation/tests/test_visualization.py`

```python
"""Tests for visualization with reduce_zero_label handling."""

import os
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch

from dinov3.eval.segmentation.eval import save_visualization
from dinov3.eval.segmentation.metrics import restore_original_labels


class TestSaveVisualization:
    """Tests for the save_visualization function."""

    def test_saves_file(self):
        """Test that visualization is saved as PNG."""
        pred_mask = torch.tensor([[0, 1], [2, 0]], dtype=torch.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            save_visualization(pred_mask, output_path)

            assert os.path.exists(output_path)
            img = Image.open(output_path)
            assert img.mode == 'L'  # Grayscale

    def test_preserves_class_indices(self):
        """Test that pixel values equal class indices."""
        pred_mask = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            save_visualization(pred_mask, output_path)

            img = Image.open(output_path)
            arr = np.array(img)

            expected = pred_mask.numpy()
            assert np.array_equal(arr, expected)


class TestVisualizationWithReduceZeroLabel:
    """Integration tests for visualization with label restoration."""

    def test_visualization_restores_labels_when_reduce_true(self):
        """Test that predictions are restored before saving when reduce_zero_label=True."""
        # Model predicts in reduced space: [0, 1, 2]
        reduced_pred = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)

        # Restore to original space: [1, 2, 3]
        restored_pred = restore_original_labels(reduced_pred)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            save_visualization(restored_pred, output_path)

            img = Image.open(output_path)
            arr = np.array(img)

            # Should be [1, 2, 3, 1], not [0, 1, 2, 0]
            expected = torch.tensor([[1, 2], [3, 1]]).numpy()
            assert np.array_equal(arr, expected)

    def test_visualization_preserves_when_reduce_false(self):
        """Test that predictions are saved directly when reduce_zero_label=False."""
        # Model predicts in original space: [0, 1, 2, 3]
        pred = torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            save_visualization(pred, output_path)

            img = Image.open(output_path)
            arr = np.array(img)

            # Should preserve [0, 1, 2, 3]
            expected = pred.numpy()
            assert np.array_equal(arr, expected)
```

#### 4. Create conftest for shared fixtures
**File**: `dinov3/eval/segmentation/tests/conftest.py`

```python
"""Pytest configuration and shared fixtures for segmentation tests."""

import pytest
import torch


@pytest.fixture
def sample_labels_3class():
    """Sample labels with 3 classes (0, 1, 2)."""
    return torch.tensor([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ])


@pytest.fixture
def sample_labels_with_ignore():
    """Sample labels with ignore_index (255)."""
    return torch.tensor([
        [0, 1, 255],
        [2, 255, 0],
        [255, 1, 2],
    ])


@pytest.fixture
def sample_predictions_3class():
    """Sample predictions matching 3-class labels."""
    return torch.tensor([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ]).float()
```

### Success Criteria:

#### Automated Verification:
- [x] All tests pass: `PYTHONPATH=. pytest dinov3/eval/segmentation/tests/ -v`
- [x] Test coverage is reasonable: `PYTHONPATH=. pytest dinov3/eval/segmentation/tests/ --cov=dinov3.eval.segmentation.metrics --cov-report=term-missing`

---

## Testing Strategy

### Unit Tests:
- `preprocess_nonzero_labels`: Verify label transformation [0,1,2,3] -> [255,0,1,2]
- `restore_original_labels`: Verify reverse mapping [0,1,2] -> [1,2,3]
- `calculate_intersect_and_union`: Verify metrics with both `reduce_zero_label` values
- `save_visualization`: Verify PNG output preserves class indices

### Integration Tests:
- End-to-end label flow consistency
- Visualization with restored labels

### Run All Tests:
```bash
# Install pytest if needed
pip install pytest pytest-cov

# Run all tests
PYTHONPATH=. pytest dinov3/eval/segmentation/tests/ -v

# Run with coverage
PYTHONPATH=. pytest dinov3/eval/segmentation/tests/ --cov=dinov3.eval.segmentation -v
```

## Performance Considerations

- `restore_original_labels` is a simple +1 operation, negligible overhead
- No changes to the core inference or training loops
- Tests use small synthetic tensors, run quickly

## Migration Notes

- No data migration needed
- Existing checkpoints remain compatible
- Config changes are backwards compatible (default value unchanged)

## References

- Research document: `thoughts/shared/research/2026-01-15-reduce-zero-label-bug-analysis.md`
- Hardcoded bug location: `dinov3/eval/segmentation/eval.py:103`
- Visualization bug location: `dinov3/eval/segmentation/eval.py:92-97`
- Config definition: `dinov3/eval/segmentation/config.py:115`
