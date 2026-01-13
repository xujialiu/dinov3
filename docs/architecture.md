# DINOv3 Architecture & Patterns

## Configuration Pattern (OmegaConf + Dataclasses)

All modules use structured dataclasses with OmegaConf for hierarchical config management:

```python
# Pattern: Nested dataclasses in config.py
@dataclass
class ModelConfig:
    dino_hub: str = "dinov3_vitl14"

@dataclass
class SegmentationConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    decoder_head: DecoderConfig = field(default_factory=DecoderConfig)
```

Config loading merges YAML files with CLI overrides:
```python
cfg = OmegaConf.merge(OmegaConf.structured(SegmentationConfig), yaml_cfg, cli_cfg)
```

## Backbone + Head Architecture

**Pattern:** Frozen backbone with trainable task-specific head

```
Input Image → Backbone (frozen) → Intermediate Features → Head (trainable) → Output
```

- `ModelWithIntermediateLayers` wraps backbone to extract multi-level features
- Backbone output selection via `BackboneLayersSet` enum:
  - `LAST`: Single final layer (linear head)
  - `FOUR_EVEN_INTERVALS`: 4 evenly-spaced layers (Mask2Former)

## Distributed Training Pattern

All training uses PyTorch DDP with these conventions:

```python
# Wrap model
model = DistributedDataParallel(model, device_ids=[gpu])

# Use SyncBatchNorm for cross-GPU normalization
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# Distributed sampler
sampler = DistributedSampler(dataset, shuffle=True)
```

Checkpoints save only decoder head weights (backbone is pretrained).

## Iteration-Based Training

Uses `InfiniteDataloader` wrapper instead of epoch-based:

```python
class InfiniteDataloader:
    """Cycles through dataloader indefinitely for fixed iteration count."""
    def __iter__(self):
        while True:
            for batch in self.dataloader:
                yield batch
```

Training loop runs `total_iter` iterations with periodic validation at `eval_interval`.

## Inference Patterns

### Sliding Window Inference
For high-resolution images, uses overlapping crops:
```python
# Grid of crops with stride < crop_size creates overlap
# Predictions averaged in overlap regions using count matrix
slide_inference(model, image, crop_size=512, stride=341)
```

### Test-Time Augmentation (TTA)
Multiple scales + horizontal flip, predictions averaged:
```python
tta_ratios: [0.9, 0.95, 1.0, 1.05, 1.1]  # 5 scales
use_horizontal_flip: True
```

## Feature Extraction Pattern

Backbone adapters wrap DINOv3 for multi-scale feature extraction:

```python
# DINOv3_Adapter in models/backbone/dinov3_adapter.py
class DINOv3_Adapter:
    """
    1. Extract features from DINOv3 backbone at multiple layers
    2. Add spatial priors from input image
    3. Cross-scale interaction via deformable attention
    4. Output: 4-level feature pyramid
    """
```

## Loss Composition Pattern

`MultiSegmentationLoss` combines multiple losses with configurable weights:

```python
loss = dice_weight * DiceLoss() + celoss_weight * CrossEntropyLoss()
```

## Metrics Pattern

Segmentation metrics computed via intersection/union histograms:

```python
# Gather per-class statistics across distributed workers
intersect, union, pred_label, label = calculate_intersect_and_union(pred, target)
# Aggregate then compute final metrics
mIoU = (intersect / union).mean()
```

## PyTorch Hub Integration

Models exposed via `hubconf.py` for easy loading:
```python
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl14')
segmentor = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl14_ade20k_linear')
```

## File Organization Convention

Each eval task follows consistent structure:
```
eval/<task>/
├── run.py          # Entry point
├── train.py        # Training loop
├── eval.py         # Evaluation loop
├── config.py       # Configuration dataclasses
├── loss.py         # Loss functions
├── metrics.py      # Evaluation metrics
├── transforms.py   # Data augmentation
└── models/         # Task-specific heads
    ├── heads/
    └── backbone/   # Backbone adapters
```
