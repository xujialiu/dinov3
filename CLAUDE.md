# DINOv3 - Self-Supervised Vision Foundation Model

Meta AI Research's vision transformer for dense feature extraction. Provides pretrained ViT and ConvNeXt backbones with task-specific heads for segmentation, depth, detection, and classification.

## Tech Stack
- Python 3.11+, PyTorch, torchvision, torchmetrics
- OmegaConf (config), submitit (SLURM jobs)
- Conda environment: `nnunet`

## Key Directories
| Directory | Purpose |
|-----------|---------|
| `dinov3/models/` | Backbone architectures (ViT, ConvNeXt) |
| `dinov3/eval/segmentation/` | **Primary focus** - Semantic segmentation evaluation |
| `dinov3/eval/depth/` | Depth estimation evaluation |
| `dinov3/hub/` | PyTorch Hub model exports |
| `dinov3/train/` | Training pipeline |
| `dinov3/configs/` | Training configuration templates |

## Essential Commands

```bash
# Activate environment
conda activate dinov3

# Linear segmentation training on ADE20K
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/segmentation/run.py \
  model.dino_hub=dinov3_vit7b16 \
  config=dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
  datasets.root=<PATH> --output-dir <OUTPUT>

# Segmentation inference with Mask2Former
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/segmentation/run.py \
  config=dinov3/eval/segmentation/configs/config-ade20k-m2f-inference.yaml \
  model.load_from=<CHECKPOINT> datasets.root=<PATH>
```

## Segmentation Module (`dinov3/eval/segmentation/`)
| File | Purpose |
|------|---------|
| `run.py` | Entry point - routes to train/test |
| `train.py` | Training loop with DDP support |
| `eval.py` | Evaluation with sliding window inference |
| `config.py` | Configuration dataclasses |
| `models/heads/linear_head.py` | Simple linear decoder |
| `models/heads/mask2former_head.py` | Mask2Former decoder |

## Additional Documentation
- [docs/architecture.md](docs/architecture.md) - Architectural patterns, design decisions, conventions
- [docs/changelog.md](docs/changelog.md) - Version history and notable changes
