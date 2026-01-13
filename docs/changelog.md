# Changelog

All notable changes to DINOv3 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- ConvNeXt distillation support (PR #282)
- More configurable parameters for segmentation evaluation
- pip installability with proper `__init__.py` files (PR #260)

### Changed
- Segmentation evaluation now supports additional configurable parameters

## Recent Commits

- `54694f7` - Merge ConvNeXt distillation support
- `83318a8` - Add ConvNeXt distillation capability
- `5aa7e93` - Make segmentation evaluation parameters configurable
- `66b5c6c` - Make more segmentation parameters configurable
- `76c051f` - Add missing init files for pip installation
