#!/usr/bin/env python3
"""
Script to create ADE20K split files for DINOv3 segmentation training.

Expected ADE20K folder structure:
    ADEChallengeData2016/
    ├── images/
    │   ├── training/
    │   │   ├── ADE_train_00000001.jpg
    │   │   └── ...
    │   └── validation/
    │       ├── ADE_val_00000001.jpg
    │       └── ...
    └── annotations/
        ├── training/
        │   ├── ADE_train_00000001.png
        │   └── ...
        └── validation/
            ├── ADE_val_00000001.png
            └── ...

This script creates:
    - ADE20K_object150_train.txt
    - ADE20K_object150_val.txt
"""

import os
import argparse
from pathlib import Path


def create_split_file(root: str, split: str):
    """Create split file for train or val."""
    split_dir = "training" if split == "train" else "validation"
    images_dir = Path(root) / "images" / split_dir

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Get all image files
    image_files = sorted([
        f"{split_dir}/{f.name}"
        for f in images_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    # Write split file
    output_file = Path(root) / f"ADE20K_object150_{split}.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(image_files))

    print(f"Created {output_file} with {len(image_files)} images")
    return len(image_files)


def main():
    parser = argparse.ArgumentParser(description="Create ADE20K split files for DINOv3")
    parser.add_argument(
        "root",
        type=str,
        help="Path to ADEChallengeData2016 directory"
    )
    args = parser.parse_args()

    root = args.root

    # Verify directory structure
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root directory not found: {root}")

    print(f"Creating ADE20K split files in: {root}")

    # Create train and val split files
    train_count = create_split_file(root, "train")
    val_count = create_split_file(root, "val")

    print(f"\nDone! Created split files:")
    print(f"  - Train: {train_count} images")
    print(f"  - Val: {val_count} images")


if __name__ == "__main__":
    main()
