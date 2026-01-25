#!/usr/bin/env python3
"""
Script to prepare external Clash Royale detection dataset for training.

Usage:
    python scripts/prepare_dataset.py --input /path/to/external/dataset --output data/prepared
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_extended_label(input_path: Path, output_path: Path) -> dict:
    """Convert extended YOLO format (12 cols) to standard (5 cols)."""
    stats = {"extended": 0, "standard": 0}

    with open(input_path, "r") as f:
        lines = f.readlines()

    converted = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if len(parts) >= 12:
            # Extended format: class x y w h belonging + 6 state columns
            class_id, x, y, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
            converted.append(f"{class_id} {x} {y} {w} {h}\n")
            stats["extended"] += 1
        elif len(parts) >= 5:
            # Already standard format
            converted.append(line)
            stats["standard"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(converted)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare external Clash Royale detection dataset for training"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/home/claude/workspace/CS175-Project-Ralph/Clash-Royale-Detection-Dataset",
        help="Path to external dataset root"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/prepared",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/dataset.yaml",
        help="Output path for dataset config YAML"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip label conversion (use if labels are already standard format)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Find all image/label pairs from part2 directory
    print("Scanning for image/label pairs...")
    pairs = []  # List of (image_path, label_path) tuples

    part2_dir = input_path / "images" / "part2"
    if part2_dir.exists():
        for img_file in part2_dir.rglob("*.jpg"):
            if "background" in str(img_file):
                continue
            label_file = img_file.with_suffix(".txt")
            if label_file.exists():
                pairs.append((img_file, label_file))

        for img_file in part2_dir.rglob("*.png"):
            if "background" in str(img_file):
                continue
            label_file = img_file.with_suffix(".txt")
            if label_file.exists():
                pairs.append((img_file, label_file))

    if not pairs:
        print(f"Error: No image/label pairs found in {input_path}")
        sys.exit(1)

    print(f"Found {len(pairs)} image/label pairs")

    # Shuffle and split
    random.shuffle(pairs)
    n_train = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    print(f"\nSplit: {len(train_pairs)} train, {len(val_pairs)} val")

    # Create output directories
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    val_img_dir = output_dir / "images" / "val"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Process train set
    print("\nProcessing training set...")
    total_extended = 0
    total_standard = 0
    class_counts = defaultdict(int)

    for i, (img_path, lbl_path) in enumerate(train_pairs):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(train_pairs)}...")

        # Generate unique filename based on index
        new_name = f"train_{i:06d}"
        new_img = train_img_dir / f"{new_name}{img_path.suffix}"
        new_lbl = train_lbl_dir / f"{new_name}.txt"

        # Copy image
        shutil.copy2(img_path, new_img)

        # Convert and copy label
        if not args.no_convert:
            stats = convert_extended_label(lbl_path, new_lbl)
            total_extended += stats["extended"]
            total_standard += stats["standard"]
        else:
            shutil.copy2(lbl_path, new_lbl)

        # Count classes
        with open(new_lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1

    # Process val set
    print("\nProcessing validation set...")
    for i, (img_path, lbl_path) in enumerate(val_pairs):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(val_pairs)}...")

        new_name = f"val_{i:06d}"
        new_img = val_img_dir / f"{new_name}{img_path.suffix}"
        new_lbl = val_lbl_dir / f"{new_name}.txt"

        shutil.copy2(img_path, new_img)

        if not args.no_convert:
            stats = convert_extended_label(lbl_path, new_lbl)
            total_extended += stats["extended"]
            total_standard += stats["standard"]
        else:
            shutil.copy2(lbl_path, new_lbl)

        with open(new_lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1

    print(f"\nLabel conversion: {total_extended} extended -> standard, {total_standard} already standard")
    print(f"Total annotations: {sum(class_counts.values())}")
    print(f"Classes used: {len(class_counts)}")

    # Generate dataset.yaml config
    import yaml

    # Load class names from external dataset config
    external_yaml = input_path / "images" / "part2" / "ClashRoyale_detection.yaml"
    class_names = {}
    if external_yaml.exists():
        with open(external_yaml, "r") as f:
            ext_config = yaml.safe_load(f)
            if "names" in ext_config:
                class_names = ext_config["names"]
                print(f"Loaded {len(class_names)} class names from external config")

    # Create dataset config
    config = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names) if class_names else max(class_counts.keys()) + 1,
        "names": class_names if class_names else {i: str(i) for i in range(max(class_counts.keys()) + 1)},
    }

    config_path = Path(args.config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nConfig saved to: {config_path}")

    # Summary
    print("\n" + "=" * 50)
    print("Dataset preparation complete!")
    print("=" * 50)
    print(f"  Train images: {len(train_pairs)}")
    print(f"  Val images: {len(val_pairs)}")
    print(f"  Total annotations: {sum(class_counts.values())}")
    print(f"  Number of classes: {len(class_names)}")
    print(f"\nTo train the model, run:")
    print(f"  python -m src.detection.train --data {args.config} --epochs 50")


if __name__ == "__main__":
    main()
