"""
Dataset preparation module for YOLO training.

This module handles:
1. Creating train/val splits from annotated data
2. Organizing data into YOLO directory structure
3. Generating dataset configuration files
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def split_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    copy_files: bool = True,
) -> Dict[str, int]:
    """Split dataset into train/val/test sets.

    Args:
        image_dir: Directory containing images.
        label_dir: Directory containing label files.
        output_dir: Output directory for split dataset.
        train_ratio: Ratio of training data (default: 0.8).
        val_ratio: Ratio of validation data (default: 0.2).
        test_ratio: Ratio of test data (default: 0.0).
        seed: Random seed for reproducibility.
        image_extensions: Valid image file extensions.
        copy_files: If True, copy files. If False, create symlinks.

    Returns:
        Dict with counts for each split.
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    # Find all images with corresponding labels
    image_files = []
    for ext in image_extensions:
        for img_file in image_dir.glob(f"*{ext}"):
            label_file = label_dir / (img_file.stem + ".txt")
            if label_file.exists():
                image_files.append((img_file, label_file))

    if not image_files:
        raise ValueError(f"No image-label pairs found in {image_dir} and {label_dir}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(image_files)

    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:] if test_ratio > 0 else []

    # Create output directories
    splits = [("train", train_files), ("val", val_files)]
    if test_files:
        splits.append(("test", test_files))

    stats = {}

    for split_name, files in splits:
        img_out_dir = output_dir / split_name / "images"
        lbl_out_dir = output_dir / split_name / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_file, lbl_file in files:
            img_dest = img_out_dir / img_file.name
            lbl_dest = lbl_out_dir / lbl_file.name

            if copy_files:
                shutil.copy2(img_file, img_dest)
                shutil.copy2(lbl_file, lbl_dest)
            else:
                # Create symlinks (relative)
                if img_dest.exists():
                    img_dest.unlink()
                if lbl_dest.exists():
                    lbl_dest.unlink()
                img_dest.symlink_to(os.path.relpath(img_file, img_dest.parent))
                lbl_dest.symlink_to(os.path.relpath(lbl_file, lbl_dest.parent))

        stats[split_name] = len(files)

    return stats


def generate_dataset_yaml(
    output_dir: str,
    yaml_path: str,
    class_names: Optional[Dict[int, str]] = None,
    nc: Optional[int] = None,
) -> str:
    """Generate YOLO dataset configuration file.

    Args:
        output_dir: Directory containing train/val/test splits.
        yaml_path: Path to save the YAML configuration.
        class_names: Dict mapping class_id to class_name.
        nc: Number of classes (required if class_names not provided).

    Returns:
        Path to the generated YAML file.
    """
    output_dir = Path(output_dir).resolve()

    # Default tower classes
    if class_names is None:
        class_names = {
            0: "king_tower_player",
            1: "king_tower_enemy",
            2: "princess_tower_left_player",
            3: "princess_tower_left_enemy",
            4: "princess_tower_right_player",
            5: "princess_tower_right_enemy",
        }

    if nc is None:
        nc = len(class_names)

    config = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": nc,
        "names": class_names,
    }

    # Add test if it exists
    test_dir = output_dir / "test" / "images"
    if test_dir.exists():
        config["test"] = "test/images"

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return yaml_path


def count_class_distribution(label_dir: str) -> Dict[int, int]:
    """Count the distribution of classes in label files.

    Args:
        label_dir: Directory containing YOLO label files.

    Returns:
        Dict mapping class_id to count.
    """
    label_dir = Path(label_dir)
    class_counts: Dict[int, int] = {}

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

    return class_counts


def analyze_dataset(dataset_dir: str) -> Dict:
    """Analyze a YOLO-format dataset.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        Dict with dataset statistics.
    """
    dataset_dir = Path(dataset_dir)
    stats = {
        "splits": {},
        "total_images": 0,
        "total_annotations": 0,
        "class_distribution": {},
    }

    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        n_images = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        class_dist = count_class_distribution(str(lbl_dir)) if lbl_dir.exists() else {}
        n_annotations = sum(class_dist.values())

        stats["splits"][split] = {
            "images": n_images,
            "labels": n_labels,
            "annotations": n_annotations,
            "class_distribution": class_dist,
        }

        stats["total_images"] += n_images
        stats["total_annotations"] += n_annotations

        # Merge class distributions
        for class_id, count in class_dist.items():
            stats["class_distribution"][class_id] = (
                stats["class_distribution"].get(class_id, 0) + count
            )

    return stats


def prepare_yolo_dataset(
    raw_image_dir: str,
    annotation_dir: str,
    output_dir: str,
    config_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> Dict:
    """Complete pipeline to prepare YOLO dataset.

    Args:
        raw_image_dir: Directory with raw images.
        annotation_dir: Directory with YOLO label files.
        output_dir: Output directory for processed dataset.
        config_path: Path to save dataset YAML config.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        seed: Random seed.

    Returns:
        Dict with dataset statistics.
    """
    # Split dataset
    split_stats = split_dataset(
        image_dir=raw_image_dir,
        label_dir=annotation_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Generate YAML config
    generate_dataset_yaml(output_dir, config_path)

    # Analyze final dataset
    analysis = analyze_dataset(output_dir)
    analysis["split_counts"] = split_stats

    return analysis


def verify_dataset_integrity(dataset_dir: str) -> Dict[str, List[str]]:
    """Verify dataset integrity - check for missing files and format issues.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        Dict with lists of issues found.
    """
    dataset_dir = Path(dataset_dir)
    issues = {
        "missing_images": [],
        "missing_labels": [],
        "invalid_labels": [],
        "empty_labels": [],
    }

    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        # Check for missing labels
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                label_file = lbl_dir / (img_file.stem + ".txt")
                if not label_file.exists():
                    issues["missing_labels"].append(f"{split}/{img_file.name}")

        # Check for missing images and invalid labels
        for label_file in lbl_dir.glob("*.txt"):
            # Check if corresponding image exists
            img_found = False
            for ext in (".png", ".jpg", ".jpeg"):
                if (img_dir / (label_file.stem + ext)).exists():
                    img_found = True
                    break

            if not img_found:
                issues["missing_images"].append(f"{split}/{label_file.name}")

            # Check label format
            try:
                with open(label_file, "r") as f:
                    content = f.read().strip()
                    if not content:
                        issues["empty_labels"].append(f"{split}/{label_file.name}")
                    else:
                        for line_num, line in enumerate(content.split("\n"), 1):
                            parts = line.strip().split()
                            if len(parts) < 5:
                                issues["invalid_labels"].append(
                                    f"{split}/{label_file.name}:L{line_num}"
                                )
                            else:
                                # Verify values are in valid range
                                class_id = int(parts[0])
                                coords = [float(p) for p in parts[1:5]]
                                if not all(0 <= c <= 1 for c in coords):
                                    issues["invalid_labels"].append(
                                        f"{split}/{label_file.name}:L{line_num} (out of range)"
                                    )
            except Exception as e:
                issues["invalid_labels"].append(f"{split}/{label_file.name}: {e}")

    return issues


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset from annotated images"
    )
    parser.add_argument(
        "--images", "-i",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        required=True,
        help="Directory containing YOLO label files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to save dataset YAML config"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training data ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation data ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze existing dataset, don't process"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset integrity"
    )

    args = parser.parse_args()

    if args.analyze:
        stats = analyze_dataset(args.output)
        print("Dataset Analysis:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Class distribution: {stats['class_distribution']}")
        for split, info in stats["splits"].items():
            print(f"  {split}: {info['images']} images, {info['annotations']} annotations")
    elif args.verify:
        issues = verify_dataset_integrity(args.output)
        has_issues = any(issues.values())
        if has_issues:
            print("Issues found:")
            for issue_type, items in issues.items():
                if items:
                    print(f"  {issue_type}:")
                    for item in items[:10]:  # Show first 10
                        print(f"    - {item}")
                    if len(items) > 10:
                        print(f"    ... and {len(items) - 10} more")
        else:
            print("Dataset integrity check passed!")
    else:
        config_path = args.config or str(Path(args.output) / "dataset.yaml")
        stats = prepare_yolo_dataset(
            raw_image_dir=args.images,
            annotation_dir=args.labels,
            output_dir=args.output,
            config_path=config_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print("Dataset prepared successfully!")
        print(f"  Split counts: {stats['split_counts']}")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Config saved to: {config_path}")
