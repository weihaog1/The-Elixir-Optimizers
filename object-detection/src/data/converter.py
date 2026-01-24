"""
Data conversion utilities for extended YOLO format.

The external Clash Royale detection dataset uses an extended format:
<class> <x> <y> <w> <h> <belonging> <state1> <state2> <state3> <state4> <state5> <state6>

This module provides tools to convert to standard YOLO format (5 columns).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def convert_extended_yolo_to_standard(
    input_label_dir: str,
    output_label_dir: str,
    keep_belonging: bool = False,
) -> Dict[str, int]:
    """Convert extended YOLO format (12 columns) to standard format (5 columns).

    The external Clash Royale detection dataset uses an extended format:
    <class> <x> <y> <w> <h> <belonging> <state1> <state2> <state3> <state4> <state5> <state6>

    This function converts it to standard YOLO format:
    <class> <x> <y> <w> <h>

    Args:
        input_label_dir: Directory containing extended format label files.
        output_label_dir: Directory to save standard format label files.
        keep_belonging: If True, encode belonging into class ID (class*2 + belonging).

    Returns:
        Dict with conversion statistics.
    """
    input_dir = Path(input_label_dir)
    output_dir = Path(output_label_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "total_annotations": 0,
        "extended_format": 0,
        "standard_format": 0,
    }

    for label_file in input_dir.glob("*.txt"):
        try:
            with open(label_file, "r") as f:
                lines = f.readlines()

            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                stats["total_annotations"] += 1

                # Extended format has 12 columns
                if len(parts) >= 12:
                    stats["extended_format"] += 1
                    class_id = int(parts[0])
                    x, y, w, h = parts[1:5]
                    belonging = int(parts[5])

                    if keep_belonging:
                        # Encode belonging into class ID
                        new_class_id = class_id * 2 + belonging
                        converted_lines.append(f"{new_class_id} {x} {y} {w} {h}\n")
                    else:
                        converted_lines.append(f"{class_id} {x} {y} {w} {h}\n")

                # Standard format has 5 columns
                elif len(parts) >= 5:
                    stats["standard_format"] += 1
                    converted_lines.append(line)
                else:
                    # Invalid format, skip
                    continue

            # Write converted labels
            output_file = output_dir / label_file.name
            with open(output_file, "w") as f:
                f.writelines(converted_lines)

            stats["files_processed"] += 1

        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            stats["files_skipped"] += 1

    return stats


def prepare_external_dataset(
    external_dataset_dir: str,
    output_dir: str,
    config_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
    convert_labels: bool = True,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict:
    """Prepare external Clash Royale detection dataset for training.

    This function handles the external dataset which has:
    - Images in 'images/' subdirectory
    - Labels in 'labels/' subdirectory with extended YOLO format (12 columns)

    Args:
        external_dataset_dir: Root directory of the external dataset.
        output_dir: Output directory for processed dataset.
        config_path: Path to save dataset YAML config.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        seed: Random seed.
        convert_labels: Whether to convert extended labels to standard format.
        class_names: Optional dict mapping class_id to class_name.

    Returns:
        Dict with dataset statistics.
    """
    from .dataset import split_dataset, generate_dataset_yaml, analyze_dataset

    external_dir = Path(external_dataset_dir)
    output_dir = Path(output_dir)

    # Find images and labels directories
    image_dir = external_dir / "images"
    label_dir = external_dir / "labels"

    if not image_dir.exists():
        # Try without subdirectory
        image_dir = external_dir
        label_dir = external_dir

    if not image_dir.exists():
        raise ValueError(f"Could not find images in {external_dataset_dir}")

    # Convert labels if needed
    if convert_labels:
        converted_label_dir = output_dir / "converted_labels"
        print(f"Converting extended YOLO labels to standard format...")
        conversion_stats = convert_extended_yolo_to_standard(
            str(label_dir),
            str(converted_label_dir),
            keep_belonging=False,
        )
        print(f"  Processed: {conversion_stats['files_processed']} files")
        print(f"  Extended format: {conversion_stats['extended_format']} annotations")
        print(f"  Standard format: {conversion_stats['standard_format']} annotations")
        label_dir = converted_label_dir

    # Split dataset
    print(f"Splitting dataset (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
    split_stats = split_dataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        output_dir=str(output_dir),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        copy_files=True,  # Copy files to avoid modifying original
    )
    print(f"  Train: {split_stats.get('train', 0)} images")
    print(f"  Val: {split_stats.get('val', 0)} images")
    if test_ratio > 0:
        print(f"  Test: {split_stats.get('test', 0)} images")

    # Load class names from external dataset if available
    if class_names is None:
        classes_yaml = external_dir / "classes.yaml"
        if classes_yaml.exists():
            with open(classes_yaml, "r") as f:
                classes_data = yaml.safe_load(f)
                if "names" in classes_data:
                    class_names = classes_data["names"]
                    print(f"  Loaded {len(class_names)} class names from classes.yaml")

    # Generate YAML config
    generate_dataset_yaml(
        str(output_dir),
        config_path,
        class_names=class_names,
    )
    print(f"  Config saved to: {config_path}")

    # Analyze final dataset
    analysis = analyze_dataset(str(output_dir))
    analysis["split_counts"] = split_stats
    if convert_labels:
        analysis["conversion_stats"] = conversion_stats

    return analysis


def main():
    """CLI for data conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert and prepare external Clash Royale detection dataset"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory (external dataset root or labels directory)"
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
        "--convert-only",
        action="store_true",
        help="Only convert labels, don't prepare full dataset"
    )

    args = parser.parse_args()

    if args.convert_only:
        # Just convert labels
        print(f"Converting labels from {args.input} to {args.output}")
        stats = convert_extended_yolo_to_standard(args.input, args.output)
        print(f"Conversion complete:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Extended format: {stats['extended_format']} annotations")
        print(f"  Standard format: {stats['standard_format']} annotations")
    else:
        # Full dataset preparation
        config_path = args.config or str(Path(args.output) / "dataset.yaml")
        stats = prepare_external_dataset(
            external_dataset_dir=args.input,
            output_dir=args.output,
            config_path=config_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(f"\nDataset prepared successfully!")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total annotations: {stats['total_annotations']}")


if __name__ == "__main__":
    main()
