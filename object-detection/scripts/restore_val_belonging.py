"""Re-prepare validation labels with belonging column preserved.

Runs the same dataset preparation as prepare_dataset.py (same seed=42, same
directory walk, same train/val split) but outputs 6-column labels:
  cls_id cx cy w h belonging

Must be run on the remote instance where the original KataCR dataset is
available at Clash-Royale-Detection-Dataset/images/part2/.

Usage:
    python scripts/restore_val_belonging.py \
        --input /workspace/Clash-Royale-Detection-Dataset \
        --output data/prepared_belong
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.generation.label_list import unit2idx, idx2unit


def convert_label_with_belonging(input_path, output_path, names_155):
    """Convert extended YOLO labels to 6-column with belonging.

    Input format (12 cols): class x y w h belonging state1..state6
    Output format (6 cols): remapped_cls x y w h belonging

    Skips classes >= 155 (padding/belong classes in the original 201-class scheme).
    """
    names_inv = {n: i for i, n in names_155.items()}

    with open(input_path, 'r') as f:
        lines = f.readlines()

    converted = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        cls_id = int(parts[0])
        # Skip padding classes (155+)
        if cls_id >= len(idx2unit):
            continue

        cls_name = idx2unit[cls_id]
        if cls_name not in names_inv:
            continue

        remapped_cls = names_inv[cls_name]
        x, y, w, h = parts[1], parts[2], parts[3], parts[4]

        if len(parts) >= 6:
            belonging = parts[5]
        else:
            belonging = '0'

        converted.append(f"{remapped_cls} {x} {y} {w} {h} {belonging}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(converted)

    return len(converted)


def main():
    parser = argparse.ArgumentParser(
        description="Re-prepare val labels with belonging preserved"
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default="/workspace/Froked-KataCR-Clash-Royale-Detection-Dataset",
        help="Path to KataCR dataset root",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="data/prepared_belong",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (must match prepare_dataset.py)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Validation split ratio (must match prepare_dataset.py)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Build the 155-class name mapping from our config
    import yaml
    config_path = project_root / "configs" / "synthetic_data.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    names_155 = config['names']

    # Walk the same directory structure as prepare_dataset.py
    part2_dir = input_path / "images" / "part2"
    if not part2_dir.exists():
        print(f"Error: {part2_dir} not found")
        sys.exit(1)

    print("Scanning for image/label pairs...")
    pairs = []
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
        print(f"Error: No pairs found in {part2_dir}")
        sys.exit(1)

    print(f"Found {len(pairs)} image/label pairs")

    # Same shuffle and split as prepare_dataset.py
    random.shuffle(pairs)
    n_train = int(len(pairs) * (1.0 - args.val_ratio))
    val_pairs = pairs[n_train:]

    print(f"Val split: {len(val_pairs)} images (same as prepare_dataset.py)")

    # Create output directories
    val_img_dir = output_dir / "images" / "val"
    val_lbl_dir = output_dir / "labels" / "val"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Process val set with belonging
    total_annotations = 0
    for i, (img_path, lbl_path) in enumerate(val_pairs):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(val_pairs)}...")

        new_name = f"val_{i:06d}"
        new_img = val_img_dir / f"{new_name}{img_path.suffix}"
        new_lbl = val_lbl_dir / f"{new_name}.txt"

        # Copy image
        if not new_img.exists():
            shutil.copy2(img_path, new_img)

        # Convert label with belonging
        n = convert_label_with_belonging(lbl_path, new_lbl, names_155)
        total_annotations += n

    avg_anns = total_annotations / len(val_pairs) if val_pairs else 0
    print(f"\nDone! Processed {len(val_pairs)} val images")
    print(f"Total annotations: {total_annotations} (avg {avg_anns:.1f}/image)")
    print(f"Label format: cls_id cx cy w h belonging (6 columns)")
    print(f"Output: {output_dir}")

    # Verify a sample
    sample_lbl = val_lbl_dir / "val_000000.txt"
    if sample_lbl.exists():
        with open(sample_lbl) as f:
            lines = f.readlines()
        print(f"\nSample label ({sample_lbl.name}, {len(lines)} annotations):")
        for line in lines[:3]:
            parts = line.strip().split()
            cls_id = int(parts[0])
            bel = int(parts[5])
            cls_name = names_155.get(cls_id, f"cls_{cls_id}")
            bel_str = "ally" if bel == 0 else "enemy"
            print(f"  {cls_name} ({bel_str}): {' '.join(parts[1:5])}")
        if len(lines) > 3:
            print(f"  ... ({len(lines) - 3} more)")


if __name__ == "__main__":
    main()
