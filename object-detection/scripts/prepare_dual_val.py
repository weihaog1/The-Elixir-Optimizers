"""Prepare per-detector validation labels for dual-detector training.

Reads the shared val set (global class indices) and creates per-detector
copies with filtered and remapped labels. Images are symlinked to save space.

Usage:
    python scripts/prepare_dual_val.py --split-config configs/split_config.json \
        --val-images data/prepared_belong/images/val \
        --val-labels data/prepared_belong/labels/val \
        --output-dir data
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare_detector_val(det_key, split_config, val_images_dir, val_labels_dir, output_dir):
    """Create per-detector val set with remapped labels.

    Args:
        det_key: "detector1" or "detector2".
        split_config: Parsed split_config.json dict.
        val_images_dir: Path to shared val images.
        val_labels_dir: Path to shared val labels (global indices).
        output_dir: Base output dir (e.g. "data").
    """
    det_cfg = split_config[det_key]
    global_indices = set(det_cfg["global_indices"])
    # global_to_local: maps str(global_idx) -> local_idx
    g2l = {int(k): v for k, v in det_cfg["global_to_local"].items()}

    det_num = det_key[-1]  # "1" or "2"
    out_images = Path(output_dir) / f"synthetic_d{det_num}" / "val" / "images"
    out_labels = Path(output_dir) / f"synthetic_d{det_num}" / "val" / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    val_images_dir = Path(val_images_dir)
    val_labels_dir = Path(val_labels_dir)

    label_files = sorted(val_labels_dir.glob("*.txt"))
    n_images = 0
    n_labels_total = 0
    n_labels_kept = 0

    for lf in label_files:
        stem = lf.stem
        # Find matching image
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = val_images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        # Read and filter/remap labels
        lines = lf.read_text().strip().splitlines()
        remapped = []
        for line in lines:
            parts = line.split()
            if len(parts) < 6:
                continue
            n_labels_total += 1
            global_cls = int(parts[0])
            if global_cls not in global_indices:
                continue
            local_cls = g2l.get(global_cls)
            if local_cls is None:
                continue
            # Remap class index, keep rest unchanged
            rest = " ".join(parts[1:])
            remapped.append(f"{local_cls} {rest}")
            n_labels_kept += 1

        # Only include image if it has at least one valid detection
        if not remapped:
            continue

        # Symlink image
        dst_img = out_images / img_path.name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        os.symlink(img_path.resolve(), dst_img)

        # Write remapped labels
        (out_labels / lf.name).write_text("\n".join(remapped) + "\n")
        n_images += 1

    print(f"{det_key}: {n_images} images, {n_labels_kept}/{n_labels_total} labels kept")
    print(f"  Output: {out_images}")
    return n_images


def main():
    parser = argparse.ArgumentParser(description="Prepare per-detector val labels")
    parser.add_argument("--split-config", default="configs/split_config.json")
    parser.add_argument("--val-images", default="data/prepared_belong/images/val")
    parser.add_argument("--val-labels", default="data/prepared_belong/labels/val")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    with open(args.split_config) as f:
        split_config = json.load(f)

    for det_key in ["detector1", "detector2"]:
        prepare_detector_val(
            det_key, split_config,
            args.val_images, args.val_labels, args.output_dir,
        )


if __name__ == "__main__":
    main()
