"""Auto-split 155 classes into two detectors by average sprite pixel area.

Follows KataCR's dual-detector approach (model_setup.py):
- 13 valid base classes (indices 0-14, minus invalid) duplicated in BOTH detectors
- 5 invalid classes excluded: selected, text, mirror, tesla-evolution-shock, zap-evolution
- Remaining ~137 classes sorted by sprite size, split in half
- Related unit families kept together (skeleton-king + skeleton-king-skill)
- Each detector padded to max_detect_num with padding_N, last class = padding_belong

Outputs:
  configs/split_config.json  - per-detector class lists + global<->local remapping
  configs/detector1_data.yaml - YOLO data config for detector 1 (small sprites)
  configs/detector2_data.yaml - YOLO data config for detector 2 (large sprites)
"""

import json
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.generation.label_list import idx2unit, unit2idx

BASE_IDXS = 15  # indices 0..14 are base classes (towers + UI)
MAX_DETECT_NUM = 85  # max classes per detector (KataCR default for 2-detector)
INVALID_UNITS = ['selected', 'text', 'mirror', 'tesla-evolution-shock', 'zap-evolution']
RELATED_UNITS = [
    {'tesla-evolution', 'tesla-evolution-shock'},
    {'skeleton-king-skill', 'skeleton-king'},
]


def measure_sprite_sizes(segment_dir):
    """Measure average pixel area per class from sprite PNGs.

    Args:
        segment_dir: Path to .../images/segment/ containing per-class subdirs.

    Returns:
        dict mapping class name -> mean pixel area (width * height).
    """
    segment_dir = Path(segment_dir)
    unit2size = {}
    no_img = []

    for name in unit2idx:
        if name in INVALID_UNITS:
            continue
        if unit2idx[name] < BASE_IDXS:
            continue
        cls_dir = segment_dir / name
        if not cls_dir.exists():
            no_img.append(name)
            continue
        sizes = []
        for p in cls_dir.glob('*.png'):
            img = Image.open(str(p))
            sizes.append(img.size[0] * img.size[1])
        if not sizes:
            no_img.append(name)
        else:
            unit2size[name] = float(np.mean(sizes))

    if no_img:
        print(f"Warning: no sprites found for: {no_img}")

    return dict(sorted(unit2size.items(), key=lambda x: x[1]))


def build_detection_ranges(unit2size):
    """Split non-base classes into two detector groups by sprite size.

    Returns:
        (detector1_global_indices, detector2_global_indices) where base classes
        appear in both, and each list excludes invalid classes.
    """
    ranked = list(unit2size.keys())
    n = len(ranked)
    step = (n + 1) // 2  # first half = small, second half = large

    small_names = ranked[:step]
    large_names = ranked[step:]

    # Move related units together
    for related in RELATED_UNITS:
        # Check if any member is in small vs large
        in_small = [u for u in related if u in small_names]
        in_large = [u for u in related if u in large_names]
        if in_small and in_large:
            # Move all to whichever group has the majority
            if len(in_small) >= len(in_large):
                for u in in_large:
                    large_names.remove(u)
                    small_names.append(u)
            else:
                for u in in_small:
                    small_names.remove(u)
                    large_names.append(u)

    # Also handle skeleton-king-bar (base class 14) needing skeleton-king in same detector
    # skeleton-king-bar is base (idx 14), skeleton-king (idx 101) should be in both if possible
    # Since base classes appear in both detectors, this is fine.

    # Build valid base indices (0..14 minus invalid)
    base_indices = []
    for i in range(BASE_IDXS):
        name = idx2unit[i]
        if name not in INVALID_UNITS:
            base_indices.append(i)

    d1_indices = list(base_indices) + [unit2idx[n] for n in small_names]
    d2_indices = list(base_indices) + [unit2idx[n] for n in large_names]

    return d1_indices, d2_indices


def build_config(d1_indices, d2_indices, val_path="prepared_belong/images/val",
                 train_d1="synthetic_d1/train/images", train_d2="synthetic_d2/train/images"):
    """Build split_config.json and per-detector data.yaml files.

    Returns:
        split_config dict ready for JSON serialization.
    """
    config = {}

    for det_num, global_indices in [(1, d1_indices), (2, d2_indices)]:
        # Build local name list: valid classes + padding + padding_belong
        names_list = [idx2unit[i] for i in global_indices]
        n_real = len(names_list)

        # Pad to MAX_DETECT_NUM - 1 with padding_N
        for i in range(n_real, MAX_DETECT_NUM - 1):
            names_list.append(f"padding_{i - n_real}")

        # Last class is padding_belong
        names_list.append("padding_belong")
        nc = len(names_list)

        # Build global <-> local mappings
        global_to_local = {}
        local_to_global = {}
        for local_idx, name in enumerate(names_list):
            if name.startswith("padding"):
                continue
            global_idx = unit2idx[name]
            global_to_local[str(global_idx)] = local_idx
            local_to_global[str(local_idx)] = global_idx

        key = f"detector{det_num}"
        config[key] = {
            "global_indices": global_indices,
            "global_to_local": global_to_local,
            "local_to_global": local_to_global,
            "nc": nc,
            "names": {i: name for i, name in enumerate(names_list)},
            "n_real_classes": n_real,
        }

    # Add metadata
    config["base_indices"] = [i for i in range(BASE_IDXS) if idx2unit[i] not in INVALID_UNITS]
    config["invalid_units"] = INVALID_UNITS
    config["max_detect_num"] = MAX_DETECT_NUM

    return config


def write_data_yamls(config, output_dir, val_path="prepared_belong/images/val",
                     train_d1="synthetic_d1/train/images", train_d2="synthetic_d2/train/images"):
    """Write per-detector YOLO data.yaml files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = {"detector1": train_d1, "detector2": train_d2}

    for det_key in ["detector1", "detector2"]:
        det = config[det_key]
        data = {
            'path': 'data',
            'train': train_paths[det_key],
            'val': val_path,
            'nc': det['nc'],
            'names': det['names'],
        }
        yaml_path = output_dir / f"{det_key}_data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
        print(f"Wrote {yaml_path} (nc={det['nc']}, {det['n_real_classes']} real classes)")


def auto_split(segment_dir, output_dir="configs"):
    """Run full auto-split pipeline.

    Args:
        segment_dir: Path to .../images/segment/ with sprite PNGs.
        output_dir: Directory for split_config.json and data.yaml files.
    """
    print("Measuring sprite sizes...")
    unit2size = measure_sprite_sizes(segment_dir)
    print(f"  Measured {len(unit2size)} classes")

    # Print size distribution summary
    sizes = list(unit2size.values())
    print(f"  Size range: {min(sizes):.0f} - {max(sizes):.0f} px^2")
    print(f"  Median: {np.median(sizes):.0f} px^2")

    print("Building detection ranges...")
    d1_indices, d2_indices = build_detection_ranges(unit2size)
    print(f"  Detector 1 (small): {len(d1_indices)} classes")
    print(f"  Detector 2 (large): {len(d2_indices)} classes")

    # Show base class overlap
    base = set(d1_indices) & set(d2_indices)
    print(f"  Shared base classes: {len(base)}")

    print("Building config...")
    config = build_config(d1_indices, d2_indices)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write split_config.json
    config_path = output_dir / "split_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {config_path}")

    # Write data.yaml files
    write_data_yamls(config, output_dir)

    return config


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Auto-split classes for dual detector")
    parser.add_argument("--segment-dir", type=str, required=True,
                        help="Path to images/segment/ directory with sprite PNGs")
    parser.add_argument("--output-dir", type=str, default="configs",
                        help="Output directory for config files")
    args = parser.parse_args()

    auto_split(args.segment_dir, args.output_dir)
