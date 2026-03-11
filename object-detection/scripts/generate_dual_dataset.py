"""Pre-generate synthetic dataset for a single dual-detector.

Wraps generate_dataset.py logic with class filtering and index remapping.
Each generated image includes all sprites, but labels are filtered to only
include classes belonging to the specified detector, with global indices
remapped to local indices.
"""

import argparse
import json
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.generation.generator import Generator
from src.generation.label_list import idx2unit


def _generate_chunk(args):
    """Worker function: generate a chunk of images with class filtering."""
    (start_idx, count, output_dir, background_index, seed, unit_nums,
     noise_unit_ratio, intersect_ratio_thre, img_size, ally_classes,
     global_to_local, valid_global_set, worker_id) = args

    img_dir = Path(output_dir) / "images"
    lbl_dir = Path(output_dir) / "labels"

    gen_kwargs = dict(
        seed=seed + worker_id,
        background_index=background_index,
        intersect_ratio_thre=intersect_ratio_thre,
        augment=True,
        dynamic_unit=True,
        noise_unit_ratio=noise_unit_ratio,
    )
    if ally_classes:
        gen_kwargs['ally_classes'] = ally_classes
    generator = Generator(**gen_kwargs)

    total_dets = 0
    for i in range(start_idx, start_idx + count):
        generator.reset()
        generator.add_tower()
        generator.add_unit(unit_nums)

        img, box, _ = generator.build(box_format='cxcywh', img_size=img_size)

        Image.fromarray(img).save(str(img_dir / f"{i:06d}.jpg"), quality=95)

        det_count = 0
        with open(lbl_dir / f"{i:06d}.txt", 'w') as f:
            for b in box:
                global_cls = int(b[-1])
                if global_cls < 0:
                    continue
                # Filter: skip classes not in this detector
                global_str = str(global_cls)
                if global_str not in global_to_local:
                    continue
                local_cls = global_to_local[global_str]
                cx, cy, w, h = b[0], b[1], b[2], b[3]
                belonging = int(b[4])
                f.write(f"{local_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {belonging}\n")
                det_count += 1

        total_dets += det_count

    return total_dets


def generate_dual_dataset(
    detector,
    split_config_path,
    num_images=10000,
    output_dir=None,
    background_index=15,
    seed=42,
    unit_nums=40,
    noise_unit_ratio=0.25,
    intersect_ratio_thre=0.5,
    img_size=(576, 896),
    ally_classes=None,
    workers=1,
):
    """Generate synthetic dataset for one detector with class filtering + remapping.

    Args:
        detector: Detector number (1 or 2).
        split_config_path: Path to split_config.json.
        num_images: Number of images to generate.
        output_dir: Output directory. Defaults to data/synthetic_d{N}/train.
        Other args: Same as generate_dataset.py.
    """
    with open(split_config_path) as f:
        config = json.load(f)

    det_key = f"detector{detector}"
    det_config = config[det_key]
    global_to_local = det_config["global_to_local"]
    valid_global_set = set(global_to_local.keys())

    if output_dir is None:
        output_dir = f"data/synthetic_d{detector}/train"

    output_path = Path(output_dir)
    img_dir = output_path / "images"
    lbl_dir = output_path / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_images} images for detector {detector}")
    print(f"  Classes: {det_config['n_real_classes']} real + padding (nc={det_config['nc']})")
    print(f"  Label format: 6-col (local_cls x y w h belonging)")
    print(f"  Background: {background_index}, Units/image: {unit_nums}")
    print(f"  Workers: {workers}")
    print(f"  Output: {output_path}")

    if workers <= 1:
        gen_kwargs = dict(
            seed=seed,
            background_index=background_index,
            intersect_ratio_thre=intersect_ratio_thre,
            augment=True,
            dynamic_unit=True,
            noise_unit_ratio=noise_unit_ratio,
        )
        if ally_classes:
            gen_kwargs['ally_classes'] = ally_classes
        generator = Generator(**gen_kwargs)

        total_dets = 0
        for i in tqdm(range(num_images), desc=f"D{detector}"):
            generator.reset()
            generator.add_tower()
            generator.add_unit(unit_nums)

            img, box, _ = generator.build(box_format='cxcywh', img_size=img_size)

            Image.fromarray(img).save(str(img_dir / f"{i:06d}.jpg"), quality=95)

            det_count = 0
            with open(lbl_dir / f"{i:06d}.txt", 'w') as f:
                for b in box:
                    global_cls = int(b[-1])
                    if global_cls < 0:
                        continue
                    global_str = str(global_cls)
                    if global_str not in global_to_local:
                        continue
                    local_cls = global_to_local[global_str]
                    cx, cy, w, h = b[0], b[1], b[2], b[3]
                    belonging = int(b[4])
                    f.write(f"{local_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {belonging}\n")
                    det_count += 1

            total_dets += det_count
    else:
        chunk_size = num_images // workers
        remainder = num_images % workers
        chunks = []
        start = 0
        for w in range(workers):
            count = chunk_size + (1 if w < remainder else 0)
            chunks.append((
                start, count, output_dir, background_index, seed,
                unit_nums, noise_unit_ratio, intersect_ratio_thre,
                img_size, ally_classes, global_to_local, valid_global_set, w,
            ))
            start += count

        print(f"  Chunks: {[c[1] for c in chunks]}")
        with Pool(workers) as pool:
            results = pool.map(_generate_chunk, chunks)
        total_dets = sum(results)

    avg_dets = total_dets / num_images if num_images > 0 else 0
    print(f"\nDone! {num_images} images, avg {avg_dets:.1f} dets/image (filtered for detector {detector})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-detector synthetic dataset")
    parser.add_argument("--detector", type=int, required=True, choices=[1, 2],
                        help="Detector number (1=small sprites, 2=large sprites)")
    parser.add_argument("--split-config", type=str, default="configs/split_config.json",
                        help="Path to split_config.json")
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: data/synthetic_d{N}/train)")
    parser.add_argument("--background", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--units", type=int, default=40)
    parser.add_argument("--noise-ratio", type=float, default=0.25)
    parser.add_argument("--img-width", type=int, default=576)
    parser.add_argument("--img-height", type=int, default=896)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ally-classes", type=str, default=None,
                        help="Comma-separated ally class names")
    args = parser.parse_args()

    ally_list = args.ally_classes.split(',') if args.ally_classes else None

    generate_dual_dataset(
        detector=args.detector,
        split_config_path=args.split_config,
        num_images=args.num_images,
        output_dir=args.output,
        background_index=args.background,
        seed=args.seed,
        unit_nums=args.units,
        noise_unit_ratio=args.noise_ratio,
        img_size=(args.img_width, args.img_height),
        ally_classes=ally_list,
        workers=args.workers,
    )
