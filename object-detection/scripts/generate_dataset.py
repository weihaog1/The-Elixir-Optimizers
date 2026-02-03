"""Pre-generate synthetic training dataset using KataCR's generator.

Supports optional belonging labels (6-column format), ally deck restriction,
and multi-worker parallel generation.
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.generation.generator import Generator


def _generate_chunk(args):
    """Worker function: generate a chunk of images."""
    (start_idx, count, output_dir, background_index, seed, unit_nums,
     noise_unit_ratio, intersect_ratio_thre, img_size, ally_classes,
     include_belonging, worker_id) = args

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

        with open(lbl_dir / f"{i:06d}.txt", 'w') as f:
            for b in box:
                cls_id = int(b[-1])
                if cls_id < 0:
                    continue
                cx, cy, w, h = b[0], b[1], b[2], b[3]
                if include_belonging:
                    belonging = int(b[4])
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {belonging}\n")
                else:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        total_dets += len([b for b in box if int(b[-1]) >= 0])

    return total_dets


def generate_dataset(
    num_images: int = 10000,
    output_dir: str = "data/synthetic/train",
    background_index: int = 15,
    seed: int = 42,
    unit_nums: int = 40,
    noise_unit_ratio: float = 0.25,
    intersect_ratio_thre: float = 0.5,
    img_size: tuple = (576, 896),
    ally_classes: list = None,
    include_belonging: bool = False,
    workers: int = 1,
):
    """Generate synthetic training images in YOLO format.

    Args:
        num_images: Number of images to generate.
        output_dir: Output directory for images and labels.
        background_index: Background arena index (15 = stone/railroad).
        seed: Random seed for reproducibility.
        unit_nums: Number of sprite units per image.
        noise_unit_ratio: Ratio of noise/hard-negative units.
        intersect_ratio_thre: Occlusion NMS threshold.
        img_size: Output image size (width, height).
        ally_classes: If set, restrict bel=0 sprites to these class names.
        include_belonging: If True, write 6-column labels with belonging.
        workers: Number of parallel workers (each ~300MB RAM).
    """
    output_path = Path(output_dir)
    img_dir = output_path / "images"
    lbl_dir = output_path / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    label_fmt = "6-col (cls x y w h bel)" if include_belonging else "5-col (cls x y w h)"
    print(f"Generating {num_images} synthetic images...")
    print(f"  Background: {background_index}")
    print(f"  Units/image: {unit_nums}")
    print(f"  Noise ratio: {noise_unit_ratio}")
    print(f"  Image size: {img_size}")
    print(f"  Label format: {label_fmt}")
    print(f"  Workers: {workers}")
    if ally_classes:
        print(f"  Ally classes ({len(ally_classes)}): {', '.join(ally_classes)}")
    print(f"  Output: {output_path}")

    if workers <= 1:
        # Single-threaded path with progress bar
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

        total_detections = 0
        for i in tqdm(range(num_images), desc="Generating"):
            generator.reset()
            generator.add_tower()
            generator.add_unit(unit_nums)

            img, box, pil_img = generator.build(
                box_format='cxcywh',
                img_size=img_size,
            )

            img_path = img_dir / f"{i:06d}.jpg"
            Image.fromarray(img).save(str(img_path), quality=95)

            lbl_path = lbl_dir / f"{i:06d}.txt"
            with open(lbl_path, 'w') as f:
                for b in box:
                    cls_id = int(b[-1])
                    if cls_id < 0:
                        continue
                    cx, cy, w, h = b[0], b[1], b[2], b[3]
                    if include_belonging:
                        belonging = int(b[4])
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {belonging}\n")
                    else:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            total_detections += len([b for b in box if int(b[-1]) >= 0])
    else:
        # Multi-worker parallel generation
        chunk_size = num_images // workers
        remainder = num_images % workers
        chunks = []
        start = 0
        for w in range(workers):
            count = chunk_size + (1 if w < remainder else 0)
            chunks.append((
                start, count, output_dir, background_index, seed,
                unit_nums, noise_unit_ratio, intersect_ratio_thre,
                img_size, ally_classes, include_belonging, w,
            ))
            start += count

        print(f"  Chunks: {[c[1] for c in chunks]}")
        with Pool(workers) as pool:
            results = pool.map(_generate_chunk, chunks)
        total_detections = sum(results)

    avg_dets = total_detections / num_images
    print(f"\nDone! Generated {num_images} images with avg {avg_dets:.1f} detections/image")
    print(f"Images: {img_dir}")
    print(f"Labels: {lbl_dir}")


def setup_validation(val_source: str = "data/prepared", val_dest: str = "data/synthetic/val"):
    """Symlink or copy validation set from prepared data.

    Args:
        val_source: Source directory with existing val images/labels.
        val_dest: Destination in synthetic dataset structure.
    """
    src = Path(val_source)
    dst = Path(val_dest)

    src_images = src / "images" / "val"
    src_labels = src / "labels" / "val"
    dst_images = dst / "images"
    dst_labels = dst / "labels"

    if not src_images.exists():
        print(f"Warning: Validation source not found at {src_images}")
        return

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    import shutil
    for img_file in sorted(src_images.glob("*.jpg")):
        dest_file = dst_images / img_file.name
        if not dest_file.exists():
            shutil.copy2(img_file, dest_file)

    for lbl_file in sorted(src_labels.glob("*.txt")):
        dest_file = dst_labels / lbl_file.name
        if not dest_file.exists():
            shutil.copy2(lbl_file, dest_file)

    n_imgs = len(list(dst_images.glob("*.jpg")))
    n_lbls = len(list(dst_labels.glob("*.txt")))
    print(f"Validation set: {n_imgs} images, {n_lbls} labels in {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset")
    parser.add_argument("--num-images", type=int, default=10000, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="data/synthetic/train", help="Output directory")
    parser.add_argument("--background", type=int, default=15, help="Background index (15=stone/railroad)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--units", type=int, default=40, help="Units per image")
    parser.add_argument("--noise-ratio", type=float, default=0.25, help="Noise unit ratio")
    parser.add_argument("--img-width", type=int, default=576, help="Output image width")
    parser.add_argument("--img-height", type=int, default=896, help="Output image height")
    parser.add_argument("--setup-val", action="store_true", help="Also set up validation symlinks")
    parser.add_argument("--val-source", type=str, default="data/prepared", help="Validation source dir")
    parser.add_argument("--ally-classes", type=str, default=None,
                        help="Comma-separated ally class names for fixed deck restriction")
    parser.add_argument("--include-belonging", action="store_true",
                        help="Write 6-column labels with belonging (cls x y w h bel)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (~300MB RAM each)")
    args = parser.parse_args()

    ally_list = args.ally_classes.split(',') if args.ally_classes else None

    generate_dataset(
        num_images=args.num_images,
        output_dir=args.output,
        background_index=args.background,
        seed=args.seed,
        unit_nums=args.units,
        noise_unit_ratio=args.noise_ratio,
        img_size=(args.img_width, args.img_height),
        ally_classes=ally_list,
        include_belonging=args.include_belonging,
        workers=args.workers,
    )

    if args.setup_val:
        setup_validation(val_source=args.val_source)
