"""Train YOLOv8 on the 8 RR Hog deck classes with belonging (ally/enemy).

CUSTOM sprites are ally (belonging=0), KataCR sprites are enemy (belonging=1).
Uses CRDetectionModel + CRDetectionLoss for belonging-aware training.
Pastes sprites onto background15 at random positions.
"""

import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import RANK

from src.yolov8_custom.custom_model import CRDetectionModel

DATASET_ROOT = Path(__file__).resolve().parents[1]
SEGMENT_DIR = Path(
    "/Users/alanguo/Codin/CS175/"
    "Froked-KataCR-Clash-Royale-Detection-Dataset/images/segment"
)
BG_PATH = SEGMENT_DIR / "backgrounds" / "background15.jpg"

# The 8 classes with custom sprites
CLASS_MAP = {
    "arrows": 0,
    "barbarian-barrel": 1,
    "electro-spirit": 2,
    "flying-machine": 3,
    "goblin-cage": 4,
    "royal-hog": 5,
    "royal-recruit": 6,
    "zappy": 7,
}
NC = len(CLASS_MAP) + 1  # 8 classes + 1 padding_belong = 9


def find_sprites():
    """Find CUSTOM (ally) and KataCR (enemy) sprites for the 8 classes."""
    ally = {cls: [] for cls in CLASS_MAP}
    enemy = {cls: [] for cls in CLASS_MAP}
    for cls_name in CLASS_MAP:
        cls_dir = SEGMENT_DIR / cls_name
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob("*.png")):
            if "CUSTOM" in f.name:
                ally[cls_name].append(f)
            else:
                enemy[cls_name].append(f)
    return ally, enemy


def paste_sprite(bg_img, sprite_path, max_scale=1.5):
    """Paste a sprite onto the background at a random position.

    Returns (image, bbox) where bbox is (cx, cy, w, h) normalized.
    """
    sprite = Image.open(sprite_path).convert("RGBA")

    scale = random.uniform(0.8, max_scale)
    new_w = max(1, int(sprite.width * scale))
    new_h = max(1, int(sprite.height * scale))
    sprite = sprite.resize((new_w, new_h), Image.BILINEAR)

    bg_w, bg_h = bg_img.size

    max_x = max(0, bg_w - new_w)
    max_y = max(0, bg_h - new_h)
    x = random.randint(0, max_x) if max_x > 0 else 0
    y = random.randint(0, max_y) if max_y > 0 else 0

    img = bg_img.copy()
    img.paste(sprite, (x, y), sprite)

    cx = (x + new_w / 2) / bg_w
    cy = (y + new_h / 2) / bg_h
    w = new_w / bg_w
    h = new_h / bg_h

    return img, (cx, cy, w, h)


def generate_dataset(output_dir, ally_sprites, enemy_sprites,
                     num_images=500, sprites_per_image=3):
    """Generate training images with ally and enemy sprites."""
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    bg = Image.open(BG_PATH).convert("RGB").resize((640, 640), Image.BILINEAR)

    # Build flat list: (cls_name, path, belonging)
    all_sprites = []
    for cls_name, paths in ally_sprites.items():
        for p in paths:
            all_sprites.append((cls_name, p, 0))  # ally
    for cls_name, paths in enemy_sprites.items():
        for p in paths:
            all_sprites.append((cls_name, p, 1))  # enemy

    n_ally = sum(len(v) for v in ally_sprites.values())
    n_enemy = sum(len(v) for v in enemy_sprites.values())
    print(f"  {len(all_sprites)} sprites total ({n_ally} ally, {n_enemy} enemy)")

    for i in range(num_images):
        img = bg.copy()
        labels = []

        chosen = random.choices(all_sprites, k=sprites_per_image)
        for cls_name, sprite_path, belonging in chosen:
            cls_id = CLASS_MAP[cls_name]
            img, bbox = paste_sprite(img, sprite_path)
            cx, cy, w, h = bbox
            labels.append(
                f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {belonging}"
            )

        img.save(img_dir / f"{i:06d}.jpg", quality=95)
        with open(lbl_dir / f"{i:06d}.txt", "w") as f:
            f.write("\n".join(labels) + "\n")

    print(f"  Generated {num_images} images in {output_dir}")


class CutoutTrainer(DetectionTrainer):
    """Trainer that uses CRDetectionModel for belonging prediction."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CRDetectionModel(
            cfg, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        from copy import copy
        from src.yolov8_custom.custom_validator import CRDetectionValidator
        return CRDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        """Use CRDataset with use_belonging for 6-column label support."""
        from ultralytics.utils import colorstr
        from src.generation.synthetic_dataset import CRDataset
        return CRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            cache=self.args.cache,
            augment=False,
            hyp=self.args,
            prefix=colorstr(f"{mode}: "),
            rect=True,
            batch_size=batch,
            stride=32,
            pad=0.0,
            single_cls=False,
            classes=None,
            fraction=1.0,
            data=self.data,
            seed=self.args.seed,
            use_belonging=True,
        )


def main():
    random.seed(42)
    np.random.seed(42)

    data_root = DATASET_ROOT / "data" / "cutout_test_belong"

    ally_sprites, enemy_sprites = find_sprites()

    print("Ally sprites (CUSTOM):")
    for cls_name in CLASS_MAP:
        print(f"  {cls_name}: {len(ally_sprites[cls_name])}")
    print("Enemy sprites (KataCR):")
    for cls_name in CLASS_MAP:
        print(f"  {cls_name}: {len(enemy_sprites[cls_name])}")

    print("\nGenerating training set...")
    generate_dataset(
        data_root / "train", ally_sprites, enemy_sprites,
        num_images=400, sprites_per_image=3,
    )
    print("\nGenerating validation set...")
    generate_dataset(
        data_root / "val", ally_sprites, enemy_sprites,
        num_images=100, sprites_per_image=3,
    )

    # Write YAML config with nc = 8 + 1 (padding_belong)
    yaml_path = DATASET_ROOT / "configs" / "cutout_test_belong.yaml"
    yaml_content = f"""path: {data_root}
train: train/images
val: val/images
nc: {NC}
names:
"""
    for name, idx in sorted(CLASS_MAP.items(), key=lambda x: x[1]):
        yaml_content += f"  {idx}: {name}\n"
    yaml_content += f"  {NC - 1}: padding_belong\n"

    yaml_path.write_text(yaml_content)
    print(f"\nConfig written to {yaml_path}")
    print(f"nc={NC} (8 classes + 1 padding_belong)")

    # Train with CRDetectionModel
    print(f"\nStarting training (10 epochs, imgsz=640, belonging)...")
    trainer = CutoutTrainer(
        overrides=dict(
            data=str(yaml_path),
            model="yolov8s.pt",
            epochs=10,
            batch=8,
            imgsz=640,
            device="mps",
            workers=0,
            project=str(DATASET_ROOT / "runs" / "cutout_test"),
            name="v2_belong",
            mosaic=0.0,
            mixup=0.0,
            erasing=0.0,
            copy_paste=0.0,
            patience=20,
            verbose=True,
        ),
    )
    trainer.train()

    print("\nTraining complete!")
    print(f"Results: {DATASET_ROOT / 'runs' / 'cutout_test' / 'v2_belong'}")


if __name__ == "__main__":
    main()
