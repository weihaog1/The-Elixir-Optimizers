"""Train YOLOv8 with belonging (ally/enemy) prediction.

Uses custom CRDetectionModel with padding_belong as the last class channel.
The model learns to predict both class and belonging from visual features.
Ally sprites are restricted to a fixed deck during training.
"""

import argparse
import sys
from copy import copy
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.build import build_dataloader
from ultralytics.utils import colorstr, RANK, LOGGER

from src.yolov8_custom.custom_model import CRDetectionModel
from src.yolov8_custom.custom_validator import CRDetectionValidator
from src.yolov8_custom.custom_utils import plot_images
from src.generation.synthetic_dataset import CRDataset

# RR Hogs deck: 10 ally classes (8 cards + 2 spawned units)
ALLY_CLASSES = [
    'royal-hog',
    'royal-recruit',
    'goblin-cage',
    'goblin-brawler',  # spawned by goblin-cage
    'flying-machine',
    'electro-spirit',
    'bowler',
    'barbarian-barrel',
    'barbarian',  # spawned by barbarian-barrel
    'arrows',
]


class BelongingTrainer(DetectionTrainer):
    """Detection trainer with belonging prediction and fixed deck restriction."""

    # Configure before calling train()
    background_index = 15
    noise_unit_ratio = 0.25
    unit_nums = 40
    ally_classes = ALLY_CLASSES

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return CRDetectionModel with belonging-aware loss."""
        model = CRDetectionModel(
            cfg, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return CRDetectionValidator for belonging-aware validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CRDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        return CRDataset(
            img_path=img_path if mode == "val" else None,
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
            unit_nums=self.unit_nums,
            noise_unit_ratio=self.noise_unit_ratio,
            background_index=self.background_index,
            ally_classes=None,  # Don't restrict ally classes; most deck classes lack ally sprites
            use_belonging=True,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in ["train", "val"]
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def plot_training_labels(self):
        pass  # Skip for generated data

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"],
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
            names=self.data['names'],
        )


def train(
    data_yaml="configs/synthetic_belonging_data.yaml",
    model_type="yolov8s.pt",
    epochs=50,
    batch=16,
    imgsz=960,
    device=0,
    workers=4,
    project="runs/synthetic",
    name="yolov8s_belonging_v1",
    resume=False,
    background_index=15,
    noise_ratio=0.25,
    unit_nums=40,
    seed=42,
):
    # Ensure train dir exists for YAML path validation
    Path("data/synthetic_belong/train/images").mkdir(parents=True, exist_ok=True)
    Path("data/synthetic_belong/train/labels").mkdir(parents=True, exist_ok=True)

    BelongingTrainer.background_index = background_index
    BelongingTrainer.noise_unit_ratio = noise_ratio
    BelongingTrainer.unit_nums = unit_nums

    overrides = {
        'data': data_yaml,
        'model': model_type,
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'device': device,
        'workers': workers,
        'project': project,
        'name': name,
        'seed': seed,

        # Generator handles compositing, disable mosaic/mixup
        'mosaic': 0.0,
        'mixup': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5,
        'translate': 0.05,
        'scale': 0.5,
        'flipud': 0.0,
        'fliplr': 0.5,
        'erasing': 0.4,

        'patience': 15,
        'lr0': 0.01,
        'lrf': 0.01,
        'amp': True,
        'close_mosaic': 0,
    }

    if resume:
        ckpt = Path(project) / name / "weights" / "last.pt"
        if ckpt.exists():
            overrides['model'] = str(ckpt)
            overrides['resume'] = True
            print(f"Resuming from {ckpt}")
        else:
            print(f"No checkpoint at {ckpt}, starting fresh")

    from src.generation.synthetic_dataset import TRAIN_DATASIZE
    print(f"Training with belonging prediction (ally/enemy)")
    print(f"  Model: {model_type}, ImgSz: {imgsz}")
    print(f"  Epochs: {epochs}, Batch: {batch}")
    print(f"  Images/epoch: {TRAIN_DATASIZE} (unique)")
    print(f"  Units/image: {unit_nums}, Background: {background_index}")
    print(f"  Noise ratio: {noise_ratio}, Workers: {workers}")
    print(f"  Ally classes: {ALLY_CLASSES}")
    print(f"  nc: 156 (155 classes + padding_belong)")

    trainer = BelongingTrainer(overrides=overrides)
    trainer.train()

    results_dir = Path(project) / name
    print(f"\nTraining complete!")
    print(f"Best weights: {results_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 with belonging prediction"
    )
    parser.add_argument("--data", type=str,
                        default="configs/synthetic_belonging_data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/synthetic")
    parser.add_argument("--name", type=str, default="yolov8s_belonging_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--background", type=int, default=15)
    parser.add_argument("--noise-ratio", type=float, default=0.25)
    parser.add_argument("--units", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_type=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        background_index=args.background,
        noise_ratio=args.noise_ratio,
        unit_nums=args.units,
        seed=args.seed,
    )
