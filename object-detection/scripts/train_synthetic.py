"""Train YOLOv8 on synthetic data generated on-the-fly.

Ported from KataCR's katacr/yolov8/custom_trainer.py.
Uses their CRDataset (subclasses YOLODataset, overrides get_image_and_label)
so every training image is unique. Validation uses human-labeled images.
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

from src.generation.synthetic_dataset import CRDataset
from src.yolov8_custom.custom_model import CRDetectionModel
from src.yolov8_custom.custom_validator import CRDetectionValidator


class SyntheticTrainer(DetectionTrainer):
    """Detection trainer using on-the-fly synthetic data for training.

    Ported from KataCR's CRTrainer. Key override: build_dataset() returns
    CRDataset which generates unique images via the Generator.
    """

    # Configure before calling train()
    background_index = 15
    noise_unit_ratio = 0.25
    unit_nums = 40
    use_belonging = False
    use_pregen = False

    def build_dataset(self, img_path, mode="train", batch=None):
        use_disk = (mode == "val") or self.use_pregen
        return CRDataset(
            img_path=img_path if use_disk else None,
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
            use_belonging=self.use_belonging,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.use_belonging:
            model = CRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
            if weights:
                model.load(weights)
            return model
        return super().get_model(cfg, weights, verbose)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        if self.use_belonging:
            return CRDetectionValidator(
                self.test_loader, save_dir=self.save_dir,
                args=copy(self.args), _callbacks=self.callbacks,
            )
        return super().get_validator()

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in ["train", "val"]
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def plot_training_labels(self):
        pass  # Skip for generated data


def train(
    data_yaml="configs/synthetic_data.yaml",
    model_type="yolov8s.pt",
    epochs=50,
    batch=16,
    imgsz=960,
    device=0,
    workers=4,
    project="runs/synthetic",
    name="yolov8s_gen_v1",
    resume=False,
    background_index=15,
    noise_ratio=0.25,
    unit_nums=40,
    seed=42,
    use_belonging=False,
    use_pregen=False,
):
    # Ensure train dir exists for YAML path validation
    if use_belonging:
        Path("data/synthetic_belong/train/images").mkdir(parents=True, exist_ok=True)
        Path("data/synthetic_belong/train/labels").mkdir(parents=True, exist_ok=True)
    else:
        Path("data/synthetic/train/images").mkdir(parents=True, exist_ok=True)
        Path("data/synthetic/train/labels").mkdir(parents=True, exist_ok=True)

    SyntheticTrainer.background_index = background_index
    SyntheticTrainer.noise_unit_ratio = noise_ratio
    SyntheticTrainer.unit_nums = unit_nums
    SyntheticTrainer.use_belonging = use_belonging
    SyntheticTrainer.use_pregen = use_pregen

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
    print(f"Training with on-the-fly synthetic generation")
    print(f"  Model: {model_type}, ImgSz: {imgsz}")
    print(f"  Epochs: {epochs}, Batch: {batch}")
    print(f"  Images/epoch: {TRAIN_DATASIZE} (unique)")
    print(f"  Units/image: {unit_nums}, Background: {background_index}")
    print(f"  Noise ratio: {noise_ratio}, Workers: {workers}")
    print(f"  Belonging: {use_belonging}")
    print(f"  Pre-gen: {use_pregen}")

    trainer = SyntheticTrainer(overrides=overrides)
    trainer.train()

    results_dir = Path(project) / name
    print(f"\nTraining complete!")
    print(f"Best weights: {results_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 with on-the-fly synthetic data")
    parser.add_argument("--data", type=str, default="configs/synthetic_data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/synthetic")
    parser.add_argument("--name", type=str, default="yolov8s_gen_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--background", type=int, default=15)
    parser.add_argument("--noise-ratio", type=float, default=0.25)
    parser.add_argument("--units", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--belonging", action="store_true",
                        help="Train with belonging (ally/enemy) prediction head")
    parser.add_argument("--pregen", action="store_true",
                        help="Load training images from disk instead of on-the-fly generation")
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
        use_belonging=args.belonging,
        use_pregen=args.pregen,
    )
