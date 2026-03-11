"""Train one detector of the dual YOLOv8-M system.

Based on train_synthetic.py's SyntheticTrainer pattern. Key differences:
- Uses YOLOv8m (25.9M params) instead of YOLOv8s (11.2M)
- Trains on per-detector dataset with remapped class indices
- batch=8 default for 16GB VRAM (YOLOv8m + imgsz=960 + AMP)
- patience=30 (val belonging mismatch causes noisy metrics)
- augment=(mode == "train") fix (was hardcoded False)
- Always uses belonging (CRDetectionModel + CRDetectionLoss)
"""

import argparse
import json
import sys
from copy import copy
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.build import build_dataloader
from ultralytics.utils import colorstr, RANK

from src.generation.synthetic_dataset import CRDataset
from src.yolov8_custom.custom_model import CRDetectionModel
from src.yolov8_custom.custom_validator import CRDetectionValidator


class DualTrainer(DetectionTrainer):
    """Trainer for one detector of the dual-detector system.

    Always uses belonging. Supports both pre-generated (pregen) and
    on-the-fly generation with class filtering.
    """

    background_index = 15
    noise_unit_ratio = 0.25
    unit_nums = 40
    use_pregen = True  # default to pregen for dual training
    class_filter = None  # set of global class indices for this detector
    global_to_local = None  # dict mapping str(global_idx) -> local_idx

    def build_dataset(self, img_path, mode="train", batch=None):
        use_disk = (mode == "val") or self.use_pregen
        return CRDataset(
            img_path=img_path if use_disk else None,
            imgsz=self.args.imgsz,
            cache=self.args.cache,
            augment=(mode == "train"),
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
            use_belonging=True,
            class_filter=self.class_filter,
            global_to_local=self.global_to_local,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CRDetectionValidator(
            self.test_loader, save_dir=self.save_dir,
            args=copy(self.args), _callbacks=self.callbacks,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in ["train", "val"]
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def plot_training_labels(self):
        pass


def train(
    detector,
    split_config_path="configs/split_config.json",
    data_yaml=None,
    model_type="yolov8m.pt",
    epochs=50,
    batch=8,
    imgsz=960,
    device=0,
    workers=4,
    project="runs/dual",
    name=None,
    resume=False,
    background_index=15,
    noise_ratio=0.25,
    unit_nums=40,
    seed=42,
    use_pregen=True,
):
    """Train one detector of the dual system.

    Args:
        detector: 1 or 2.
        split_config_path: Path to split_config.json.
        data_yaml: Override data.yaml path. Defaults to configs/detector{N}_data.yaml.
        model_type: Base model. Default yolov8m.pt.
        use_pregen: Use pre-generated dataset (default True).
    """
    with open(split_config_path) as f:
        split_config = json.load(f)

    det_key = f"detector{detector}"
    det_config = split_config[det_key]

    if data_yaml is None:
        data_yaml = f"configs/{det_key}_data.yaml"

    if name is None:
        name = f"d{detector}_v1"

    # Create dummy train dir for YAML path validation if using on-the-fly
    train_dir = f"data/synthetic_d{detector}/train/images"
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(train_dir.replace("/images", "/labels")).mkdir(parents=True, exist_ok=True)

    # Set up class filter for on-the-fly generation path
    class_filter = set(det_config["global_indices"])
    global_to_local = det_config["global_to_local"]

    DualTrainer.background_index = background_index
    DualTrainer.noise_unit_ratio = noise_ratio
    DualTrainer.unit_nums = unit_nums
    DualTrainer.use_pregen = use_pregen
    DualTrainer.class_filter = class_filter
    DualTrainer.global_to_local = global_to_local

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

        # Generator handles compositing
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

        'patience': 30,
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
    print(f"Training detector {detector} ({det_key})")
    print(f"  Model: {model_type}, ImgSz: {imgsz}")
    print(f"  Epochs: {epochs}, Batch: {batch}, Patience: 30")
    print(f"  Classes: {det_config['n_real_classes']} real (nc={det_config['nc']})")
    print(f"  Pre-gen: {use_pregen}")
    print(f"  Data YAML: {data_yaml}")

    trainer = DualTrainer(overrides=overrides)
    trainer.train()

    results_dir = Path(project) / name
    print(f"\nTraining complete!")
    print(f"Best weights: {results_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one dual-detector model")
    parser.add_argument("--detector", type=int, required=True, choices=[1, 2],
                        help="Detector number (1=small, 2=large)")
    parser.add_argument("--split-config", type=str, default="configs/split_config.json")
    parser.add_argument("--data", type=str, default=None,
                        help="Override data.yaml (default: configs/detector{N}_data.yaml)")
    parser.add_argument("--model", type=str, default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/dual")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--background", type=int, default=15)
    parser.add_argument("--noise-ratio", type=float, default=0.25)
    parser.add_argument("--units", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pregen", action="store_true", default=True,
                        help="Use pre-generated dataset (default)")
    parser.add_argument("--no-pregen", dest="pregen", action="store_false",
                        help="Use on-the-fly generation with class filtering")
    args = parser.parse_args()

    train(
        detector=args.detector,
        split_config_path=args.split_config,
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
        use_pregen=args.pregen,
    )
