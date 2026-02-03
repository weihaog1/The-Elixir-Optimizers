"""
Training Script for Clash Royale Object Detection
Train custom YOLOv8 models on Clash Royale dataset.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml: str,
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "clash_royale",
    pretrained: bool = True,
    resume: bool = False,
):
    """
    Train a YOLOv8 model on Clash Royale dataset.
    
    Args:
        data_yaml: Path to data.yaml file with dataset configuration.
        model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge).
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        img_size: Image size for training.
        device: GPU device (0, 1, etc.) or 'cpu'.
        project: Project directory for saving results.
        name: Experiment name.
        pretrained: Whether to use pretrained weights.
        resume: Whether to resume from last checkpoint.
    """
    
    # Load model
    if resume:
        # Resume from last checkpoint
        model = YOLO(f"{project}/{name}/weights/last.pt")
    elif pretrained:
        # Start from pretrained COCO weights
        model = YOLO(f"yolov8{model_size}.pt")
    else:
        # Train from scratch (not recommended)
        model = YOLO(f"yolov8{model_size}.yaml")
    
    print(f"Training YOLOv8{model_size} on {data_yaml}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    print(f"Device: {device}")
    print("-" * 50)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        
        # Data augmentation (similar to KataCR settings)
        hsv_h=0.015,      # HSV-Hue augmentation
        hsv_s=0.7,        # HSV-Saturation augmentation
        hsv_v=0.4,        # HSV-Value augmentation
        degrees=0.0,      # Rotation (disabled for CR)
        translate=0.1,    # Translation
        scale=0.5,        # Scale
        shear=0.0,        # Shear
        perspective=0.0,  # Perspective
        flipud=0.0,       # Flip up-down (disabled)
        fliplr=0.5,       # Flip left-right
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.0,        # Mixup augmentation
        
        # Training settings
        patience=50,      # Early stopping patience
        save=True,        # Save checkpoints
        save_period=10,   # Save every N epochs
        val=True,         # Run validation
        plots=True,       # Generate plots
        
        # Optimizer settings
        optimizer="SGD",
        lr0=0.01,         # Initial learning rate
        lrf=0.01,         # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Other
        workers=8,
        seed=42,
        verbose=True,
    )
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    print(f"Last model saved to: {project}/{name}/weights/last.pt")
    print("=" * 50)
    
    return results


def validate(model_path: str, data_yaml: str, device: str = "0"):
    """Validate a trained model."""
    model = YOLO(model_path)
    results = model.val(data=data_yaml, device=device)
    return results


def export_model(model_path: str, format: str = "onnx"):
    """Export model to different formats."""
    model = YOLO(model_path)
    model.export(format=format)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Clash Royale Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train.py --data path/to/data.yaml
  
  # Train larger model for more accuracy
  python train.py --data path/to/data.yaml --model-size m --epochs 150
  
  # Train on specific GPU
  python train.py --data path/to/data.yaml --device 1
  
  # Resume interrupted training
  python train.py --data path/to/data.yaml --resume
  
  # Validate trained model
  python train.py --validate --model runs/train/clash_royale/weights/best.pt --data path/to/data.yaml
"""
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml configuration file"
    )
    
    parser.add_argument(
        "--model-size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for training"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device (0, 1, etc.) or 'cpu'"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="clash_royale",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation instead of training"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path for validation"
    )
    
    args = parser.parse_args()
    
    if args.validate:
        if args.model is None:
            print("Error: --model required for validation")
            return
        validate(args.model, args.data, args.device)
    else:
        train(
            data_yaml=args.data,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            name=args.name,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
