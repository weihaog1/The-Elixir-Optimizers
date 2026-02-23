"""
Inference script for Clash Royale object detection.

Usage:
    python -m src.detection.inference --model models/best.pt --source image.png
    python -m src.detection.inference --model models/best.pt --source data/test/ --save
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import cv2

from .model import CRDetector, Detection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with trained Clash Royale detector"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Image file or directory to run inference on"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda, cpu, or None for auto)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save visualized results"
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detections as YOLO format labels"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detections as JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="runs/detect",
        help="Output directory for results (default: runs/detect)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in window"
    )
    return parser.parse_args()


def detections_to_dict(detections: List[Detection], img_width: int, img_height: int) -> List[Dict]:
    """Convert detections to JSON-serializable format."""
    results = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        results.append({
            "class_id": det.class_id,
            "class_name": det.class_name,
            "confidence": round(det.confidence, 4),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "center": {
                "x": det.center[0],
                "y": det.center[1],
            },
            "size": {
                "width": det.width,
                "height": det.height,
            },
            "yolo_format": {
                "x_center": round((x1 + x2) / 2 / img_width, 6),
                "y_center": round((y1 + y2) / 2 / img_height, 6),
                "width": round(det.width / img_width, 6),
                "height": round(det.height / img_height, 6),
            }
        })
    return results


def process_image(
    detector: CRDetector,
    image_path: Path,
    output_dir: Path,
    save_viz: bool = False,
    save_txt: bool = False,
    save_json: bool = False,
    show: bool = False,
) -> List[Detection]:
    """Process a single image and optionally save results."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image: {image_path}")
        return []

    height, width = img.shape[:2]
    detections = detector.detect(img)

    print(f"{image_path.name}: {len(detections)} detections")
    for det in detections:
        print(f"  - {det.class_name}: {det.confidence:.3f} @ {det.bbox}")

    # Save visualized image
    if save_viz:
        viz_path = output_dir / f"{image_path.stem}_det.jpg"
        detector.visualize(img, detections, output_path=str(viz_path))

    # Save YOLO format labels
    if save_txt:
        label_path = output_dir / "labels" / f"{image_path.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w") as f:
            for det in detections:
                yolo = det.to_yolo_format(width, height)
                f.write(f"{yolo[0]} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f} {yolo[4]:.6f}\n")

    # Save JSON
    if save_json:
        json_path = output_dir / "json" / f"{image_path.stem}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image": str(image_path),
            "width": width,
            "height": height,
            "detections": detections_to_dict(detections, width, height),
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    # Display
    if show:
        viz_img = detector.visualize(img, detections)
        cv2.imshow("Detections", viz_img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            return detections

    return detections


def main():
    args = parse_args()

    # Verify model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {args.model}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    print(f"Loading model: {args.model}")
    detector = CRDetector(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
    )

    # Get source images
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_paths = list(source_path.glob("*.png")) + \
                      list(source_path.glob("*.jpg")) + \
                      list(source_path.glob("*.jpeg"))
    else:
        print(f"Error: Source not found: {args.source}")
        return 1

    if not image_paths:
        print(f"Error: No images found in {args.source}")
        return 1

    print(f"Processing {len(image_paths)} images...")
    print("-" * 40)

    all_results = {}
    for image_path in sorted(image_paths):
        detections = process_image(
            detector=detector,
            image_path=image_path,
            output_dir=output_dir,
            save_viz=args.save,
            save_txt=args.save_txt,
            save_json=args.save_json,
            show=args.show,
        )
        all_results[str(image_path)] = len(detections)

    print("-" * 40)
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total detections: {sum(all_results.values())}")

    if args.save or args.save_txt or args.save_json:
        print(f"Results saved to: {output_dir}")

    if args.show:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    exit(main())
