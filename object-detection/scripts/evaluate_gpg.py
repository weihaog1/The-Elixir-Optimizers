#!/usr/bin/env python3
"""
Evaluate trained model on Google Play Games resolution screenshots.

This script is part of Phase 5: Fine-tuning for Google Play Games.
It evaluates model performance on screenshots captured at ~540x960 resolution
and identifies areas where fine-tuning may be needed.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


def load_model(model_path: str) -> Optional['YOLO']:
    """Load a trained YOLO model."""
    if not HAS_ULTRALYTICS:
        return None

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None

    return YOLO(model_path)


def analyze_resolution(image_path: str) -> Dict:
    """Analyze image resolution and aspect ratio."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}

    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Target: 540x960 (aspect ratio: 0.5625)
    target_aspect = 540 / 960
    aspect_diff = abs(aspect_ratio - target_aspect)

    return {
        "path": image_path,
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 4),
        "target_aspect": target_aspect,
        "aspect_difference": round(aspect_diff, 4),
        "resolution_match": width == 540 and height == 960,
        "needs_resize": width != 540 or height != 960,
    }


def run_inference(
    model: 'YOLO',
    image_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict:
    """Run inference on a single image and return results."""
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        det = {
            "class_id": int(box.cls.item()),
            "class_name": results.names[int(box.cls.item())],
            "confidence": round(float(box.conf.item()), 4),
            "bbox": [round(x, 2) for x in box.xyxy[0].tolist()],
        }
        detections.append(det)

    # Group by class
    class_counts = {}
    for det in detections:
        name = det["class_name"]
        if name not in class_counts:
            class_counts[name] = 0
        class_counts[name] += 1

    return {
        "path": image_path,
        "num_detections": len(detections),
        "detections": detections,
        "class_counts": class_counts,
    }


def evaluate_tower_detection(detections: List[Dict]) -> Dict:
    """Evaluate tower detection quality."""
    tower_classes = [
        "king_tower_player", "king_tower_enemy",
        "princess_tower_left_player", "princess_tower_left_enemy",
        "princess_tower_right_player", "princess_tower_right_enemy",
        # Also check alternative naming from full_dataset.yaml
        "0-bar-bar-king-tower", "1-bar-bar-princess-tower",
        "2-bar-bar-king-tower-enemy", "3-bar-bar-princess-tower-enemy",
    ]

    tower_detections = [
        d for d in detections
        if d["class_name"].lower() in [t.lower() for t in tower_classes]
        or "tower" in d["class_name"].lower()
    ]

    # Expected: 1 king tower + 2 princess towers per side = 6 total
    # But some may be destroyed
    return {
        "tower_count": len(tower_detections),
        "tower_detections": tower_detections,
        "expected_max": 6,
        "detection_rate": len(tower_detections) / 6 if tower_detections else 0,
    }


def evaluate_batch(
    model: 'YOLO',
    image_dir: str,
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.25,
) -> Dict:
    """Evaluate model on a batch of images."""
    image_dir = Path(image_dir)

    # Find images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    images = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not images:
        return {"error": f"No images found in {image_dir}"}

    print(f"Found {len(images)} images in {image_dir}")

    results = {
        "total_images": len(images),
        "resolution_analysis": [],
        "inference_results": [],
        "summary": {},
    }

    # Analyze and run inference
    all_detections = []
    total_towers = 0

    for img_path in images:
        # Resolution analysis
        res_info = analyze_resolution(str(img_path))
        results["resolution_analysis"].append(res_info)

        # Inference
        if model:
            inf_result = run_inference(model, str(img_path), conf_threshold)
            results["inference_results"].append(inf_result)
            all_detections.extend(inf_result["detections"])

            # Tower evaluation
            tower_eval = evaluate_tower_detection(inf_result["detections"])
            total_towers += tower_eval["tower_count"]

    # Summary statistics
    results["summary"] = {
        "total_images": len(images),
        "resolution_matches": sum(1 for r in results["resolution_analysis"] if r.get("resolution_match", False)),
        "needs_resize_count": sum(1 for r in results["resolution_analysis"] if r.get("needs_resize", True)),
        "total_detections": len(all_detections) if model else 0,
        "avg_detections_per_image": len(all_detections) / len(images) if model and images else 0,
        "tower_detection_total": total_towers if model else 0,
    }

    # Class distribution
    if all_detections:
        class_dist = {}
        for det in all_detections:
            name = det["class_name"]
            if name not in class_dist:
                class_dist[name] = {"count": 0, "avg_conf": 0, "confs": []}
            class_dist[name]["count"] += 1
            class_dist[name]["confs"].append(det["confidence"])

        # Calculate averages
        for name in class_dist:
            confs = class_dist[name]["confs"]
            class_dist[name]["avg_conf"] = round(sum(confs) / len(confs), 4)
            del class_dist[name]["confs"]

        results["summary"]["class_distribution"] = class_dist

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to {report_path}")

    return results


def print_summary(results: Dict):
    """Print evaluation summary."""
    summary = results.get("summary", {})

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nImages Analyzed: {summary.get('total_images', 0)}")
    print(f"Resolution Matches (540x960): {summary.get('resolution_matches', 0)}")
    print(f"Images Needing Resize: {summary.get('needs_resize_count', 0)}")

    if summary.get("total_detections", 0) > 0:
        print(f"\nTotal Detections: {summary['total_detections']}")
        print(f"Avg Detections/Image: {summary.get('avg_detections_per_image', 0):.1f}")
        print(f"Tower Detections: {summary.get('tower_detection_total', 0)}")

        class_dist = summary.get("class_distribution", {})
        if class_dist:
            print("\nClass Distribution:")
            sorted_classes = sorted(class_dist.items(), key=lambda x: x[1]["count"], reverse=True)
            for name, info in sorted_classes[:15]:  # Top 15
                print(f"  {name}: {info['count']} (avg conf: {info['avg_conf']:.3f})")
    else:
        print("\nNo inference results (model not loaded)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on Google Play Games screenshots"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--images", "-i",
        type=str,
        required=True,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--resolution-only",
        action="store_true",
        help="Only analyze resolution, skip model inference"
    )

    args = parser.parse_args()

    # Load model
    model = None
    if not args.resolution_only:
        model = load_model(args.model)
        if model is None and not args.resolution_only:
            print("Warning: Running without model (resolution analysis only)")

    # Run evaluation
    results = evaluate_batch(
        model=model,
        image_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf,
    )

    # Print summary
    print_summary(results)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    summary = results.get("summary", {})

    if summary.get("needs_resize_count", 0) > 0:
        print("- Some images have different resolution. Consider standardizing to 540x960")

    if model and summary.get("avg_detections_per_image", 0) < 5:
        print("- Low detection count. Model may need fine-tuning for GPG resolution")

    if model and summary.get("tower_detection_total", 0) < summary.get("total_images", 0) * 3:
        print("- Low tower detection rate. Verify tower classes in training data")

    print("\nUse --output to save detailed results as JSON")


if __name__ == "__main__":
    main()
