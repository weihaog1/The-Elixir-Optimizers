"""
Annotation helper for generating YOLO format labels.

This module provides tools to:
1. Generate initial bounding box annotations for towers based on known regions
2. Visualize annotations on images for verification
3. Convert annotations to YOLO format
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .screen_regions import (
    Region,
    ScreenConfig,
    TOWER_CLASS_IDS,
    CLASS_ID_TO_NAME,
    get_config_for_image,
    get_default_config,
)


def generate_tower_annotations(
    image_path: str,
    config: Optional[ScreenConfig] = None,
    verify_visibility: bool = True,
) -> List[Tuple[int, float, float, float, float]]:
    """Generate initial tower annotations for an image.

    Args:
        image_path: Path to the image file.
        config: Screen configuration (auto-detected if None).
        verify_visibility: If True, only include towers that appear to be present.

    Returns:
        List of (class_id, x_center, y_center, width, height) in YOLO format.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if config is None:
        config = get_config_for_image(image)

    height, width = image.shape[:2]
    annotations = []

    tower_regions = config.get_tower_regions()

    for tower_name, region in tower_regions.items():
        class_id = TOWER_CLASS_IDS[tower_name]

        # Convert to YOLO format
        x_center, y_center, w, h = region.to_yolo_format(width, height)

        if verify_visibility:
            # Simple check: see if the region has significant content
            # (i.e., not all black/uniform color)
            crop = region.crop_image(image)
            if crop.size == 0:
                continue
            std_dev = np.std(crop)
            if std_dev < 10:  # Nearly uniform, likely no tower
                continue

        annotations.append((class_id, x_center, y_center, w, h))

    return annotations


def write_yolo_label(
    label_path: str,
    annotations: List[Tuple[int, float, float, float, float]],
) -> None:
    """Write annotations to YOLO format label file.

    Args:
        label_path: Path to output label file (.txt).
        annotations: List of (class_id, x_center, y_center, width, height).
    """
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as f:
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def read_yolo_label(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Read annotations from YOLO format label file.

    Args:
        label_path: Path to label file (.txt).

    Returns:
        List of (class_id, x_center, y_center, width, height).
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append((class_id, x_center, y_center, width, height))

    return annotations


def yolo_to_bbox(
    annotation: Tuple[int, float, float, float, float],
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int, int]:
    """Convert YOLO annotation to pixel bounding box.

    Args:
        annotation: (class_id, x_center, y_center, width, height) normalized.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        (class_id, x1, y1, x2, y2) in pixels.
    """
    class_id, x_center, y_center, width, height = annotation
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return (class_id, x1, y1, x2, y2)


def visualize_annotations(
    image_path: str,
    annotations: Optional[List[Tuple[int, float, float, float, float]]] = None,
    label_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = False,
) -> np.ndarray:
    """Draw annotations on an image for verification.

    Args:
        image_path: Path to the image file.
        annotations: List of YOLO annotations (optional if label_path provided).
        label_path: Path to YOLO label file (optional if annotations provided).
        output_path: Path to save visualized image (optional).
        show: If True, display the image in a window.

    Returns:
        Image with annotations drawn.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if annotations is None and label_path:
        annotations = read_yolo_label(label_path)
    elif annotations is None:
        annotations = []

    height, width = image.shape[:2]

    # Color palette for different classes
    colors = [
        (0, 255, 0),    # Green - king_tower_player
        (0, 0, 255),    # Red - king_tower_enemy
        (0, 200, 100),  # Light green - princess_tower_left_player
        (0, 100, 200),  # Orange-red - princess_tower_left_enemy
        (100, 255, 0),  # Yellow-green - princess_tower_right_player
        (0, 50, 255),   # Dark red - princess_tower_right_enemy
    ]

    for ann in annotations:
        class_id, x1, y1, x2, y2 = yolo_to_bbox(ann, width, height)
        color = colors[class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = CLASS_ID_TO_NAME.get(class_id, f"class_{class_id}")
        cv2.putText(
            image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    if output_path:
        cv2.imwrite(output_path, image)

    if show:
        cv2.imshow("Annotations", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def process_directory(
    image_dir: str,
    output_label_dir: str,
    config: Optional[ScreenConfig] = None,
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    visualize: bool = False,
    vis_output_dir: Optional[str] = None,
) -> Dict[str, int]:
    """Process all images in a directory and generate labels.

    Args:
        image_dir: Directory containing images.
        output_label_dir: Directory to save label files.
        config: Screen configuration (auto-detected if None).
        image_extensions: Tuple of valid image extensions.
        visualize: If True, generate visualization images.
        vis_output_dir: Directory for visualization images.

    Returns:
        Statistics dict with counts of processed images and annotations.
    """
    image_dir = Path(image_dir)
    output_label_dir = Path(output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    if visualize and vis_output_dir:
        vis_output_dir = Path(vis_output_dir)
        vis_output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images_processed": 0,
        "total_annotations": 0,
        "class_counts": {name: 0 for name in TOWER_CLASS_IDS.keys()},
    }

    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    for image_file in image_files:
        try:
            annotations = generate_tower_annotations(str(image_file), config)

            # Write label file
            label_file = output_label_dir / (image_file.stem + ".txt")
            write_yolo_label(str(label_file), annotations)

            # Update stats
            stats["images_processed"] += 1
            stats["total_annotations"] += len(annotations)

            for ann in annotations:
                class_name = CLASS_ID_TO_NAME.get(ann[0], "unknown")
                if class_name in stats["class_counts"]:
                    stats["class_counts"][class_name] += 1

            # Generate visualization
            if visualize and vis_output_dir:
                vis_file = vis_output_dir / (image_file.stem + "_annotated.jpg")
                visualize_annotations(
                    str(image_file),
                    annotations=annotations,
                    output_path=str(vis_file),
                )

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    return stats


def create_empty_label(image_path: str, label_dir: str) -> str:
    """Create an empty label file for images with no objects.

    Args:
        image_path: Path to the image file.
        label_dir: Directory to save the label file.

    Returns:
        Path to the created label file.
    """
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem
    label_path = label_dir / (image_name + ".txt")

    # Create empty file
    label_path.touch()

    return str(label_path)


def validate_annotations(
    image_dir: str,
    label_dir: str,
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> Dict[str, List[str]]:
    """Validate that all images have corresponding labels and vice versa.

    Args:
        image_dir: Directory containing images.
        label_dir: Directory containing label files.
        image_extensions: Tuple of valid image extensions.

    Returns:
        Dict with 'missing_labels' and 'orphan_labels' lists.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_stems = {
        f.stem for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    }

    label_stems = {
        f.stem for f in label_dir.iterdir()
        if f.suffix == ".txt"
    }

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    return {
        "missing_labels": sorted(list(missing_labels)),
        "orphan_labels": sorted(list(orphan_labels)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate YOLO annotations for Clash Royale screenshots"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input image or directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for labels"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualization images"
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help="Directory for visualization images"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Process single image
        annotations = generate_tower_annotations(str(input_path))
        label_path = Path(args.output) / (input_path.stem + ".txt")
        write_yolo_label(str(label_path), annotations)
        print(f"Generated {len(annotations)} annotations -> {label_path}")

        if args.visualize:
            vis_dir = args.vis_dir or args.output
            vis_path = Path(vis_dir) / (input_path.stem + "_annotated.jpg")
            visualize_annotations(str(input_path), annotations, output_path=str(vis_path))
            print(f"Visualization saved to {vis_path}")
    else:
        # Process directory
        vis_dir = args.vis_dir if args.visualize else None
        stats = process_directory(
            str(input_path),
            args.output,
            visualize=args.visualize,
            vis_output_dir=vis_dir,
        )
        print(f"Processed {stats['images_processed']} images")
        print(f"Total annotations: {stats['total_annotations']}")
        print(f"Class counts: {stats['class_counts']}")
