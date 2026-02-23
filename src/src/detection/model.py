"""
YOLOv8 model wrapper for Clash Royale object detection.

Provides training, inference, and evaluation capabilities using ultralytics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    side: int = -1  # 0=ally (bottom), 1=enemy (top), -1=unknown

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[int, float, float, float, float]:
        """Convert to YOLO format (class_id, x_center, y_center, width, height) normalized."""
        x1, y1, x2, y2 = self.bbox
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = self.width / img_width
        height = self.height / img_height
        return (self.class_id, x_center, y_center, width, height)


class CRDetector:
    """Clash Royale object detector using YOLOv8."""

    # Default class names for Phase 1 (tower detection)
    DEFAULT_CLASSES = {
        0: "king_tower_player",
        1: "king_tower_enemy",
        2: "princess_tower_left_player",
        3: "princess_tower_left_enemy",
        4: "princess_tower_right_player",
        5: "princess_tower_right_enemy",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "yolov8n",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes: Optional[Dict[int, str]] = None,
        belonging_model: bool = False,
    ):
        """Initialize the detector.

        Args:
            model_path: Path to trained model weights. If None, uses pretrained model.
            model_type: YOLO model variant (yolov8n, yolov8s, yolov8m, etc.)
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
            confidence_threshold: Minimum confidence for detections.
            iou_threshold: IoU threshold for NMS.
            classes: Dictionary mapping class IDs to names.
            belonging_model: If True, use custom NMS that outputs belonging.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or self.DEFAULT_CLASSES
        self.belonging_model = belonging_model

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Load pretrained model for fine-tuning
            self.model = YOLO(f"{model_type}.pt")

        self.model.to(self.device)

        # Use model's own class names if no explicit mapping was provided
        if classes is None and hasattr(self.model, "names"):
            self.classes = dict(self.model.names)

    def detect(
        self,
        image: Union[str, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Detection]:
        """Run detection on a single image.

        Args:
            image: Image path or numpy array (BGR format).
            conf: Override confidence threshold.
            iou: Override IoU threshold.

        Returns:
            List of Detection objects.
        """
        conf = conf or self.confidence_threshold
        iou = iou or self.iou_threshold

        if self.belonging_model:
            return self._detect_with_belonging(image, conf, iou)

        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf_val, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.classes.get(class_id, f"class_{class_id}")

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(conf_val),
                    bbox=(x1, y1, x2, y2),
                ))

        return detections

    def _detect_with_belonging(
        self,
        image: Union[str, np.ndarray],
        conf: float,
        iou: float,
    ) -> List[Detection]:
        """Run detection with belonging-aware custom NMS.

        Uses CRDetectionPredictor for 7-column output:
        (x1, y1, x2, y2, conf, cls, belonging)
        """
        from src.yolov8_custom.custom_utils import non_max_suppression

        # Run raw inference
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            # For belonging models, boxes.data has 7 columns
            data = result.boxes.data.cpu().numpy()
            for row in data:
                if len(row) >= 7:
                    x1, y1, x2, y2 = map(int, row[:4])
                    conf_val = float(row[4])
                    class_id = int(row[5])
                    belonging = int(row[6])
                else:
                    x1, y1, x2, y2 = map(int, row[:4])
                    conf_val = float(row[4])
                    class_id = int(row[5])
                    belonging = -1

                class_name = self.classes.get(class_id, f"class_{class_id}")
                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf_val,
                    bbox=(x1, y1, x2, y2),
                    side=belonging,
                ))

        return detections

    def detect_batch(
        self,
        images: List[Union[str, np.ndarray]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[List[Detection]]:
        """Run detection on multiple images.

        Args:
            images: List of image paths or numpy arrays.
            conf: Override confidence threshold.
            iou: Override IoU threshold.

        Returns:
            List of detection lists (one per image).
        """
        conf = conf or self.confidence_threshold
        iou = iou or self.iou_threshold

        results = self.model.predict(
            source=images,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        all_detections = []
        for result in results:
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf_val, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.classes.get(class_id, f"class_{class_id}")

                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf_val),
                        bbox=(x1, y1, x2, y2),
                    ))

            all_detections.append(detections)

        return all_detections

    def visualize(
        self,
        image: Union[str, np.ndarray],
        detections: Optional[List[Detection]] = None,
        output_path: Optional[str] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw detections on an image.

        Args:
            image: Image path or numpy array.
            detections: Detections to draw. If None, runs detection first.
            output_path: Path to save output image.
            show_labels: Whether to show class names.
            show_confidence: Whether to show confidence scores.

        Returns:
            Image with drawn detections.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        if detections is None:
            detections = self.detect(img)

        # Color palette (BGR)
        colors = [
            (0, 255, 0),    # Green - player towers
            (0, 0, 255),    # Red - enemy towers
            (0, 200, 100),  # Light green
            (0, 100, 200),  # Orange-red
            (100, 255, 0),  # Yellow-green
            (0, 50, 255),   # Dark red
        ]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors[det.class_id % len(colors)]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(det.class_name)
                if show_confidence:
                    label_parts.append(f"{det.confidence:.2f}")
                label = " ".join(label_parts)

                # Background for text
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img,
                    (x1, y1 - text_height - 8),
                    (x1 + text_width + 4, y1),
                    color,
                    -1
                )
                cv2.putText(
                    img, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )

        if output_path:
            cv2.imwrite(output_path, img)

        return img

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        project: str = "runs/train",
        name: str = "cr_detector",
        resume: bool = False,
        **kwargs,
    ) -> str:
        """Train the model on custom data.

        Args:
            data_yaml: Path to data configuration YAML file.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            imgsz: Input image size.
            project: Project directory for saving results.
            name: Experiment name.
            resume: Whether to resume training from last checkpoint.
            **kwargs: Additional training arguments.

        Returns:
            Path to best model weights.
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=project,
            name=name,
            resume=resume,
            device=self.device,
            **kwargs,
        )

        # Return path to best weights
        return str(Path(project) / name / "weights" / "best.pt")

    def evaluate(
        self,
        data_yaml: str,
        batch_size: int = 16,
        imgsz: int = 640,
        split: str = "val",
    ) -> Dict:
        """Evaluate model on validation/test set.

        Args:
            data_yaml: Path to data configuration YAML file.
            batch_size: Batch size for evaluation.
            imgsz: Input image size.
            split: Dataset split to evaluate on ('val' or 'test').

        Returns:
            Dictionary with evaluation metrics.
        """
        results = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=imgsz,
            split=split,
            device=self.device,
        )

        return {
            "map50": results.box.map50,
            "map75": results.box.map75,
            "map": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 640,
        output_path: Optional[str] = None,
    ) -> str:
        """Export model to different formats.

        Args:
            format: Export format (onnx, torchscript, engine, etc.).
            imgsz: Input image size.
            output_path: Optional output path.

        Returns:
            Path to exported model.
        """
        return self.model.export(format=format, imgsz=imgsz)


def load_detector(
    model_path: str,
    device: Optional[str] = None,
    confidence_threshold: float = 0.5,
) -> CRDetector:
    """Load a trained detector.

    Args:
        model_path: Path to trained model weights.
        device: Device to run on ('cuda', 'cpu', or None for auto).
        confidence_threshold: Minimum confidence for detections.

    Returns:
        CRDetector instance.
    """
    return CRDetector(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )
