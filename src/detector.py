"""
Object Detection Module for Clash Royale
Based on KataCR approach using YOLOv8
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

from config import (
    CONF_THRESHOLD, 
    IOU_THRESHOLD, 
    IMAGE_SIZE,
    UNIT_CATEGORIES,
    COLORS
)


class Detection:
    """Represents a single object detection."""
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],  # (x1, y1, x2, y2)
        confidence: float,
        class_id: int,
        class_name: str,
        belonging: int = 0  # 0: unknown, 1: friendly, 2: enemy
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.belonging = belonging
        
        # Calculate derived properties
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
        self.area = self.width * self.height
    
    def __repr__(self):
        return (
            f"Detection({self.class_name}, "
            f"conf={self.confidence:.2f}, "
            f"bbox=[{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}])"
        )
    
    def to_dict(self) -> dict:
        """Convert detection to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": (self.center_x, self.center_y),
            "belonging": self.belonging
        }


class ClashRoyaleDetector:
    """
    YOLOv8-based object detector for Clash Royale.
    Based on KataCR's detection approach.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: str = "auto"  # "auto", "cpu", "cuda", or "0", "1", etc.
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLOv8 model weights (.pt file).
                       If None, uses a pre-trained YOLOv8n model.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IOU threshold for NMS.
            device: Device to run inference on.
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics library is required")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        if model_path is None or not Path(model_path).exists():
            print("No custom model provided. Using pre-trained YOLOv8n...")
            print("Note: For best results, train a custom model on Clash Royale data.")
            self.model = YOLO("yolov8n.pt")
            self.custom_classes = False
        else:
            print(f"Loading custom model: {model_path}")
            self.model = YOLO(model_path)
            self.custom_classes = True
        
        # Set device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Class mapping
        self.class_names = self.model.names if hasattr(self.model, 'names') else UNIT_CATEGORIES
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for detection.
        
        Args:
            image: BGR image as numpy array.
            
        Returns:
            Preprocessed image.
        """
        # Resize to target size while maintaining aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = IMAGE_SIZE
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        dw = (target_w - new_w) // 2
        dh = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[dh:dh+new_h, dw:dw+new_w] = resized
        
        return padded, (scale, dw, dh, w, h)
    
    def detect(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> List[Detection]:
        """
        Perform object detection on an image.
        
        Args:
            image: BGR image as numpy array.
            preprocess: Whether to preprocess the image.
            
        Returns:
            List of Detection objects.
        """
        # Store original dimensions for coordinate conversion
        orig_h, orig_w = image.shape[:2]
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, classes):
                # Get class name
                if cls_id in self.class_names:
                    cls_name = self.class_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"
                
                # Create detection object
                detection = Detection(
                    bbox=tuple(box),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=cls_name
                )
                
                detections.append(detection)
        
        return detections
    
    def detect_with_tracking(
        self,
        image: np.ndarray,
        tracker: str = "bytetrack"
    ) -> List[Detection]:
        """
        Perform detection with object tracking.
        
        Args:
            image: BGR image as numpy array.
            tracker: Tracking algorithm ("bytetrack", "botsort").
            
        Returns:
            List of Detection objects with track IDs.
        """
        # Run tracking
        results = self.model.track(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            tracker=f"{tracker}.yaml",
            persist=True,
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            # Get track IDs if available
            track_ids = None
            if results.boxes.id is not None:
                track_ids = results.boxes.id.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
                # Get class name
                if cls_id in self.class_names:
                    cls_name = self.class_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"
                
                # Create detection object
                detection = Detection(
                    bbox=tuple(box),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=cls_name
                )
                
                # Add track ID if available
                if track_ids is not None:
                    detection.track_id = int(track_ids[i])
                
                detections.append(detection)
        
        return detections
    
    def get_class_names(self) -> Dict[int, str]:
        """Get the class ID to name mapping."""
        return dict(self.class_names)


class ComboDetector:
    """
    Multi-model detector combining multiple YOLOv8 models.
    Based on KataCR's combo detection approach for handling 
    many classes (150+) by splitting into multiple specialized detectors.
    
    KataCR splits detection by unit size:
    - Detector 1: Small/medium units (skeletons, goblins, archers, etc.)
    - Detector 2: Large units and buildings (giants, towers, golems, etc.)
    
    This approach improves detection accuracy for the wide range of 
    object sizes in Clash Royale.
    """
    
    def __init__(
        self,
        model_paths: List[str],
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: str = "auto",
        verbose: bool = True
    ):
        """
        Initialize combo detector with multiple models.
        
        Args:
            model_paths: List of paths to YOLOv8 model weights.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IOU threshold for NMS.
            device: Device to run inference on.
            verbose: Whether to print loading messages.
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics library is required")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        
        # Set device first
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if verbose:
            print(f"ComboDetector using device: {self.device}")
        
        # Load all models
        self.models = []
        self.model_names = []
        for path in model_paths:
            path = Path(path)
            if path.exists():
                model = YOLO(str(path))
                self.models.append(model)
                self.model_names.append(path.stem)
                if verbose:
                    print(f"  Loaded model: {path.name} ({len(model.names)} classes)")
            else:
                print(f"  Warning: Model not found: {path}")
        
        if not self.models:
            raise ValueError("No valid models loaded. Please check model paths.")
        
        # Build unified class name mapping
        self._build_class_mapping()
        
        # Tracking state
        self._track_history = {}
        self._next_track_id = 1
    
    def _build_class_mapping(self):
        """Build a unified class name mapping from all models."""
        self.class_names = {}
        self.name_to_id = {}
        
        # Collect all unique class names
        current_id = 0
        for model in self.models:
            for model_cls_id, name in model.names.items():
                if name not in self.name_to_id:
                    self.name_to_id[name] = current_id
                    self.class_names[current_id] = name
                    current_id += 1
        
        if self.verbose:
            print(f"  Total unique classes: {len(self.class_names)}")
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run all models and combine detections with NMS.
        
        Args:
            image: BGR image as numpy array.
            
        Returns:
            Combined list of detections.
        """
        import torch
        import torchvision
        
        all_boxes = []
        all_scores = []
        all_class_ids = []
        all_class_names = []
        
        # Run each model
        for model_idx, model in enumerate(self.models):
            results = model.predict(
                image,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )[0]
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu()
                scores = results.boxes.conf.cpu()
                classes = results.boxes.cls.cpu().int()
                
                for box, score, cls_id in zip(boxes, scores, classes):
                    cls_name = model.names.get(cls_id.item(), f"class_{cls_id.item()}")
                    unified_cls_id = self.name_to_id.get(cls_name, cls_id.item())
                    
                    all_boxes.append(box)
                    all_scores.append(score.item())
                    all_class_ids.append(unified_cls_id)
                    all_class_names.append(cls_name)
        
        if not all_boxes:
            return []
        
        # Stack tensors for NMS
        boxes_tensor = torch.stack(all_boxes)
        scores_tensor = torch.tensor(all_scores)
        
        # Apply NMS to remove duplicate detections
        keep_indices = torchvision.ops.nms(
            boxes_tensor,
            scores_tensor,
            self.iou_threshold
        )
        
        # Create detection objects
        detections = []
        for idx in keep_indices:
            idx = idx.item()
            detection = Detection(
                bbox=tuple(all_boxes[idx].numpy()),
                confidence=float(all_scores[idx]),
                class_id=all_class_ids[idx],
                class_name=all_class_names[idx]
            )
            detections.append(detection)
        
        return detections
    
    def detect_with_tracking(
        self,
        image: np.ndarray,
        max_distance: float = 50.0
    ) -> List[Detection]:
        """
        Run detection with simple centroid-based tracking.
        
        Note: For more robust tracking, consider using ByteTrack or BoTSORT
        directly with a single model.
        
        Args:
            image: BGR image as numpy array.
            max_distance: Maximum distance to match detections across frames.
            
        Returns:
            List of detections with track_id attribute.
        """
        detections = self.detect(image)
        
        # Simple centroid tracking
        current_centroids = {}
        for det in detections:
            current_centroids[id(det)] = (det.center_x, det.center_y, det.class_name)
        
        # Match with previous tracks
        matched = set()
        for det in detections:
            det_centroid = (det.center_x, det.center_y)
            best_track_id = None
            best_distance = max_distance
            
            for track_id, (tx, ty, tname) in self._track_history.items():
                if track_id in matched:
                    continue
                if tname != det.class_name:
                    continue
                    
                distance = ((det_centroid[0] - tx)**2 + (det_centroid[1] - ty)**2)**0.5
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                det.track_id = best_track_id
                matched.add(best_track_id)
            else:
                det.track_id = self._next_track_id
                self._next_track_id += 1
        
        # Update track history
        self._track_history = {
            det.track_id: (det.center_x, det.center_y, det.class_name)
            for det in detections
        }
        
        return detections
    
    def get_class_names(self) -> Dict[int, str]:
        """Get the unified class ID to name mapping."""
        return dict(self.class_names)
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "num_models": len(self.models),
            "model_names": self.model_names,
            "total_classes": len(self.class_names),
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold
        }


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    show_confidence: bool = True,
    show_labels: bool = True,
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw detection boxes on an image.
    
    Args:
        image: BGR image as numpy array.
        detections: List of Detection objects.
        show_confidence: Whether to show confidence scores.
        show_labels: Whether to show class labels.
        thickness: Line thickness for boxes.
        font_scale: Font scale for labels.
        
    Returns:
        Image with drawn detections.
    """
    result = image.copy()
    
    # Generate colors for classes
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)
    
    for det in detections:
        # Get color based on class ID
        color = tuple(map(int, colors[det.class_id % len(colors)]))
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        if show_labels:
            label = det.class_name
            if show_confidence:
                label += f" {det.confidence:.2f}"
            
            # Calculate label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Draw label background
            cv2.rectangle(
                result,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                result,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    return result


def filter_detections(
    detections: List[Detection],
    class_names: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    min_area: Optional[float] = None,
    region: Optional[Tuple[int, int, int, int]] = None
) -> List[Detection]:
    """
    Filter detections based on various criteria.
    
    Args:
        detections: List of Detection objects.
        class_names: Only keep detections with these class names.
        min_confidence: Minimum confidence threshold.
        min_area: Minimum bounding box area.
        region: Only keep detections within (x1, y1, x2, y2) region.
        
    Returns:
        Filtered list of detections.
    """
    filtered = detections
    
    if class_names is not None:
        filtered = [d for d in filtered if d.class_name in class_names]
    
    if min_confidence is not None:
        filtered = [d for d in filtered if d.confidence >= min_confidence]
    
    if min_area is not None:
        filtered = [d for d in filtered if d.area >= min_area]
    
    if region is not None:
        rx1, ry1, rx2, ry2 = region
        filtered = [
            d for d in filtered
            if d.center_x >= rx1 and d.center_x <= rx2
            and d.center_y >= ry1 and d.center_y <= ry2
        ]
    
    return filtered


if __name__ == "__main__":
    # Test detection with a sample image
    import sys
    
    print("Clash Royale Object Detection Test")
    print("-" * 40)
    
    # Initialize detector
    detector = ClashRoyaleDetector()
    
    # Test with camera or image
    if len(sys.argv) > 1:
        # Load image from argument
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            sys.exit(1)
        
        # Run detection
        detections = detector.detect(image)
        
        print(f"Found {len(detections)} detections:")
        for det in detections:
            print(f"  {det}")
        
        # Draw and show results
        result = draw_detections(image, detections)
        cv2.imshow("Detections", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Usage: python detector.py <image_path>")
        print("Or run main.py for real-time detection")
