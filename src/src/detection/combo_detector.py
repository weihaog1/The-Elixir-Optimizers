"""Dual-detector combo inference for Clash Royale.

Runs two YOLOv8m models (small-sprite D1 + large-sprite D2) in parallel,
remaps local class indices to the global 155-class space, and deduplicates
shared base classes via cross-detector NMS.

Each model outputs belonging (ally=0 / enemy=1) as an extra channel,
eliminating the Y-position heuristic used by the single-model pipeline.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from ..yolov8_custom.custom_utils import non_max_suppression


class ComboDetector:
    """Dual YOLOv8m detector with belonging prediction.

    Loads two models split by sprite size, runs them in parallel,
    remaps to global class indices, and merges with deduplication.

    Output: (N, 7) ndarray = [x1, y1, x2, y2, confidence, global_class_id, belonging]
    """

    def __init__(
        self,
        model_paths: List[str],
        split_config_path: str,
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 960,
    ):
        assert len(model_paths) == 2, "ComboDetector requires exactly 2 model paths"

        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        # Load split config
        with open(split_config_path) as f:
            self.split_config = json.load(f)

        # Build local-to-global remapping for each detector
        self._d1_local_to_global = {
            int(k): int(v)
            for k, v in self.split_config["detector1"]["local_to_global"].items()
        }
        self._d2_local_to_global = {
            int(k): int(v)
            for k, v in self.split_config["detector2"]["local_to_global"].items()
        }

        self._d1_nc = self.split_config["detector1"]["nc"]
        self._d2_nc = self.split_config["detector2"]["nc"]

        self._base_indices = set(self.split_config["base_indices"])

        # Build global class name map from label_list
        from ..generation.label_list import idx2unit
        self.names: Dict[int, str] = dict(idx2unit)

        # Load models
        self._models: List[YOLO] = []
        for path in model_paths:
            model = YOLO(path)
            model.to(self.device)
            self._models.append(model)

        self._letterbox = LetterBox(self.imgsz, auto=True, stride=32)
        self._executor = ThreadPoolExecutor(max_workers=2)

    def warmup(self):
        """Run a dummy inference to warm up both models."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for model in self._models:
            model.predict(source=dummy, device=self.device, verbose=False, imgsz=self.imgsz)

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Letterbox and normalize a frame for inference."""
        img = self._letterbox(image=frame)
        img = img.transpose(2, 0, 1)[::-1]  # HWC->CHW, BGR->RGB
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).float().to(self.device) / 255.0
        return tensor.unsqueeze(0)

    def _run_single(
        self,
        model: YOLO,
        tensor: torch.Tensor,
        nc: int,
        local_to_global: Dict[int, int],
        orig_shape: Tuple[int, int],
        img_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Run one detector and return remapped (N, 7) detections."""
        with torch.no_grad():
            raw = model.model(tensor)

        # Custom NMS with belonging
        preds = non_max_suppression(
            raw, conf_thres=self.conf, iou_thres=self.iou, nc=nc,
        )

        pred = preds[0]  # single image
        if len(pred) == 0:
            return np.zeros((0, 7), dtype=np.float32)

        # Scale boxes from letterboxed coords to original frame coords
        from ultralytics.utils import ops
        pred[:, :4] = ops.scale_boxes(img_shape, pred[:, :4], orig_shape)

        data = pred.cpu().numpy()

        # Remap local class indices to global
        remapped = []
        n_real_d1 = self.split_config["detector1"].get("n_real_classes", nc)
        n_real_d2 = self.split_config["detector2"].get("n_real_classes", nc)
        n_real = n_real_d1 if local_to_global is self._d1_local_to_global else n_real_d2

        for row in data:
            local_cls = int(row[5])
            # Skip padding classes and belonging class
            if local_cls >= n_real:
                continue
            global_cls = local_to_global.get(local_cls)
            if global_cls is None:
                continue
            remapped.append([
                row[0], row[1], row[2], row[3],  # x1, y1, x2, y2
                row[4],                            # confidence
                float(global_cls),                  # global class id
                row[6],                             # belonging
            ])

        if not remapped:
            return np.zeros((0, 7), dtype=np.float32)
        return np.array(remapped, dtype=np.float32)

    def _cross_detector_nms(
        self,
        dets1: np.ndarray,
        dets2: np.ndarray,
        iou_thresh: float = 0.45,
    ) -> np.ndarray:
        """Merge detections from both detectors, deduplicating base classes.

        For base classes (shared between D1 and D2), keeps the higher-confidence
        detection when IoU exceeds threshold. Non-base classes pass through as-is.
        """
        if len(dets1) == 0 and len(dets2) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        if len(dets1) == 0:
            return dets2
        if len(dets2) == 0:
            return dets1

        # Separate base vs exclusive detections
        d1_base_mask = np.array([int(d[5]) in self._base_indices for d in dets1])
        d2_base_mask = np.array([int(d[5]) in self._base_indices for d in dets2])

        d1_exclusive = dets1[~d1_base_mask] if d1_base_mask.any() else dets1
        d2_exclusive = dets2[~d2_base_mask] if d2_base_mask.any() else dets2

        d1_base = dets1[d1_base_mask] if d1_base_mask.any() else np.zeros((0, 7), dtype=np.float32)
        d2_base = dets2[d2_base_mask] if d2_base_mask.any() else np.zeros((0, 7), dtype=np.float32)

        # NMS on base classes across detectors
        if len(d1_base) > 0 and len(d2_base) > 0:
            all_base = np.vstack([d1_base, d2_base])
            # Class-aware NMS: only suppress same-class detections
            keep = self._numpy_class_nms(all_base, iou_thresh)
            merged_base = all_base[keep]
        elif len(d1_base) > 0:
            merged_base = d1_base
        elif len(d2_base) > 0:
            merged_base = d2_base
        else:
            merged_base = np.zeros((0, 7), dtype=np.float32)

        # Combine all
        parts = [p for p in [d1_exclusive, d2_exclusive, merged_base] if len(p) > 0]
        if not parts:
            return np.zeros((0, 7), dtype=np.float32)

        merged = np.vstack(parts)
        # Sort by confidence descending
        merged = merged[merged[:, 4].argsort()[::-1]]
        return merged

    @staticmethod
    def _numpy_class_nms(dets: np.ndarray, iou_thresh: float) -> List[int]:
        """Class-aware greedy NMS on (N, 7) detections."""
        if len(dets) == 0:
            return []

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        classes = dets[:, 5].astype(int)

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            rest = order[1:]

            # Only suppress same-class detections
            same_class = classes[rest] == classes[i]

            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-6)

            # Suppress if same class AND high IoU
            suppress = same_class & (iou > iou_thresh)
            order = rest[~suppress]

        return keep

    def infer(
        self,
        frame: np.ndarray,
        arena_cutoff: int = 1550,
    ) -> np.ndarray:
        """Run dual-detector inference on a frame.

        Args:
            frame: BGR image (1080x1920 expected).
            arena_cutoff: Y-coordinate below which to crop (UI region).

        Returns:
            (N, 7) ndarray: [x1, y1, x2, y2, confidence, global_class_id, belonging]
            Coordinates are in the original frame space.
        """
        # Crop to arena region
        arena = frame[:arena_cutoff]
        orig_shape = arena.shape[:2]  # (H, W)

        # Preprocess once, share between both detectors
        tensor = self._preprocess(arena)
        img_shape = tensor.shape[2:]  # (H, W) after letterbox

        # Run both detectors in parallel
        future1 = self._executor.submit(
            self._run_single,
            self._models[0], tensor, self._d1_nc,
            self._d1_local_to_global, orig_shape, img_shape,
        )
        future2 = self._executor.submit(
            self._run_single,
            self._models[1], tensor, self._d2_nc,
            self._d2_local_to_global, orig_shape, img_shape,
        )

        dets1 = future1.result()
        dets2 = future2.result()

        # Cross-detector deduplication
        merged = self._cross_detector_nms(dets1, dets2, self.iou)

        return merged

    def detect_to_list(
        self,
        frame: np.ndarray,
        arena_cutoff: int = 1550,
    ):
        """Run inference and return Detection objects (for StateBuilder compatibility).

        Returns:
            List of Detection objects with belonging populated from model output.
        """
        from .model import Detection

        raw = self.infer(frame, arena_cutoff=arena_cutoff)
        detections = []
        for row in raw:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            conf = float(row[4])
            cls_id = int(row[5])
            belonging = int(row[6])
            class_name = self.names.get(cls_id, f"class_{cls_id}")
            detections.append(Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                side=belonging,
            ))
        return detections
