"""Dual-model inference merger for the two-detector YOLOv8m system.

Follows KataCR's combo_detect.py pattern:
1. Load both detector models + split_config for index remapping
2. Forward pass each model with LetterBox(auto=True) preprocessing
3. Apply per-detector custom NMS (belonging-aware)
4. Remap local class indices back to global 155-class space
5. Concatenate detections from both models
6. Cross-detector NMS to deduplicate base classes (shared between detectors)
7. Return 7-column detections: x1, y1, x2, y2, conf, global_cls, belonging
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes

from src.yolov8_custom.custom_utils import non_max_suppression


class ComboDetector:
    """Dual YOLOv8m detector that merges results from small-sprite and large-sprite models.

    Args:
        model_paths: Tuple/list of (detector1_path, detector2_path).
        split_config_path: Path to split_config.json with index remapping.
        conf: Confidence threshold for per-detector NMS.
        iou: IoU threshold for per-detector NMS.
        iou_merge: IoU threshold for cross-detector deduplication of base classes.
        device: Inference device ("mps", "cuda", "cpu").
        imgsz: Inference image size.
    """

    def __init__(self, model_paths, split_config_path, conf=0.25, iou=0.45,
                 iou_merge=0.6, device="mps", imgsz=960):
        self.conf = conf
        self.iou = iou
        self.iou_merge = iou_merge
        self.imgsz = imgsz

        # Load split config
        with open(split_config_path) as f:
            self.split_config = json.load(f)

        self.base_global_indices = set(self.split_config["base_indices"])

        # Build local->global remapping for each detector
        self.local_to_global = {}
        self.nc = {}
        for det_key in ["detector1", "detector2"]:
            det = self.split_config[det_key]
            # local_to_global maps str(local_idx) -> global_idx
            l2g = det["local_to_global"]
            self.local_to_global[det_key] = {int(k): v for k, v in l2g.items()}
            self.nc[det_key] = det["nc"]

        # Load models and move to target device
        self.models = {}
        self.device = device
        for det_key, path in zip(["detector1", "detector2"], model_paths):
            model = YOLO(path)
            model.model.to(device)
            self.models[det_key] = model

        # LetterBox with auto=True (KataCR approach - pads to stride multiple)
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=32)

        # Build global names dict from label_list for display
        from src.generation.label_list import idx2unit
        self.names = dict(idx2unit)

    def warmup(self):
        """Run dummy inference to warm up both models."""
        dummy = np.zeros((480, 270, 3), dtype=np.uint8)
        for det_key, model in self.models.items():
            model.predict(dummy, verbose=False, imgsz=self.imgsz)

    def _forward_one(self, det_key, frame):
        """Run one detector on a frame, return 7-column dets in LOCAL indices.

        Returns:
            numpy array of shape (N, 7): x1,y1,x2,y2,conf,local_cls,belonging
            in original frame coordinates.
        """
        model = self.models[det_key]
        nc = self.nc[det_key]

        # Preprocess: letterbox with auto=True
        img_lb = self.letterbox(image=frame)
        img_lb = img_lb.transpose(2, 0, 1)[::-1].copy()  # HWC->CHW, BGR->RGB
        img_t = torch.from_numpy(img_lb).unsqueeze(0).float() / 255.0
        img_t = img_t.to(next(model.model.parameters()).device)

        # Forward pass
        with torch.no_grad():
            preds = model.model(img_t)

        # Custom NMS with belonging
        dets = non_max_suppression(
            preds, conf_thres=self.conf, iou_thres=self.iou, nc=nc
        )[0]

        if len(dets) > 0:
            # Scale boxes back to original frame coordinates
            dets[:, :4] = scale_boxes(img_t.shape[2:], dets[:, :4], frame.shape[:2])
            return dets.cpu().numpy()
        return np.zeros((0, 7))

    def _remap_to_global(self, dets, det_key):
        """Remap local class indices to global 155-class indices.

        Args:
            dets: (N, 7) array with local class indices in column 5.
            det_key: "detector1" or "detector2".

        Returns:
            (N, 7) array with global class indices.
        """
        if len(dets) == 0:
            return dets

        l2g = self.local_to_global[det_key]
        result = dets.copy()
        keep = []
        for i in range(len(result)):
            local_cls = int(result[i, 5])
            if local_cls in l2g:
                result[i, 5] = l2g[local_cls]
                keep.append(i)
            # Drop padding classes that have no global mapping
        return result[keep] if keep else np.zeros((0, 7))

    def _cross_detector_nms(self, all_dets):
        """Deduplicate base class detections that appear in both detectors.

        For non-base classes (unique to one detector), all detections are kept.
        For base classes (shared), apply NMS at iou_merge threshold to remove
        duplicate detections of the same object from both detectors.

        Args:
            all_dets: (N, 7) concatenated detections from both detectors.

        Returns:
            (M, 7) deduplicated detections.
        """
        if len(all_dets) == 0:
            return all_dets

        # Separate base and non-base detections
        is_base = np.array([int(d[5]) in self.base_global_indices for d in all_dets])
        non_base = all_dets[~is_base]
        base = all_dets[is_base]

        if len(base) <= 1:
            return all_dets

        # Class-aware NMS on base class detections: offset boxes by class ID
        # so overlapping detections of different classes are preserved
        import torchvision
        boxes_t = torch.from_numpy(base[:, :4]).float()
        scores_t = torch.from_numpy(base[:, 4]).float()
        cls_offset = torch.from_numpy(base[:, 5]).float() * 4096
        boxes_offset = boxes_t.clone()
        boxes_offset[:, 0] += cls_offset
        boxes_offset[:, 1] += cls_offset
        boxes_offset[:, 2] += cls_offset
        boxes_offset[:, 3] += cls_offset
        keep = torchvision.ops.nms(boxes_offset, scores_t, self.iou_merge)
        base_kept = base[keep.numpy()]

        if len(non_base) == 0:
            return base_kept
        return np.vstack([non_base, base_kept])

    def infer(self, frame, arena_cutoff=1550):
        """Run dual-model inference on a frame.

        Args:
            frame: BGR numpy array (full resolution, e.g. 1080x1920).
            arena_cutoff: Y coordinate to crop arena (exclude UI below).

        Returns:
            (N, 7) numpy array: x1, y1, x2, y2, conf, global_cls, belonging.
            Coordinates are in the arena_frame coordinate space (y offset not added).
        """
        arena_frame = frame[:arena_cutoff, :]

        # Forward pass both detectors in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut1 = pool.submit(self._forward_one, "detector1", arena_frame)
            fut2 = pool.submit(self._forward_one, "detector2", arena_frame)
            dets1 = fut1.result()
            dets2 = fut2.result()

        # Remap to global indices
        dets1 = self._remap_to_global(dets1, "detector1")
        dets2 = self._remap_to_global(dets2, "detector2")

        # Concatenate
        if len(dets1) == 0 and len(dets2) == 0:
            return np.zeros((0, 7))
        elif len(dets1) == 0:
            all_dets = dets2
        elif len(dets2) == 0:
            all_dets = dets1
        else:
            all_dets = np.vstack([dets1, dets2])

        # Cross-detector NMS for base classes
        all_dets = self._cross_detector_nms(all_dets)

        # Sort by confidence (highest first)
        if len(all_dets) > 0:
            all_dets = all_dets[all_dets[:, 4].argsort()[::-1]]

        return all_dets
