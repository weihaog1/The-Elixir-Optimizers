# src/detection/ - YOLOv8 Model Wrapper

Wraps ultralytics YOLOv8 for Clash Royale object detection. Used for inference and evaluation. Training is now handled by `scripts/train_synthetic.py` (which uses the generation pipeline directly).

## Files

**model.py** (448 lines) - Core module. Actively used.
- `Detection` dataclass: class_id, class_name, confidence, bbox (x1,y1,x2,y2), side (0=ally, 1=enemy, default -1). Properties: center, width, height. Methods: to_yolo_format().
- `CRDetector`: Wraps ultralytics YOLO. Key methods:
  - `detect(image, conf, iou)` -> list of Detection (6-column, side always -1)
  - `detect_batch(images)` -> list of list of Detection
  - `visualize(image, detections)` -> annotated image
  - `evaluate(data_yaml, imgsz)` -> mAP metrics dict
  - `train(data, epochs, ...)` -- legacy, use train_synthetic.py instead
  - `_detect_with_belonging()` -- exists for 7-column output, but unused (model not trained for it)
- `belonging_model=False` flag: If True, uses custom NMS from `src/yolov8_custom/custom_utils.py` for 7-column output (x1,y1,x2,y2,conf,cls,belonging). Current model was NOT trained with belonging labels, so this is False.
- Default thresholds: confidence=0.5, IoU=0.45.
- Note: F1-optimal conf for v12 model is 0.765 (conservative). For gameplay where recall matters more than precision, use conf=0.3-0.4.

**inference.py** (244 lines) - CLI inference tool. Actively used.
- `python -m src.detection.inference --model best.pt --source image.jpg --conf 0.25`
- Saves annotated images, YOLO-format labels, and JSON detection results.
- Supports single image or directory batch processing.

**train.py** (166 lines) - Legacy training script.
- Superseded by `scripts/train_synthetic.py` for synthetic data training.
- Was used for the original YOLOv8n training on KataCR's val set.
- Kept for reference. Do not use for new training runs.

## Usage

```python
from src.detection.model import CRDetector, Detection

detector = CRDetector("models/best_yolov8s_50epochs_fixed_pregen_set.pt")
detections = detector.detect(image, conf=0.35)  # lower conf for better recall
annotated = detector.visualize(image, detections)

for det in detections:
    print(f"{det.class_name} ({det.confidence:.2f}) at {det.center}")
    print(f"  Side: {'ally' if det.side == 0 else 'enemy'}")
```

## Belonging Output (not yet active)

`src/yolov8_custom/custom_utils.py` (217 lines) contains a custom NMS function ported from KataCR that outputs 7-column detections: (x1, y1, x2, y2, conf, cls, belonging). The belonging column is a binary classification head (0=player, 1=enemy) trained alongside detection. To use it:

1. Retrain YOLOv8s with 12-column labels (class, x, y, w, h, belonging, state1-6) from KataCR's dataset
2. Set `belonging_model=True` on CRDetector
3. `_detect_with_belonging()` will use the custom NMS and populate `Detection.side` from the model output

Until retraining, StateBuilder uses a Y-position heuristic for belonging (fails when troops cross the river).

## v12 Model Performance Notes

- Best model: `models/best_yolov8s_50epochs_fixed_pregen_set.pt` (22MB)
- Inference: 15.3 FPS on M1 Pro MPS at imgsz=960
- Towers/structures detected at 0.90-0.97 confidence (reliable)
- Common troops at 0.60-0.85 confidence (good)
- Small troops (skeleton, spirits) at 0.40-0.65 (weaker)
- Zero "the-log" hallucinations (was the main FP problem with old model)
- Card icons in hand tray sometimes detected as troops (minor inflation)
