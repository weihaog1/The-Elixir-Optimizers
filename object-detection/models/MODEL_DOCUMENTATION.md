# Clash Royale Object Detection Model

## Model Card: cr_detector_yolov8n_20260206_235726

### Overview

YOLOv8 nano object detection model trained to detect 155 classes of Clash Royale
game entities from screenshots. Designed as the perception module for a real-time
RL agent that plays Clash Royale.

---

### Model Specifications

| Spec | Value |
|------|-------|
| Architecture | YOLOv8n (nano) |
| Parameters | 3,357,389 (3.4M) |
| Layers | 73 (fused) |
| GFLOPs | 9.7 |
| Input size | 640x640 (letterbox-resized) |
| Output shape | (1, 159, 8400) |
| Weight file | best.pt (6.7 MB) |
| Framework | Ultralytics 8.4.12, PyTorch 2.10.0 |
| Training precision | AMP (mixed FP16/FP32) |

### Performance

| Metric | Value |
|--------|-------|
| mAP50 | 0.944 |
| mAP50-95 | 0.722 |
| Precision | 0.929 |
| Recall | 0.902 |
| Inference (RTX 5070 Ti) | 0.4ms/image |
| Inference (M1 Pro MPS) | ~5-15ms/image |
| Inference (M1 Pro CPU) | ~20-40ms/image |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Batch size | 32 |
| Image size | 640 |
| Optimizer | MuSGD (lr=0.01, momentum=0.9) |
| Learning rate | 0.01 (initial), 0.01 (final ratio) |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| Early stopping patience | 30 |
| Close mosaic | 10 (last 10 epochs) |
| GPU | NVIDIA GeForce RTX 5070 Ti (16GB VRAM) |
| Training time | ~2.5 hours |

### Augmentation

| Augmentation | Value | Notes |
|-------------|-------|-------|
| Mosaic | 1.0 | Full mosaic augmentation |
| Mixup | 0.1 | Light mixup |
| HSV hue | 0.015 | Slight color shift |
| HSV saturation | 0.7 | Moderate saturation jitter |
| HSV value | 0.4 | Moderate brightness jitter |
| Horizontal flip | 0.5 | OK for game screenshots |
| Vertical flip | 0.0 | Disabled (game is directional) |
| Scale | 0.5 | Random scale augmentation |
| Translate | 0.1 | Slight translation |
| Erasing | 0.4 | Random erasing |
| Auto augment | randaugment | Random augmentation policy |

---

### Dataset

| Item | Value |
|------|-------|
| Source | KataCR Clash-Royale-Detection-Dataset |
| Total images | 6,939 |
| Training images | 5,551 (80%) |
| Validation images | 1,388 (20%) |
| Total annotations | ~93,000+ bounding boxes |
| Classes | 155 (reduced from 201, padding classes removed) |
| Label format | YOLO (class_id x_center y_center width height, normalized) |
| Image resolution | ~568x896 (varies slightly, KataCR Clash Royale screenshots) |
| Split seed | 42 |

### What This Model Detects

Arena entities only. Does NOT detect cards in hand or elixir bar (those require
separate OCR/classifier pipeline).

**Categories:**
- Towers: king-tower, queen-tower, cannoneer-tower, dagger-duchess-tower
- Tower UI: tower-bar (HP bars), king-tower-bar, bar, bar-level
- Game UI: clock, emote, text, elixir (indicator), selected
- Troops: 80+ unit types (skeletons, goblins, archers, knights, etc.)
- Spells: zap, fireball, arrows, rocket, lightning, freeze, poison, rage, etc.
- Buildings: cannon, tesla, inferno-tower, bomb-tower, x-bow, mortar, etc.
- Evolution variants: skeleton-evolution, knight-evolution, archer-evolution, etc.

### Per-Class Performance (Key Classes)

#### Towers and UI (Near Perfect)
| Class | mAP50 | mAP50-95 | Samples |
|-------|-------|----------|---------|
| king-tower | 0.995 | 0.990 | 2,773 |
| queen-tower | 0.995 | 0.986 | 5,228 |
| cannoneer-tower | 0.994 | 0.892 | 135 |
| dagger-duchess-tower | 0.995 | 0.954 | 62 |
| tower-bar | 0.995 | 0.971 | 4,808 |
| clock | 0.989 | 0.842 | 603 |
| text | 0.979 | 0.844 | 404 |

#### Common Troops (Strong)
| Class | mAP50 | mAP50-95 | Samples |
|-------|-------|----------|---------|
| musketeer | 0.978 | 0.696 | 496 |
| ice-golem | 0.982 | 0.768 | 361 |
| cannon | 0.995 | 0.959 | 331 |
| hog-rider | 0.962 | 0.655 | 240 |
| barbarian | 0.914 | 0.558 | 207 |
| skeleton | 0.955 | 0.570 | 692 |
| knight | 0.974 | 0.756 | 29 |
| goblin | 0.934 | 0.506 | 52 |

#### Heavy Units (Strong)
| Class | mAP50 | mAP50-95 | Samples |
|-------|-------|----------|---------|
| pekka | 0.950 | 0.782 | 68 |
| mega-knight | 0.977 | 0.741 | 19 |
| golem | 0.986 | 0.771 | 36 |
| lava-hound | 0.995 | 0.799 | 15 |
| giant | 0.995 | 0.825 | 21 |
| electro-giant | 0.995 | 0.797 | 9 |
| royal-giant | 0.995 | 0.847 | 9 |

#### Spells (Variable)
| Class | mAP50 | mAP50-95 | Samples |
|-------|-------|----------|---------|
| rage | 0.960 | 0.945 | 50 |
| poison | 0.995 | 0.978 | 38 |
| freeze | 0.995 | 0.926 | 4 |
| tornado | 0.995 | 0.951 | 6 |
| fireball | 0.797 | 0.521 | 54 |
| arrows | 0.754 | 0.366 | 11 |
| zap | 0.777 | 0.745 | 4 |

#### Weaker Classes (Low Samples)
| Class | mAP50 | mAP50-95 | Samples |
|-------|-------|----------|---------|
| barbarian-barrel | 0.558 | 0.384 | 15 |
| skeleton-king-bar | 0.968 | 0.419 | 10 |
| skeleton-evolution | 0.803 | 0.414 | 29 |
| arrows | 0.754 | 0.366 | 11 |

---

### Class List (155 Classes)

```
  0: king-tower            1: queen-tower           2: cannoneer-tower
  3: dagger-duchess-tower  4: dagger-duchess-bar    5: tower-bar
  6: king-tower-bar        7: bar                   8: bar-level
  9: clock                10: emote                 11: text
 12: elixir               13: selected              14: skeleton-king-bar
 15: skeleton             16: skeleton-evolution    17: electro-spirit
 18: fire-spirit          19: ice-spirit            20: heal-spirit
 21: goblin               22: spear-goblin          23: bomber
 24: bat                  25: bat-evolution          26: zap
 27: giant-snowball       28: ice-golem             29: barbarian-barrel
 30: barbarian            31: barbarian-evolution   32: wall-breaker
 33: rage                 34: the-log               35: archer
 36: arrows               37: knight                38: knight-evolution
 39: minion               40: cannon                41: skeleton-barrel
 42: firecracker          43: firecracker-evolution 44: royal-delivery
 45: royal-recruit        46: royal-recruit-evol    47: tombstone
 48: mega-minion          49: dart-goblin           50: earthquake
 51: elixir-golem-big     52: elixir-golem-mid     53: elixir-golem-small
 54: goblin-barrel        55: guard                 56: clone
 57: tornado              58: miner                 59: dirt
 60: princess             61: ice-wizard            62: royal-ghost
 63: bandit               64: fisherman             65: skeleton-dragon
 66: mortar               67: mortar-evolution      68: tesla
 69: fireball             70: mini-pekka            71: musketeer
 72: goblin-cage          73: goblin-brawler        74: valkyrie
 75: battle-ram           76: battle-ram-evolution  77: bomb-tower
 78: bomb                 79: flying-machine        80: hog-rider
 81: battle-healer        82: furnace               83: zappy
 84: baby-dragon          85: dark-prince           86: freeze
 87: poison               88: hunter                89: goblin-drill
 90: electro-wizard       91: inferno-dragon        92: phoenix-big
 93: phoenix-egg          94: phoenix-small         95: magic-archer
 96: lumberjack           97: night-witch           98: mother-witch
 99: hog                 100: golden-knight        101: skeleton-king
102: mighty-miner        103: rascal-boy           104: rascal-girl
105: giant               106: goblin-hut           107: inferno-tower
108: wizard              109: royal-hog            110: witch
111: balloon             112: prince               113: electro-dragon
114: bowler              115: executioner          116: axe
117: cannon-cart         118: ram-rider            119: graveyard
120: archer-queen        121: monk                 122: royal-giant
123: royal-giant-evol    124: elite-barbarian      125: rocket
126: barbarian-hut       127: elixir-collector     128: giant-skeleton
129: lightning           130: goblin-giant         131: x-bow
132: sparky              133: pekka                134: electro-giant
135: mega-knight         136: lava-hound           137: lava-pup
138: golem               139: golemite             140: little-prince
141: royal-guardian      142: archer-evolution     143: ice-spirit-evolution
144: valkyrie-evolution  145: bomber-evolution     146: wall-breaker-evol
147: evolution-symbol    148: mirror               149: tesla-evolution
150: goblin-ball         151: skeleton-king-skill  152: tesla-evol-shock
153: ice-spirit-evol-sym 154: zap-evolution
```

---

### Usage

#### Basic Inference (Python)

```python
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Run on image (auto-detects device: cuda > mps > cpu)
results = model.predict("screenshot.png", conf=0.5, iou=0.45)

# Parse detections
for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"{class_name}: {confidence:.2f} at [{x1},{y1},{x2},{y2}]")
```

#### Force Specific Device

```python
# Apple M1/M2/M3 (Metal Performance Shaders)
results = model.predict("screenshot.png", conf=0.5, device="mps")

# NVIDIA GPU
results = model.predict("screenshot.png", conf=0.5, device="cuda")

# CPU
results = model.predict("screenshot.png", conf=0.5, device="cpu")
```

#### Real-Time Game Loop

```python
from ultralytics import YOLO
import time

model = YOLO("best.pt")

while True:
    screenshot = capture_screen()  # your screen capture function
    results = model.predict(
        screenshot,
        conf=0.5,
        iou=0.45,
        device="mps",
        verbose=False,
        imgsz=640,
    )

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": list(map(int, box.xyxy[0])),
            })

    # Feed detections to game state builder / RL agent
    process_detections(detections)
    time.sleep(0.1)  # 10 FPS is sufficient for CR
```

#### Export to Other Formats (on Mac)

```python
from ultralytics import YOLO
model = YOLO("best.pt")

# CoreML (optimized for Apple Silicon)
model.export(format="coreml", imgsz=640)

# ONNX (cross-platform)
model.export(format="onnx", imgsz=640, simplify=True)

# TorchScript
model.export(format="torchscript", imgsz=640)
```

#### Using the Project's CRDetector Wrapper

```python
import sys
sys.path.insert(0, "path/to/cr-object-detection")

from src.detection.model import CRDetector, load_detector

detector = load_detector("best.pt", device="mps", confidence_threshold=0.5)
detections = detector.detect("screenshot.png")

for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f} at {det.bbox}")
    print(f"  Center: {det.center}, Size: {det.width}x{det.height}")
```

---

### Limitations

1. **Arena entities only** — Does not detect cards in hand or elixir bar. Those
   require a separate card classifier (ResNet) and OCR pipeline (PaddleOCR),
   both of which are scaffolded in `src/pipeline/` and `src/ocr/`.

2. **KataCR data domain** — Trained on KataCR screenshots (~568x896). Performance
   on different resolutions (e.g., 1080x1920) is good due to YOLO letterboxing,
   but visual differences from different devices/game versions may reduce accuracy.

3. **Rare classes** — Classes with few training samples (<20) have lower recall
   (e.g., barbarian-barrel 0.558 mAP50, arrows 0.754). These are uncommon in
   gameplay so impact is minimal.

4. **Transient spell effects** — Fast-moving spell animations (arrows, zap) are
   harder to detect consistently compared to persistent units.

5. **No player/enemy distinction** — The model detects entity type but does NOT
   classify whether a unit belongs to the player or enemy. KataCR's extended
   format had a "belonging" column (0=player, 1=enemy) that was stripped during
   data conversion. Belonging must be inferred by vertical position on screen.

### Files in This Directory

| File | Description |
|------|-------------|
| weights/best.pt | Best model weights (6.7 MB) — use this |
| weights/last.pt | Final epoch weights (6.7 MB) |
| results.csv | Per-epoch metrics (loss, mAP, precision, recall) |
| results.png | Training curves plot |
| args.yaml | Full training configuration |
| confusion_matrix.png | Class confusion matrix |
| confusion_matrix_normalized.png | Normalized confusion matrix |
| BoxPR_curve.png | Precision-Recall curve |
| BoxP_curve.png | Precision curve |
| BoxR_curve.png | Recall curve |
| BoxF1_curve.png | F1 curve |
| labels.jpg | Dataset label distribution visualization |
| train_batch*.jpg | Sample training batches with augmentations |
| val_batch*_labels.jpg | Validation ground truth |
| val_batch*_pred.jpg | Validation predictions (compare with labels) |

### Reproducibility

To reproduce this training run:

```bash
cd /workspace/project-alan/cr-object-detection

python -m src.detection.train \
  --data /path/to/dataset_reduced.yaml \
  --pretrained /path/to/previous/last.pt \
  --epochs 150 \
  --batch 32 \
  --imgsz 640 \
  --device cuda \
  --project /output/directory \
  --patience 30 \
  --lr0 0.01 \
  --workers 8
```

The dataset config (dataset_reduced.yaml) must have `path:` set to the absolute
path of `data/prepared/`. The pretrained weights came from an earlier 11-epoch
CPU training run (cr_detector_yolov8n_20260129_014736).
