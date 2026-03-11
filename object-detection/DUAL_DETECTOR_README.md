# Dual YOLOv8m Detector - Clash Royale Object Detection

## Overview

Two YOLOv8m models split the 155 Clash Royale classes by sprite size (following KataCR's approach). D1 handles small sprites (skeletons, spirits, projectiles), D2 handles large sprites (golems, pekka, buildings). 13 base classes (towers, bars, UI) appear in both and are deduplicated at inference time.

Each model predicts **belonging** (ally vs enemy) as an extra output channel, so detections include side assignment without relying on Y-position heuristics.

## Files Included

```
dual_demo.py                      # Real-time demo script
models/dual_d1_best.pt            # Detector 1 weights (small sprites, YOLOv8m)
models/dual_d2_best.pt            # Detector 2 weights (large sprites, YOLOv8m)
configs/split_config.json          # Class split + index remapping config
models/card_classifier/
  card_classifier.pt               # Card-in-hand classifier (optional)
```

## Training Details

| Property | D1 (Small Sprites) | D2 (Large Sprites) |
|----------|--------------------|--------------------|
| Architecture | YOLOv8m + belonging head | YOLOv8m + belonging head |
| Classes | 85 (72 unique + 13 base) | 85 (72 unique + 13 base) |
| Training data | 20,000 synthetic images | 20,000 synthetic images |
| Validation data | 1,369 images (remapped) | 1,369 images (remapped) |
| Epochs | 27 (early stopped) | 50 |
| Best mAP50 | 0.798 | 0.853 |
| Best mAP50-95 | 0.547 | 0.665 |
| Precision | 0.885 | 0.868 |
| Recall | 0.730 | 0.827 |
| imgsz | 960 | 960 |
| Batch size | 24 | 24 |
| GPU | RTX 5080 (16GB) | RTX 5080 (16GB) |

## Prerequisites

```bash
pip install ultralytics torch torchvision opencv-python numpy
# For card classification (optional):
pip install timm
# For OCR (optional):
pip install paddleocr paddlepaddle
```

The demo also requires the project's `src/` package on the Python path. Run from the `cr-object-detection/` directory.

## Quick Start

```bash
cd cr-object-detection

# Basic - starts paused on first frame
python dual_demo.py --video path/to/gameplay.mp4

# Autoplay at half speed with card detection
python dual_demo.py --video path/to/gameplay.mp4 --autoplay --speed 0.5 \
    --card-model models/card_classifier/card_classifier.pt

# Full features (cards + OCR for elixir/timer)
python dual_demo.py --video path/to/gameplay.mp4 --autoplay --speed 0.5 \
    --card-model models/card_classifier/card_classifier.pt --ocr

# On CUDA GPU
python dual_demo.py --video path/to/gameplay.mp4 --device cuda

# Lower confidence for more detections (noisier)
python dual_demo.py --video path/to/gameplay.mp4 --conf 0.15
```

## Controls

| Key | Action |
|-----|--------|
| Space | Toggle pause/play |
| Right arrow / `.` | Step forward (both modes) |
| Left arrow / `,` | Step backward (both modes) |
| `+` / `-` | Increase/decrease frame skip (1-30) |
| `]` / `[` | Increase/decrease playback speed (0.25x - 4x) |
| `c` | Toggle confidence display |
| `d` | Toggle detector source labels (D1/D2/Base) |
| `s` | Save current frame as JPEG |
| `q` | Quit |

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | (required) | Path to gameplay video |
| `--d1` | `models/dual_d1_best.pt` | Detector 1 weights |
| `--d2` | `models/dual_d2_best.pt` | Detector 2 weights |
| `--split-config` | `configs/split_config.json` | Class split config |
| `--device` | `mps` | Inference device: mps, cuda, cpu |
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.45 | IoU threshold for NMS |
| `--imgsz` | 960 | Inference resolution |
| `--autoplay` | off | Start playing immediately |
| `--speed` | 1.0 | Playback speed multiplier |
| `--card-model` | none | Card classifier weights path |
| `--ocr` | off | Enable elixir/timer OCR |
| `--ocr-interval` | 5 | Run OCR every N frames |

## Display

- **Orange boxes** = ally units (belonging=0)
- **Red boxes** = enemy units (belonging=1)
- **Gray boxes** = non-combat elements (bars, clock, emotes)
- **Green line** = arena cutoff (UI excluded below this)
- **HUD** shows: play state, frame number, inference time (ms), detection count, skip value
- **Stats line** shows: D1-only / D2-only / Base detection counts
- **Bottom panel** (when enabled): card hand with confidence bars, elixir bar, game timer

## How It Works

1. Frame is cropped to arena region (above y=1550 in 1080x1920)
2. Both detectors run in parallel (ThreadPoolExecutor)
3. Each detector: LetterBox(auto=True) -> forward pass -> custom NMS with belonging
4. Local class indices remapped to global 155-class space via split_config.json
5. Cross-detector class-aware NMS deduplicates shared base classes
6. Results merged and sorted by confidence

## Performance

- M1 Pro (MPS): ~400ms per frame (~2.5 FPS) for both models combined
- CUDA GPUs will be significantly faster
- Card classification adds ~5ms per frame
- OCR adds ~50-100ms every N frames (configurable interval)

## Using ComboDetector Programmatically

```python
from src.detection.combo_detector import ComboDetector
import cv2

combo = ComboDetector(
    model_paths=["models/dual_d1_best.pt", "models/dual_d2_best.pt"],
    split_config_path="configs/split_config.json",
    device="mps",  # or "cuda"
    conf=0.25,
    imgsz=960,
)
combo.warmup()

frame = cv2.imread("screenshot.png")
dets = combo.infer(frame, arena_cutoff=1550)
# dets shape: (N, 7) = [x1, y1, x2, y2, confidence, global_class_id, belonging]
# belonging: 0=ally, 1=enemy

for d in dets:
    x1, y1, x2, y2, conf, cls_id, belonging = d
    class_name = combo.names[int(cls_id)]
    side = "ally" if int(belonging) == 0 else "enemy"
    print(f"{class_name} ({conf:.0%}) [{side}] at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
```
