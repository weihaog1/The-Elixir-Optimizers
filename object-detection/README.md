# Clash Royale Object Detection Pipeline

YOLOv8-based object detection for Clash Royale game state extraction.

## Project Structure

```
cr-object-detection/
├── src/
│   ├── data/           # Data processing (frame extraction, format conversion)
│   ├── detection/      # YOLOv8 wrapper, training, inference
│   ├── ocr/            # PaddleOCR for elixir/timer extraction
│   └── pipeline/       # Game state builder (combines detection + OCR)
├── configs/            # YOLO dataset configurations
├── scripts/            # Dataset preparation scripts
├── tests/              # Unit tests
└── requirements.txt    # Python dependencies
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 weights (optional - auto-downloads on first run)
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Dataset

The dataset is not included in this repo due to size (551MB).

**Option 1: Use KataCR dataset**
Download from: https://github.com/wty-yy/Clash-Royale-Detection-Dataset

**Option 2: Prepare your own**
```bash
python scripts/prepare_dataset.py --input /path/to/raw/data --output data/prepared
```

## Usage

### Training
```bash
python src/detection/train.py --data configs/dataset.yaml --epochs 30 --batch 8
```

### Inference
```bash
python src/detection/inference.py --model runs/detect/train/weights/best.pt --source image.jpg
```

### Game State Extraction
```python
from src.pipeline.state_builder import StateBuilder

builder = StateBuilder(model_path="path/to/model.pt")
state = builder.process_frame(image)
print(state.to_json())
```

## Pre-trained Models

Instead of training from scratch, you can use KataCR's pre-trained weights:
- [detector1 v0.7.13](https://drive.google.com/file/d/1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_/view)
- [detector2 v0.7.13](https://drive.google.com/file/d/1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD/view)

## Training Results

Training was run on Apple M1 Pro (MPS backend):
- Model: YOLOv8n (nano)
- Dataset: 6,939 images (5,551 train / 1,388 val)
- Classes: 201 (full unit coverage)
- Current mAP@0.5: 0.031 (needs more training or fewer classes)

Sample validation outputs are available in the `results/` folder (if included).

## Key Components

### Detection (`src/detection/model.py`)
- `CRDetector` class wrapping YOLOv8
- Methods: `detect()`, `train()`, `evaluate()`, `visualize()`

### OCR (`src/ocr/text_extractor.py`)
- `GameTextExtractor` for elixir count, timer, tower HP
- Uses PaddleOCR with preprocessing

### State Builder (`src/pipeline/state_builder.py`)
- Combines detection + OCR into structured game state
- Outputs JSON with units, towers, elixir, time

## Class Configuration

**MVP (6 classes):** `configs/classes.yaml`
- King Tower (player/enemy)
- Princess Tower Left/Right (player/enemy)

**Full (201 classes):** `configs/dataset.yaml`
- All troops, buildings, spells, UI elements

## References

- [KataCR](https://github.com/wty-yy/KataCR) - Original Clash Royale RL project
- [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset) - Training data
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Detection framework
