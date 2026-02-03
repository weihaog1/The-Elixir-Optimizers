# Clash Royale Object Detection - Non-Embedded AI

Based on the [KataCR](https://github.com/wty-yy/KataCR) project, this module implements object detection for Clash Royale that runs on your computer (non-embedded), analyzing the game screen in real-time.

## Overview

This system uses computer vision and deep learning to:
- Capture game screen (from emulator, phone mirror, or video)
- Detect all game objects (troops, spells, buildings, towers)
- Track unit positions and team affiliation
- Extract game state (elixir, cards, time)

## Architecture

```
┌─────────────────┐
│  Screen Capture │ ← Emulator / Phone Mirror / Video
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLOv8 Detector│ ← Object Detection Model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visual Fusion  │ ← Combine detections + Game State
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Game State   │ → Units, Cards, Elixir, Time
└─────────────────┘
```

## Installation

### Prerequisites
- Python 3.9+
- NVIDIA GPU (recommended) or CPU
- Windows 10/11

### Setup
       
1. **Install dependencies:**
```bash
cd src
pip install -r requirements.txt
```

2. **Install PyTorch** (choose based on your system):
```bash
# CPU only
pip install torch torchvision

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. **(Recommended) Download KataCR trained models:**
   - **detector1 v0.7.13**: [Google Drive](https://drive.google.com/file/d/1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_/view?usp=drive_link) - Small/medium units
   - **detector2 v0.7.13**: [Google Drive](https://drive.google.com/file/d/1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD/view?usp=drive_link) - Large units/buildings
   - Place `.pt` files in `src/models/` directory

## Combo Detection (Recommended)

KataCR uses a **combo detector approach** because Clash Royale has 150+ unit types with vastly different sizes:

| Model | Purpose | Unit Types |
|-------|---------|------------|
| **Detector 1** | Small/medium units | Skeletons, goblins, archers, minions, spirits |
| **Detector 2** | Large units & buildings | Giants, towers, golems, P.E.K.K.A, buildings |

**Why Combo Detection?**
- Each model is optimized for specific unit sizes
- Single YOLOv8 struggles with 150+ classes
- Better accuracy for both small and large units

## Usage

### Quick Start

```bash
# Combo detection with both models (best results)
python main.py --models models/detector1_v0.7.13.pt models/detector2_v0.7.13.pt --select-region

# Single model detection
python main.py --model models/detector1_v0.7.13.pt --select-region

# Run with specific screen region
python main.py --models models/detector1_v0.7.13.pt models/detector2_v0.7.13.pt \
               --region 100 200 540 1200

# Run on a video file
python main.py --models models/detector1_v0.7.13.pt models/detector2_v0.7.13.pt \
               --source video --source-arg gameplay.mp4

# Run on emulator window
python main.py --models models/detector1_v0.7.13.pt models/detector2_v0.7.13.pt \
               --source window --source-arg "BlueStacks"
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--models` | **Paths to multiple YOLOv8 models for combo detection** |
| `--model`, `-m` | Path to single YOLOv8 model weights |
| `--source`, `-s` | Capture source: `screen`, `window`, `image`, `video` |
| `--source-arg` | Window title or file path |
| `--region` | Screen capture region (LEFT TOP WIDTH HEIGHT) |
| `--conf` | Detection confidence threshold (default: 0.5) |
| `--output`, `-o` | Save output video to file |
| `--no-display` | Run without display window |
| `--select-region` | Interactively select screen region |

### Examples

```bash
# Combo detection from BlueStacks emulator
python main.py --models models/detector1_v0.7.13.pt models/detector2_v0.7.13.pt \
               --source window --source-arg "BlueStacks App Player"

# Process video with combo detection and save output
python main.py --models models/detector1_v0.7.13.pt models/detector2_v0.7.13.pt \
               --source video --source-arg match.mp4 \
               --output detection_output.mp4

# Single model with high confidence threshold
python main.py --model models/detector1_v0.7.13.pt --conf 0.7 --select-region

# Process video without display (faster)
python main.py --models models/detector1_v0.7.13.pt models/detector2_v0.7.13.pt \
               --source video --source-arg match.mp4 \
               --output result.mp4 --no-display
```

## Project Structure

```
src/
├── main.py           # Main entry point
├── config.py         # Configuration settings
├── capture.py        # Screen/window/video capture
├── detector.py       # YOLOv8 object detection
├── visual_fusion.py  # Game state extraction
├── requirements.txt  # Python dependencies
└── models/           # Model weights (create this folder)
    └── *.pt          # YOLOv8 model files
```

## Training Custom Models

For best results, train a custom YOLOv8 model on Clash Royale data:

1. **Get training data:**
   - Use [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)
   - Or create your own labeled dataset

2. **Train model:**
```python
from ultralytics import YOLO

# Load a base model
model = YOLO('yolov8n.pt')

# Train on Clash Royale data
model.train(
    data='path/to/clash_royale.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

3. **Use trained model:**
```bash
python main.py --model runs/train/best.pt --select-region
```

## Detection Categories

The system detects 100+ game objects including:

### Troops
- Knight, Archers, Giant, Musketeer, Mini P.E.K.K.A
- Valkyrie, Hog Rider, Wizard, Witch, Prince
- Skeleton Army, Minions, Balloon, Baby Dragon
- And many more...

### Spells
- Fireball, Arrows, Rage, Freeze, Lightning
- Zap, Poison, Graveyard, Tornado, Rocket, Log

### Buildings
- Cannon, Tesla, Inferno Tower, Bomb Tower
- Goblin Cage, Tombstone, Furnace, Elixir Collector

### UI Elements
- Towers (King, Princess)
- Health bars
- Elixir bar
- Cards

## Tips for Best Results

1. **Screen Capture:**
   - Use a fixed window position
   - Ensure consistent resolution (1080x2400 recommended)
   - Minimize UI overlays

2. **Performance:**
   - Use GPU for faster inference
   - Lower resolution for faster processing
   - Adjust confidence threshold based on needs

3. **Accuracy:**
   - Train custom model on your screen resolution
   - Calibrate screen regions for your setup
   - Use good lighting conditions

## Based on KataCR

This implementation is inspired by [KataCR](https://github.com/wty-yy/KataCR), an excellent undergraduate thesis project that created a full non-embedded AI for Clash Royale using:
- YOLOv8 for object detection
- ResNet for card classification
- PaddleOCR for text recognition
- Offline RL for decision making

## License

MIT License - See LICENSE file for details.
