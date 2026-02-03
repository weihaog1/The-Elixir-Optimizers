# Training Your Own Clash Royale Detection Model

This guide explains how to train custom YOLOv8 models for Clash Royale object detection.

## Quick Start

```bash
# 1. Download the dataset
git clone https://github.com/wty-yy/Clash-Royale-Detection-Dataset

# 2. Train the model
python train.py --data path/to/Clash-Royale-Detection-Dataset/data.yaml --epochs 100

# 3. Use your trained model
python main.py --model runs/train/clash_royale/weights/best.pt --select-region
```

## Step 1: Get the Dataset

### Option A: Use KataCR's Dataset (Recommended)

Download the pre-built dataset from [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset):

```bash
git clone https://github.com/wty-yy/Clash-Royale-Detection-Dataset
```

The dataset includes:
- **150+ unit classes** (troops, spells, buildings, towers)
- **Synthetic training images** generated from unit sprites
- **Real validation images** from actual gameplay

### Option B: Create Your Own Dataset

1. **Collect screenshots** from Clash Royale gameplay
2. **Label images** using [Label Studio](https://labelstud.io/) or [Roboflow](https://roboflow.com/)
3. **Create data.yaml** configuration file

Example `data.yaml`:
```yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: king-tower
  1: queen-tower
  2: knight
  3: archers
  4: giant
  # ... add all your classes
```

## Step 2: Train the Model

### Basic Training

```bash
python train.py --data path/to/data.yaml
```

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Path to data.yaml | Required |
| `--model-size` | Model size: n, s, m, l, x | n |
| `--epochs` | Training epochs | 100 |
| `--batch-size` | Batch size | 16 |
| `--img-size` | Image size | 640 |
| `--device` | GPU device (0, 1, cpu) | 0 |
| `--name` | Experiment name | clash_royale |
| `--resume` | Resume from checkpoint | False |

### Recommended Settings

**For quick testing:**
```bash
python train.py --data data.yaml --model-size n --epochs 50 --batch-size 16
```

**For better accuracy:**
```bash
python train.py --data data.yaml --model-size m --epochs 150 --batch-size 8
```

**For best results (requires good GPU):**
```bash
python train.py --data data.yaml --model-size l --epochs 200 --batch-size 4
```

### Model Size Comparison

| Size | Parameters | Speed | Accuracy | GPU Memory |
|------|------------|-------|----------|------------|
| n (nano) | 3.2M | Fastest | Good | ~4 GB |
| s (small) | 11.2M | Fast | Better | ~6 GB |
| m (medium) | 25.9M | Medium | Great | ~8 GB |
| l (large) | 43.7M | Slow | Excellent | ~12 GB |
| x (xlarge) | 68.2M | Slowest | Best | ~16 GB |

## Step 3: Monitor Training

Training progress is saved to `runs/train/{name}/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Latest checkpoint
- `results.csv` - Training metrics
- `confusion_matrix.png` - Class confusion matrix
- `results.png` - Training curves

### Using Weights & Biases (Optional)

```bash
pip install wandb
wandb login

python train.py --data data.yaml --name my_experiment
```

## Step 4: Validate Your Model

```bash
python train.py --validate --model runs/train/clash_royale/weights/best.pt --data data.yaml
```

## Step 5: Use Your Trained Model

### Single Model
```bash
python main.py --model runs/train/clash_royale/weights/best.pt --select-region
```

### Training Multiple Models for Combo Detection

Following KataCR's approach, train separate models for different unit sizes:

**Detector 1 (small units):**
```bash
python train.py --data small_units_data.yaml --name detector1 --model-size s
```

**Detector 2 (large units):**
```bash
python train.py --data large_units_data.yaml --name detector2 --model-size s
```

**Use combo detection:**
```bash
python main.py --models runs/train/detector1/weights/best.pt \
                       runs/train/detector2/weights/best.pt \
               --select-region
```

## Tips for Better Results

### 1. Data Quality
- Use diverse gameplay footage (different arenas, times of day)
- Include various unit combinations
- Balance classes (don't have 1000 skeletons and 10 golems)

### 2. Training
- Start with a smaller model (n or s) to iterate quickly
- Use early stopping (`patience=50`) to avoid overfitting
- Monitor validation mAP to track progress

### 3. Augmentation
The training script includes augmentation optimized for CR:
- Horizontal flip (units can face either direction)
- Color jitter (different lighting in arenas)
- Scale variation (units at different distances)

### 4. Common Issues

**Out of Memory:**
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Use smaller model: `--model-size n`

**Poor Accuracy:**
- Train longer: `--epochs 200`
- Use larger model: `--model-size m`
- Check dataset quality

**Overfitting:**
- Use more training data
- Increase augmentation
- Try smaller model

## Resources

- [KataCR Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow (for labeling)](https://roboflow.com/)
- [KataCR Training Curves](https://wandb.ai/wty-yy/YOLOv8)
