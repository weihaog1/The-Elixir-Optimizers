# Training Plan: YOLOv8s Belonging Model on vast.ai RTX 5080

**Date:** 2026-03-02
**Goal:** Train a YOLOv8s model with belonging prediction (ally/enemy) using pre-generated synthetic data on a rented RTX 5080 instance.
**Connection:** `ssh -p 44409 root@74.48.78.46 -L 8080:localhost:8080`

---

## What's Changed Since v12 (Current Best)

| Change | Before (v12) | Now |
|--------|-------------|-----|
| Belonging | Not trained | Training with belonging (nc=156) |
| Ally sprites (deck) | 235 (old batch, mixed quality) | 383 deck + 78 spawned units (113 CUSTOMv1 + 348 CUSTOMv2) |
| Non-deck allies | 553 present (confusing signal) | Removed (only deck cards + spawns can be allies) |
| Total ally sprites | ~1,158 | ~831 (370 towers/UI + 383 deck + 78 spawned) |
| Ally restriction | None | 10 classes: 8 deck cards + 2 spawned units |
| Spawned unit sprites | Misclassified in parent class | Reclassified: 41 goblin-brawler, 37 barbarian ally sprites |

---

## Phase 1: Upload to vast.ai

Upload the sprite dataset (~50 MB) and project code, then pre-generate on the remote server. This is much faster than pre-generating locally (~1.4 GB) and uploading the result.

### 1A. Verify instance is ready

```bash
ssh -p 44409 root@74.48.78.46

# Check GPU and CUDA
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('Available:', torch.cuda.is_available())"
```

**CRITICAL:** RTX 5080 (Blackwell, sm_120) requires:
- CUDA 12.8+
- NVIDIA driver 570+
- PyTorch 2.7.0+

If PyTorch doesn't see the GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 1B. Install dependencies

```bash
pip install ultralytics tqdm
```

(torch, torchvision, opencv, numpy, Pillow should already be on the vast.ai image)

### 1C. Upload files from local machine

```bash
cd /Users/alanguo/Codin/CS175

# 1. Project code (~small)
tar czf /tmp/cr-code.tar.gz \
  -C Project/cr-object-detection \
  src/ scripts/ configs/ requirements.txt

# 2. Sprite dataset (~50 MB -- much smaller than uploading 1.4 GB of pre-gen images)
tar czf /tmp/cr-sprites.tar.gz \
  Froked-KataCR-Clash-Royale-Detection-Dataset/images/segment/

# 3. Full forked dataset (needed for restore_val_belonging.py -- has part2/ labels)
tar czf /tmp/cr-val-source.tar.gz \
  Froked-KataCR-Clash-Royale-Detection-Dataset/images/part2/

# Upload all (total ~200-300 MB vs 1.6 GB if pre-gen locally)
scp -P 44409 /tmp/cr-code.tar.gz /tmp/cr-sprites.tar.gz /tmp/cr-val-source.tar.gz root@74.48.78.46:/workspace/
```

### 1D. Extract on remote

```bash
ssh -p 44409 root@74.48.78.46

cd /workspace
mkdir -p cr-object-detection
tar xzf cr-code.tar.gz -C cr-object-detection/
tar xzf cr-sprites.tar.gz
tar xzf cr-val-source.tar.gz

# Verify
ls Froked-KataCR-Clash-Royale-Detection-Dataset/images/segment/ | head -5
```

### 1E. Set environment variables

```bash
export CR_DATASET_PATH=/workspace/Froked-KataCR-Clash-Royale-Detection-Dataset
export CR_TRAIN_DATASIZE=20000
```

---

## Phase 2: Pre-Generate Dataset (on vast.ai)

Pre-generation is CPU-bound (no GPU needed), but running it on the remote avoids uploading 1.4 GB of images. The GPU sits idle for ~30-60 min -- costs ~$0.25-0.50 at vast.ai rates.

### 2A. Generate belonging training data

```bash
cd /workspace/cr-object-detection

export CR_DATASET_PATH=/workspace/Froked-KataCR-Clash-Royale-Detection-Dataset

python scripts/generate_dataset.py \
  --num-images 20000 \
  --output data/synthetic_belong/train \
  --background 15 \
  --seed 42 \
  --units 40 \
  --noise-ratio 0.25 \
  --img-width 576 \
  --img-height 896 \
  --include-belonging \
  --ally-classes arrows,barbarian,barbarian-barrel,electro-spirit,flying-machine,goblin-brawler,goblin-cage,royal-hog,royal-recruit,zappy \
  --workers 8
```

**Output:** `data/synthetic_belong/train/{images,labels}/` (~1.4 GB, 20k images with 6-column labels)

**Why 20k images:** v12 used 20k and mAP50 plateaued by epoch 10. More images would only help if the domain gap were smaller. The bottleneck is synthetic-to-real transfer, not data volume.

**Why --ally-classes restriction:** We cleaned the sprite dataset to only have ally sprites for our 10 ally classes (8 deck cards + 2 spawned units) + towers + UI. The --ally-classes flag adds an extra enforcement layer in the generator, ensuring only these 10 classes produce `bel=0` sprites. Towers are handled separately (add_tower()), and UI elements are in drop_units (excluded from unit sampling).

**The 10 ally classes:**
- 8 deck cards: arrows, barbarian-barrel, electro-spirit, flying-machine, goblin-cage, royal-hog, royal-recruit, zappy
- 2 spawned units: barbarian (from barbarian-barrel), goblin-brawler (from goblin-cage)

### 2B. Prepare belonging validation data

```bash
python scripts/restore_val_belonging.py \
  --input /workspace/Froked-KataCR-Clash-Royale-Detection-Dataset \
  --output data/prepared_belong
```

**Output:** `data/prepared_belong/{images/val, labels/val}/` (1,388 images with 6-column labels)

**Verify:**
```bash
ls data/synthetic_belong/train/images/ | wc -l     # Should be 20000
ls data/prepared_belong/images/val/ | wc -l         # Should be 1388
head -3 data/prepared_belong/labels/val/*.txt | head -20  # Should show 6 columns
```

---

## Phase 3: Train (on vast.ai RTX 5080)

### 3A. Start tmux (survives SSH disconnects)

```bash
tmux new -s train
```

### 3B. Run training

```bash
cd /workspace/cr-object-detection

export CR_TRAIN_DATASIZE=20000

python scripts/train_synthetic.py \
  --data configs/synthetic_belonging_data.yaml \
  --model yolov8s.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 960 \
  --device 0 \
  --workers 8 \
  --belonging \
  --pregen \
  --background 15 \
  --noise-ratio 0.25 \
  --units 40 \
  --seed 42 \
  --project runs/synthetic \
  --name yolov8s_belonging_v3
```

**Why these settings:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | YOLOv8s | 3.3x capacity over nano, proven at 0.804 mAP50. YOLOv8m would halve batch size on 16GB VRAM for marginal gain. |
| imgsz | 960 | Critical for small units (skeletons, spirits). Default 640 misses them. |
| batch | 16 | Safe for 16GB VRAM with AMP. Can try 20 if GPU memory allows. |
| epochs | 50 | mAP50 plateaus by epoch 10, but mAP50-95 continues improving slowly. 50 epochs captures the box precision tail. |
| workers | 8 | Pre-gen data is disk I/O bound, not CPU bound. 8 workers saturates typical vast.ai disk throughput. |
| belonging | flag | Enables nc=156 with CRDetectionModel + CRDetectionLoss. |
| pregen | flag | Loads from disk instead of generating on-the-fly. Eliminates CPU bottleneck. |
| seed | 42 | Matches all previous runs and val split preparation. |
| lr0 | 0.01 (default) | Standard SGD learning rate, proven in v12. |
| patience | 15 (default) | Early stopping if val loss stalls for 15 epochs. |
| mosaic | 0.0 (hardcoded) | Disabled - generator handles compositing. |
| AMP | True (hardcoded) | Mixed precision for 1.5-2x speedup on Tensor Cores. |

**Detach from tmux:** Press `Ctrl+B`, then `D`. Training continues in background.

### 3C. Monitor training

```bash
# Reattach to see live output
tmux attach -t train

# Or check metrics from another terminal
ssh -p 44409 root@74.48.78.46 "tail -5 /workspace/cr-object-detection/runs/synthetic/yolov8s_belonging_v3/results.csv"

# GPU utilization
ssh -p 44409 root@74.48.78.46 "nvidia-smi"
```

**Expected timeline (RTX 5080):**
- Per epoch: ~2.5-3.5 minutes (v12 was ~4.2 min/epoch on 5070 Ti)
- Total: ~2.5-3 hours for 50 epochs
- mAP50 will plateau around epoch 10-15
- Training loss will keep decreasing (normal - model learns synthetic patterns)
- Val loss will plateau or slightly increase after epoch 15-20

### 3D. Resume after crash

```bash
cd /workspace/cr-object-detection

export CR_TRAIN_DATASIZE=20000

python scripts/train_synthetic.py \
  --data configs/synthetic_belonging_data.yaml \
  --belonging \
  --pregen \
  --project runs/synthetic \
  --name yolov8s_belonging_v3 \
  --resume
```

---

## Phase 4: Download Results

### 4A. Download weights and metrics

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection

# Best weights
scp -P 44409 root@74.48.78.46:/workspace/cr-object-detection/runs/synthetic/yolov8s_belonging_v3/weights/best.pt ./models/best_yolov8s_belonging_v3.pt

# Last checkpoint (for resume)
scp -P 44409 root@74.48.78.46:/workspace/cr-object-detection/runs/synthetic/yolov8s_belonging_v3/weights/last.pt ./models/last_yolov8s_belonging_v3.pt

# Full results directory
rsync -avz --progress -e "ssh -p 44409" \
  root@74.48.78.46:/workspace/cr-object-detection/runs/synthetic/yolov8s_belonging_v3/ \
  ./runs/synthetic/yolov8s_belonging_v3/
```

### 4B. Key files to check

```
runs/synthetic/yolov8s_belonging_v3/
  weights/best.pt        # Best mAP50 model
  weights/last.pt        # Last epoch checkpoint
  results.csv            # Epoch-by-epoch metrics
  results.png            # Loss/metric curves
  args.yaml              # Exact training config used
  train_batch0.jpg       # Sample training batch visualization
  val_batch0_labels.jpg  # Validation ground truth
  val_batch0_pred.jpg    # Validation predictions
```

---

## Phase 5: Evaluate

### 5A. Standard mAP evaluation

```bash
python -c "
from ultralytics import YOLO
model = YOLO('models/best_yolov8s_belonging_v3.pt')
results = model.val(data='configs/synthetic_belonging_data.yaml', imgsz=960, batch=16)
print(f'mAP50: {results.box.map50:.3f}')
print(f'mAP50-95: {results.box.map:.3f}')
"
```

### 5B. Gameplay video evaluation

```bash
python scripts/evaluate_on_video.py \
  --model models/best_yolov8s_belonging_v3.pt \
  --video gameplay-videos/gameplay.mp4 \
  --imgsz 960
```

### 5C. What to look for

| Metric | v12 (no belonging) | Target (with belonging) | Notes |
|--------|-------------------|------------------------|-------|
| mAP50 | 0.804 | >= 0.75 | May dip slightly - belonging adds complexity |
| mAP50-95 | 0.567 | >= 0.50 | Box precision should be similar |
| Precision | 0.822 | >= 0.75 | Acceptable range |
| Recall | 0.771 | >= 0.70 | Acceptable range |
| Belonging accuracy | N/A | > 85% | New metric - check manually on gameplay |
| Dets/frame (video) | 30.2 | >= 25 | Slight drop acceptable |
| Hallucinations | 0 | 0 | Must stay zero |

**Important:** The validation set is KataCR's data (Hog 2.6 deck), not our deck. Belonging accuracy on the val set may underperform because:
1. Val set has ally hog-riders, mustketeers etc. from KataCR's deck
2. Our model now says "those can only be enemies"
3. This is **correct for our use case** even if val numbers look worse

The real test is gameplay video evaluation where our deck's cards should correctly show as allies.

---

## Decisions and Justifications

### Why not train_belonging.py?
`train_belonging.py` does NOT support `--pregen`. It always generates on-the-fly, which wastes the GPU waiting for CPU generation. We use `train_synthetic.py --belonging --pregen` instead, which loads pre-generated images from disk for maximum GPU utilization.

### Why not YOLOv8m?
At imgsz=960, YOLOv8m would only fit batch=4-6 on 16GB VRAM. This would:
- Slow training by 2-3x (~9 hours instead of ~3)
- Reduce batch normalization effectiveness
- Not address the domain gap (the actual bottleneck)
The marginal +1-3% mAP gain doesn't justify the cost.

### Why not more than 20k images?
v12 proved mAP50 plateaus by epoch 10 regardless of data volume. The domain gap (4.7x cls loss ratio) is the ceiling, not data quantity. 20k images with YOLO augmentations (HSV jitter, rotation, erasing, flip) provide sufficient variety.

### Why 50 epochs if mAP50 plateaus at 10?
mAP50-95 (box precision) continues improving slowly through epoch 50. The extra epochs refine bounding box accuracy even though detection recall is stable. With pre-gen data and a fast GPU, the extra epochs only cost ~1.5 hours.

### Why restrict ally_classes?
Without restriction, any class with `_0_` sprites could appear as an ally in training. After our cleanup, only 10 classes (8 deck cards + 2 spawned units) + towers + UI have ally sprites. The --ally-classes flag adds defense-in-depth: even if stray ally sprites exist, the generator won't use them.

---

## Known Limitations

### Spawned units reclassified (RESOLVED)
goblin-brawler (41 ally sprites) and barbarian (37 ally sprites) were originally misclassified under their parent card directories. They have been moved to their correct class directories and added to the --ally-classes list. Both spawned units now have strong ally representation.

### arrows has only 1 ally sprite
With 1 ally sprite vs 6 enemy sprites, the model will have a strong bias toward predicting enemy for arrows. Arrows detection is already weak (AP50=0.190 in v12). This will likely not improve with belonging training.

### Validation set measures the wrong deck
The val set is from KataCR's Hog 2.6 deck. Belonging predictions will be systematically wrong for their ally units (hog-rider, musketeer, skeleton, etc.) because our model says only our 8 cards can be allies. This means val metrics may understate real-world belonging accuracy. The true test is gameplay video with our deck.

## Fallback Strategy

If mAP50 drops below 0.70, or if gameplay video evaluation shows significantly fewer detections/frame than v12's 30.2 average:

1. Check if the regression is from belonging or from the ally sprite changes
2. Consider retraining without belonging (use v12's standard config) to isolate the cause
3. If belonging itself causes regression, the v12 weights (`best_yolov8s_50epochs_fixed_pregen_set.pt`) remain the production model
4. Belonging prediction can be added as a post-processing step using the Y-position heuristic as a temporary fallback

---

## Troubleshooting

### OOM (Out of Memory)
Drop batch size: `--batch 12` or `--batch 8`

### "No kernel image is available" error
PyTorch doesn't support sm_120. Install PyTorch 2.7.0+:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Training loss not decreasing
Known RTX 5080 Blackwell issue. Try:
1. Update NVIDIA driver to latest
2. Set `CUDA_LAUNCH_BLOCKING=1` for debugging
3. Disable torch.compile if enabled

### "path_dataset not found" warning
Set the env var: `export CR_DATASET_PATH=/workspace/Froked-KataCR-Clash-Royale-Detection-Dataset`

### Validation fails with label format error
The belonging validation needs 6-column labels. Ensure `restore_val_belonging.py` was run and `data/prepared_belong/` exists with correct labels.

### SSH disconnects mid-training
Training continues in tmux. Reconnect: `ssh -p 44409 root@74.48.78.46` then `tmux attach -t train`

---

## Cost Estimate

- RTX 5080 on vast.ai: ~$0.40-0.60/hour
- Training time: ~3 hours
- Total GPU cost: ~$1.20-1.80
- Add ~1 hour for setup/upload/verification: ~$0.50
- **Total estimated cost: ~$2-2.50**
