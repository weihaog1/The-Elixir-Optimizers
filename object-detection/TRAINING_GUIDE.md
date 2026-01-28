# YOLOv8m Training Guide - vast.ai Instance

## Prerequisites

Files should already be at `/workspace/project-alan/cr-object-detection/`.

## Step 1: Verify GPU

```bash
nvidia-smi
```

Should show RTX 5070 Ti with 16GB VRAM.

## Step 2: Install Dependencies

```bash
pip install ultralytics
```

## Step 3: Verify Dataset

```bash
ls /workspace/project-alan/cr-object-detection/data/prepared/images/train/ | wc -l
# Should show 5551

ls /workspace/project-alan/cr-object-detection/data/prepared/images/val/ | wc -l
# Should show 1388
```

## Step 4: Fix Dataset Config Path

The config uses a relative path. You need to update it to an absolute path.

Edit `configs/dataset_reduced.yaml` and change the first line:

```yaml
path: /workspace/project-alan/cr-object-detection/data/prepared
```

The rest stays the same (train, val, nc: 155, names).

## Step 5: Start Training

```bash
cd /workspace/project-alan/cr-object-detection

yolo detect train \
  model=yolov8m.pt \
  data=configs/dataset_reduced.yaml \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0 \
  patience=30 \
  project=runs/detect \
  name=yolov8m_reduced \
  pretrained=True \
  mosaic=1.0 \
  mixup=0.1 \
  fliplr=0.5 \
  flipud=0.0 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  workers=8
```

If batch=16 causes OOM (out of memory), drop to batch=12 or batch=8.

Expected training time: 4-8 hours for 150 epochs.

## Step 6: Monitor Training

Training logs print to stdout. Key metrics to watch:

- box_loss: should decrease steadily
- cls_loss: should decrease steadily
- mAP50: should increase (target > 0.5)

## Step 7: Download Results

After training completes, the best weights will be at:

```
runs/detect/yolov8m_reduced/weights/best.pt
runs/detect/yolov8m_reduced/weights/last.pt
```

From your local machine:

```bash
scp -P 23815 root@74.48.78.46:/workspace/project-alan/cr-object-detection/runs/detect/yolov8m_reduced/weights/best.pt ./models/
```

## Notes

- dataset_reduced.yaml has 155 classes (201 minus 46 pad_* classes)
- No label files were changed since pad classes had zero annotations
- yolov8m.pt will auto-download COCO-pretrained weights on first run
- patience=30 means training stops if mAP doesn't improve for 30 epochs
