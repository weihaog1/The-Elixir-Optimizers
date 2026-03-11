# Model Comparison: KataCR vs Our Implementation vs Standard YOLOv8

How our YOLOv8s belonging model differs from KataCR's original and from stock ultralytics YOLOv8.

---

## The Core Trick (Shared by Both KataCR and Us)

Neither KataCR nor our project adds a separate detection head for belonging. Instead, belonging is encoded as **a fake extra class appended to the end of the class list**. The standard YOLOv8 classification head outputs `nc` channels per anchor. By setting `nc = real_classes + 1`, the last channel becomes the belonging score, trained with the same BCE loss as real classes but reinterpreted at inference.

This is elegant because it requires zero architectural changes to the YOLOv8 backbone, neck, or detection head. The only changes are in the loss function (how targets are built), NMS (how the last channel is split off), and data loading (6-column labels instead of 5).

---

## Side-by-Side Comparison

| Aspect | Standard YOLOv8 | KataCR | Our Implementation |
|--------|----------------|--------|-------------------|
| **Architecture** | YOLOv8 (any size) | YOLOv8-Large x2 (dual detector) | YOLOv8s (single) |
| **Parameters** | Varies | ~87.4M (2 x 43.7M) | 11.2M |
| **nc (num classes)** | Real classes only (e.g., 80) | 85 per detector (83 real + 1 padding + 1 pad_belong) | 156 (155 real + 1 padding_belong) |
| **no (outputs/anchor)** | nc + 64 | 85 + 64 = 149 | 156 + 64 = 220 |
| **Belonging prediction** | None | Last class channel (pad_belong) | Last class channel (padding_belong) |
| **State prediction (s1-s6)** | None | Annotated in labels but NOT predicted | Not implemented |
| **Image size** | 640 default | 896 (rect mode, 576x896) | 960 |
| **Training data** | Real images | Synthetic only (on-the-fly generation) | Synthetic only (pre-generated) |
| **Validation data** | Real images | Real human-labeled (1,388 frames) | Same real set (1,369 frames after cleanup) |
| **Pretrained** | COCO weights | From scratch (pretrained: False) | COCO pretrained (yolov8s.pt) |
| **Label format (disk)** | 5-col: cls, x, y, w, h | 12-col: cls, x, y, w, h, bel, s1-s6 | 6-col: cls, x, y, w, h, bel |
| **Label format (internal)** | cls=(N,1), bbox=(N,4) | cls=(N,2) [id,bel], bbox=(N,4) | cls=(N,2) [id,bel], bbox=(N,4) |
| **NMS output** | 6-col: x1,y1,x2,y2,conf,cls | 7-col: x1,y1,x2,y2,conf,cls,bel | 7-col: x1,y1,x2,y2,conf,cls,bel |
| **Epochs** | Varies | 80, patience 10 | 50, patience 15 |
| **Batch size** | Varies | 16 | 16 |
| **Augmentation** | Full YOLO aug | mosaic=0, mixup=0, augment=False | mosaic=0, mixup=0, HSV/flip/erasing on |
| **FPS** | Varies | 34 (4090), 18 with tracking | ~15 (M1 Pro MPS) |
| **mAP50** | Varies | 84.3 (combo) | 0.739 (belonging) / 0.804 (v12 no belonging) |

---

## Architecture Differences in Detail

### KataCR: Dual Detector Strategy

KataCR splits 155 classes across TWO YOLOv8-Large models:

```
Detector 1 (nc=85): 15 base classes + ~66 small-sprite classes + 3 padding + pad_belong
Detector 2 (nc=85): 15 base classes + ~66 large-sprite classes + 3 padding + pad_belong
```

Why: 155 classes exceed the capacity of a single detector. Small and large sprites have vastly different scales, causing the anchor-free head to bias toward larger objects. Splitting by sprite size lets each detector specialize.

At inference, both detectors run on every frame. Their outputs are remapped to global class indices and merged with NMS (iou=0.6).

### Ours: Single Detector

We use a single YOLOv8s (Small, 11.2M params) with all 155 classes + 1 belonging = nc=156.

Why: Speed and simplicity. YOLOv8s at imgsz=960 gives mAP50=0.804 (without belonging), which is sufficient for the RL agent. Dual detectors would halve throughput and double engineering complexity for a 10-week course project.

Trade-off: We sacrifice small object detection (KataCR's dual approach improved mAP_S from 35.9 to 43.9) and overall mAP (their 84.3 vs our 80.4).

---

## The Belonging Mechanism (Identical Concept, Same Code Structure)

Both implementations use the exact same trick. Here is how it flows:

### 1. Data: Labels have 6 columns

```
# Standard YOLO (5 columns):
37  0.45  0.32  0.08  0.12           # class=37(knight), bbox

# Belonging YOLO (6 columns):
37  0.45  0.32  0.08  0.12  1        # class=37(knight), bbox, enemy
37  0.45  0.32  0.08  0.12  0        # class=37(knight), bbox, ally
```

### 2. Loss: Targets are 2-column class labels

Standard v8DetectionLoss builds targets as `(batch_idx, cls, x, y, w, h)` with cls being 1 column. Both KataCR and our CRDetectionLoss build `(batch_idx, cls, bel, x, y, w, h)` -- 7 values.

After preprocessing, `gt_labels` has shape `(batch, max_obj, 2)` instead of `(batch, max_obj, 1)`.

### 3. Assigner: Belonging goes in the last target score channel

The CRTaskAlignedAssigner builds a target score matrix of shape `(batch, anchors, nc)`:

```python
target_scores.scatter_(2, target_labels[..., 0:1], 1)  # One-hot encode class ID into channels 0-154
target_scores[..., -1] = target_labels[..., 1]          # Write belonging (0 or 1) into channel 155
```

Critically, the alignment metric (which determines anchor-to-GT assignment quality) uses ONLY the class prediction, not belonging:
```python
ind[1] = gt_labels[..., 0]  # class_id only, ignoring belonging
```

This prevents belonging uncertainty from distorting which anchors get assigned to which objects.

### 4. Loss computation: BCE treats belonging like any class

The BCE loss operates over all `nc=156` channels identically. Channel 155 gets trained to predict belonging just like channels 0-154 predict classes. No special loss weighting or separate loss term.

### 5. NMS: Last channel split off as belonging

```python
# Standard NMS:
box, cls, mask = x.split((4, nc, nm), 1)      # All nc channels are class scores

# Belonging NMS:
box, cls, bel, mask = x.split((4, nc-1, 1, nm), 1)  # Last channel is belonging
conf, j = cls.max(1, keepdim=True)             # Best class from 155 real classes
bel = (bel > 0.5).float()                      # Threshold belonging at 0.5
output = torch.cat((box, conf, j.float(), bel), 1)  # 7 columns
```

### 6. Results: 7-column detections

Standard: `(x1, y1, x2, y2, confidence, class_id)` -- 6 values
Belonging: `(x1, y1, x2, y2, confidence, class_id, belonging)` -- 7 values

CRBoxes.cls returns `(N, 2)` tensor of `(class_id, belonging)` instead of `(N,)` scalar.

---

## What KataCR Has That We Don't

### States s1-s6 (Movement, Shield, Visibility, Rage, Slow, Heal/Clone)

KataCR annotated 6 state dimensions in their validation labels:

| State | Values | Example |
|-------|--------|---------|
| s1: movement | norm, attack, deploy, freeze, dash | archer-queen in deploy state |
| s2: shield | bare/charge, shield/over | royal-recruit with shield |
| s3: visibility | visible, invisible | royal-ghost cloaked |
| s4: rage | norm, rage | barbarian under rage spell |
| s5: slow | norm, slow | unit hit by poison |
| s6: heal/clone | norm, heal, clone | clone spell duplicate |

**However, KataCR does NOT predict these either.** Their model only predicts belonging (s0). States s1-s6 exist in the annotation format but are completely ignored during training (`num_state_classes = 1` in their code). The synthetic generator cannot produce state variations because sprites are single-frame static images.

### Health bar processing

KataCR stores raw 24x8 pixel health bar images (192 values per bar) and processes them with a CNN in their policy model (StARformer). We detect bar sprites but don't extract HP values from them.

### ByteTrack object tracking

KataCR runs multi-object tracking across frames for temporal consistency. We process individual frames without tracking.

### Dual detector with class splitting by sprite size

Already discussed above. Their combo detector (2x YOLOv8-L) achieves mAP50=84.3 vs our single YOLOv8s at 80.4.

### Continuous action space

KataCR's StARformer predicts continuous 2D placement coordinates. We use Discrete(2305) = 4 cards x 576 cells + no-op with action masking.

### Offline RL (Decision Transformer in JAX)

KataCR uses offline RL with a custom Decision Transformer (StARformer) in JAX. We use online RL with SB3 MaskablePPO in PyTorch, warm-started with behavior cloning.

---

## What We Have That KataCR Doesn't

### Deck-specific ally sprite optimization

KataCR's ally sprites cover their Hog 2.6 deck. We removed all non-deck allies and added high-quality cutouts for our RR Hogs deck (8 cards + 2 spawned units). This gives the belonging model a cleaner training signal -- it only learns "these 10 classes can be ally, everything else is enemy."

### Pre-generated training data

KataCR generates synthetic images on-the-fly during training (the Generator runs on CPU while the GPU trains). We pre-generate 20k images to disk, which decouples CPU generation from GPU training and maximizes GPU utilization on rented instances.

### COCO pretrained initialization

KataCR trains from scratch (`pretrained: False`). We fine-tune from COCO-pretrained yolov8s.pt, which gives better early convergence and feature initialization.

### Card classification

We have a separate MiniResNet card classifier (~25K params, 8 classes) trained on our deck. KataCR had a ResNet classifier but only for Hog 2.6.

### OCR for elixir and timer

We use PaddleOCR to read game state. KataCR used it too, but our implementation is independently developed.

---

## Performance Summary

| Model | mAP50 | mAP50-95 | FPS | Params | Notes |
|-------|-------|----------|-----|--------|-------|
| KataCR YOLOv8-L x2 | 84.3 | 67.4 | 34 (4090) | 87.4M | Dual detector, from scratch |
| Our v12 YOLOv8s (no belonging) | 80.4 | 56.7 | 15.3 (M1) | 11.2M | Single detector, COCO pretrained |
| Our v33 YOLOv8s (belonging) | 73.9 | 53.9 | ~15 (M1) | 11.2M | With belonging, val set mismatch |

The v33 mAP50 drop (80.4 -> 73.9) is expected: the validation set contains KataCR's Hog 2.6 allies, and our model classifies them as enemies. Real-world belonging accuracy on our deck should be better than the metric suggests.

---

## Code Lineage

Our custom YOLO code is a direct port of KataCR's modifications:

```
KataCR (original)                    Our Project (ported)
katacr/yolov8/custom_model.py    -> src/yolov8_custom/custom_model.py
katacr/yolov8/custom_utils.py    -> src/yolov8_custom/custom_utils.py
katacr/yolov8/custom_result.py   -> src/yolov8_custom/custom_result.py
katacr/yolov8/custom_dataset.py  -> src/generation/synthetic_dataset.py
katacr/yolov8/custom_trainer.py  -> scripts/train_synthetic.py (SyntheticTrainer)
katacr/yolov8/custom_validator.py-> src/yolov8_custom/custom_validator.py
katacr/yolov8/custom_predict.py  -> src/yolov8_custom/custom_predict.py
```

The modifications from KataCR's code to ours:
- YOLOv8-Large -> YOLOv8s (smaller model)
- Dual detector -> single detector (all 155 classes in one model)
- On-the-fly generation -> pre-generated support added
- nc=85 per detector -> nc=156 (all classes + belonging in one model)
- Ultralytics 8.1.24 -> updated to current version (compatibility patches)
- Added --ally-classes restriction in generator
- Added CUSTOMv1/v2 sprite batches
