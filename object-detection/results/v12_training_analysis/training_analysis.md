# YOLOv8s Pregen v12 Training Analysis

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8s (11.2M params) |
| Image Size | 960 |
| Batch Size | 16 |
| Epochs | 50 |
| Optimizer | auto (SGD) |
| Initial LR | 0.01 |
| Final LR factor | 0.01 |
| Patience | 15 (early stopping) |
| Warmup Epochs | 3.0 |
| Mosaic | 0.0 (disabled) |
| Close Mosaic | 0 |
| Augmentations | degrees=5, scale=0.5, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4, randaugment |
| Data | Pre-generated synthetic dataset |
| Seed | 42 |
| AMP | Enabled |

Notable: Mosaic and mixup are disabled. The `resume` field points to last.pt, indicating this run was resumed at some point (time jumps from epoch 8 to epoch 9 confirm a restart mid-training).

## Final Metrics (Epoch 50)

| Metric | Value |
|--------|-------|
| mAP50 | 0.804 |
| mAP50-95 | 0.567 |
| Precision | 0.822 |
| Recall | 0.771 |
| F1 (all classes at conf=0.765) | 0.76 |
| Train Box Loss | 0.340 |
| Train Cls Loss | 0.185 |
| Train DFL Loss | 0.793 |
| Val Box Loss | 1.330 |
| Val Cls Loss | 0.877 |
| Val DFL Loss | 1.607 |

## Loss Curve Analysis

### Training Loss
All three training losses (box, cls, dfl) show smooth, continuous downward trends across all 50 epochs with no signs of plateau. The box loss dropped from 0.934 to 0.340, classification loss from 1.189 to 0.185, and DFL loss from 0.988 to 0.793. The consistent decline suggests the model could benefit from additional training epochs.

### Validation Loss
- **Box loss**: Dropped sharply from 1.536 (epoch 1) to ~1.33 by epoch 38, then plateaued. Good convergence behavior.
- **Cls loss**: Dropped from 1.083 to ~0.86 by epoch 15, then largely plateaued with minor fluctuations. Suggests classification capacity is near its limit given the domain gap between synthetic training and real validation data.
- **DFL loss**: Erratic behavior throughout training, ranging from 1.39 to 1.68 with no clear improvement trend. Started at 1.39, peaked at 1.68 around epoch 13, and ended at 1.607. This indicates the model struggles with precise bounding box localization on validation data.

### Domain Gap Evidence
The large gap between training and validation losses is expected since the model trains on synthetic data and validates on real images. The train/val ratios at epoch 50:
- Box: 0.340 / 1.330 = 3.9x gap
- Cls: 0.185 / 0.877 = 4.7x gap
- DFL: 0.793 / 1.607 = 2.0x gap

The classification gap is the widest, suggesting that appearance differences between synthetic sprites and real in-game screenshots are the primary challenge.

## mAP Progression

| Epoch | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|----------|-----------|--------|
| 1 | 0.675 | 0.414 | 0.760 | 0.623 |
| 10 | 0.797 | 0.520 | 0.815 | 0.759 |
| 20 | 0.796 | 0.532 | 0.822 | 0.760 |
| 30 | 0.804 | 0.550 | 0.815 | 0.765 |
| 40 | 0.804 | 0.563 | 0.814 | 0.775 |
| 50 | 0.804 | 0.567 | 0.822 | 0.771 |

Key observations:
- **mAP50 plateaued around epoch 10** at ~0.80 and barely moved for the remaining 40 epochs.
- **mAP50-95 continued to improve** slowly throughout training (0.52 -> 0.567), meaning bounding box precision improved even when detection recall did not.
- **Precision was consistently higher than recall** (~0.82 vs ~0.77), meaning the model is conservative - it misses some objects rather than producing false positives.

## Convergence Assessment

The model has effectively converged for mAP50, which has been stable at ~0.804 since epoch 10. However:
- mAP50-95 was still improving at a rate of about +0.004 per 10 epochs at the end of training
- Training losses were still decreasing steadily
- No signs of overfitting (validation loss did not increase)

More epochs might yield marginal mAP50-95 improvement but are unlikely to push mAP50 significantly beyond 0.805. The mAP50 ceiling appears to be a fundamental limitation of the synthetic-to-real domain gap rather than insufficient training.

## Confusion Matrix Observations

The normalized confusion matrix shows:
- Strong diagonal pattern across most classes, indicating correct classification
- Most confusions are mild (off-diagonal values appear low)
- The "background" column (rightmost) captures missed detections - some classes have notable miss rates
- The king-tower class has high absolute counts (visible in the non-normalized matrix as the darkest cell), consistent with it appearing in every image
- Most individual class confusions are between visually similar troops or between troops and the background

## F1 and PR Curves

- **F1 curve**: Optimal confidence threshold is 0.765, achieving F1=0.76 across all classes. The wide spread of individual class curves (from ~0.2 to near 1.0) indicates highly variable per-class performance.
- **PR curve**: Overall mAP@0.5 = 0.805. Many classes cluster near perfect precision-recall in the upper-right, but a long tail of classes with poor AP pulls the average down. Several classes have near-zero area under the PR curve.

## Validation Prediction Quality

Visual inspection of val_batch0_pred and val_batch1_pred shows:
- Queen-tower and king-tower detections are reliable with high confidence (0.9-1.0)
- Tower bars (king-tower-bar) are detected well
- Troops like skeleton, musketeer, spear-goblin are detected at moderate confidence (0.4-0.8)
- Some crowded scenes show overlapping detections, but no egregious false positive clusters
- Lightning and spell effects are detected but sometimes at lower confidence

## Key Findings

1. **mAP50 of 0.804 falls short of the 0.85 target** outlined in CLAUDE.md. The gap is ~5 percentage points.

2. **The synthetic-to-real domain gap is the primary bottleneck**, as evidenced by the 3.9-4.7x train/val loss ratios and the mAP50 plateau at epoch 10.

3. **mAP50-95 of 0.567 is reasonable** for an object detection model with 155 classes and a synthetic-to-real gap, indicating decent bounding box localization.

4. **No overfitting detected** - validation losses stabilize but do not increase, suggesting the model generalizes as well as the data allows.

5. **Erratic DFL validation loss** suggests localization precision on real images is unstable, possibly due to differences in object scales or aspect ratios between synthetic and real data.

6. **High precision (0.822) relative to recall (0.771)** - the model is conservative, which may translate to fewer false positives in gameplay inference at the cost of missing some objects.

7. **Large per-class performance variance** (F1 range from ~0.2 to ~1.0) means some classes work very well while others are essentially undetectable. This is likely correlated with how well each class's synthetic sprites match their in-game appearance.

8. **Training was resumed at least once** (time jump between epochs 8-9), but this did not negatively impact convergence.

## Recommendations

1. **Domain-specific augmentation**: The erasing=0.4 and randaugment are good, but adding more augmentations that simulate in-game visual effects (shadows, transparency, particle effects) could help close the domain gap.

2. **Longer training unlikely to help mAP50**: Since mAP50 plateaued at epoch 10, additional epochs will not meaningfully improve detection recall. Focus efforts on data quality instead.

3. **Investigate worst-performing classes**: Identify which classes have near-zero AP and determine if their synthetic sprites are poor representations of in-game appearance.

4. **Consider fine-tuning on a small set of real images**: Even 50-100 labeled real gameplay frames could help bridge the domain gap for the worst-performing classes.

5. **Test optimal confidence threshold**: The F1-optimal threshold of 0.765 is quite high. For gameplay inference where recall matters, a lower threshold (0.3-0.5) with NMS tuning may give better detection coverage.
