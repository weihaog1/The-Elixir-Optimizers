# YOLOv8s v12 Comprehensive Evaluation Report

Model: best_yolov8s_50epochs_fixed_pregen_set.pt (YOLOv8s, 11.2M params)
Training: 50 epochs on pre-generated synthetic data, imgsz=960, background15 only
Validation: 1,388 human-labeled KataCR images

---

## 1. Executive Summary

The v12 model is a major improvement over our previous YOLOv8n. It eliminates the "the-log" hallucination problem entirely, detects 49% more objects per frame, and covers 85 unique classes vs 57. However, mAP50 plateaued at 0.804 (below our 0.85 target) by epoch 10, with the remaining 40 epochs contributing almost nothing. The bottleneck is the synthetic-to-real domain gap, not training duration.

| Metric | Old YOLOv8n (val-trained) | New YOLOv8s (synthetic) | Change |
|--------|--------------------------|------------------------|--------|
| Val mAP50 | 0.944 (misleading) | 0.804 | N/A (different data) |
| Avg dets/frame (video) | 20.3 | 30.2 | +49% |
| "the-log" hallucinations | 50 | 0 | -100% |
| Unique classes detected | 57 | 85 | +49% |
| Inference FPS (M1 Pro) | 18.9 | 15.3 | -19% (acceptable) |

---

## 2. Training Analysis

### Convergence
- mAP50 reached 0.797 by epoch 10 and flatlined at 0.804 for the remaining 40 epochs
- mAP50-95 continued slow improvement (0.520 -> 0.567), meaning box precision improved even when detection recall did not
- Training losses still decreasing at epoch 50, but validation losses plateaued -- classic domain gap signature
- No overfitting detected (validation loss never increased)

### Domain Gap (the core limitation)
Train/val loss ratios at epoch 50:
- Box loss: 3.9x gap
- Classification loss: 4.7x gap (largest -- appearance differences are the main problem)
- DFL loss: 2.0x gap

The model learns synthetic sprites perfectly but struggles to transfer to real gameplay screenshots. Classification gap being widest means the synthetic sprites don't look enough like real in-game units.

### Key Insight
More epochs will NOT fix this. The model has extracted all it can from the current synthetic data distribution. Improvements must come from better data or architecture changes.

---

## 3. Gameplay Video Performance

### What Works Well
- Towers and structures: 0.90-0.97 confidence, near-perfect detection
- Common troops (witch, flying-machine, baby-dragon): 0.78-0.85 confidence
- UI elements (clock, elixir, bars): reliable detection
- Zero "the-log" hallucinations (was 50 with old model)
- Zero "dagger-duchess-tower" false positives (was 20 with old model)

### What Needs Improvement
- Small troops (skeletons, spirits, goblins): detected but at lower confidence (0.40-0.65)
- Some card icons in hand tray detected as troops (minor inflation)
- Rare false positives at low confidence (princess 0.32, pekka 0.31) -- 1-2 per video, negligible

---

## 4. Per-Class Performance vs Meta Deck Coverage

### Detection Tier Distribution
| Tier | AP50 Range | Classes | % |
|------|-----------|---------|---|
| Strong | >= 0.90 | 73 | 47% |
| Adequate | 0.70-0.89 | 32 | 21% |
| Weak | 0.40-0.69 | 20 | 13% |
| Failing | < 0.40 | 14 | 9% |

### Critical Meta Failures (high usage, AP50 < 0.40)
| Class | AP50 | Meta Decks | Sprites | Root Cause |
|-------|------|------------|---------|------------|
| barbarian-barrel | 0.396 | 8/20 | 8 | Too few sprites |
| miner | 0.237 | 3/20 | 46 | Underground emergence doesn't match static sprite |
| zap | 0.245 | 3/20 | 9 | Brief visual effect, too few sprites |
| arrows | 0.190 | 1/20 | 6 | Brief visual effect, too few sprites |
| bomb-tower | 0.329 | 3/20 | 14 | Domain gap |

### Weak But Meta-Important (AP50 0.40-0.70)
| Class | AP50 | Meta Decks | Root Cause |
|-------|------|------------|------------|
| skeleton | 0.670 | 7/20 | Very small sprite, size is the limit |
| ice-spirit | 0.644 | 6/20 | Small sprite |
| electro-spirit | 0.560 | 3/20 | Small sprite |
| tesla | 0.420 | 2/20 | Underground/above states |
| barbarian/evo | ~0.47 | 1/20 | Cross-class confusion |

### Confusion Clusters
- **Wizard family**: wizard (0.152) vs ice-wizard (0.508) vs electro-wizard (0.887) -- wizard is essentially invisible
- **Barbarian family**: barbarian (0.472) vs barbarian-evo (0.474) vs elite-barbarian (0.665) -- all confused
- **Spirit family**: ice-spirit (0.644), electro-spirit (0.560), fire-spirit (0.878), heal-spirit (0.951) -- color-distinctive ones do better
- **Knight family**: knight (0.916) vs knight-evolution (0.122) -- evolution variant catastrophically fails

### Our Deck Coverage (hog/royal-hogs "pigs" deck)
- royal-hog: 0.995 -- excellent
- hog-rider: 0.869 -- good
- musketeer: 0.979 -- excellent
- fireball: 0.977 -- excellent
- the-log: 0.802 -- adequate
- skeleton: 0.670 -- weak (cycle card, will miss some)
- ice-spirit: 0.644 -- weak (cycle card, will miss some)

Our own deck is mostly well-covered. The gap is in detecting opponent cards.

---

## 5. Improvement Recommendations (Ranked by Impact/Effort)

### Tier 1: High Impact, Moderate Effort

**1. Fine-tune on real gameplay frames (estimated +3-5% mAP50)**
- The domain gap is the #1 bottleneck. Even 50-100 labeled real frames would help enormously.
- Method: Take 50 frames from our gameplay video, auto-label with current model at low conf, manually correct, fine-tune for 10-20 epochs.
- This directly attacks the 4.7x classification gap.
- Risk: Could overfit to our specific arena/phone. Mitigate by mixing with synthetic data (90% synthetic, 10% real).

**2. Lower confidence threshold for inference (immediate, no training needed)**
- F1-optimal threshold is 0.765 (very conservative). For RL gameplay where missing a troop is worse than a false positive, use conf=0.3-0.4.
- Trades precision for recall -- catches more troops at cost of occasional false detections.
- Combine with class-specific thresholds: high conf for towers (0.7), low conf for small troops (0.25).

**3. Increase sprite diversity for failing classes (estimated +2-3% on those classes)**
- barbarian-barrel (8 sprites), zap (9), arrows (6), goblin-cage (4) -- all critically low.
- Options: Extract more sprites from gameplay video using SAM, or augment existing sprites with more aggressive color/scale/rotation transforms.
- The generator's inverse frequency weighting will automatically use new sprites.

### Tier 2: Medium Impact, Higher Effort

**4. Multi-resolution inference (estimated +5-10% on small objects)**
- Run inference at two scales: imgsz=960 for large objects, imgsz=1280 for small objects (skeletons, spirits, goblins).
- Merge detections with NMS. This is KataCR's approach (they split by object size).
- Cost: 2x inference time (~8 FPS). Acceptable if we only do high-res on the arena center region.

**5. Address wizard/barbarian confusion clusters**
- The wizard family has catastrophic confusion (wizard AP50=0.152 despite 41 sprites).
- Investigate: Are wizard sprites visually distinct from ice-wizard in the training data? If not, consider merging into a single "wizard-type" class and using color-based post-processing to distinguish.
- For barbarians: Consider merging barbarian + barbarian-evolution into one class if evolution distinction isn't needed for RL decisions.

**6. Spell detection via temporal context**
- Spells like zap, arrows, freeze are brief visual effects. Single-frame detection will always be unreliable.
- For RL: Detect spell effects by monitoring area damage (units disappearing/taking damage in a region) rather than detecting the spell visual itself.
- This is an architecture change, not a model change.

### Tier 3: Lower Priority

**7. More training data (marginal impact on mAP50)**
- Current: 20k pre-generated images per epoch equivalent. More images won't help -- the model already converged.
- However, if we fix the domain gap (fine-tuning on real data), then more synthetic data becomes useful again as a data augmentation source.

**8. Model architecture upgrade (YOLOv8m or YOLOv8l)**
- YOLOv8s has 11.2M params. YOLOv8m (25.9M) or YOLOv8l (43.7M) could help with the 155-class discriminaton task.
- Cost: Slower inference. YOLOv8m at imgsz=960 is ~10 FPS on M1 Pro (borderline for 10 FPS target).
- Only worth it after domain gap is addressed.

**9. KataCR-style model split (2-3 models by object size)**
- Split classes into small (skeleton, spirit, goblin), medium (troops), large (buildings, towers).
- Each model specializes. Combined mAP improves significantly in KataCR's results (+1.2 mAP50, +4.1 mAP-small).
- Cost: 2-3x inference time, more complex pipeline.
- Only needed if single-model performance is still insufficient after other improvements.

---

## 6. Recommended Next Steps (Prioritized Action Plan)

### Immediate (no retraining)
1. Lower inference confidence to 0.35 and evaluate gameplay video again
2. Add class-specific confidence thresholds in the pipeline

### Short-term (hours of work)
3. Label 50-100 real gameplay frames (semi-auto with current model + manual correction)
4. Fine-tune v12 model on mixed dataset (synthetic + real) for 10-20 epochs
5. Extract more sprites for barbarian-barrel, zap, arrows from gameplay video using SAM

### Medium-term (days of work)
6. Implement dual-resolution inference (960 + 1280) with NMS merge
7. Investigate wizard/barbarian confusion -- potentially merge similar classes
8. Add temporal spell detection logic in the pipeline layer

### Long-term (if needed)
9. Split into 2 models by object size
10. Upgrade to YOLOv8m if 155-class discrimination remains insufficient

---

## 7. Assessment for RL Agent Pipeline

For the fixed-deck-vs-any-deck RL use case:

**What the model can already do:**
- Detect our own troops reliably (royal-hog 0.995, hog-rider 0.869, musketeer 0.979)
- Detect towers and HP bars for game state tracking
- Detect most opponent troops from common meta decks (balloon 0.995, pekka 0.870, golem 0.769)
- Run at 15.3 FPS -- sufficient for RL decision-making at 2-5 Hz

**What the RL agent will struggle with:**
- Missing cycle cards (skeleton 0.670, ice-spirit 0.644) -- affects elixir counting
- Missing spells (zap 0.245, arrows 0.190) -- can't react to opponent spells
- Missing miner (0.237) -- can't detect this win condition appearing near towers
- Barbarian-barrel (0.396) -- very common defensive card, nearly invisible

**Bottom line:** The model is good enough to start RL training on basic gameplay scenarios (troop placement, tower targeting). For competitive play against diverse meta decks, the failing classes (especially miner, barbarian-barrel, and spell detection) need to be addressed through fine-tuning on real data and temporal context.
