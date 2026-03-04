# The Elixir Optimizers: Development Status Report

**CS 175 - Project in Artificial Intelligence, UC Irvine, Winter 2026**
**Instructor:** Professor Roy Fox
**Team:** Alan Guo, Josh Talla, Lakshya Shrivastava
**Date:** February 25, 2026

---

## 1. Project Overview

The Elixir Optimizers is a reinforcement learning agent for Supercell's Clash Royale, a real-time strategy mobile game where players deploy cards from a rotating hand onto a shared arena to destroy opponent towers. The agent operates through screen capture and computer vision rather than game API access, making it applicable to the live game environment via Google Play Games on PC.

Our approach follows a two-stage training pipeline:

1. **Behavior Cloning (BC):** Supervised learning from human gameplay demonstrations to bootstrap a competent feature extractor and initial policy.
2. **Proximal Policy Optimization (PPO):** Reinforcement learning fine-tuning using the BC-pretrained feature extractor, with MaskablePPO to handle the large discrete action space.

The system is organized into six modules spanning the full pipeline from data collection through live autonomous play.

---

## 2. System Architecture

### 2.1 Module Pipeline

```
Recording --> Dataset Building --> BC Training --> PPO Fine-tuning --> Live Play
(click_logger)  (dataset_builder)  (bc_model)      (ppo_module)
                       |                |                |
                state_encoder     action_builder    state_encoder
```

### 2.2 Observation and Action Spaces

**Observation space** (Dict):
- `arena`: (32, 18, 6) float32 tensor encoding the game arena as a spatial grid
  - Channel 0: Unit class ID (155 classes, normalized)
  - Channel 1: Belonging (ally=1, enemy=-1, neutral=0)
  - Channel 2: Arena mask (valid playable cells)
  - Channel 3: Ally tower HP (normalized)
  - Channel 4: Enemy tower HP (normalized)
  - Channel 5: Spell/effect count
- `vector`: (23,) float32 scalar features (elixir, time, overtime flag, tower counts, card hand, card availability)

**Action space**: Discrete(2305)
- Actions 0-2303: Card placements (4 card slots x 576 grid cells, where each cell = row * 18 + col)
- Action 2304: No-op (do nothing this timestep)

### 2.3 Perception Pipeline

Game state is extracted from raw screen captures using:
1. **YOLOv8s object detection** (mAP50 = 0.804) for unit/tower localization across 155 unit classes
2. **Card classifier** for identifying the 4 cards currently in hand
3. **OCR/template matching** for elixir bar and timer readings
4. **StateEncoder** converts detections into the (32,18,6) + (23,) observation tensors

### 2.4 Test Coverage

| Module | Test Files | Test Functions | Status |
|--------|-----------|---------------|--------|
| state_encoder | 3 | 42 | All passing |
| action_builder | 2 | 46 | All passing |
| dataset_builder | 2 | 24 | All passing |
| ppo_module | 2 | 41 | All passing |
| **Total** | **9** | **153** | **All passing** |

---

## 3. Behavior Cloning (BC) - Status: Trained

### 3.1 Training Data

| Metric | Value |
|--------|-------|
| Training matches recorded | 40 |
| Total frames (after downsampling) | 5,366 |
| Action frames (card placements) | ~1,600 (30%) |
| No-op frames (after downsampling) | ~3,766 (70%) |
| No-op downsampling rate | 15% retention |
| Train/val split | 80/20 file-level |
| Data augmentation | Horizontal flip (2x training data) |
| Recording dates | Feb 21-23, 2026 |

File-level splitting ensures no data leakage between consecutive frames of the same match.

### 3.2 Model Architecture

**CRFeatureExtractor** (192-dim output):
- Arena branch: Class ID embedding (155 -> 8-dim) + 5 continuous channels -> 3-layer CNN (13->32->64->128) with BatchNorm, ReLU, MaxPool -> AdaptiveAvgPool -> 128-dim
- Vector branch: Card ID embeddings (4 slots, 9 classes -> 8-dim each) + 19 scalar features -> 2-layer MLP -> 64-dim
- Output: Concatenation -> 192-dim feature vector

**BCPolicy** (hierarchical 3-head decomposition):
- **Play head**: Binary classifier (play a card vs no-op) - addresses the 70:30 noop imbalance directly
- **Card head**: 4-way classifier (which card slot to play)
- **Position head**: FiLM-conditioned 576-way classifier (which grid cell), with per-card spatial modulation

**Total parameters:** 267,950

The hierarchical decomposition was a critical design choice. A flat 2305-way softmax collapsed to always-noop during early experiments because ~3,000 noop examples vastly outweighed ~547 action examples spread across 2,304 placement classes (0.24 examples per class on average). The hierarchical approach splits this into solvable sub-problems.

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 (cosine annealing to 1e-5) |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Epochs | 31 (early stopping on val F1) |
| Play loss | Weighted binary cross-entropy (noop=0.3, action=3.0) |
| Card loss | Standard cross-entropy |
| Position loss | Label-smoothed cross-entropy |
| Loss weights | 1.0 play + 1.0 card + 1.0 position |

### 3.4 Quantitative Results

#### Training Metrics (31 epochs)

| Metric | Initial (Epoch 1) | Best | Final (Epoch 31) |
|--------|--------------------|------|-------------------|
| Train loss (total) | 7.90 | 5.61 | 5.61 |
| Train play loss | 0.62 | 0.46 | 0.46 |
| Train card loss | 1.39 | 0.74 | 0.74 |
| Train position loss | 5.95 | 4.45 | 4.45 |
| Val loss | 7.08 | 6.85 (epoch 4) | 7.72 |

#### Best Validation Performance (Epoch 6, 0-indexed epoch 5)

| Metric | Value |
|--------|-------|
| **Action F1 score** | **0.322** |
| Action recall | 69.1% |
| Action precision | 21.0% |
| Overall accuracy | 49.2% |
| Noop accuracy | 57.2% |
| Card accuracy (action frames) | 22.0% |

#### Interpretation

- **Play head** converges well: The binary play/noop classifier reaches ~69% recall on action frames, meaning it correctly identifies most situations where a card should be played. The 57% noop accuracy shows it can also hold back when appropriate.

- **Card head** shows moderate learning: 22-35% accuracy on a 4-way classification with imbalanced card usage. Later epochs reach ~35% card accuracy, suggesting the model learns some card preferences.

- **Position head** is the bottleneck: Position loss dominates (4.45 of 5.61 total loss). With 576 possible grid positions and only ~1,600 action frames, each position receives very few examples. The model learns general spatial tendencies (e.g., "play near the bridge") but not precise placement.

- **Overfitting** is visible: Training loss decreases monotonically (7.90 -> 5.61) while validation loss increases after epoch 4 (6.85 -> 7.72+), indicating the model memorizes training placements rather than generalizing. This is expected with only 40 matches of data.

- **F1 plateau**: Best F1 of 0.322 occurs at epoch 6 and does not improve. The precision-recall tradeoff stabilizes with the model favoring recall (detecting actions) over precision (correct placement).

### 3.5 Live Inference Results

| Metric | Value |
|--------|-------|
| Live inference sessions | 23 |
| Total live frames | 1,594 |
| Avg inference latency | ~50ms per frame |
| Common execution reasons | mask_blocked, noop_chosen, action_executed |

During live play, the BC model demonstrates:
- Consistent ability to identify when to play cards (play head works)
- Preference for bridge-area placements (rows 14-18)
- Spatial collapse: predictions cluster in a narrow band rather than spreading across the full arena
- Card selection shows some deck-awareness but limited strategic variety

### 3.6 Evaluation Plots

Five evaluation plots were generated and saved to `eval_results/`:

1. **training_curves.png** - 4-panel view of total loss, decomposed losses (play/card/position), validation F1 progression, and recall vs precision over 31 epochs.

2. **dataset_stats.png** - 6-panel dataset analysis showing action/noop distribution, card slot usage frequency, elixir distribution at action time, and spatial heatmaps of human placements.

3. **model_evaluation.png** - 6-panel model assessment with play/noop confusion matrix, card confusion matrix, play probability distributions, predicted vs ground truth placement heatmaps, and a summary metrics table.

4. **live_inference.png** - 4-panel live deployment analysis showing execution reason breakdown, logit score distributions, inference latency histogram, and live placement spatial heatmap.

5. **model_architecture.png** - 2-panel architecture overview showing parameter distribution across components and a structural diagram of the feature extractor and action heads.

---

## 4. Reinforcement Learning (PPO) - Status: Implemented, Not Yet Trained

### 4.1 Implementation Status

The PPO module is **fully implemented and tested** (41/41 tests passing) but has **not yet been trained** on live gameplay. All infrastructure is ready to begin training runs.

### 4.2 Environment Design

**ClashRoyaleEnv** (Gymnasium-compatible):
- Wraps live screen capture, YOLO detection, state encoding, and action execution
- 500ms step interval (2 Hz decision frequency)
- Automatic phase detection (loading, battle, overtime, end)
- Action masking: only valid card placements are allowed (cards with enough elixir, valid grid positions)
- Episode termination on game end detection

### 4.3 Reward Function

| Component | Value | Trigger |
|-----------|-------|---------|
| Enemy tower destroyed | +10.0 | Enemy tower count decreases |
| Ally tower destroyed | -10.0 | Ally tower count decreases |
| Game win | +30.0 | Terminal: enemy king tower destroyed |
| Game loss | -30.0 | Terminal: ally king tower destroyed |
| Draw | -5.0 | Terminal: time expires, equal towers |
| Survival bonus | +0.02 | Every step |
| Elixir waste penalty | -0.05 | Elixir >= 9.5/10 (wasting generation) |
| Step clamp | +/-15.0 | Non-terminal steps capped |

The reward function operates on observation deltas -- comparing tower counts between consecutive steps to detect tower destruction events. An anomaly detector handles mid-episode game restarts (both tower counts suddenly jump up).

### 4.4 Training Plan

**Phase 1: Frozen Feature Extractor (~15 games)**
- Load BC-pretrained CRFeatureExtractor weights
- Freeze all feature extractor parameters
- Train only the PPO policy head and value head
- Learning rate: 1e-4
- Purpose: Learn value estimation and policy gradient updates without disrupting learned perceptual features

**Phase 2: Full Fine-tuning (~25+ games)**
- Unfreeze feature extractor
- Fine-tune entire network end-to-end
- Reduced learning rate: 3e-5
- Purpose: Adapt visual features to RL reward signal, potentially learning representations that BC didn't capture

### 4.5 PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | MaskablePPO (sb3-contrib) |
| Policy | MultiInputPolicy |
| Features dim | 192 (from CRFeatureExtractor) |
| Policy layers | [128, 64] |
| Value layers | [128, 64] |
| n_steps | 512 |
| Batch size | 64 |
| n_epochs | 10 |
| Clip range | 0.1 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.01 |
| Value function coefficient | 0.5 |
| Max gradient norm | 0.5 |

### 4.6 Visualization and Monitoring

An OpenCV-based real-time visualizer (`CVVisualizer`) has been implemented for monitoring PPO training and live inference:
- Side-by-side game frame + 32x18 color-coded arena grid
- Color coding: green=ally units, magenta=enemy units, blue=ally towers, red=enemy towers, yellow=spells
- Info panel showing channel values, game state, card hand, and performance metrics
- MP4 recording via cv2.VideoWriter for post-hoc analysis
- TensorBoard integration for training curves

---

## 5. Known Limitations and Challenges

### 5.1 Data Limitations
- **Small dataset**: 40 matches / 5,366 frames is limited for a 2305-class action space. More recordings would improve BC quality, especially for position learning.
- **Single player**: All demonstrations come from one player's style, limiting the diversity of strategies learned.
- **No-op dominance**: Even after 85% no-op downsampling, 70% of training frames are still no-ops.

### 5.2 Perception Limitations
- **YOLO domain gap**: mAP50 of 0.804 means ~20% of detections are incorrect, introducing noise into observations.
- **No belonging from YOLO**: Unit ownership (ally vs enemy) is inferred from Y-position heuristic, which fails when troops cross the river mid-battle.
- **Single-frame observations**: No temporal context -- the model cannot track troop movement trajectories or predict elixir generation timing.
- **Card classifier**: Only available during dataset building, not yet integrated into the live PPO environment.

### 5.3 Training Challenges
- **Position learning**: 576 grid positions with ~1,600 action examples means <3 examples per position on average. The position head learns spatial priors but not precise tactical placement.
- **Validation instability**: Val loss oscillates significantly, suggesting the small validation set (8 files) is not representative.
- **Live execution gap**: The model's 50ms inference time is fast enough, but perception pipeline latency and action execution delay add ~200ms per step.

---

## 6. Development Timeline

| Date | Milestone |
|------|-----------|
| Weeks 1-3 | YOLO training, state encoder, action builder implementation |
| Weeks 4-5 | Click logger, dataset builder, data collection (40 matches) |
| Week 6 | BC model design and training (hierarchical decomposition) |
| Week 7 | BC evaluation, live inference testing (23 sessions) |
| Week 8 | PPO module implementation, environment wrapper, reward function |
| Week 9 | PPO testing (41 tests), visualization tools, evaluation pipeline |
| Week 10 | PPO training runs (upcoming), final evaluation |

---

## 7. Next Steps

1. **PPO Training**: Execute Phase 1 (frozen extractor) and Phase 2 (full fine-tune) training runs on live Clash Royale matches. This is the immediate priority.

2. **Additional BC Data**: Record 20-40 more matches to improve position head learning and provide more diverse card usage examples.

3. **Temporal Features**: Add frame stacking or recurrent layers to capture troop movement patterns and elixir timing.

4. **Card Classifier Integration**: Port the card classifier from dataset_builder into the live PPO environment for more accurate hand observation.

5. **Belonging Detection**: Train a secondary classifier or add a belonging output to YOLO to improve ally/enemy discrimination during river crossings.

6. **PPO Reward Tuning**: Evaluate and iterate on reward component weights based on initial training runs. The elixir waste penalty and survival bonus may need adjustment.

---

## 8. Summary

The project has a complete end-to-end pipeline from screen capture to autonomous play. The BC model provides a functional starting point with F1=0.322, correctly identifying when to play cards ~69% of the time but struggling with precise spatial placement. The PPO infrastructure is fully implemented and tested, ready to begin reinforcement learning fine-tuning that should improve strategic decision-making beyond what imitation learning alone can achieve. The modular architecture with 153 passing tests across all modules provides a solid foundation for continued development.
