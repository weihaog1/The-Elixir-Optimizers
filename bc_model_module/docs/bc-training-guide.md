# BC Model Training Guide

Step-by-step documentation for training the Behavior Cloning model on Clash Royale gameplay data.

## Prerequisites

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.7.1 | Neural network training |
| numpy | any | Data loading |
| gymnasium | 1.2.3 | Required by StateEncoder (encoder_constants) |

All installed in the project conda environment. No additional pip installs needed.

### Training Data

28 `.npz` files in `data/bc_training/`, produced by `dataset_builder_module/process_recordings.py`:

| Statistic | Value |
|-----------|-------|
| Total sessions | 28 (20 + 8 new) |
| Total frames | 25,908 raw |
| Frames after noop downsampling | 4,429 |
| Card placement actions | 652 |
| Noop frames | 3,777 (85.3%) |
| Action frames | 652 (14.7%) |

Each `.npz` contains:
- `obs_arena`: `(N, 32, 18, 6)` float32 - spatial arena grid
- `obs_vector`: `(N, 23)` float32 - scalar features (elixir, time, cards, towers)
- `actions`: `(N,)` int64 - action indices in [0, 2304]
- `masks`: `(N, 2305)` bool - valid action masks
- `timestamps`: `(N,)` float64

## Architecture

### Why Hierarchical Decomposition

The original flat 2305-way softmax classifier **completely fails** with this data:
- 652 action examples spread across 2304 action classes = 0.28 examples per class average
- The noop class (index 2304) has 3777 examples
- Even with extreme class weighting (50x), the model always predicts noop (0% action recall)

The hierarchical approach decomposes the intractable single classification into three solvable sub-problems:

| Head | Classes | Train Examples/Class | Purpose |
|------|---------|---------------------|---------|
| Play head | 2 (noop vs play) | ~1889 noop, ~274 play | When to play a card |
| Card head | 4 (card slots) | ~69 per card | Which card to play |
| Position head | 576 (grid cells) | sparse | Where to play it |

### Model Architecture

```
CRFeatureExtractor (105,384 params)
  Arena branch: Embedding(156,8) + 3-layer CNN -> 128 features
  Vector branch: Embedding(9,8) + 2-layer MLP -> 64 features
  Output: 192 features

Shared Trunk (49,408 params)
  Linear(192, 256) + ReLU + Dropout(0.2)

Play Head (514 params)
  Linear(256, 2) -> binary play/noop

Card Head (1,028 params)
  Linear(256, 4) -> which card slot

Position Head (148,032 params)
  Linear(256, 576) -> which grid cell

Total: 304,366 parameters
```

### Training Loss

Three separate cross-entropy losses:
1. **Play loss**: Weighted binary CE on ALL frames (play_weight addresses noop:action imbalance)
2. **Card loss**: Standard CE on action frames only
3. **Position loss**: Standard CE on action frames only

Total loss = play_loss + card_loss + position_loss

### Early Stopping Criterion

**Action F1 score** (harmonic mean of precision and recall):
- Precision = true_positives / (true_positives + false_positives)
- Recall = true_positives / (true_positives + false_negatives)
- F1 = 2 * precision * recall / (precision + recall)

F1 is critical because:
- Pure recall maximization leads to "predict everything as action" (recall=1.0, precision=0.14)
- Pure precision maximization leads to "predict nothing as action" (recall=0.0)
- F1 balances both, saving the checkpoint that best identifies real actions

## Step-by-Step Training Process

### Step 1: Verify Data

```bash
# Check .npz files exist
ls data/bc_training/*.npz | wc -l
# Expected: 28

# Verify shapes
python -c "
import numpy as np
f = np.load('data/bc_training/match_20260221_203354.npz')
for k, v in f.items():
    print(f'{k}: {v.shape} {v.dtype}')
"
```

### Step 2: Run Training

```bash
python bc_model_module/train_model.py \
    --data_dir data/bc_training/ \
    --output_dir models/bc/ \
    --epochs 100 \
    --batch_size 32 \
    --patience 25 \
    --lr 1e-4 \
    --play_weight 8.0 \
    --seed 42
```

### Step 3: Monitor Output

The training loop prints per-epoch metrics:

```
Epoch  19/100 | Loss: 5.966 (p=0.600 c=1.299 g=4.067) | F1: 0.324 R/P: 0.695/0.212 Card: 0.342 | Noop: 0.589 | LR: 9.22e-05
  -> Saved best (F1=0.324, R=0.695, P=0.212, tp=73, fp=272, fn=32)
```

Column descriptions:
- **Loss**: Total (play + card + position component losses)
- **F1**: Action F1 score (early stopping criterion)
- **R/P**: Action recall / action precision
- **Card**: Card selection accuracy (among correctly predicted actions)
- **Noop**: Noop accuracy (fraction of noop frames correctly identified)
- **LR**: Current learning rate

### Step 4: Verify Output Files

```bash
ls -la models/bc/
# Expected:
#   best_bc.pt               (~1.2 MB) - Full policy checkpoint
#   bc_feature_extractor.pt  (~423 KB) - Feature extractor only (for PPO)
#   training_log.json        (~12 KB)  - Per-epoch metrics
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | (required) | Directory containing .npz files |
| `--output_dir` | `models/bc/` | Output directory for checkpoints |
| `--epochs` | 100 | Maximum training epochs |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 3e-4 | Initial learning rate (AdamW) |
| `--patience` | 15 | Early stopping patience (epochs) |
| `--play_weight` | 5.0 | Weight for "play" class in play/noop head |
| `--seed` | 42 | Random seed for reproducibility |

### Recommended Configurations

**Default (balanced):**
```bash
python bc_model_module/train_model.py --data_dir data/bc_training/ --output_dir models/bc/
```

**High recall (catch more actions, more false positives):**
```bash
python bc_model_module/train_model.py --data_dir data/bc_training/ --output_dir models/bc/ --play_weight 15.0
```

**Stable training (lower LR, smaller batch):**
```bash
python bc_model_module/train_model.py --data_dir data/bc_training/ --output_dir models/bc/ --lr 1e-4 --batch_size 32 --patience 25 --play_weight 8.0
```

## Train/Validation Split

The split is performed at the **file level** (game level), not frame level, to prevent temporal data leakage:

- 28 total .npz files (one per game session)
- 80% train: 22 games (~3,663 frames)
- 20% validation: 6 games (~766 frames)
- Files are shuffled with the seed before splitting

This means the validation games are completely unseen during training. No consecutive frames from the same game appear in both train and val.

## Training Results

### Best Checkpoint (Epoch 19)

| Metric | Value |
|--------|-------|
| Action F1 | **0.324** |
| Action Recall | 0.695 (catches 70% of real card placements) |
| Action Precision | 0.212 (21% of predicted actions are real) |
| Card Accuracy | 0.342 (above 25% random baseline) |
| Noop Accuracy | 0.589 |
| True Positives | 73 (of 105 val actions) |
| False Positives | 272 |
| False Negatives | 32 |

### Loss Progression

| Loss Component | Epoch 1 | Epoch 19 | Epoch 44 | Trend |
|---------------|---------|----------|----------|-------|
| Play (p) | 0.678 | 0.600 | 0.518 | Steady decrease |
| Card (c) | 1.394 | 1.299 | 0.782 | Strong decrease |
| Position (g) | 6.202 | 4.067 | 3.299 | Strong decrease |
| Total | 8.273 | 5.966 | 4.599 | Strong decrease |

Key observations:
- **Card loss decreasing strongly** (1.39 -> 0.78): The card head IS learning meaningful card selection
- **Position loss decreasing** (6.20 -> 3.30): Position learning is happening but sparse
- **Play loss decreasing slowly** (0.68 -> 0.52): The binary decision is harder to stabilize

### Known Limitations

1. **Low precision (21%)**: The model over-predicts actions, producing ~3 false positives per true positive. For live play, use confidence thresholding.
2. **Play head instability**: The play/noop decision oscillates between epochs. F1-based early stopping captures the best balance point.
3. **Card accuracy modest (34%)**: Better than random (25%) but limited by data volume. 652 actions across 4 cards = ~163 per card.
4. **Position accuracy untested directly**: The position head loss decreases but exact position match is rare due to 576-cell resolution.
5. **Data volume**: 28 games with 652 total card placements is very small for a neural network. More recorded games would significantly improve all metrics.

## Architecture Evolution

The training process went through several iterations to address the extreme class imbalance:

### Attempt 1: Flat 2305-way Softmax + Weighted CE
- **Result**: 0% action recall at all weight ratios (3x, 10x, 100x, 5000x)
- **Root cause**: 652 actions / 2304 classes = 0.28 examples per class. Noop always wins argmax.

### Attempt 2: Flat Softmax + Focal Loss (gamma=2.0)
- **Result**: Still 0% action recall
- **Root cause**: Focal loss down-weights easy examples but can't overcome the structural sparsity

### Attempt 3: Hierarchical Decomposition (Final)
- **Result**: F1=0.324, recall=69.5%, precision=21.2%
- **Key insight**: Splitting into play/card/position gives each head a tractable number of classes with adequate training examples

## Using the Trained Model

### Loading for Inference

```python
from src.bc.bc_policy import BCPolicy

policy = BCPolicy.load("models/bc/best_bc.pt")
action = policy.predict_action(obs, mask)
```

### PPO Transfer

```python
import torch

# Load feature extractor weights
fe_weights = torch.load("models/bc/bc_feature_extractor.pt")

# Load into MaskablePPO
ppo.policy.features_extractor.load_state_dict(fe_weights)

# Freeze during initial PPO training
for p in ppo.policy.features_extractor.parameters():
    p.requires_grad = False
ppo.learn(total_timesteps=500_000)

# Unfreeze with lower LR
for p in ppo.policy.features_extractor.parameters():
    p.requires_grad = True
ppo.learning_rate = 3e-5
ppo.learn(total_timesteps=500_000)
```

### Live Game Inference

```bash
python bc_model_module/run_live.py \
    --model-path models/bc/best_bc.pt \
    --capture-region 0,0,540,960 \
    --dry-run
```

## Improving Results

Ordered by expected impact:

1. **More training data**: Record 50+ additional games. The card and position heads are data-limited.
2. **Confidence thresholding**: At inference, only execute actions when play probability > 0.7 to reduce false positives.
3. **Coarser position grid**: Reduce from 576 cells to ~100 zones for denser training signal.
4. **Label smoothing**: Spread position probability to neighboring cells (the exact cell doesn't matter much).
5. **Temporal features**: Add frame-to-frame diff or LSTM to capture elixir generation and troop movement patterns.
