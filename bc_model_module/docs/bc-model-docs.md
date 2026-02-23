# BC Model Module - Usage Guide

This document covers training, evaluation, live game inference, and PPO transition for the behavior cloning model.

---

## 1. Overview

The BC model learns to predict card placements from recorded human gameplay. It takes encoded game observations (arena grid + scalar features) and outputs a probability distribution over 2,305 discrete actions (4 cards x 576 grid cells + no-op).

### Architecture

```
Arena (32,18,6) ──> Embedding(156,8) + 5 channels ──> 3-layer CNN ──> 128 features
                                                                          |
Vector (23,) ──> Card Embedding(9,8) + 19 scalars ──> 2-layer MLP ──> 64 features
                                                                          |
                                                           Concatenate ──> 192 features
                                                                          |
                                                   Linear(192,256) + ReLU + Dropout(0.2)
                                                                          |
                                                            Linear(256,2305) ──> logits
```

**Total parameters:** ~140K

---

## 2. Prerequisites

**Python packages:**
- `torch` (PyTorch)
- `numpy`

**Internal dependencies:**
- `src.encoder.encoder_constants` (NUM_CLASSES, ACTION_SPACE_SIZE, NOOP_ACTION, NUM_DECK_CARDS)

**Training data:**
- `.npz` files produced by `dataset_builder_module` (DatasetBuilder)
- Each file contains: `obs_arena (N,32,18,6)`, `obs_vector (N,23)`, `actions (N,)`, `masks (N,2305)`

---

## 3. Training

### CLI Usage

```bash
python -m src.bc.train_bc --data_dir data/bc_training/ --output_dir models/bc/
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | (required) | Directory containing .npz files |
| `--output_dir` | `models/bc/` | Output directory for checkpoints |
| `--epochs` | 100 | Maximum training epochs |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 3e-4 | Initial learning rate (AdamW) |
| `--patience` | 10 | Early stopping patience |
| `--seed` | 42 | Random seed |
| `--noop_weight` | 0.3 | Class weight for no-op action |
| `--action_weight` | 3.0 | Class weight for card placements |

### Programmatic Usage

```python
from pathlib import Path
from src.bc import BCTrainer, TrainConfig

npz_paths = sorted(Path("data/bc_training/").glob("*.npz"))

config = TrainConfig(
    epochs=100,
    batch_size=64,
    lr=3e-4,
    patience=10,
)

trainer = BCTrainer(config)
history = trainer.train(npz_paths, "models/bc/")

print(f"Best val loss: {history['best_val_loss']:.4f}")
print(f"Best epoch: {history['best_epoch'] + 1}")
```

### Training Details

- **Loss:** Weighted cross-entropy (noop=0.3, actions=3.0) to handle class imbalance
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **LR Schedule:** Cosine annealing over total epochs
- **Gradient clipping:** Max norm 1.0
- **Early stopping:** Patience=10 epochs on validation loss
- **Data split:** File-level 80/20 (not frame-level, to prevent leakage)
- **Device:** Automatic CUDA detection

### Expected Output

Training produces three files in `output_dir`:

| File | Description |
|------|-------------|
| `best_bc.pt` | Full BC policy checkpoint (best validation loss) |
| `bc_feature_extractor.pt` | Feature extractor weights only (for PPO transfer) |
| `training_log.json` | Per-epoch metrics: train_losses, val_losses, val_accuracies |

### Example Training Output

```
Using device: cuda
Train: 2160 frames (1512 noop, 648 action)
Val:   540 frames
Model parameters: 142,337

Starting training for up to 100 epochs...
----------------------------------------------------------------------
Epoch   1/100 | Train Loss: 4.2135 | Val Loss: 3.8721 | Val Acc: 0.652 | LR: 3.00e-04
Epoch   2/100 | Train Loss: 3.1247 | Val Loss: 3.0158 | Val Acc: 0.701 | LR: 2.99e-04
...
Epoch  47/100 | Train Loss: 1.2453 | Val Loss: 1.8912 | Val Acc: 0.312 | LR: 8.21e-05

Early stopping at epoch 47 (no improvement for 10 epochs)
----------------------------------------------------------------------
Training complete. Best val loss: 1.7234 at epoch 37
Training log saved to models/bc/training_log.json
```

---

## 4. Evaluation

### Loading a Trained Model

```python
from src.bc import BCPolicy

policy = BCPolicy.load("models/bc/best_bc.pt")
# policy is in eval mode
```

### Computing Metrics

```python
import torch
from src.bc import BCDataset
from pathlib import Path

# Load validation data
val_paths = [Path("data/bc_training/match_25.npz")]
val_dataset = BCDataset(val_paths)

# Forward pass
policy.eval()
with torch.no_grad():
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        obs = {
            "arena": sample["arena"].unsqueeze(0),
            "vector": sample["vector"].unsqueeze(0),
        }
        mask = sample["mask"]
        action = sample["action"]

        logits = policy(obs).squeeze(0)  # (2305,)

        # Apply mask
        logits[~mask] = float("-inf")
        pred = logits.argmax().item()

        # Metrics
        correct = (pred == action.item())
        pred_card = pred // 576 if pred < 2304 else -1
        true_card = action.item() // 576 if action.item() < 2304 else -1
        card_correct = (pred_card == true_card)
```

### Key Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Overall accuracy | Exact action match | 25-40% |
| Action-only accuracy | Correct among non-noop frames | 15-30% |
| Card selection accuracy | Right card (ignore placement) | 45-65% |
| Top-5 accuracy | Expert action in top 5 predictions | 50-70% |
| No-op F1 | Precision/recall balance on no-op | 75%+ |

**Why accuracy seems low:** 2,305 classes is enormous. Random chance = 0.04%. Even 25% accuracy is ~625x better than random. Being off by one grid cell is often equally valid gameplay.

---

## 5. Live Game Inference

### Pipeline

```
Screen Capture (mss at 2 FPS)
       |
       v
EnhancedStateBuilder (YOLO + OCR + CardPredictor)
       |
       v
StateEncoder.encode() -> obs dict {arena: (32,18,6), vector: (23,)}
StateEncoder.action_mask() -> mask (2305,)
       |
       v
BCPolicy.predict_action(obs, mask) -> action_idx
       |
       v
action_to_placement(action_idx) -> (card_id, col, row) or None (no-op)
cell_to_norm(col, row) -> (x_norm, y_norm)
       |
       v
PyAutoGUI click execution
```

### Inference Script Example

```python
import time
import torch
import numpy as np

from src.bc import BCPolicy
from src.encoder import StateEncoder
from src.encoder.coord_utils import action_to_placement, cell_to_norm

# Load model
policy = BCPolicy.load("models/bc/best_bc.pt")
policy.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = policy.to(device)

encoder = StateEncoder()

def run_inference(game_state):
    """Run one inference step. Returns (card_id, x_norm, y_norm) or None."""
    obs = encoder.encode(game_state)
    mask = encoder.action_mask(game_state)

    # Convert to tensors with batch dim
    obs_tensor = {
        "arena": torch.from_numpy(obs["arena"]).float().unsqueeze(0).to(device),
        "vector": torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(device),
    }
    mask_tensor = torch.from_numpy(mask).bool().to(device)

    action_idx = policy.predict_action(obs_tensor, mask_tensor)

    result = action_to_placement(action_idx)
    if result is not None:
        card_id, col, row = result
        x_norm, y_norm = cell_to_norm(col, row)
        return card_id, x_norm, y_norm
    return None  # no-op
```

### Timing Budget (500ms per frame at 2 FPS)

| Component | Time |
|-----------|------|
| mss screen capture | ~5ms |
| YOLO inference | ~65ms |
| OCR (timer/elixir) | ~50ms |
| Card classifier | ~10ms |
| StateEncoder.encode() | ~2ms |
| BC forward pass | ~5ms |
| PyAutoGUI clicks | ~100ms |
| **Total** | **~237ms** (well within 500ms) |

### Safety Measures

- **Confidence threshold:** Only act if `max_logit > threshold`. Below threshold, default to no-op. Start with a conservative threshold and lower it as confidence in the model grows.
- **Action cooldown:** After playing a card, wait at least 0.5s before the next action to prevent rapid-fire plays that waste elixir.
- **Manual override:** Implement a keyboard shortcut (e.g., Ctrl+Q) to pause/resume the bot.
- **Logging:** Save every observation, prediction, and action to a log file for post-game analysis.

### Known Limitations

- **Single-frame observation:** No temporal context (can't see troop movement direction)
- **Belonging heuristic:** Units near the river may be mislabeled as ally/enemy
- **Distribution shift:** One bad action leads to unfamiliar states, causing cascading errors. This is the fundamental limitation of BC that PPO fine-tuning addresses.

---

## 6. PPO Transition

The BC model is designed as a warm-start for PPO. The feature extractor (arena CNN + vector MLP) transfers to MaskablePPO; the BC action head is discarded (PPO creates its own action_net and value_net).

### Step-by-Step Weight Transfer

```python
import torch
from sb3_contrib import MaskablePPO
from src.bc import CRFeatureExtractor

# 1. Create MaskablePPO with the same feature extractor architecture
model = MaskablePPO(
    "MultiInputPolicy",
    env,  # Your ClashRoyaleEnv gym wrapper
    policy_kwargs={
        "features_extractor_class": CRFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
        "net_arch": [128, 64],  # Shared MLP after extractor
    },
    learning_rate=1e-4,   # Conservative LR for fine-tuning
    clip_range=0.1,       # Conservative clip range
)

# 2. Load BC-pretrained feature extractor weights
bc_weights = torch.load("models/bc/bc_feature_extractor.pt")
model.policy.features_extractor.load_state_dict(bc_weights)

# 3. (Optional) Freeze extractor initially
for param in model.policy.features_extractor.parameters():
    param.requires_grad = False

# 4. Train PPO with frozen extractor (learns policy/value heads)
model.learn(total_timesteps=500_000)

# 5. Unfreeze and fine-tune everything with lower LR
for param in model.policy.features_extractor.parameters():
    param.requires_grad = True
model.learning_rate = 3e-5
model.learn(total_timesteps=500_000)
```

### Conservative PPO Hyperparameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `learning_rate` | 1e-4 | Lower than default to preserve BC features |
| `clip_range` | 0.1 | Tighter than default (0.2) for stability |
| `net_arch` | [128, 64] | Shared layers after 192-dim features |
| `n_steps` | 2048 | Standard PPO rollout length |
| `batch_size` | 64 | Match BC batch size |

### What Transfers vs What's New

| Component | BC | PPO |
|-----------|---------|---------|
| Arena embedding (156x8) | Trained | Transferred |
| Arena CNN (3 layers) | Trained | Transferred |
| Card embedding (9x8) | Trained | Transferred |
| Vector MLP (2 layers) | Trained | Transferred |
| Action head (2 layers) | Trained | **Discarded** (PPO creates action_net) |
| Value network | N/A | **New** (PPO creates value_net) |

---

## 7. Troubleshooting

### Model always predicts no-op

- **Cause:** Class imbalance not handled. No-op dominates training data.
- **Fix:** Ensure `noop_weight=0.3` and `action_weight=3.0` are set. Check that the dataset has enough action frames (at least 15-20% after downsampling).

### Validation loss doesn't improve

- **Cause:** Too few training games or severe overfitting.
- **Fix:** Collect more gameplay recordings. Reduce model complexity (try `hidden_dim=128`). Increase dropout to 0.3.

### CUDA out of memory

- **Cause:** Batch size too large for GPU memory.
- **Fix:** Reduce `--batch_size` to 32 or 16.

### Import errors

- **Cause:** Module paths not configured correctly.
- **Fix:** Run from the repository root. Ensure `src.encoder.encoder_constants` is importable. For tests, use the provided `conftest.py`.

### Card embedding indices look wrong

- **Cause:** Card class normalization uses `/ 7` (NUM_DECK_CARDS - 1), not `/ 8`.
- **Verification:** Card 0 (arrows) = 0.0, card 7 (zappies) = 1.0. After denormalization: `round(0.0 * 7) = 0`, `round(1.0 * 7) = 7`. With +1 shift: embed indices 1-8 for present cards, 0 for empty.

### Arena embedding indices look wrong

- **Cause:** Class ID normalization uses `/ 155` (NUM_CLASSES).
- **Verification:** Empty = 0.0 -> embed[0]. Class 1 = 1/155 = 0.00645 -> `round(0.00645 * 155) = 1` -> embed[1]. Class 155 = 155/155 = 1.0 -> embed[155].
- **Critical:** Always use `torch.round()`, never `int()` truncation.
