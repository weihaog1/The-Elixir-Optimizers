# DatasetBuilder Developer Documentation

## What Is the DatasetBuilder?

The DatasetBuilder is the bridge between raw gameplay recordings and neural network training data. It takes session directories produced by the click_logger (screenshots + paired click actions) and converts them into .npz files containing observation tensors, action labels, and action masks.

In the full pipeline:

```
Human plays Clash Royale
    |
    +--- Thread A: mss screen capture at 2 FPS
    |    -> screenshots/frame_000000.jpg, ...
    |    -> frames.jsonl (timestamp manifest)
    |
    +--- Thread B: pynput click logger
    |    -> actions.jsonl (pre-paired card placements)
    |
    v
Session directory: recordings/match_YYYYMMDD_HHMMSS/
    |
    v  (offline, post-game)
    |
DatasetBuilder  <--- this module
    |
    +--- screenshots/ -> EnhancedStateBuilder (YOLO + OCR + CardPredictor)
    |                  -> GameState -> StateEncoder -> obs tensors + masks
    |
    +--- actions.jsonl -> norm_to_cell() + placement_to_action()
    |                  -> Discrete(2305) action indices
    |
    +--- Timestamp matching: assign actions to nearest frames
    |
    +--- No-op downsampling: keep 15% of wait frames
    |
    v
.npz file: {obs_arena, obs_vector, actions, masks, timestamps}
    |
    v
BC Training (SB3 MaskableMultiInputPolicy or custom PyTorch)
```

Without the DatasetBuilder, there is no way to get from raw recordings to training data. It handles the entire offline processing pipeline: loading session files, running the perception pipeline on every screenshot, converting click actions to discrete action indices, matching actions to frames by timestamp, downsampling the ~90% no-op frames, and saving everything in a compact numpy format.

---

## Thought Process

### Why offline processing?

The click_logger records raw screenshots and raw click events during gameplay. No YOLO inference, no OCR, no encoding runs during the game. The DatasetBuilder processes everything offline. This was a deliberate choice for three reasons:

1. **Speed during recording.** YOLO inference takes ~65ms per frame and OCR takes ~30ms per region. Running these during gameplay would consume 20%+ of each 500ms capture interval, risking dropped frames or missed clicks. The recording loop stays lightweight (~5ms per frame for JPEG encoding).

2. **Flexibility.** If we improve the YOLO model, retrain the card classifier, or change the encoding scheme, we can re-run the DatasetBuilder on existing recordings without replaying games. Raw screenshots are the most reusable format.

3. **Debuggability.** When the BC model misbehaves, we can visually inspect the exact screenshot at any training frame. With pre-processed state, we would need to reverse-engineer what the agent "saw."

The trade-off is disk space (~120MB per match for raw screenshots) and processing time (~100ms per frame offline). Both are acceptable.

### Why EnhancedStateBuilder wrapper?

The existing StateBuilder produces GameState objects from screenshots, but it does not populate `GameState.cards` (always an empty list). The MiniResNet card classifier exists as a separate `CardPredictor` class but is not wired into the StateBuilder.

Rather than modifying the main StateBuilder (which other modules depend on), the EnhancedStateBuilder wraps it:

```
image -> StateBuilder.build_state() -> GameState (cards empty)
      -> CardPredictor.predict(crop_1..crop_4) -> [Card, Card, Card, Card]
      -> GameState.cards = [Card, Card, Card, Card]
```

This keeps the change isolated. The EnhancedStateBuilder delegates all detection and OCR to the base StateBuilder, then adds card classification on top. If the card classifier is not provided, it falls back to the base behavior (empty cards list).

### Why .npz format?

numpy-native, requires no extra dependencies, loads directly into arrays with `np.load()`, and is compact (~1-2MB per session). HDF5 adds a dependency (h5py) and is overkill for our data size. Pickle has security concerns and version dependencies.

### Why timestamp-based action assignment?

Actions and frames are recorded by independent threads with independent timing. The DatasetBuilder assigns each action to the frame with the closest timestamp. This is simpler than fixed-window matching (would miss boundary cases) or forward-only assignment (introduces systematic lag).

### Why no-op downsampling?

In a 3-minute match at 2 FPS, ~93% of frames are no-ops (waiting). Training on this raw distribution produces a model that always predicts "wait." Downsampling no-ops to 15% brings the ratio to ~30% actions / ~70% no-ops - still imbalanced but manageable with weighted loss at training time. The 0.15 default comes from the pipeline design document.

---

## How to Use It - Complete Step-by-Step Guide

### Step 1: Record match data using click_logger

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection/docs/josh/click_logger/
python record_bc.py
# Play a full match in Clash Royale
# Press Enter in the terminal when done
# Output: recordings/match_YYYYMMDD_HHMMSS/
```

Each session directory contains:
```
recordings/match_20260222_143000/
  screenshots/
    frame_000000.jpg
    frame_000001.jpg
    ...
  actions.jsonl       # Pre-paired card placements
  frames.jsonl        # Frame timestamp manifest
  metadata.json       # Session summary
```

### Step 2: Configure and run DatasetBuilder

```python
from src.pipeline.state_builder import StateBuilder
from src.classification.card_classifier import CardPredictor
from src.dataset import DatasetBuilder, EnhancedStateBuilder
from src.encoder import StateEncoder

# Set up the perception pipeline
state_builder = StateBuilder(
    detection_model_path="models/best_yolov8s_50epochs_fixed_pregen_set.pt"
)
card_predictor = CardPredictor(
    weights_path="models/card_classifier/card_classifier.pt"
)
enhanced_sb = EnhancedStateBuilder(state_builder, card_predictor)
encoder = StateEncoder()

# Build dataset from one session
builder = DatasetBuilder(
    enhanced_state_builder=enhanced_sb,
    state_encoder=encoder,
)
stats = builder.build_dataset(
    session_dir="docs/josh/click_logger/recordings/match_20260222_143000",
    output_dir="data/bc_training",
    noop_keep_ratio=0.15,
)
print(stats)
# DatasetStats(total_frames=360, total_actions=22, noop_frames=338,
#              action_frames=22, kept_after_downsample=73, ...)
```

### Step 3: Process multiple matches

```python
import glob

sessions = sorted(glob.glob(
    "docs/josh/click_logger/recordings/match_*"
))
all_stats = builder.build_from_multiple(
    sessions,
    output_dir="data/bc_training",
    noop_keep_ratio=0.15,
)

for s in all_stats:
    print(f"{s.session_dir}: {s.kept_after_downsample} frames "
          f"({s.action_frames} actions, {s.noop_frames} no-ops)")
```

### Step 4: Load dataset for BC training

```python
import numpy as np

data = np.load("data/bc_training/match_20260222_143000.npz")
obs_arena = data["obs_arena"]    # (N, 32, 18, 6) - spatial grid
obs_vector = data["obs_vector"]  # (N, 23) - scalar features
actions = data["actions"]        # (N,) - Discrete(2305) labels
masks = data["masks"]            # (N, 2305) - valid action mask

print(f"Frames: {len(actions)}")
print(f"Actions: {(actions != 2304).sum()}")
print(f"No-ops: {(actions == 2304).sum()}")
```

---

## BC Model Training Guide

### Custom PyTorch Dataset

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BCDataset(Dataset):
    def __init__(self, npz_paths):
        arenas, vectors, all_actions, all_masks = [], [], [], []
        for path in npz_paths:
            data = np.load(path)
            arenas.append(data["obs_arena"])
            vectors.append(data["obs_vector"])
            all_actions.append(data["actions"])
            all_masks.append(data["masks"])
        self.arena = np.concatenate(arenas)
        self.vector = np.concatenate(vectors)
        self.actions = np.concatenate(all_actions)
        self.masks = np.concatenate(all_masks)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            "arena": torch.FloatTensor(self.arena[idx]),
            "vector": torch.FloatTensor(self.vector[idx]),
            "action": torch.LongTensor([self.actions[idx]]),
            "mask": torch.BoolTensor(self.masks[idx]),
        }

# Usage
npz_files = sorted(glob.glob("data/bc_training/match_*.npz"))
dataset = BCDataset(npz_files)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch in loader:
    arena = batch["arena"]    # (64, 32, 18, 6)
    vector = batch["vector"]  # (64, 23)
    action = batch["action"]  # (64, 1)
    mask = batch["mask"]      # (64, 2305)
    # Forward pass through policy network
    break
```

### SB3 MaskablePPO Approach

```python
from sb3_contrib import MaskablePPO

# Load BC-pretrained weights into MaskablePPO for fine-tuning
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    learning_rate=1e-4,
    clip_range=0.1,
    ent_coef=0.01,
)
# Load BC weights into the policy network
# Then train with PPO for online fine-tuning
```

### Handling Class Imbalance

Even after no-op downsampling (15%), the dataset is still imbalanced (~70% no-op, ~30% actions). Two approaches:

**Weighted cross-entropy loss:**
```python
import torch.nn as nn

# Compute weights from dataset statistics
noop_weight = 0.3
action_weight = 3.0

weights = torch.ones(2305)
weights[2304] = noop_weight  # NOOP_ACTION
weights[:2304] = action_weight

criterion = nn.CrossEntropyLoss(weight=weights)
```

**Why these weights:** With ~70% no-ops and ~30% actions after downsampling, the inverse frequency ratio is roughly 0.3/3.0. The no-op class gets downweighted so the model does not learn to always predict "wait." Card placement actions get upweighted so the model actually learns when and where to play cards.

---

## Next Steps on the Pipeline

After the DatasetBuilder is built, the remaining pipeline steps are:

1. **Record 30-40 matches.** Each match produces ~360 frames at 2 FPS over 3 minutes. Raw: ~13,000 frames total. After no-op downsampling at 15%: ~2,500 frames per match, ~90,000 total training frames.

2. **Run DatasetBuilder on all recordings.** Process each session directory into a .npz file. At ~100ms per frame (YOLO + OCR + encoding), 13,000 frames takes ~22 minutes.

3. **Train BC model.** Use SB3 MaskableMultiInputPolicy or a custom PyTorch training loop with weighted cross-entropy loss. Target: beat a random baseline (random wins ~2% vs Trainer AI).

4. **Evaluate BC model.** Play 20-30 games against Trainer Cheddar. Measure win rate and qualitative card placement quality (does it place troops at the bridge? does it defend pushes?).

5. **Build Gym environment wrapper.** `ClashRoyaleEnv(gym.Env)` with `observation_space`, `action_space`, `step()`, `reset()`. Uses mss for live capture, EnhancedStateBuilder for perception, StateEncoder for encoding, ActionExecutor for PyAutoGUI execution.

6. **PPO fine-tuning.** Initialize PPO from BC weights. Train with reward shaping against Trainer AI. Use conservative hyperparameters (`clip_range=0.1`, `learning_rate=1e-4`) to preserve the BC prior.

7. **Reward design.** Primary: tower HP delta (measured via OCR). Terminal: +10 for win, -10 for loss. Clip per-step HP delta to [-500, +500] to guard against OCR errors.

---

## Assumptions and Limitations

### 1. Pre-paired actions from ClickLogger

The DatasetBuilder consumes `actions.jsonl` from the ClickLogger, where each line is already a complete card placement (`{card_id, x_norm, y_norm}`). It does NOT use ActionBuilder's click pairing state machine. This means:
- No need to handle raw ClickEvent pairing
- Actions that ClickLogger paired incorrectly (e.g., due to drag) propagate to training data
- The DatasetBuilder trusts ClickLogger's output

### 2. Y-position belonging heuristic

The base StateBuilder uses `frame_height * 0.42` as the midpoint for unit belonging classification. Units above this line are labeled enemy, below are labeled ally. When troops cross the river, they get mislabeled. The DatasetBuilder faithfully passes these through - it does not correct belonging errors.

### 3. CardPredictor fixed 8-card deck

EnhancedStateBuilder uses the MiniResNet card classifier trained on 8 specific cards (Royal Hogs / Royal Recruits deck). If the player uses a different deck, card predictions will be wrong. To change decks:
- Retrain CardPredictor with new reference card images
- Update `DECK_CARDS`, `CARD_ELIXIR_COST`, `CARD_IS_SPELL` in `encoder_constants.py`

### 4. Linear timestamp scan

The `_assign_actions_to_frames()` method uses a linear scan (O(A * F) where A = actions, F = frames). For typical sessions (360 frames, 20 actions), this is 7,200 comparisons - negligible. For very long sessions (1000+ frames), binary search would be more efficient but is not needed at current scale.

### 5. No validation of action plausibility

The DatasetBuilder does not check whether an action was actually valid (enough elixir, card in hand, placement on correct half). It records exactly what the human clicked. Invalid actions appear in training data. The action mask (from StateEncoder) provides validity information, but the action label itself is unfiltered. The BC model must learn from both valid and occasionally invalid actions.

### 6. Random no-op downsampling

No-op frames are selected uniformly at random for retention. This means:
- No temporal structure (the kept no-ops may cluster in one part of the match)
- Results are non-deterministic across runs (different random seed)
- At least 1 no-op frame is always kept

### 7. Single-frame observations

Each frame is processed independently. There is no frame stacking or temporal context. The BC model sees one snapshot at a time. This is sufficient for reactive play (the pipeline design document recommends starting with single-frame and adding 2-frame stacking only if performance plateaus).

### 8. StateBuilder must be provided for meaningful data

If `enhanced_state_builder=None`, the DatasetBuilder produces zero-filled observations for every frame (a default empty GameState is encoded). This is useful for testing the pipeline without loading YOLO/OCR models, but the resulting training data has no perceptual content.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/dataset/__init__.py` | Module exports: DatasetBuilder, DatasetStats, EnhancedStateBuilder |
| `src/dataset/dataset_builder.py` | Core DatasetBuilder class: session loading, action assignment, processing, downsampling, saving |
| `src/dataset/card_integration.py` | EnhancedStateBuilder: wraps StateBuilder + CardPredictor for complete GameState |
| `tests/conftest.py` | Pytest path setup: injects josh's module copies into src.__path__ |
| `tests/test_card_integration.py` | Tests for EnhancedStateBuilder: delegation, card extraction, scaling |
| `tests/test_dataset_builder.py` | Tests for DatasetBuilder: session loading, action conversion, downsampling, end-to-end |
| `docs/dataset-builder-docs.md` | This developer documentation |
| `src/dataset/CLAUDE.md` | Technical reference (API, formats, algorithm) |

### Dependencies

- `cv2` (OpenCV) - image loading from JPEG screenshots
- `numpy` - array operations and .npz file I/O
- `src.encoder.coord_utils` - `norm_to_cell()`, `placement_to_action()` for action conversion
- `src.encoder.encoder_constants` - `NOOP_ACTION` (2304)
- `src.encoder.state_encoder` - `StateEncoder` for encoding GameState to observation tensors
- `src.pipeline.state_builder` - `StateBuilder` for YOLO + OCR perception
- `src.classification.card_classifier` - `CardPredictor` for card hand classification
- `src.data.screen_regions` - `ScreenConfig` for card slot crop regions
- `src.pipeline.game_state` - `GameState`, `Card` dataclasses

### What depends on this module

- **BC training script** (not yet built) - loads .npz files into PyTorch Dataset for training
- **Evaluation tools** (not yet built) - may inspect .npz files to debug training data quality
- **Pipeline orchestration** (not yet built) - will call `build_from_multiple()` to batch-process all recordings

---

## Common Questions

**Q: How much disk space does one .npz file use?**
A 3-minute match produces ~73 frames after downsampling, at ~16KB per frame = ~1.2MB per session. For 40 sessions: ~48MB total.

**Q: How long does processing one session take?**
YOLO (~65ms) + OCR (~40ms) per frame. A 360-frame session takes ~38 seconds. Action assignment and downsampling are negligible.

**Q: What resolution should screenshots be?**
Any resolution works. ScreenConfig scales card slot regions proportionally. YOLO downscales internally to imgsz=960. The click_logger typically records at 1080x1920.

**Q: Can I change the no-op downsampling ratio?**
Yes. Pass `noop_keep_ratio=X` (0 to 1). Default 0.15 keeps 15% of no-ops. At 0.0, at least 1 no-op frame is always preserved.

**Q: What happens if actions.jsonl is empty?**
All frames get NOOP_ACTION (2304). After downsampling, the .npz contains only no-op training data.

**Q: What if a screenshot is missing or corrupted?**
`cv2.imread()` returns None, so the DatasetBuilder creates a zero-filled observation from a default empty GameState. No error is raised.

**Q: Can I run DatasetBuilder without YOLO/OCR models?**
Yes. Pass `enhanced_state_builder=None`. Every frame produces zero-filled observations. Useful for testing pipeline mechanics.

**Q: What if two actions map to the same frame?**
The later action overwrites the earlier one. Extremely rare at 2 FPS (would require two placements within 250ms).

**Q: How does the DatasetBuilder handle different resolutions across frames?**
It relies on actual image dimensions from cv2, not the width/height in frames.jsonl. The StateEncoder normalizes everything to the 18x32 grid regardless of resolution.

**Q: Can I re-run DatasetBuilder on the same session?**
Yes. The .npz file is overwritten with the new settings. Raw session data is never modified.
