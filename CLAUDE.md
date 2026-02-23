# Josh's BC Pipeline - Top-Level Reference

Behavior Cloning data collection and training pipeline for CS175 "The Elixir Optimizers" - an RL agent that plays Clash Royale via screen capture and computer vision.

**Team:** Alan Guo (weihaog1), Josh Talla (tallaj), Lakshya Shrivastava (Lshrivas)
**Course:** CS 175 - Project in Artificial Intelligence, UC Irvine, Winter 2026
**Instructor:** Professor Roy Fox

## Overview

This pipeline takes human gameplay recordings and trains a neural network to imitate expert card placements. All code lives in `docs/josh/` within the `cr-object-detection` repo. Five modules cover the full path from recording to training to live play.

## Data Flow

```
1. RECORD:  record_bc.py -> MatchRecorder -> ClickLogger + ScreenCapture
   Output: session/ with screenshots/, actions.jsonl, frames.jsonl, metadata.json

2. BUILD:   DatasetBuilder -> loads session -> EnhancedStateBuilder (YOLO+OCR+CardPredictor) -> StateEncoder
   Output: .npz with obs_arena, obs_vector, actions, masks, timestamps

3. TRAIN:   BCTrainer -> loads .npz files -> BCDataset -> BCPolicy (CRFeatureExtractor + action head)
   Output: best_bc.pt, bc_feature_extractor.pt, training_log.json

4. PLAY:    Live inference -> mss capture -> EnhancedStateBuilder -> StateEncoder -> BCPolicy -> action_to_placement -> PyAutoGUI
   Timing: ~237ms total per frame (well within 500ms budget at 2 FPS)

5. PPO:     MaskablePPO + CRFeatureExtractor loaded with BC weights -> online RL fine-tuning
```

## The 5 Modules

### 1. click_logger/ (Recording)

- **Entry point:** `record_bc.py`
- **ClickLogger:** OS-level mouse capture via pynput. State machine pairs card slot clicks with arena placement clicks.
- **ScreenCapture:** Threaded mss screen capture at 2 FPS, saves JPEG screenshots + frames.jsonl manifest.
- **MatchRecorder:** Orchestrates both threads, writes metadata.json on stop.
- **Output:** session directory with `screenshots/`, `actions.jsonl`, `frames.jsonl`, `metadata.json`
- Card positions calibrated for Josh's Google Play Games window (1080x1920).
- Actions are pre-paired: `{timestamp, card_id, x_norm, y_norm}`

### 2. state_encoder_module/ (Encoding)

- **StateEncoder:** Converts GameState dataclass -> SB3-compatible observation tensors.
- **Observation space:** Dict with `"arena"` (32,18,6 float32) and `"vector"` (23, float32).
- **Arena channels:** class_id (normalized), belonging (-1/+1), arena_mask, ally_tower_hp, enemy_tower_hp, spell_count.
- **Vector:** elixir, time, overtime, 6 tower HP fracs, 2 tower counts, 4 card present, 4 card class, 4 card elixir.
- **Action space:** Discrete(2305) = 4 cards x 576 grid cells + no-op.
- **PositionFinder:** Collision resolution via scipy cdist (at most 1 unit per cell).
- **coord_utils:** Pixel/norm/cell/action coordinate conversions.
- **encoder_constants:** Single source of truth for ALL constants.
- 42 tests passing.

### 3. action_builder_module/ (Action Processing)

- **ActionBuilder:** Classifies raw clicks (card/arena/other), pairs them via state machine, assigns to frames by nearest timestamp.
- **ActionExecutor:** Converts agent action indices back to PyAutoGUI two-click sequences for live play.
- **action_constants:** Card slot normalized positions, arena bounds.
- **Click pairing state machine:** idle -> card click -> card_selected -> arena click -> emit action -> idle.
- 46 tests passing.

### 4. dataset_builder_module/ (Dataset Building)

- **DatasetBuilder:** Processes click_logger sessions -> .npz training files.
- **Pipeline:** load session -> convert actions to indices -> assign to frames -> run each frame through EnhancedStateBuilder + StateEncoder -> downsample no-ops -> save .npz.
- **EnhancedStateBuilder:** Wraps StateBuilder + CardPredictor to populate GameState.cards.
- **card_integration.py:** Crops 4 card slots via `ScreenConfig.scale_to_resolution()`, classifies each with CardPredictor.
- Card slot keys are 1-indexed in ScreenConfig: `card_1` through `card_4`.
- **.npz output:** `obs_arena` (N,32,18,6), `obs_vector` (N,23), `actions` (N,) int64, `masks` (N,2305) bool, `timestamps` (N,) float64.
- **No-op downsampling:** keeps all action frames + random 15% of no-ops.
- 24 tests passing.

### 5. bc_model_module/ (BC Model - Implementation Planned)

- **CRFeatureExtractor:** SB3 BaseFeaturesExtractor subclass with arena CNN + vector MLP.
- **Embeddings:** `nn.Embedding(156, 8)` for arena unit classes, `nn.Embedding(9, 8)` for card classes.
- **Arena branch:** extract class channel -> embedding -> concat with 5 other channels -> 3-layer CNN (3x3 kernels, BatchNorm, MaxPool) -> AdaptiveAvgPool -> 128 features.
- **Vector branch:** extract card indices -> embedding -> concat with 19 scalar features -> 2-layer MLP -> 64 features.
- **Combined:** 192 features.
- **BCPolicy:** CRFeatureExtractor + 2-layer action head (Linear(192,256) + ReLU + Dropout + Linear(256,2305)).
- **BCDataset:** PyTorch Dataset loading .npz files, 80/20 file-level split, class weight computation.
- **BCTrainer:** Custom PyTorch training loop with weighted cross-entropy (noop=0.3, action=3.0), cosine annealing, early stopping, checkpoint saving.
- **PPO transition:** save feature extractor weights separately, load into MaskablePPO.
- **bc-analysis.md:** 964-line comprehensive analysis document.

## Key Constants (from encoder_constants.py)

```
Grid: GRID_COLS=18, GRID_ROWS=32, GRID_CELLS=576
Action: ACTION_SPACE_SIZE=2305, NOOP_ACTION=2304, NUM_CARD_SLOTS=4
Arena channels: CH_CLASS_ID=0, CH_BELONGING=1, CH_ARENA_MASK=2,
                CH_ALLY_TOWER_HP=3, CH_ENEMY_TOWER_HP=4, CH_SPELL=5
Classes: NUM_CLASSES=155, NUM_DECK_CARDS=8
Features: NUM_ARENA_CHANNELS=6, NUM_VECTOR_FEATURES=23
Arena bounds: ARENA_Y_START_FRAC=0.0521, ARENA_Y_END_FRAC=0.7813
Normalization: MAX_ELIXIR=10, MAX_TIME_SECONDS=300
Tower HP: DEFAULT_KING_MAX_HP=6408, DEFAULT_PRINCESS_MAX_HP=4032
Deployment: PLAYER_HALF_ROW_START=17
Deck: arrows(3), barbarian-barrel(2), eletro-spirit(1), flying-machine(4),
      goblin-cage(4), royal-hogs(5), royal-recruits(7), zappies(4)
```

## Directory Structure

```
docs/josh/
  CLAUDE.md                        <- THIS FILE
  click_logger/
    CLAUDE.md                      # Recording module technical reference
    click_logger.py                # OS-level mouse capture (142 lines)
    screen_capture.py              # Threaded mss capture (136 lines)
    match_recorder.py              # Orchestrator (159 lines)
    record_bc.py                   # Entry point (52 lines)
    docs/recording-script-docs.md  # Usage guide
  state_encoder_module/
    CLAUDE.md                      # Module overview
    src/encoder/
      CLAUDE.md                    # Package technical reference
      state_encoder.py             # GameState -> obs tensors (326 lines)
      encoder_constants.py         # ALL constants (167 lines)
      coord_utils.py               # Coordinate conversions (149 lines)
      position_finder.py           # Cell collision resolution (96 lines)
    tests/                         # 42 tests
    docs/state-encoder-docs.md     # Usage guide
  action_builder_module/
    CLAUDE.md                      # Module overview
    src/action/
      CLAUDE.md                    # Package technical reference
      action_builder.py            # Click -> action pairing (165 lines)
      action_executor.py           # Action -> PyAutoGUI execution
      action_constants.py          # Card slot positions, arena bounds
    tests/                         # 46 tests
    docs/
      action-builder-docs.md       # Usage guide
      bc-data-collection-guide.md  # End-to-end collection guide
  dataset_builder_module/
    CLAUDE.md                      # Module overview
    src/dataset/
      CLAUDE.md                    # Package technical reference
      dataset_builder.py           # Session -> .npz processing (311 lines)
      card_integration.py          # EnhancedStateBuilder wrapper (101 lines)
    tests/                         # 24 tests
    docs/dataset-builder-docs.md   # Usage guide
  bc_model_module/
    CLAUDE.md                      # Module overview
    src/bc/
      CLAUDE.md                    # Package technical reference
      feature_extractor.py         # CRFeatureExtractor (planned)
      bc_policy.py                 # BCPolicy (planned)
      bc_dataset.py                # BCDataset (planned)
      train_bc.py                  # Training script (planned)
    tests/                         # (planned, ~30 tests)
    docs/
      bc-analysis.md               # 964-line comprehensive analysis
      bc-model-docs.md             # Usage guide (planned)
```

## Import Path Pattern (conftest.py)

Each module's `tests/conftest.py` injects the module's `src/` directories into `src.__path__` so that:

- `from src.encoder import ...` resolves to josh's state_encoder_module copy
- `from src.action import ...` resolves to josh's action_builder_module copy
- `from src.dataset import ...` resolves to josh's dataset_builder_module copy
- `from src.bc import ...` resolves to josh's bc_model_module copy
- Other subpackages (`src.pipeline`, `src.generation`, `src.data`, `src.classification`) resolve from the real codebase at `Project/cr-object-detection/`

Pattern used in each conftest.py:

```python
import os, sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _repo_root)
import src
_module_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
src.__path__.insert(0, _module_src)
```

## Dependencies on Main Codebase

Josh's modules import from the main cr-object-detection codebase:

| Import | Source | Used By |
|--------|--------|---------|
| `src.pipeline.game_state` | GameState, Tower, Unit, Card dataclasses | state_encoder, dataset_builder |
| `src.pipeline.state_builder` | StateBuilder (YOLO + OCR perception) | dataset_builder |
| `src.generation.label_list` | Unit class lists (ground, flying, spell, tower, other) | state_encoder |
| `src.data.screen_regions` | ScreenConfig with card slot positions (1-indexed: card_1..card_4) | dataset_builder |
| `src.classification.card_classifier` | CardPredictor (MiniResNet, 8-class) | dataset_builder |
| `src.detection.model` | YOLO detection wrapper | dataset_builder (via StateBuilder) |

## Running Tests

```bash
# Individual modules (from cr-object-detection root):
python -m pytest docs/josh/state_encoder_module/tests/ -v
python -m pytest docs/josh/action_builder_module/tests/ -v
python -m pytest docs/josh/dataset_builder_module/tests/ -v
python -m pytest docs/josh/bc_model_module/tests/ -v

# All at once:
python -m pytest docs/josh/ -v --ignore=docs/josh/click_logger
```

## Test Counts

| Module | Tests |
|--------|-------|
| state_encoder_module | 42 |
| action_builder_module | 46 |
| dataset_builder_module | 24 |
| bc_model_module | 30 (planned) |
| **Total** | **142** |

## Known Limitations

1. **No belonging output from YOLO:** Model outputs 6-column detections. StateBuilder uses Y-position heuristic (`arena_mid = frame_height * 0.42`) which fails when troops cross the river.
2. **Domain gap (mAP50=0.804):** Synthetic training data does not perfectly match real Google Play Games screenshots.
3. **Single-frame observations:** No temporal context (troop movement direction, elixir generation rate).
4. **Card classifier not wired into main StateBuilder:** Only available via EnhancedStateBuilder wrapper in dataset_builder_module.
5. **Coordinate inconsistencies:** 540x960 base resolution in screen_regions.py vs 1080x1920 pixel coords in some files. Normalized coords (0-1) bridge the gap.

## Phase Status

| Phase | Module | Status |
|-------|--------|--------|
| Recording | click_logger | Complete (4 files, manual calibration for Josh's window) |
| Encoding | state_encoder_module | Complete (42 tests passing) |
| Action Processing | action_builder_module | Complete (46 tests passing) |
| Dataset Building | dataset_builder_module | Complete (24 tests passing) |
| BC Model | bc_model_module | Design complete (bc-analysis.md), implementation planned |
| PPO Fine-tuning | (not started) | Requires Gym env wrapper + reward function |
