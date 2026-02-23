# Dataset Builder Module

Processes click_logger recording sessions into training-ready `.npz` files for behavior cloning.

## Pipeline

1. Load session directory (screenshots/, actions.jsonl, frames.jsonl)
2. Convert pre-paired actions (card_id, x_norm, y_norm) to Discrete(2305) indices
3. Assign actions to frames by nearest timestamp matching
4. Run each frame through EnhancedStateBuilder + StateEncoder
5. Downsample no-op frames (keep ALL action frames + random 15% of no-ops)
6. Save consolidated `.npz` file

## Key Files

| File | Lines | Description |
|------|-------|-------------|
| `src/dataset/dataset_builder.py` | 311 | DatasetBuilder class, DatasetStats dataclass. `build_dataset()`, `build_from_multiple()`, internal helpers for loading, converting, assigning, processing, downsampling |
| `src/dataset/card_integration.py` | 101 | EnhancedStateBuilder wraps StateBuilder + CardPredictor. Crops 4 card slots from screenshot using `ScreenConfig.scale_to_resolution()`, classifies each. Card slot keys are 1-indexed (`card_1` through `card_4`) |
| `src/dataset/__init__.py` | - | Exports DatasetBuilder, DatasetStats, EnhancedStateBuilder |
| `src/dataset/CLAUDE.md` | - | Detailed technical reference |

## .npz Output Format

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `obs_arena` | (N, 32, 18, 6) | float32 | Arena grid observation |
| `obs_vector` | (N, 23) | float32 | Scalar features vector |
| `actions` | (N,) | int64 | Discrete(2305) action indices |
| `masks` | (N, 2305) | bool | Valid action mask per frame |
| `timestamps` | (N,) | float64 | Frame timestamps |

## No-op Downsampling

Keeps ALL action frames plus a random sample of no-op frames (default 15%) to reduce class imbalance.

## Tests

- 24 tests total: `test_dataset_builder.py` (17 tests), `test_card_integration.py` (7 tests)
- Run: `python -m pytest docs/josh/dataset_builder_module/tests/ -v`

## Documentation

- `docs/dataset-builder-docs.md` - Full module documentation

## Dependencies

cv2, numpy, `src.encoder` (StateEncoder, coord_utils, encoder_constants), `src.pipeline` (GameState, StateBuilder), `src.data` (ScreenConfig), `src.classification` (CardPredictor)

## Used By

- `bc_model_module` - BCDataset loads the `.npz` files produced by this module
