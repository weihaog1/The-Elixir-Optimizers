# State Encoder Module

Converts the structured `GameState` dataclass into fixed-shape numpy tensors for Stable Baselines 3 (SB3). This is the bridge between perception (StateBuilder) and learning (BC/PPO).

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/encoder/state_encoder.py` | 325 | Main encoder class. `encode()` returns obs dict, `action_mask()` returns valid actions. |
| `src/encoder/encoder_constants.py` | 166 | ALL shared constants. Single source of truth for grid dims, action space, channels, deck, tower HP, class mapping, unit type mapping. |
| `src/encoder/coord_utils.py` | 148 | Coordinate conversions between pixel, normalized, grid cell, and action index spaces. |
| `src/encoder/position_finder.py` | 95 | Resolves cell collisions via `scipy.spatial.distance.cdist` nearest-neighbor. |
| `src/encoder/CLAUDE.md` | - | Detailed technical reference (obs channels, vector indices, deck config, coordinate system). |

## Observation Space

- **"arena"**: `(32, 18, 6)` float32 - Spatial grid with 6 channels:
  - Ch 0: class_id (normalized), Ch 1: belonging (-1 ally, +1 enemy)
  - Ch 2: arena_mask (unit present), Ch 3: ally tower HP fraction
  - Ch 4: enemy tower HP fraction, Ch 5: spell effect count
- **"vector"**: `(23,)` float32 - Scalar features:
  - Elixir, time remaining, overtime flag
  - 6 tower HP fractions (player/enemy king + 2 princess each)
  - Player/enemy tower counts
  - Card slots 0-3: present flag, class index, elixir cost

## Action Space

`Discrete(2305)` = 4 cards x 576 grid cells (18 cols x 32 rows) + 1 no-op.

```
action = card_id * 576 + row * 18 + col    (0..2303)
action = 2304                                (no-op)
```

Action mask disables cards that are empty or too expensive (elixir check). No-op is always valid.

## Tests

42 tests across 3 files:
- `tests/test_state_encoder.py` - Observation shapes, values, action masking
- `tests/test_coord_utils.py` - Pixel/norm/cell/action conversions, roundtrips
- `tests/test_position_finder.py` - Collision resolution, edge cases

```bash
python -m pytest docs/josh/state_encoder_module/tests/ -v
```

## Documentation

- `docs/state-encoder-docs.md` - Full module documentation

## Dependencies

- `gymnasium`, `numpy`, `scipy`
- `src.pipeline.game_state` (GameState, Tower, Unit, Card)
- `src.generation.label_list` (ground/flying/spell/tower class lists for UNIT_TYPE_MAP)

## Used By

- **dataset_builder_module** - DatasetBuilder calls `encode()` to generate training observations
- **bc_model_module** - BCTrainer/BCDataset consume the observation and action spaces
