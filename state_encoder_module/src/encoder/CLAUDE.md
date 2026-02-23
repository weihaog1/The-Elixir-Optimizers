# src/encoder/ - StateEncoder (GameState -> RL Observations)

Converts the structured `GameState` dataclass from `StateBuilder` into fixed-shape numpy tensors compatible with Stable Baselines 3 (SB3). This module is the bridge between perception and learning -- used identically by behavior cloning and PPO.

Uses **per-cell unit identity encoding** (KataCR approach): each arena grid cell holds at most one unit with its class ID and belonging flag. A `PositionFinder` resolves collisions when multiple units map to the same cell.

## Files

**state_encoder.py** - Core class.
- `StateEncoder`: Stateless encoder with `encode()` and `action_mask()`.
  - `encode(state: GameState) -> dict["arena": np.float32, "vector": np.float32]`
  - `action_mask(state: GameState) -> np.bool_ array (2305,)`
  - `observation_space`: gymnasium.spaces.Dict (arena + vector)
  - `action_space`: gymnasium.spaces.Discrete(2305)

**position_finder.py** - Grid cell collision resolution.
- `PositionFinder(rows=32, cols=18)`: Fresh instance per frame.
  - `find_position(col_f, row_f) -> (col, row)`: Assigns nearest free cell.
  - Uses `scipy.spatial.distance.cdist` for Euclidean nearest-neighbor lookup.
  - Pre-computes all 576 cell center coordinates at init.
  - Raises `RuntimeError` if all cells are occupied.

**encoder_constants.py** - All constants and lookups.
- Grid: GRID_COLS=18, GRID_ROWS=32, GRID_CELLS=576
- Channels: CH_CLASS_ID=0, CH_BELONGING=1, CH_ARENA_MASK=2, CH_ALLY_TOWER_HP=3, CH_ENEMY_TOWER_HP=4, CH_SPELL=5, NUM_ARENA_CHANNELS=6
- Class mapping: NUM_CLASSES=155, CLASS_NAME_TO_ID (1-indexed, 0 = empty)
- Action: ACTION_SPACE_SIZE=2305, NOOP_ACTION=2304, NUM_CARD_SLOTS=4
- Arena bounds: ARENA_Y_START_FRAC, ARENA_Y_END_FRAC (from screen_regions.py 540x960)
- UNIT_TYPE_MAP: dict mapping 155 detection class names -> "ground"/"flying"/"spell"/"tower"/"other"
  - Built at import time from `src.generation.label_list`
- DECK_CARDS: 8 card names matching CardPredictor class names
- CARD_ELIXIR_COST: elixir cost per card
- CARD_IS_SPELL: whether card is a spell (placement rules)
- DEFAULT_KING_MAX_HP, DEFAULT_PRINCESS_MAX_HP: level-14 tower HP

**coord_utils.py** - Coordinate conversion utilities.
- `pixel_to_cell(cx, cy, frame_w, frame_h) -> (col, row)`: Pixel bbox center to 18x32 grid cell (clamped integers)
- `pixel_to_cell_float(cx, cy, frame_w, frame_h) -> (col_f, row_f)`: Pixel bbox center to continuous grid coordinates (for PositionFinder input)
- `norm_to_cell(x_norm, y_norm) -> (col, row)`: Normalized screen coords to grid cell
- `cell_to_norm(col, row) -> (x_norm, y_norm)`: Grid cell center to normalized screen coords
- `action_to_placement(action_idx) -> (card_id, col, row) | None`: Decode action index
- `placement_to_action(card_id, col, row) -> int`: Encode placement to action index

## Observation Space

### Arena Grid: shape (32, 18, 6), float32

Per-cell identity encoding. Each cell holds at most one unit (enforced by PositionFinder). Channels 0-2 are per-cell unit features. Channels 3-5 are continuous/additive features that do not participate in PositionFinder.

| Channel | Constant | Range | Description |
|---------|----------|-------|-------------|
| 0 | CH_CLASS_ID | 0-1 | Normalized class ID: class_idx / 155 (0.0 = empty cell) |
| 1 | CH_BELONGING | -1, 0, +1 | -1.0 = ally, +1.0 = enemy, 0.0 = empty cell |
| 2 | CH_ARENA_MASK | 0-1 | 1.0 = unit present, 0.0 = empty |
| 3 | CH_ALLY_TOWER_HP | 0-1 | Allied tower HP fraction (0 = no tower or destroyed) |
| 4 | CH_ENEMY_TOWER_HP | 0-1 | Enemy tower HP fraction |
| 5 | CH_SPELL | 0+ | Spell effect count (additive, bypasses PositionFinder) |

### CLASS_NAME_TO_ID Mapping

155 detection class names from `label_list.unit_list` mapped to 1-indexed integers. ID 0 is reserved for empty cells. Example values:
- `CLASS_NAME_TO_ID["knight"]` = 49
- `CLASS_NAME_TO_ID["skeleton"]` = 27
- `CLASS_NAME_TO_ID["royal-hog"]` = some index

To recover the class name from a normalized class_id in the arena tensor:
```python
class_idx = int(round(arena[row, col, CH_CLASS_ID] * NUM_CLASSES))
# class_idx is 0 for empty, 1-155 for actual classes
```

### Vector: shape (23,), float32

All values normalized to [0, 1]. None/unknown defaults to 0.0.

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Elixir | / 10 |
| 1 | Time remaining | / 300 |
| 2 | Is overtime | 0 or 1 |
| 3 | Player king tower HP | fraction 0-1 |
| 4 | Player left princess HP | fraction 0-1 |
| 5 | Player right princess HP | fraction 0-1 |
| 6 | Enemy king tower HP | fraction 0-1 |
| 7 | Enemy left princess HP | fraction 0-1 |
| 8 | Enemy right princess HP | fraction 0-1 |
| 9 | Player tower count | / 3 |
| 10 | Enemy tower count | / 3 |
| 11-14 | Card present (slots 0-3) | 0 or 1 |
| 15-18 | Card class index (slots 0-3) | / (num_deck_cards - 1) |
| 19-22 | Card elixir cost (slots 0-3) | / 10 |

## Action Space: Discrete(2305)

```
action = card_id * 576 + row * 18 + col    (0..2303 for card placements)
action = 2304                                (no-op / wait)
```

Decoding: `action_to_placement(action_idx) -> (card_id, col, row) | None`
Encoding: `placement_to_action(card_id, col, row) -> int`

### Action Mask

- If card slot is empty or "empty-slot": all 576 cells for that card are masked out
- If elixir < card cost: all 576 cells for that card are masked out
- No-op (2304) is always valid

## PositionFinder Algorithm

1. Fresh `PositionFinder` instance created per frame (no state carried between frames)
2. Units sorted before placement: enemy first (belonging=1), then ally (belonging=0); within each faction, bottom-to-top (higher pixel y first)
3. For each unit, `pixel_to_cell_float()` gives continuous (col_f, row_f) coordinates
4. `find_position(col_f, row_f)` checks if the natural cell (int truncation) is free:
   - If free: assign it, mark occupied, return
   - If occupied: compute Euclidean distance to all 576 cell centers, mask occupied cells with infinity, pick closest free cell
5. Spells bypass PositionFinder entirely -- they go to CH_SPELL (additive count)
6. Tower/other unit types are skipped (towers handled via dedicated HP channels)
7. Unknown classes (not in CLASS_NAME_TO_ID) are skipped

## Coordinate System

The 18x32 grid maps to the arena portion of the screen (not the full screen):
- X: full screen width divided into 18 columns
- Y: arena region only (screen y=[50, 750] at 540x960 base resolution) divided into 32 rows
- Row 0 = top of arena (enemy king tower area)
- Row 31 = bottom of arena (player king tower area)
- Rows 15-16 = river area
- Player's deployable half: rows 17-31

Pixel-to-cell conversion accounts for non-arena screen regions (timer bar, card bar) via ARENA_Y_START_FRAC and ARENA_Y_END_FRAC constants.

Two conversion functions:
- `pixel_to_cell()`: Returns clamped integers -- used for spells and towers
- `pixel_to_cell_float()`: Returns raw floats -- used as PositionFinder input for ground/flying units

## Dependencies

```
state_encoder.py
  -> src.pipeline.game_state (GameState, Tower, Unit, Card)
  -> encoder_constants (grid, action, deck, unit type, class ID constants)
  -> coord_utils (pixel_to_cell, pixel_to_cell_float)
  -> position_finder (PositionFinder)

position_finder.py
  -> numpy, scipy.spatial.distance.cdist

encoder_constants.py
  -> src.generation.label_list (ground/flying/spell/tower unit lists, unit_list)

coord_utils.py
  -> encoder_constants (grid dimensions, arena fractions)
```

## Usage

```python
from src.encoder import StateEncoder, PositionFinder
from src.pipeline.game_state import GameState, Tower, Unit, Card

encoder = StateEncoder()

# In BC data collection or Gym env:
obs = encoder.encode(game_state)   # {"arena": (32,18,6), "vector": (23,)}
mask = encoder.action_mask(game_state)  # bool (2305,)

# For SB3 policy definition:
policy_kwargs = dict(
    observation_space=encoder.observation_space,
    action_space=encoder.action_space,
)

# Decode agent's chosen action for execution:
from src.encoder import action_to_placement, cell_to_norm
result = action_to_placement(action_idx)
if result is not None:
    card_id, col, row = result
    x_norm, y_norm = cell_to_norm(col, row)
    play_card(card_id, x_norm, y_norm)
```

### Inspecting per-cell unit encoding

```python
import numpy as np
from src.encoder.encoder_constants import (
    CH_CLASS_ID, CH_BELONGING, CH_ARENA_MASK, CH_SPELL, NUM_CLASSES,
    CLASS_NAME_TO_ID,
)

obs = encoder.encode(state)
arena = obs["arena"]

# Find all occupied cells
mask = arena[:, :, CH_ARENA_MASK] > 0
for row, col in np.argwhere(mask):
    class_idx = int(round(arena[row, col, CH_CLASS_ID] * NUM_CLASSES))
    belonging = arena[row, col, CH_BELONGING]
    side = "ally" if belonging < 0 else "enemy"
    # Reverse-lookup class name
    name = next((k for k, v in CLASS_NAME_TO_ID.items() if v == class_idx), "?")
    print(f"  ({col}, {row}): {name} ({side})")

# Check spell activity
spell_cells = np.argwhere(arena[:, :, CH_SPELL] > 0)
print(f"Spell-active cells: {len(spell_cells)}")
```

## Deck Configuration

Current deck: Royal Hogs / Royal Recruits (8 cards).

| Card | Elixir | Type | Note |
|------|--------|------|------|
| arrows | 3 | Spell | Area damage |
| barbarian-barrel | 2 | Spell | Rolling barrel |
| eletro-spirit | 1 | Troop | Filename typo preserved for CardPredictor compatibility |
| flying-machine | 4 | Troop | Flying unit |
| goblin-cage | 4 | Troop | Building (spawner) |
| royal-hogs | 5 | Troop | 4 hog riders |
| royal-recruits | 7 | Troop | 6 recruits (most expensive) |
| zappies | 4 | Troop | 3 zappies |

To change decks: update DECK_CARDS, CARD_ELIXIR_COST, and CARD_IS_SPELL in encoder_constants.py. Retrain the CardPredictor with new reference card images.

## BC Network Integration (Future)

The per-cell encoding is designed for a custom SB3 feature extractor (`CRFeatureExtractor`) that:
1. Reads the normalized class_id float from channel 0
2. Denormalizes: `class_idx = round(class_id * NUM_CLASSES)` to get integer in [0, 155]
3. Feeds integer into `nn.Embedding(156, 8)` for a learned 8-dimensional representation
4. Concatenates embedding with remaining channels (belonging, arena_mask, tower HP, spell)
5. Processes through Conv2D layers

This embedding approach lets the network learn meaningful relationships between unit types (e.g., wizard and musketeer are both ranged splash), rather than relying on arbitrary integer ordering.
