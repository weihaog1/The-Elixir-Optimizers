# src/encoder/ - StateEncoder (GameState -> RL Observations)

Converts the structured `GameState` dataclass from `StateBuilder` into fixed-shape numpy tensors compatible with Stable Baselines 3 (SB3). This module is the bridge between perception and learning -- used identically by behavior cloning and PPO.

## Files

**state_encoder.py** - Core class.
- `StateEncoder`: Stateless encoder with `encode()` and `action_mask()`.
  - `encode(state: GameState) -> dict["arena": np.float32, "vector": np.float32]`
  - `action_mask(state: GameState) -> np.bool_ array (2305,)`
  - `observation_space`: gymnasium.spaces.Dict (arena + vector)
  - `action_space`: gymnasium.spaces.Discrete(2305)

**encoder_constants.py** - All constants and lookups.
- Grid: GRID_COLS=18, GRID_ROWS=32, GRID_CELLS=576
- Action: ACTION_SPACE_SIZE=2305, NOOP_ACTION=2304, NUM_CARD_SLOTS=4
- Arena bounds: ARENA_Y_START_FRAC, ARENA_Y_END_FRAC (from screen_regions.py 540x960)
- UNIT_TYPE_MAP: dict mapping 155 detection class names -> "ground"/"flying"/"spell"/"tower"/"other"
  - Built at import time from `src.generation.label_list`
- DECK_CARDS: 8 card names matching CardPredictor class names
- CARD_ELIXIR_COST: elixir cost per card
- CARD_IS_SPELL: whether card is a spell (placement rules)
- DEFAULT_KING_MAX_HP, DEFAULT_PRINCESS_MAX_HP: level-14 tower HP

**coord_utils.py** - Coordinate conversion utilities.
- `pixel_to_cell(cx, cy, frame_w, frame_h) -> (col, row)`: Pixel bbox center to 18x32 grid cell
- `norm_to_cell(x_norm, y_norm) -> (col, row)`: Normalized screen coords to grid cell
- `cell_to_norm(col, row) -> (x_norm, y_norm)`: Grid cell center to normalized screen coords
- `action_to_placement(action_idx) -> (card_id, col, row) | None`: Decode action index
- `placement_to_action(card_id, col, row) -> int`: Encode placement to action index

## Observation Space

### Arena Grid: shape (32, 18, 7), float32

Each cell in the 18-column x 32-row grid represents a spatial tile of the arena.

| Channel | Name | Range | Description |
|---------|------|-------|-------------|
| 0 | ally_ground | 0+ | Count of allied ground units in cell |
| 1 | ally_flying | 0+ | Count of allied flying units in cell |
| 2 | enemy_ground | 0+ | Count of enemy ground units in cell |
| 3 | enemy_flying | 0+ | Count of enemy flying units in cell |
| 4 | ally_tower_hp | 0-1 | Allied tower HP fraction (0 = no tower or destroyed) |
| 5 | enemy_tower_hp | 0-1 | Enemy tower HP fraction |
| 6 | spell | 0+ | Spell effect presence in cell |

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

## Coordinate System

The 18x32 grid maps to the arena portion of the screen (not the full screen):
- X: full screen width divided into 18 columns
- Y: arena region only (screen y=[50, 750] at 540x960 base resolution) divided into 32 rows
- Row 0 = top of arena (enemy king tower area)
- Row 31 = bottom of arena (player king tower area)
- Rows 15-16 = river area
- Player's deployable half: rows 17-31

Pixel-to-cell conversion accounts for non-arena screen regions (timer bar, card bar) via ARENA_Y_START_FRAC and ARENA_Y_END_FRAC constants.

## Dependencies

```
state_encoder.py
  -> src.pipeline.game_state (GameState, Tower, Unit, Card)
  -> encoder_constants (grid, action, deck, unit type constants)
  -> coord_utils (pixel_to_cell)

encoder_constants.py
  -> src.generation.label_list (ground/flying/spell/tower unit lists)

coord_utils.py
  -> encoder_constants (grid dimensions, arena fractions)
```

## Usage

```python
from src.encoder import StateEncoder
from src.pipeline.game_state import GameState, Tower, Unit, Card

encoder = StateEncoder()

# In BC data collection or Gym env:
obs = encoder.encode(game_state)   # {"arena": (32,18,7), "vector": (23,)}
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
