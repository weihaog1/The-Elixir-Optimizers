# StateEncoder Developer Documentation

## What Is the StateEncoder?

The StateEncoder is a translation layer that converts human-readable game state (Python dataclasses with strings, bounding boxes, HP values) into fixed-shape numeric tensors that a neural network can process.

In our pipeline, the flow is:

```
Screenshot (pixels)
    |
    v
StateBuilder (YOLO + OCR)
    |
    v
GameState (Python dataclass -- variable-length lists, optional fields, pixel coords)
    |
    v
StateEncoder  <--- this module
    |
    v
Observation dict (fixed-shape numpy arrays -- always the same dimensions)
    + Action mask (which actions are legal right now)
    |
    v
Neural network policy (BC or PPO via Stable Baselines 3)
```

Without the StateEncoder, you cannot train any learning algorithm. Neural networks require fixed-size numeric inputs. A `GameState` with 3 units one frame and 12 the next is not fixed-size. The StateEncoder solves this by projecting everything onto a fixed spatial grid and a fixed-length feature vector.

---

## Thought Process

### Why a spatial grid?

Clash Royale is fundamentally a spatial game. Where you place a card matters as much as which card you play. A hog rider at the bridge is completely different from a hog rider behind the king tower.

We needed a representation that preserves spatial relationships. Two options:

1. **Flat vector**: List all unit positions as (x, y, class) tuples, padded to some max count. Problem: the network has to learn spatial relationships from raw coordinates. Also, unit ordering is arbitrary, which wastes capacity.

2. **Spatial grid**: Divide the arena into cells and encode what's in each cell. The network can use convolutional layers to learn spatial patterns (pushes, defenses, bridge spam). This is how game-playing AIs typically represent board-like state (AlphaGo, OpenAI Five, etc.).

We chose option 2. The grid is 18 columns by 32 rows, matching KataCR's generator grid from `generation_config.py`. This resolution gives roughly 2-tile precision in-game, which is enough to distinguish bridge placements, split pushes, and lane assignments.

### Why per-cell identity encoding?

The original encoder used **additive counting**: each cell held counts of ally ground, ally flying, enemy ground, and enemy flying units. This was simpler but lost critical information:

- **No unit identity**: The network couldn't distinguish a knight from a PEKKA in the same cell. A cell with "1 ally ground" could be a 3-elixir knight or an 11-elixir three musketeers -- very different strategic situations.
- **Swarm confusion**: 15 skeletons stacking in one cell showed count=15, but the network learned nothing about what kind of units they were.

KataCR's approach uses **per-cell identity encoding**: each cell holds at most one unit, identified by its class ID. When multiple units map to the same cell, a `PositionFinder` displaces extras to nearby free cells. This preserves:

- **Unit identity**: The network knows exactly which unit type is in each cell
- **Spatial accuracy**: Units near each other stay near each other (displaced to adjacent cells, not lost)
- **Embedding potential**: Class IDs can feed into a learned embedding layer, letting the network discover unit relationships (e.g., wizard and musketeer are both ranged splash)

The trade-off: with per-cell encoding, at most 576 units can be represented (one per cell). In practice, Clash Royale rarely has more than 30-40 units on screen at once, so this limit is never hit.

### Why separate arena + vector?

Some information is spatial (unit positions, tower locations), and some is scalar (elixir count, time remaining, which cards are in hand). Forcing scalar info into the grid would waste channels and make learning harder. SB3 supports `Dict` observation spaces, so we use:

- `"arena"`: spatial tensor (32, 18, 6) -- processed by conv layers
- `"vector"`: scalar tensor (23,) -- processed by dense layers

The SB3 policy network fuses these automatically via `MultiInputPolicy`.

### Why 6 arena channels?

Each grid cell needs to answer: "what's here, whose is it, and what towers/spells are active?"

| Channel | Constant | Purpose |
|---------|----------|---------|
| 0 | CH_CLASS_ID | Which unit type (normalized class index 0-1) |
| 1 | CH_BELONGING | Ally (-1) or enemy (+1), 0 if empty |
| 2 | CH_ARENA_MASK | Is there a unit here? (binary presence flag) |
| 3 | CH_ALLY_TOWER_HP | How healthy the allied tower at this position is |
| 4 | CH_ENEMY_TOWER_HP | How healthy the enemy tower at this position is |
| 5 | CH_SPELL | Active spell effects (additive count) |

Channels 0-2 are the per-cell unit identity features (one unit per cell, enforced by PositionFinder). Channels 3-5 are continuous/additive features that do not participate in PositionFinder.

We deliberately kept spells in a separate additive channel rather than per-cell. Spells are area effects, not individual units with positions -- a fireball covers many cells simultaneously. Putting them through PositionFinder would misrepresent their spatial extent.

### Why Discrete(2305)?

The action space is: pick one of 4 cards and place it at one of 576 grid cells, or do nothing.

```
4 cards x 576 cells + 1 no-op = 2305 actions
```

This seems large, but it's fine for policy gradient methods (PPO). Unlike DQN, PPO doesn't enumerate all actions -- it just outputs logits over the action space and samples. The action mask zeros out invalid actions, so the effective action space is much smaller at any given moment (typically 576-1152 valid actions when 1-2 cards are playable).

---

## PositionFinder: How Collision Resolution Works

When multiple units map to the same grid cell, PositionFinder displaces later units to the nearest free cell.

### Algorithm

1. Create a fresh `PositionFinder(rows=32, cols=18)` for each frame
2. Pre-compute all 576 cell center coordinates: `(col + 0.5, row + 0.5)` for every cell
3. Sort units before placement:
   - Enemy units first (belonging=1), then ally units (belonging=0)
   - Within each faction, sort by center_y descending (bottom of screen first)
   - This gives priority placement to enemy units (the ones you need to react to)
4. For each unit:
   - Convert pixel coordinates to continuous grid position via `pixel_to_cell_float()`
   - Call `find_position(col_f, row_f)`:
     - Clamp to grid bounds
     - If the natural cell (integer truncation) is free: use it
     - If occupied: compute Euclidean distance from the continuous position to all 576 cell centers using `scipy.spatial.distance.cdist`, mask occupied cells with infinity, pick the closest free cell
   - Write class_id / NUM_CLASSES, belonging flag, and arena_mask=1 to the assigned cell

### Example

Two knights at pixel (270, 480) on a 540x960 frame both map to approximately grid cell (9, 18). The first knight gets cell (9, 18). The second knight's natural cell is occupied, so PositionFinder finds the nearest free cell -- perhaps (9, 17) or (10, 18) depending on which is closer. Both knights are represented in the arena tensor, just in adjacent cells.

### What bypasses PositionFinder

- **Spells**: Go directly to CH_SPELL with additive counting. Multiple spells can stack in the same cell.
- **Towers**: Go to CH_ALLY_TOWER_HP or CH_ENEMY_TOWER_HP. Towers have fixed positions and don't collide with units.
- **"tower" and "other" unit types**: Skipped entirely (tower detections handled by tower HP channels, UI elements like "bar" are irrelevant).
- **Unknown classes**: Units with class_name not in CLASS_NAME_TO_ID are skipped.

### Displacement caveats

- A displaced unit appears in a cell it doesn't actually occupy in-game. For a ground unit pushed 1-2 cells away, this is acceptable -- the network still sees it nearby. For extreme cases (many units in one spot), displacement can push units several cells away. In practice, Clash Royale gameplay rarely has more than 4-5 units overlapping at the same position.
- The sort order means enemy units get priority for their natural cells. Allied units are more likely to be displaced. This is intentional -- accurate enemy positioning matters more for decision-making.

---

## Assumptions

### 1. Single fixed deck

The encoder assumes you always play the same 8-card deck (Royal Hogs / Royal Recruits). Card names, elixir costs, and spell/troop classification are hard-coded in `encoder_constants.py`. If you change decks, you must update:
- `DECK_CARDS` list
- `CARD_ELIXIR_COST` dict
- `CARD_IS_SPELL` dict
- Retrain the CardPredictor with new reference card images

### 2. CardPredictor class names match filename stems

The card classifier (`src/classification/card_classifier.py`) loads class names from PNG filenames in the training directory. The reference images in `data/deck-card-crops/frames/` have a typo: `eletro-spirit.png` (missing 'c'). The encoder uses `"eletro-spirit"` to match. If someone fixes the filename and retrains, the encoder constants must be updated too.

### 3. Arena occupies a fixed fraction of the screen

The encoder maps pixel coordinates to grid cells using fixed arena bounds:
- Arena top: 50/960 = 5.2% of screen height (below the timer bar)
- Arena bottom: 750/960 = 78.1% of screen height (above the card bar)

These fractions come from `src/data/screen_regions.py` at 540x960 base resolution. They scale proportionally to any resolution. If the game UI changes (different arena position), these constants need updating.

### 4. Y-position heuristic for unit belonging

The current YOLO model does not output which side a unit belongs to. StateBuilder uses a Y-coordinate heuristic: units above 42% of frame height are enemy, below are ally. This means:
- **When your hog rider crosses the river**, it gets classified as enemy in the observation
- **When enemy troops push deep**, they might get classified as ally

This is a known limitation. The encoder faithfully encodes whatever belonging StateBuilder assigns. Fixing this requires retraining the YOLO model with belonging labels (code is ported in `src/yolov8_custom/` but unused).

### 5. Tower HP defaults to level 14

When OCR detects tower HP but `max_hp` is unknown, the encoder assumes level 14 tournament standard:
- King tower: 6408 HP
- Princess tower: 4032 HP

If you play at a different level, the HP fractions will be slightly off. This is acceptable because the network learns relative patterns, not absolute HP values.

### 6. Unknown values default to 0.0

When OCR fails (returns None for elixir/timer) or a tower isn't detected, the encoder outputs 0.0 for those features. This is safe for neural networks -- 0.0 is a neutral value that doesn't push activations in any direction. The network learns to handle missing data implicitly through training on noisy observations.

### 7. Action mask only checks card availability and elixir

The mask does NOT enforce:
- Troop placement only on your half of the arena
- Building placement distance restrictions
- Spell targeting rules

The agent learns valid placements from demonstration data (BC) and experience (PPO). This keeps the mask simple and avoids encoding game rules that might be wrong or incomplete.

### 8. At most 576 units can be represented

With per-cell encoding, each cell holds one unit. The arena has 576 cells (18x32). If more than 576 ground/flying units were on screen, PositionFinder would raise a RuntimeError. In practice, Clash Royale never has anywhere near this many units (30-40 is a crowded board).

---

## How to Use It

### Basic usage

```python
from src.encoder import StateEncoder
from src.pipeline.game_state import GameState, Tower, Unit, Card

# Create once, reuse for all frames
encoder = StateEncoder()

# Get a GameState from the perception pipeline
state = GameState(
    time_remaining=120,
    elixir=7,
    player_king_tower=Tower("king", "center", 0, hp=4000, max_hp=6408),
    enemy_king_tower=Tower("king", "center", 1, hp=6408, max_hp=6408),
    frame_width=540,
    frame_height=960,
)
state.units = [
    Unit("royal-hog", 0, (200, 400, 230, 430), 0.9),
    Unit("knight", 0, (300, 500, 320, 520), 0.85),
    Unit("skeleton", 1, (100, 200, 120, 220), 0.8),
]
state.cards = [
    Card(0, "royal-hogs", 5, 0.95),
    Card(1, "arrows", 3, 0.90),
]

# Encode
obs = encoder.encode(state)
mask = encoder.action_mask(state)

# obs["arena"] is a (32, 18, 6) float32 numpy array
# obs["vector"] is a (23,) float32 numpy array
# mask is a (2305,) bool numpy array
```

### For behavior cloning data collection

```python
from src.encoder import StateEncoder, placement_to_action
from src.encoder.coord_utils import norm_to_cell

encoder = StateEncoder()

# For each (state, click) pair from the click logger:
obs = encoder.encode(game_state)
mask = encoder.action_mask(game_state)

# Convert the click logger's (card_id, x_norm, y_norm) to an action index
if click is None:
    action = 2304  # no-op
else:
    card_id, x_norm, y_norm = click
    col, row = norm_to_cell(x_norm, y_norm)
    action = placement_to_action(card_id, col, row)

# Save as training sample
dataset.append((obs, action, mask))
```

### For a Gym environment (PPO)

```python
import gymnasium as gym
from src.encoder import StateEncoder

class ClashRoyaleEnv(gym.Env):
    def __init__(self):
        self.encoder = StateEncoder()
        self.observation_space = self.encoder.observation_space
        self.action_space = self.encoder.action_space

    def reset(self, **kwargs):
        state = self._start_new_game()
        obs = self.encoder.encode(state)
        return obs, {"action_mask": self.encoder.action_mask(state)}

    def step(self, action):
        self._execute_action(action)
        state = self._capture_state()
        obs = self.encoder.encode(state)
        mask = self.encoder.action_mask(state)
        reward = self._compute_reward(state)
        done = self._check_game_over(state)
        return obs, reward, done, False, {"action_mask": mask}
```

### For executing an action chosen by the agent

```python
from src.encoder import action_to_placement
from src.encoder.coord_utils import cell_to_norm

# Agent outputs an action index (0..2304)
action = policy.predict(obs, action_masks=mask)

result = action_to_placement(action)
if result is None:
    pass  # no-op, wait
else:
    card_id, col, row = result
    x_norm, y_norm = cell_to_norm(col, row)
    # x_norm, y_norm are screen fractions (0-1)
    play_card(card_id, x_norm, y_norm)
```

### Inspecting observations (debugging)

```python
import numpy as np
from src.encoder.encoder_constants import (
    CH_CLASS_ID, CH_BELONGING, CH_ARENA_MASK, CH_SPELL,
    NUM_CLASSES, CLASS_NAME_TO_ID,
)

obs = encoder.encode(state)
arena = obs["arena"]

# Find all cells with units
occupied = arena[:, :, CH_ARENA_MASK] > 0
print(f"Occupied cells: {occupied.sum()} / 576")

# List each unit with its identity
for row, col in np.argwhere(occupied):
    class_idx = int(round(arena[row, col, CH_CLASS_ID] * NUM_CLASSES))
    belonging = arena[row, col, CH_BELONGING]
    side = "ally" if belonging < 0 else "enemy"
    # Reverse-lookup class name
    name = next((k for k, v in CLASS_NAME_TO_ID.items() if v == class_idx), "unknown")
    print(f"  cell ({col}, {row}): {name} ({side})")

# Check spell activity
spell_cells = np.argwhere(arena[:, :, CH_SPELL] > 0)
for row, col in spell_cells:
    count = arena[row, col, CH_SPELL]
    print(f"  spell at ({col}, {row}): count={count:.0f}")

# What's in the vector?
labels = [
    "elixir", "time", "overtime",
    "p_king_hp", "p_left_hp", "p_right_hp",
    "e_king_hp", "e_left_hp", "e_right_hp",
    "p_towers", "e_towers",
    "card0", "card1", "card2", "card3",
    "card0_cls", "card1_cls", "card2_cls", "card3_cls",
    "card0_cost", "card1_cost", "card2_cost", "card3_cost",
]
for i, (label, val) in enumerate(zip(labels, obs["vector"])):
    if val != 0:
        print(f"  [{i:2d}] {label}: {val:.4f}")
```

---

## File Reference

| File | Purpose |
|------|---------|
| `src/encoder/__init__.py` | Module exports (StateEncoder, PositionFinder, coord functions, constants) |
| `src/encoder/state_encoder.py` | StateEncoder class: encode() + action_mask() |
| `src/encoder/position_finder.py` | PositionFinder: grid cell collision resolution via cdist |
| `src/encoder/encoder_constants.py` | All constants: grid, channels, actions, deck, unit types, class IDs |
| `src/encoder/coord_utils.py` | Coordinate conversions: pixel/norm/cell/action, including pixel_to_cell_float |
| `src/encoder/CLAUDE.md` | Technical reference (observation/action specs, channel tables) |

### Dependencies

- `numpy` -- array operations
- `scipy` -- `scipy.spatial.distance.cdist` for PositionFinder nearest-neighbor lookup
- `gymnasium` -- observation/action space definitions (for SB3 compatibility)
- `src.pipeline.game_state` -- GameState, Tower, Unit, Card dataclasses
- `src.generation.label_list` -- unit_list (155 classes), ground/flying/spell/tower/other sublists

### What depends on this module

- **BC DatasetBuilder** (not yet built) -- will call `encode()` and `action_mask()` per frame
- **Gym Environment** (not yet built) -- will use `observation_space`, `action_space`, `encode()`, `action_mask()`
- **Action Executor** (not yet built) -- will use `action_to_placement()` and `cell_to_norm()`
- **CRFeatureExtractor** (not yet built) -- custom SB3 feature extractor that reads class_id from channel 0, denormalizes to integer, and feeds into `nn.Embedding(156, 8)`

---

## BC Network Integration (Future)

The per-cell encoding is designed to work with a custom SB3 feature extractor:

```python
import torch
import torch.nn as nn
from src.encoder.encoder_constants import NUM_CLASSES

class CRFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_CLASSES + 1, 8)  # 156 entries, 8 dims
        # ... conv layers, etc.

    def forward(self, obs):
        arena = obs["arena"]  # (B, 32, 18, 6)

        # Extract and denormalize class IDs
        class_id_norm = arena[:, :, :, 0]  # (B, 32, 18)
        class_idx = torch.round(class_id_norm * NUM_CLASSES).long()  # integers 0-155

        # Learned embedding
        embedded = self.embedding(class_idx)  # (B, 32, 18, 8)

        # Concatenate with remaining channels (belonging, mask, tower HP, spell)
        other_channels = arena[:, :, :, 1:]  # (B, 32, 18, 5)
        grid_input = torch.cat([embedded, other_channels], dim=-1)  # (B, 32, 18, 13)

        # ... process with Conv2D layers
```

This embedding approach lets the network learn meaningful relationships between unit types (e.g., wizard and musketeer are both ranged splash), rather than relying on arbitrary integer ordering.

---

## Changing the Deck

If you switch to a different deck (e.g., 2.6 Hog Cycle):

1. Create new reference card images in `data/deck-card-crops/frames/` (one PNG per card, named after the card)
2. Retrain the CardPredictor: `python src/classification/card_classifier.py train --data data/deck-card-crops/frames/`
3. Update `encoder_constants.py`:
   - `DECK_CARDS` -- list of 8 card name strings (must match PNG filenames without extension)
   - `CARD_ELIXIR_COST` -- elixir cost for each card
   - `CARD_IS_SPELL` -- True for spell cards, False for troops/buildings
4. The StateEncoder, coord_utils, and action space do not change -- only the card metadata does

---

## Common Questions

**Q: Why is the arena shape (32, 18, 6) and not (18, 32, 6)?**
Convention: the first dimension is rows (height/y), second is columns (width/x). This matches numpy/image conventions where `array[row, col]` gives you the value at position (x=col, y=row). It also matches what Conv2D layers expect (height, width, channels).

**Q: Can the network handle the large action space (2305)?**
Yes. PPO uses policy gradients, not value enumeration. The output layer has 2305 logits, which is comparable to Atari (18 actions) scaled up. The action mask zeros out ~50-80% of actions at any moment, so the effective space is smaller. For comparison, OpenAI Five had ~170,000 possible actions per step.

**Q: What if a unit class isn't in UNIT_TYPE_MAP?**
It defaults to "ground". This is the safest fallback since most Clash Royale units are ground troops. The UNIT_TYPE_MAP covers all detection classes in the type sublists.

**Q: What if a unit class isn't in CLASS_NAME_TO_ID?**
It gets class_idx=0 (the default for `.get()`), which causes the unit to be skipped (the encoder checks `if class_idx == 0: continue`). This means units with unknown class names are silently dropped from the arena tensor.

**Q: Why not use one-hot encoding for card identity in the vector?**
With 8 deck cards, one-hot would add 32 features (4 slots x 8 classes). The normalized index approach uses 4 features. Since the deck is fixed (same 8 cards every game), the network can learn the mapping from a single normalized index to card identity. If we later support multiple decks, one-hot might be worth reconsidering.

**Q: How fast is encode()?**
The encoder itself is negligible (<0.1ms per frame on an M1 Pro). The main cost is PositionFinder's `cdist` call when collisions occur, but even with 30+ units and several collisions, this takes under 1ms. The bottleneck is always YOLO inference (65ms) and OCR (30ms), not encoding.

**Q: What happens if PositionFinder displaces a unit far from its real position?**
In extreme cases (many units clustered), a unit could be displaced several cells away. The sort order mitigates this: enemy units get placed first (better positional accuracy for threats), and within each faction, bottom-of-screen units go first. For typical Clash Royale boards (20-40 units), displacement is usually 0-2 cells.

**Q: Why does the observation space have low=-1.0?**
The belonging channel (CH_BELONGING) uses -1.0 for allied units. Without low=-1.0, SB3 would flag observations outside the Box bounds. The high=10.0 accommodates the spell channel which can accumulate multiple spell effects in one cell.
