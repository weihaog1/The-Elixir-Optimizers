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

### Why separate arena + vector?

Some information is spatial (unit positions, tower locations), and some is scalar (elixir count, time remaining, which cards are in hand). Forcing scalar info into the grid would waste channels and make learning harder. SB3 supports `Dict` observation spaces, so we use:

- `"arena"`: spatial tensor (32, 18, 7) -- processed by conv layers
- `"vector"`: scalar tensor (23,) -- processed by dense layers

The SB3 policy network fuses these automatically via `MultiInputPolicy`.

### Why 7 arena channels?

Each grid cell needs to answer: "what's here, and whose is it?" We chose channels that capture the most strategically relevant distinctions:

| Channel | Why it matters |
|---------|---------------|
| Ally ground | Your ground troops -- the core of pushes and defenses |
| Ally flying | Your air units -- different targeting rules than ground |
| Enemy ground | Threats you need to respond to |
| Enemy flying | Air threats require specific counters (musketeer, flying machine) |
| Ally tower HP | How healthy your defenses are at this position |
| Enemy tower HP | Targeting priority -- low HP towers are worth pushing |
| Spell effect | Active spells on the field (freeze, poison, etc.) |

We deliberately did NOT add channels for:
- **Buildings vs troops**: Buildings are in the ground lists already. Adding a separate channel doubles the ground channels for minimal strategic gain at MVP stage.
- **Unit identity**: Encoding which specific unit is in each cell (one-hot over 155 classes) would create a 155+ channel tensor. The network doesn't need to know if it's a knight vs a valkyrie in cell (5, 20) -- it needs to know "there's an allied ground unit here."
- **Terrain/bridges**: These are static and never change. The network can learn this from data. We can add it later if needed.

### Why 23 vector features?

The vector encodes everything that doesn't have a spatial component:

- **Elixir (1 feature)**: How much you can spend right now. Normalized to [0, 1] by dividing by 10.
- **Time (2 features)**: Time remaining and overtime flag. Normalized by dividing by 300 (max game duration).
- **Tower HP (6 features)**: HP fraction for each of the 6 tower slots. This duplicates info from the arena grid, but having it in the vector lets the dense layers directly reason about tower health without spatial processing.
- **Tower counts (2 features)**: Quick summary of the tower situation (0/3, 1/3, 2/3, or 3/3 for each side).
- **Card hand (12 features = 4 slots x 3 features)**: For each of 4 card slots: is a card present (binary), which card (class index normalized), and its elixir cost (normalized). This tells the policy what options it has.

### Why Discrete(2305)?

The action space is: pick one of 4 cards and place it at one of 576 grid cells, or do nothing.

```
4 cards x 576 cells + 1 no-op = 2305 actions
```

This seems large, but it's fine for policy gradient methods (PPO). Unlike DQN, PPO doesn't enumerate all actions -- it just outputs logits over the action space and samples. The action mask zeros out invalid actions, so the effective action space is much smaller at any given moment (typically 576-1152 valid actions when 1-2 cards are playable).

Alternative designs we considered:
- **MultiDiscrete([5, 576])**: Separate card choice and placement. Problem: the two choices are correlated (spells go anywhere, troops go on your half), and SB3's masking support for MultiDiscrete is weaker.
- **Coarser grid (e.g., 9x16 = 144 cells)**: Would give Discrete(577). We rejected this because bridge placement precision matters -- the difference between placing at the left bridge vs center vs right bridge is 2-3 cells at 18-column resolution.

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

---

## How to Use It

### Basic usage

```python
from src.encoder import StateEncoder
from src.pipeline.game_state import GameState, Tower, Unit, Card

# Create once, reuse for all frames
encoder = StateEncoder()

# Get a GameState from the perception pipeline
# (In practice, this comes from StateBuilder.build_state(screenshot))
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
]
state.cards = [
    Card(0, "royal-hogs", 5, 0.95),
    Card(1, "arrows", 3, 0.90),
]

# Encode
obs = encoder.encode(state)
mask = encoder.action_mask(state)

# obs["arena"] is a (32, 18, 7) float32 numpy array
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
    # Pass to PyAutoGUI play_card function
    play_card(card_id, x_norm, y_norm)
```

### Inspecting observations (debugging)

```python
import numpy as np

obs = encoder.encode(state)

# Which cells have units?
occupied = (obs["arena"] != 0).any(axis=2)
print(f"Occupied cells: {occupied.sum()} / 576")

# Where are allied ground troops?
ally_ground = obs["arena"][:, :, 0]
rows, cols = np.where(ally_ground > 0)
for r, c in zip(rows, cols):
    print(f"  Allied ground at cell ({c}, {r}), count={ally_ground[r, c]}")

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

| File | Lines | Purpose |
|------|-------|---------|
| `src/encoder/__init__.py` | 11 | Module exports (StateEncoder, coord functions, constants) |
| `src/encoder/encoder_constants.py` | 120 | All constants: grid, actions, deck, unit types, arena bounds |
| `src/encoder/coord_utils.py` | 100 | Coordinate conversions: pixel/norm/cell/action |
| `src/encoder/state_encoder.py` | 200 | StateEncoder class: encode() + action_mask() |
| `src/encoder/CLAUDE.md` | 130 | Technical reference (observation/action specs, channel tables) |

### Dependencies

- `numpy` -- array operations
- `gymnasium` -- observation/action space definitions (for SB3 compatibility)
- `src.pipeline.game_state` -- GameState, Tower, Unit, Card dataclasses
- `src.generation.label_list` -- ground/flying/spell/tower unit classification lists

### What depends on this module

- **BC DatasetBuilder** (not yet built) -- will call `encode()` and `action_mask()` per frame
- **Gym Environment** (not yet built) -- will use `observation_space`, `action_space`, `encode()`, `action_mask()`
- **Action Executor** (not yet built) -- will use `action_to_placement()` and `cell_to_norm()`

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

**Q: Why is the arena shape (32, 18, 7) and not (18, 32, 7)?**
Convention: the first dimension is rows (height/y), second is columns (width/x). This matches numpy/image conventions where `array[row, col]` gives you the value at position (x=col, y=row). It also matches what Conv2D layers expect (height, width, channels).

**Q: Can the network handle the large action space (2305)?**
Yes. PPO uses policy gradients, not value enumeration. The output layer has 2305 logits, which is comparable to Atari (18 actions) scaled up. The action mask zeros out ~50-80% of actions at any moment, so the effective space is smaller. For comparison, OpenAI Five had ~170,000 possible actions per step.

**Q: What if a unit class isn't in UNIT_TYPE_MAP?**
It defaults to "ground". This is the safest fallback since most Clash Royale units are ground troops. The UNIT_TYPE_MAP covers 152 of 155 detection classes (the 3 missing are edge cases like `"selected"`, `"mirror"`, `"evolution-symbol"` which appear rarely in real gameplay).

**Q: Why not use one-hot encoding for card identity in the vector?**
With 8 deck cards, one-hot would add 32 features (4 slots x 8 classes). The normalized index approach uses 4 features. Since the deck is fixed (same 8 cards every game), the network can learn the mapping from a single normalized index to card identity. If we later support multiple decks, one-hot might be worth reconsidering.

**Q: How fast is encode()?**
Negligible. The encoder does simple array indexing and arithmetic -- no ML inference, no I/O. On an M1 Pro, `encode()` + `action_mask()` takes <0.1ms per frame. The bottleneck is always YOLO inference (65ms) and OCR (30ms), not encoding.
