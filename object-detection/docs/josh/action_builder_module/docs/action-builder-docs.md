# ActionBuilder Developer Documentation

## What Is the ActionBuilder?

The ActionBuilder is the action counterpart to the StateEncoder. While the StateEncoder converts `GameState` into observation tensors, the ActionBuilder handles the action side of the pipeline: converting raw click events from a human player into `Discrete(2305)` action indices, and converting agent action indices back into PyAutoGUI commands.

In the full pipeline:

```
Human plays game
    |
    v
Click Logger (OS-level mouse capture) -> (timestamp, x_norm, y_norm) events
    |
    v
ActionBuilder.clicks_to_actions() -> ActionEvent list (card_id, col, row, action_idx)
    |
    v
ActionBuilder.build_action_timeline() -> per-frame action labels [int]
    |
    v
DatasetBuilder merges with StateEncoder observations -> (obs, action_idx, mask) training data
```

For agent execution (the reverse direction):

```
Agent policy -> action_idx (0..2304)
    |
    v
ActionExecutor.execute(action_idx)
    |
    v
action_to_placement() -> (card_id, col, row)
    |
    v
cell_to_norm(col, row) -> (x_norm, y_norm)
    |
    v
PyAutoGUI: click card slot, wait 150ms, click arena position
```

The ActionBuilder lives in `src/action/` and depends on the shared coordinate utilities and constants from `src/encoder/`.

---

## Thought Process

### Why a click logger?

The hardest problem in behavior cloning for Clash Royale is knowing what the human intended. If we only record game state, we have to infer the action from frame differences -- which card was played, where it landed, when exactly the placement happened. That is an unsolved perception problem layered on top of another unsolved perception problem.

A click logger sidesteps this entirely. We directly record the human's mouse clicks with timestamps. Since Clash Royale on PC uses a two-click interaction (click card slot, then click arena), the clicks tell us exactly: which card (slot 0-3), where (x, y on screen), and when (wall-clock timestamp). No inference needed.

### Why two-click pairing?

In Clash Royale on Google Play Games, playing a card is a two-step interaction:

1. Click a card slot (bottom of screen) to select it
2. Click a position in the arena to place it

The click logger captures all mouse clicks as raw `(timestamp, x_norm, y_norm)` events. The ActionBuilder's job is to pair these into meaningful card placements. A card-slot click alone is not an action. An arena click without a prior card selection is not an action. Only the pair together forms a complete placement.

### Why a state machine?

Real gameplay is messy. Players might:
- Click card slot 0, change their mind, click card slot 2, then click the arena (the placement should use card 2)
- Click the arena without selecting a card first (nothing should happen)
- Click outside both the card bar and arena (UI interaction, ignore)
- Click a card slot and then click another UI element (cancel the selection)

A state machine handles all these cases cleanly with two states (`idle` and `card_selected`) and deterministic transitions based on click classification.

---

## Click Classification

Every click is classified into one of three categories based on its normalized screen coordinates:

| Category | Rule |
|----------|------|
| `card_0` through `card_3` | Click falls within one of the 4 card slot bounding boxes |
| `arena` | Click y-coordinate is within the arena region (above the card bar, below the timer) |
| `other` | Anything else (timer bar, next-card area, outside game window) |

### Card Slot Regions

Derived from `src/data/screen_regions.py` at 540x960 base resolution. Normalized to 0-1 range:

| Slot | x_min | x_max | y_min | y_max |
|------|-------|-------|-------|-------|
| 0 | 0.2037 | 0.3889 | 0.8021 | 0.9583 |
| 1 | 0.3889 | 0.5741 | 0.8021 | 0.9583 |
| 2 | 0.5741 | 0.7593 | 0.8021 | 0.9583 |
| 3 | 0.7593 | 0.9444 | 0.8021 | 0.9583 |

Source pixel values at 540x960:
- Card bar starts at x=110, each card is 100px wide
- Y range: 770 to 920

### Arena Region

The arena occupies the middle portion of the screen:
- Y range: `ARENA_Y_START_FRAC` (0.0521) to `ARENA_Y_END_FRAC` (0.7813)
- X range: full width (0.0 to 1.0)

A click is classified as `arena` if its `y_norm` is between these bounds and it does not fall within a card slot.

### Boundary Handling

Card slot regions are checked first. If a click matches a card slot, it is classified as that card regardless of whether the y-coordinate also falls in the arena range. This avoids ambiguity at the boundary between the card bar and the bottom of the arena.

---

## Click Pairing (State Machine)

The state machine has two states:

```
State: idle
  - Card click (card_0..card_3) -> transition to card_selected, remember card_id
  - Arena click -> ignore (no card selected, nothing to place)
  - Other click -> ignore

State: card_selected(card_id)
  - Arena click -> emit ActionEvent(card_id, col, row), transition to idle
  - Different card click -> update card_id, stay in card_selected
  - Same card click -> stay in card_selected (no change)
  - Other click -> cancel selection, transition to idle
```

### Walk-through Example

Given this click sequence from a gameplay session:

```
t=1.0  click at (0.30, 0.88)   -> card_0 (within slot 0)
t=1.2  click at (0.50, 0.40)   -> arena
t=3.5  click at (0.65, 0.90)   -> card_2 (within slot 2)
t=3.7  click at (0.85, 0.92)   -> card_3 (changed mind)
t=4.0  click at (0.30, 0.55)   -> arena
t=6.0  click at (0.50, 0.50)   -> arena (no card selected, ignored)
```

Result:
1. ActionEvent at t=1.2: card_0 placed at arena position (0.50, 0.40)
2. ActionEvent at t=4.0: card_3 placed at arena position (0.30, 0.55)
3. Click at t=6.0: ignored (no card was selected)

---

## Action Encoding

Every card placement maps to a single integer in `Discrete(2305)`:

```
action_idx = card_id * 576 + row * 18 + col
```

Where:
- `card_id`: 0-3 (which card slot)
- `row`: 0-31 (grid row, 0 = top of arena)
- `col`: 0-17 (grid column, 0 = left edge)
- `576` = `GRID_ROWS * GRID_COLS` = 32 * 18

The no-op action (do nothing / wait) is index 2304.

### Decoding

```python
action_to_placement(action_idx) -> (card_id, col, row) | None
```

Returns `None` for the no-op action (2304). For card placements:
- `card_id = action_idx // 576`
- `cell = action_idx % 576`
- `row = cell // 18`
- `col = cell % 18`

### Full range

| Action Range | Meaning |
|-------------|---------|
| 0 - 575 | Card slot 0, all 576 grid cells |
| 576 - 1151 | Card slot 1, all 576 grid cells |
| 1152 - 1727 | Card slot 2, all 576 grid cells |
| 1728 - 2303 | Card slot 3, all 576 grid cells |
| 2304 | No-op (wait) |

These functions are shared with the StateEncoder via `src/encoder/coord_utils.py`.

---

## Data Structures

### ClickEvent

A raw mouse click captured by the OS-level logger.

```python
@dataclass
class ClickEvent:
    timestamp: float   # Wall-clock time (time.time() when click occurred)
    x_norm: float      # Normalized window X coordinate (0=left edge, 1=right edge)
    y_norm: float      # Normalized window Y coordinate (0=top edge, 1=bottom edge)
```

The click logger produces a flat list of these. The ActionBuilder's job is to pair them into meaningful card placements.

### ActionEvent

A complete card placement action, produced by the state machine after pairing a card-slot click with an arena click.

```python
@dataclass
class ActionEvent:
    timestamp: float      # Time of the arena click (the second click in the pair)
    action_idx: int       # Discrete(2305) action index
    card_id: int          # Card slot index (0-3)
    col: int              # Grid column (0-17)
    row: int              # Grid row (0-31)
    x_norm: float         # Original arena click x_norm
    y_norm: float         # Original arena click y_norm
```

The timestamp corresponds to the arena click (the moment of placement), not the card selection click. This is the behaviorally meaningful instant -- when the unit actually appears on the field.

---

## Usage: Recording Clicks (BC Data Collection)

### Step 1: Capture clicks during gameplay

The click logger (Thread B) records all mouse clicks:

```python
from src.action import ClickEvent

clicks = []  # populated by the click logger thread

# Example: after a gameplay session, clicks might look like:
clicks = [
    ClickEvent(timestamp=100.0, x_norm=0.30, y_norm=0.88),  # card_0
    ClickEvent(timestamp=100.2, x_norm=0.50, y_norm=0.40),  # arena
    ClickEvent(timestamp=103.5, x_norm=0.65, y_norm=0.90),  # card_2
    ClickEvent(timestamp=104.0, x_norm=0.30, y_norm=0.55),  # arena
    # ... more clicks
]
```

### Step 2: Convert clicks to actions

```python
from src.action import ActionBuilder

builder = ActionBuilder()
action_events = builder.clicks_to_actions(clicks)

for event in action_events:
    print(f"t={event.timestamp:.1f}: card {event.card_id} at "
          f"grid ({event.col}, {event.row}), action_idx={event.action_idx}")
```

### Step 3: Assign per-frame action labels

```python
# frame_timestamps come from Thread A (the state capture thread)
frame_timestamps = [100.0, 100.5, 101.0, 101.5, ...]

action_labels = builder.build_action_timeline(clicks, frame_timestamps)
# action_labels[i] is the action index for frame i
# Most frames will be 2304 (no-op)
# Frames where a card was placed will have the corresponding action index
```

---

## Usage: Executing Actions (Agent Playback)

### Basic execution

```python
from src.action import ActionExecutor

executor = ActionExecutor(frame_w=540, frame_h=960)

# Agent outputs an action index
action_idx = 42  # some card placement

success = executor.execute(action_idx)
# Returns True if a card was played, False if no-op
```

### Internals of execute()

For a card placement action, `execute()` performs:
1. Decode: `action_to_placement(42)` -> `(card_id=0, col=6, row=2)`
2. Convert to normalized: `cell_to_norm(6, 2)` -> `(x_norm, y_norm)`
3. Call `play_card(card_id=0, x_norm, y_norm)` which does the two-click sequence

You can also call `play_card()` directly if you already have card_id and normalized coordinates:

```python
executor.play_card(card_id=2, x_norm=0.5, y_norm=0.4)
```

---

## Building the DatasetBuilder

The DatasetBuilder (not yet implemented) will merge perception data (Thread A) with action data (Thread B) into training samples. Here is the full data flow:

### Thread A: State Capture (2-4 FPS)

Captures screenshots via `mss`, runs them through `StateBuilder`, stores `(timestamp, GameState)` pairs.

### Thread B: Click Logging (OS-level)

Records all mouse clicks via `pynput` or similar, stores `ClickEvent(timestamp, x_norm, y_norm)` list.

### Post-game: Merge into Training Data

```python
from src.action import ActionBuilder
from src.encoder import StateEncoder

action_builder = ActionBuilder()
encoder = StateEncoder()

# Convert clicks to ActionEvents, then to per-frame labels
action_events = action_builder.clicks_to_actions(clicks)
frame_timestamps = [t for t, _ in frames]
action_labels = action_builder.build_action_timeline(clicks, frame_timestamps)

# Encode each frame into a training sample
dataset = []
for i, (timestamp, game_state) in enumerate(frames):
    obs = encoder.encode(game_state)
    mask = encoder.action_mask(game_state)
    dataset.append((obs, action_labels[i], mask))
```

### Key Design Decisions

**No downsampling or deduplication.** Every captured frame produces a training sample, even if the action is no-op for 90%+ of frames. The rationale: no-op is a valid and important action (knowing when NOT to play a card is half the game). The class imbalance (many no-ops vs few placements) will be handled at training time via weighted loss, not at data collection time.

**Timestamp-based assignment.** `build_action_timeline()` assigns each `ActionEvent` to the nearest frame by timestamp. An action event is assigned to the frame whose timestamp is closest to the action's timestamp. Frames without a nearby action get the no-op label (2304).

**Future: weighted loss.** During BC training, the loss function should weight card-placement actions higher than no-ops to counteract class imbalance. A typical game has ~180 frames at 2 FPS over 3 minutes, but only 10-20 card placements. Without weighting, the model would learn to always predict no-op.

---

## PyAutoGUI Execution Notes

### Two-click sequence timing

`play_card()` performs: (1) click card slot center, (2) wait 150ms for selection animation, (3) click arena position. The 150ms delay (`CLICK_DELAY_SECONDS`) is configurable in `action_constants.py`. Too fast and the game may not register the card selection; too slow and the card might deselect.

### Coordinate conversion and resolution scaling

Normalized coordinates (0-1) are converted to pixels: `pixel_x = int(x_norm * frame_w)`. The `frame_w` and `frame_h` passed to `ActionExecutor.__init__()` must match the actual game window size. Card slot positions are defined at 540x960 base resolution and scale proportionally via `ScreenConfig.scale_to_resolution()`.

### PyAutoGUI safety

- `pyautogui.FAILSAFE = True` (default): moving the mouse to the top-left corner aborts the script. Keep this enabled.
- `pyautogui.PAUSE = 0.05`: set low via `PYAUTOGUI_PAUSE` to minimize inter-call overhead. The 150ms delay is handled explicitly in `play_card()`.

### Potential issues

- **Game lag:** If the game lags, 150ms may not be enough. Increase `CARD_SELECT_DELAY` to 200-250ms.
- **Window focus:** PyAutoGUI clicks whatever window is in the foreground. The game must be focused.
- **Card selection state:** The executor always performs both clicks atomically (card + arena), so stale card selection should not occur.

---

## Assumptions and Limitations

### 1. Two-click interaction model

The ActionBuilder assumes card placement requires exactly two clicks: card slot then arena. This matches Google Play Games input. If the platform changes (e.g., drag-and-drop), the click pairing logic would need to be updated.

### 2. Fixed card slot positions

Card slot bounding boxes are derived from `screen_regions.py` at 540x960 and normalized to 0-1. If the game UI changes (card bar moves, card sizes change), the normalized regions in `action_constants.py` need updating.

### 3. Single game window

The click logger captures mouse clicks within the game window. It assumes the game window position and size are known and stable. Click coordinates are normalized relative to the game window, not the full desktop.

### 4. No validation of card availability

The ActionBuilder does not check whether the selected card is playable (enough elixir, card in hand). It faithfully records what the human clicked. The action mask in the StateEncoder handles validity at training time.

### 5. Click timing assumptions

The state machine assumes clicks arrive in chronological order. Out-of-order delivery (should not happen with a synchronous listener) may produce incorrect pairings.

### 6. No drag support

The ActionBuilder only handles discrete click pairs, not drag-and-drop card placement.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/action/__init__.py` | Module exports (ActionBuilder, ActionExecutor, data classes, constants) |
| `src/action/action_builder.py` | ActionBuilder class: click classification, pairing state machine, timeline building |
| `src/action/action_executor.py` | ActionExecutor class: PyAutoGUI execution of action indices |
| `src/action/action_constants.py` | Card slot regions, arena bounds, timing constants |
| `src/action/CLAUDE.md` | Technical reference (regions, state machine, dependencies) |

### Dependencies

- `src.encoder.coord_utils`: `norm_to_cell`, `cell_to_norm`, `placement_to_action`, `action_to_placement`
- `src.encoder.encoder_constants`: `NOOP_ACTION`, `ACTION_SPACE_SIZE`, `GRID_COLS`, `GRID_ROWS`
- `pyautogui`: for `ActionExecutor` (optional dependency, only needed for execution)
- `src.data.screen_regions`: `ScreenConfig` (source of card slot pixel coordinates)

### What depends on this module

- **DatasetBuilder** (not yet built) - will use `ActionBuilder.clicks_to_actions()` and `build_action_timeline()`
- **Gym Environment** (not yet built) - will use `ActionExecutor.execute()` inside `env.step()`
- **BC training script** (not yet built) - will use `ActionEvent` data to label training samples

---

## Common Questions

**Q: Why is the action space Discrete(2305) and not Discrete(577)?**
Each of the 4 card slots maps to a separate set of 576 grid cells. Card 0 at cell (5, 10) is a different action from card 1 at cell (5, 10). The agent needs to learn both which card to play and where to place it. 4 * 576 + 1 no-op = 2305.

**Q: What if the human clicks the same card slot twice?**
The state machine stays in `card_selected` with the same `card_id`. The second click is effectively a no-op. The next arena click will still use that card.

**Q: What if the human clicks a card slot but never clicks the arena?**
The selection eventually gets replaced (by clicking another card) or reset (by clicking something outside the card bar and arena). No ActionEvent is emitted for an unpaired card-slot click.

**Q: How does `build_action_timeline()` assign actions to frames?**
Each ActionEvent has a timestamp (from the arena click). The method finds the frame whose timestamp is closest to the action's timestamp and assigns that action index to that frame. All other frames get the no-op label (2304).

**Q: Can two ActionEvents map to the same frame?**
In theory, yes - if two card placements happen within half a frame interval. In practice, this is extremely rare at 2-4 FPS capture rate. If it happens, the later action overwrites the earlier one for that frame.

**Q: What if `execute()` is called with an invalid action index?**
The `action_to_placement()` function from `coord_utils` handles the range check. Action indices outside [0, 2304] would produce incorrect card_id values. The executor should validate the range. For the no-op action (2304), `execute()` returns `False` without clicking anything.

**Q: Why does ActionExecutor need frame_w and frame_h?**
PyAutoGUI works in pixel coordinates. The executor needs to convert normalized coordinates (0-1) to actual pixel positions on screen. The frame dimensions tell it the size of the game window.

**Q: Is the 150ms delay between clicks configurable?**
Yes, it is defined as `CLICK_DELAY_SECONDS` in `action_constants.py`. If the game consistently fails to register placements, try increasing it to 200-250ms. If responsiveness is more important, you could try reducing it to 100ms.
