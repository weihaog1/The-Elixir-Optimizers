# src/action/ - ActionBuilder (Clicks -> Actions) and ActionExecutor (Actions -> PyAutoGUI)

Converts raw mouse clicks from a human player into Discrete(2305) action indices for BC training, and converts agent action indices back into PyAutoGUI two-click sequences for execution.

## Files

**action_builder.py** - Click processing and action labeling.
- `ActionBuilder`: Stateless processor for click-to-action conversion.
  - `classify_click(x_norm, y_norm) -> str`: Returns `"card_0"`.."card_3"`, `"arena"`, or `"other"`
  - `clicks_to_actions(clicks: list[ClickEvent]) -> list[ActionEvent]`: Pairs card-slot + arena clicks via state machine
  - `build_action_timeline(clicks: list[ClickEvent], frame_timestamps: list[float]) -> list[int]`: Per-frame action labels (2304 = no-op)

**action_executor.py** - PyAutoGUI execution.
- `ActionExecutor(frame_w=540, frame_h=960)`: Converts action indices to mouse clicks.
  - `execute(action_idx: int) -> bool`: Full action execution. Returns True for card play, False for no-op.
  - `play_card(card_id: int, x_norm: float, y_norm: float) -> None`: Two-click sequence (card slot then arena position)

**action_constants.py** - Card slot regions and timing.
- `CARD_SLOT_REGIONS`: Normalized bounding boxes for 4 card slots
- `ARENA_Y_MIN`, `ARENA_Y_MAX`: Arena click region bounds
- `CLICK_DELAY_SECONDS`: Delay between card click and arena click (0.15s)

**__init__.py** - Module exports (ActionBuilder, ActionExecutor, ClickEvent, ActionEvent, constants).

## Data Structures

### ClickEvent
```python
@dataclass
class ClickEvent:
    timestamp: float   # Wall-clock time
    x_norm: float      # Normalized window X (0=left, 1=right)
    y_norm: float      # Normalized window Y (0=top, 1=bottom)
```

### ActionEvent
```python
@dataclass
class ActionEvent:
    timestamp: float   # Time of arena click (second click in pair)
    action_idx: int    # Discrete(2305) index
    card_id: int       # Card slot (0-3)
    col: int           # Grid column (0-17)
    row: int           # Grid row (0-31)
    x_norm: float      # Original arena click x_norm
    y_norm: float      # Original arena click y_norm
```

## Card Slot Regions (Normalized from 540x960)

| Slot | x_min | x_max | y_min | y_max |
|------|-------|-------|-------|-------|
| 0 | 0.2037 | 0.3889 | 0.8021 | 0.9583 |
| 1 | 0.3889 | 0.5741 | 0.8021 | 0.9583 |
| 2 | 0.5741 | 0.7593 | 0.8021 | 0.9583 |
| 3 | 0.7593 | 0.9444 | 0.8021 | 0.9583 |

Source: `screen_regions.py` card_start_x=110, card_width=100, y=[770,920] at 540x960.

## Click Classification Rules

1. Check card slots first (highest priority)
2. If not a card slot and `ARENA_Y_MIN <= y_norm <= ARENA_Y_MAX`: classify as `"arena"`
3. Otherwise: `"other"`

Arena Y bounds: 0.0521 (top, below timer) to 0.7813 (bottom, above card bar).

## Click Pairing State Machine

```
idle -> card click       -> card_selected(card_id)
idle -> arena/other      -> ignore, stay idle

card_selected -> arena click       -> emit ActionEvent, go to idle
card_selected -> different card    -> update card_id, stay card_selected
card_selected -> other click       -> cancel, go to idle
```

## Action Encoding

```
action = card_id * 576 + row * 18 + col    (0..2303)
action = 2304                                (no-op)
```

Uses `placement_to_action()` and `action_to_placement()` from `src.encoder.coord_utils`.

## Dependencies

```
action_builder.py
  -> src.encoder.coord_utils (norm_to_cell, placement_to_action)
  -> src.encoder.encoder_constants (NOOP_ACTION)
  -> action_constants (CARD_SLOT_REGIONS, ARENA_Y_MIN, ARENA_Y_MAX)

action_executor.py
  -> src.encoder.coord_utils (action_to_placement, cell_to_norm)
  -> src.encoder.encoder_constants (NOOP_ACTION, GRID_COLS, GRID_ROWS)
  -> action_constants (CARD_SLOT_REGIONS, CARD_SELECT_DELAY)
  -> pyautogui (optional, only for execution)

action_constants.py
  -> src.encoder.encoder_constants (ARENA_Y_START_FRAC, ARENA_Y_END_FRAC)
```

## Usage

### Recording clicks for BC

```python
from src.action import ActionBuilder, ClickEvent

builder = ActionBuilder()

# After gameplay session:
clicks = [ClickEvent(100.0, 0.30, 0.88), ClickEvent(100.2, 0.50, 0.40), ...]
action_events = builder.clicks_to_actions(clicks)

# Per-frame labels:
frame_timestamps = [100.0, 100.5, 101.0, ...]
labels = builder.build_action_timeline(clicks, frame_timestamps)
# labels[i] is int in [0, 2304]
```

### Executing agent actions

```python
from src.action import ActionExecutor

executor = ActionExecutor(frame_w=540, frame_h=960)
success = executor.execute(action_idx)  # True if card played, False if no-op
```
