# BC Data Collection Guide

## What Needs to Run During a Match

Recording match data requires two concurrent data streams during gameplay:

| Thread | What it does | Runs during game? | Output |
|--------|-------------|-------------------|--------|
| **Thread A: State Capture** | `mss.capture()` at 2-4 FPS -> save screenshots | **YES** | List of `(timestamp, screenshot)` |
| **Thread B: Click Logger** | OS-level mouse listener (pynput) | **YES** | List of `(timestamp, x_norm, y_norm)` |
| **ActionBuilder** | Pairs clicks into action indices | **NO** -- post-game | List of `ActionEvent` |
| **StateEncoder** | Converts GameState to tensors | **NO** -- post-game | `(obs_dict, action_mask)` per frame |
| **DatasetBuilder** | Merges everything into training tuples | **NO** -- post-game | `(obs, action_idx, mask)` dataset |

The click logger alone is not enough. Without Thread A capturing screenshots and producing GameStates, you have actions but no observations. The training data is `(observation, action)` pairs -- both sides are required.

---

## Can You Start Recording With Just the Click Logger?

Partially, with a lightweight alternative for Thread A. There are two approaches:

**Option A: Lightweight recording (recommended to start)**

- Thread A saves raw screenshots (PNG/JPG) + timestamps via `mss` -- no YOLO inference during gameplay
- Thread B logs clicks with timestamps
- Post-game: run StateBuilder on each saved screenshot offline
- Advantage: zero latency impact on gameplay, simpler to implement
- Disadvantage: uses more disk space (raw images), slower post-processing

**Option B: Full live pipeline**

- Thread A runs `mss` -> StateBuilder (YOLO + OCR) live at 2-4 FPS
- Thread B logs clicks with timestamps
- Post-game: data is already structured, just needs encoding
- Advantage: faster post-processing
- Disadvantage: YOLO inference (65ms) + OCR (30-80ms per region) adds CPU/GPU load during gameplay, potential frame drops

**Recommendation: Start with Option A.** Just capture screenshots + clicks. You can start recording matches as soon as you have:

1. A click logger (captures mouse events with normalized coordinates)
2. An `mss` screen capture loop (saves timestamped screenshots)

StateBuilder, ActionBuilder, StateEncoder, and DatasetBuilder all run offline after the match.

---

## The Click Logger (NOT YET IMPLEMENTED)

The click logger is the single biggest blocker for recording match data. It needs to:

1. Listen for all mouse clicks via `pynput.mouse.Listener` (OS-level)
2. Check if the click is within the game window bounds
3. Normalize coordinates to 0-1 relative to the game window: `x_norm = (click_x - window_x) / window_w`
4. Store as `ClickEvent(timestamp=time.time(), x_norm, y_norm)`
5. Save the full click log to disk when recording stops

Minimal implementation sketch:

```python
from pynput import mouse
import time

class ClickLogger:
    def __init__(self, window_x, window_y, window_w, window_h):
        self.wx, self.wy = window_x, window_y
        self.ww, self.wh = window_w, window_h
        self.clicks = []
        self.listener = None

    def start(self):
        self.listener = mouse.Listener(on_click=self._on_click)
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()

    def _on_click(self, x, y, button, pressed):
        if not pressed:  # only capture press, not release
            return
        # Check if click is within game window
        if (self.wx <= x < self.wx + self.ww and
            self.wy <= y < self.wy + self.wh):
            x_norm = (x - self.wx) / self.ww
            y_norm = (y - self.wy) / self.wh
            self.clicks.append(ClickEvent(
                timestamp=time.time(),
                x_norm=x_norm,
                y_norm=y_norm,
            ))
```

---

## Detailed Steps to Record Match Data

### Prerequisites

Before your first recording session:

1. Google Play Games running Clash Royale at known resolution (e.g., 540x960 or 1080x1920)
2. Game window position is stable (not moving/resizing during play)
3. Click logger implemented and tested
4. Screen capture script implemented and tested
5. Know which deck you're using (must match `encoder_constants.py` DECK_CARDS)

### Step 1: Pre-match setup

1. Open Clash Royale in Google Play Games
2. Position the game window (do NOT move it during recording)
3. Start the recording script (launches Thread A + Thread B)
4. Verify the script detects the game window (prints resolution, confirms capture area)
5. Navigate to a match (Trainer AI or ladder)

### Step 2: During the match

1. Play normally, following the recording rules below
2. The recording script silently captures screenshots + clicks in the background
3. Play your natural game -- do not play differently because you're recording
4. When the match ends (VICTORY/DEFEAT screen), stop the recording script

### Step 3: Post-match processing (once DatasetBuilder exists)

1. Run StateBuilder on each saved screenshot -> `(timestamp, GameState)` list
2. Run ActionBuilder on the click log -> `ActionEvent` list + per-frame action labels
3. Run StateEncoder on each GameState -> observation tensors + action masks
4. DatasetBuilder merges everything into `(obs, action_idx, mask)` training tuples
5. Save to disk (npz, HDF5, or torch format)

---

## Recording Rules

These rules ensure clean training data. Violating them produces noisy or incorrect action labels.

### Card interaction

1. **Always use two distinct clicks** to play a card: click the card slot, then click the arena position
2. **Never drag** cards from the card bar to the arena -- drags produce ambiguous coordinate streams that the state machine cannot pair
3. **Do not double-click** a card slot -- one deliberate click to select, one deliberate click to place
4. **If you change your mind**, click the new card slot (do not click random spots to "cancel")

### Click discipline

5. **Only perform deliberate, intentional clicks** -- every click should either select a card or place a unit
6. **Do not spam-click** the arena or card bar -- rapid repeated clicks create noise that the state machine must filter
7. **Do not click UI elements** (emotes, menu buttons, spectate) during the match -- these produce "other" classifications that reset the state machine
8. **Wait for the card selection animation** (~150ms) before clicking the arena -- if you click too fast, the game may not register the card selection, creating a mismatch between what you intended and what happened

### Gameplay

9. **Play at your natural pace** -- the click logger captures at OS-level, so even fast plays are recorded. But don't rush clicks to the point where they blur together
10. **Do not alt-tab** or switch windows during the match -- this may capture clicks outside the game window
11. **Do not resize or move** the game window during recording -- this invalidates the coordinate normalization
12. **Play complete matches** -- partial recordings are usable but less valuable. Start recording before the match begins and stop after the end screen

### What's OK

- Hovering the mouse (no click = no event, harmless)
- Clicking a card and then clicking a different card (state machine handles this -- uses the last card selected)
- Playing cards quickly in succession (each two-click pair is independent)
- Losing the match (losses are valid training data)

---

## Purpose of the DatasetBuilder

The DatasetBuilder is purely a post-processing pipeline that transforms raw recorded data into the format needed for BC/RL training. It does NOT run during gameplay.

### What it does

```
Raw Data (from recording):
  - Screenshots + timestamps (Thread A output)
  - Click events + timestamps (Thread B output)

         |
         v  DatasetBuilder

Step 1: StateBuilder(screenshot) -> GameState per frame
Step 2: ActionBuilder.clicks_to_actions(clicks) -> ActionEvent list
Step 3: ActionBuilder.build_action_timeline(clicks, frame_timestamps) -> per-frame labels
Step 4: StateEncoder.encode(game_state) -> obs dict per frame
Step 5: StateEncoder.action_mask(game_state) -> mask per frame
Step 6: Merge into training tuples: (obs, action_idx, mask)

         |
         v

Training Dataset:
  - obs["arena"]: (N, 32, 18, 6) float32
  - obs["vector"]: (N, 23) float32
  - actions: (N,) int64 -- values in [0, 2304]
  - masks: (N, 2305) bool
```

### Is it only for formatting?

Yes. The DatasetBuilder adds no new information. It coordinates StateBuilder, ActionBuilder, and StateEncoder in the right order and produces the tensor format that SB3's BC training loop expects.

### Can it combine Thread A and Thread B features?

Yes. The DatasetBuilder is the natural place to orchestrate everything:

```python
class DatasetBuilder:
    def __init__(self):
        self.state_builder = StateBuilder(detector, ocr_extractor)
        self.action_builder = ActionBuilder()
        self.encoder = StateEncoder()

    def process_match(self, screenshots, click_log):
        """Process one recorded match into training data."""
        # Thread A: screenshots -> GameStates
        frames = []
        for timestamp, image in screenshots:
            game_state = self.state_builder.build_state(image)
            frames.append((timestamp, game_state))

        # Thread B: clicks -> per-frame action labels
        frame_timestamps = [t for t, _ in frames]
        action_labels = self.action_builder.build_action_timeline(
            click_log, frame_timestamps
        )

        # Encode everything
        dataset = []
        for i, (timestamp, game_state) in enumerate(frames):
            obs = self.encoder.encode(game_state)
            mask = self.encoder.action_mask(game_state)
            dataset.append((obs, action_labels[i], mask))

        return dataset
```

If you go with Option A (save raw screenshots), the DatasetBuilder also runs StateBuilder on each image. If you go with Option B (live StateBuilder), it just encodes pre-computed GameStates.

---

## Pipeline Status and Next Steps

### What exists vs what's missing

| Component | Status | Location |
|-----------|--------|----------|
| YOLOv8s detector | Done (mAP50=0.804) | `src/detection/` |
| StateBuilder | Done | `src/pipeline/state_builder.py` |
| CardPredictor | Trained, NOT integrated | `src/classification/card_classifier.py` |
| StateEncoder | Done (josh), 42 tests | `docs/josh/state_encoder_module/src/encoder/` |
| ActionBuilder | Done (josh), 46 tests | `docs/josh/action_builder_module/src/action/` |
| **Click Logger** | **NOT BUILT** | -- |
| **Screen Capture Script** | **NOT BUILT** | -- |
| **DatasetBuilder** | **NOT BUILT** | -- |
| **BC Training Script** | **NOT BUILT** | -- |
| Gym Environment | NOT BUILT (needed for PPO, not BC) | -- |

### Development order (after ActionBuilder)

**1. Click Logger** (highest priority -- single blocker for recording)

- Implement using `pynput.mouse.Listener`
- Captures `(timestamp, x_norm, y_norm)` for each click
- Normalizes coordinates relative to game window bounds

**2. Screen Capture Script**

- `mss.capture()` loop at 2-4 FPS
- Saves timestamped screenshots to disk
- Can be as simple as: capture, save PNG, record timestamp, sleep

**3. Wire CardPredictor into StateBuilder**

- Without this, `GameState.cards` is always empty
- The action mask will be all-False (no cards detected = no valid actions)
- 30-minute task: crop 4 card slot regions, run `CardPredictor.predict()` on each, populate `GameState.cards`
- Card slot coordinates at 540x960: x starts at 110, each card 100px wide, y=[770, 920]

**4. DatasetBuilder**

- Coordinates StateBuilder + ActionBuilder + StateEncoder
- Processes recorded matches into training tuples
- Saves in a format SB3 can load

**5. BC Training Script**

- SB3 `MaskableMultiInputPolicy` with cross-entropy loss
- Weighted loss (no-op weight=0.3, card actions weight=3.0)
- Custom `CRFeatureExtractor` with `nn.Embedding` for class IDs
- Load dataset from DatasetBuilder output, train, evaluate

### Critical path to a trained BC agent

```
Click Logger + Screen Capture  ->  Record 30+ games  ->  DatasetBuilder  ->  BC Training
     (build these first)          (play matches)        (post-process)     (train model)
```

CardPredictor integration is a parallel task that can happen anytime before DatasetBuilder runs.

---

## Data Flow Summary

### During gameplay (recording)

```
Human plays Clash Royale
    |
    +-- Thread A: mss.capture() every 0.25-0.5s
    |       -> saves screenshot + timestamp to disk
    |
    +-- Thread B: pynput mouse listener
            -> saves (timestamp, x_norm, y_norm) to click log
```

### After gameplay (post-processing)

```
Saved screenshots + click log
    |
    v
DatasetBuilder.process_match()
    |
    +-- StateBuilder: screenshot -> GameState (runs YOLO + OCR per frame)
    |
    +-- ActionBuilder: click log -> per-frame action labels (Discrete 2305)
    |
    +-- StateEncoder: GameState -> obs tensors + action masks
    |
    v
Training dataset: list of (obs, action_idx, mask)
    |
    v
Save to disk (npz / torch)
```

### During agent execution (future)

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

---

## Assumptions

### 1. Fixed game window

The game window must be at a fixed, known position and resolution during recording. The click logger normalizes coordinates relative to this window. Moving or resizing the window mid-game invalidates all click coordinates.

### 2. Base resolution 540x960

All coordinate constants (card slots, arena bounds) are derived from 540x960. If you capture at 1080x1920, coordinates scale proportionally via normalization (all values are 0-1).

### 3. Single fixed deck

The encoder constants, action mask, and card classifier are hardcoded for the Royal Hogs / Royal Recruits deck (8 cards). Changing decks requires updating `DECK_CARDS`, `CARD_ELIXIR_COST`, `CARD_IS_SPELL` in `encoder_constants.py` and retraining the CardPredictor.

### 4. CardPredictor must be wired in

Before DatasetBuilder can produce useful training data, CardPredictor must be integrated into StateBuilder. Without it, `GameState.cards` is always empty, the action mask is trivially all-False for card placements, and the vector observation lacks card features.

### 5. Capture rate of 2-4 FPS is sufficient

At 3 minutes per match, that is 360-720 frames. With 10-20 card placements per game, roughly 97% of frames are no-ops. This imbalance is handled at training time via weighted loss, not at collection time.

### 6. No downsampling or deduplication at collection time

Every captured frame produces a training sample, even if the action is no-op for 90%+ of frames. No-op is a valid and important action (knowing when NOT to play a card is half the game). Class imbalance is handled at training time.

### 7. Post-game processing is acceptable

We don't need real-time training data. The workflow is: play games, save raw data, process later. This is standard for behavior cloning.

### 8. Target: 30+ recorded games

At ~180 frames per game (2 FPS, 3 min match) with 10-20 card placements each, 30 games yields roughly 5,400 frames with 300-600 card placement actions. After weighted sampling, this should be sufficient for a BC agent that beats random baseline.

---

## Common Questions

**Q: Do I need YOLO running while I play?**
No. With Option A (recommended), you save raw screenshots during gameplay and run YOLO offline. This has zero performance impact on the game.

**Q: What if I accidentally drag a card?**
The click logger only captures discrete click events (press), not drag trajectories. A drag will register as a click at the drag start position, which may be classified as a card slot click with no subsequent arena click -- the state machine will leave this unpaired. The drag endpoint is not captured. The action is lost, but it doesn't corrupt other data.

**Q: What if I click outside the game window?**
The click logger checks if the click is within the game window bounds. Clicks outside the window are not recorded.

**Q: How much disk space per game?**
At 540x960 resolution, each PNG screenshot is roughly 200-400KB. At 2 FPS for 3 minutes, that is 360 frames = 72-144MB per game. At 4 FPS, double that. The click log is negligible (a few KB). For 30 games: 2-4 GB total.

**Q: Can I record at 1080x1920 and process at 540x960?**
Yes. All coordinates are normalized to 0-1 before processing. The StateBuilder accepts any resolution. The encoder's coordinate utilities handle scaling via the frame_width and frame_height parameters.

**Q: What happens if StateBuilder fails on a screenshot?**
If YOLO inference or OCR fails on a frame, that frame gets a degraded GameState (missing detections, None values for timer/elixir). The StateEncoder handles None gracefully (defaults to 0.0). The frame still produces a training sample, just with a noisier observation. A few bad frames per game are acceptable.
