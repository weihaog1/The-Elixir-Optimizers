# Recording Script Developer Documentation

## What Is the Recording Script?

The recording script captures raw BC (behavior cloning) training data during live Clash Royale gameplay. It runs two concurrent threads -- screen capture and click logging -- and produces a self-contained session directory with everything needed for offline post-processing.

In the pipeline:

```
Human plays Clash Royale
    |
    +--- Thread A: mss screen capture at 2 FPS
    |    -> screenshots/frame_000000.jpg, frame_000001.jpg, ...
    |    -> frames.jsonl (timestamp-to-filename manifest)
    |
    +--- Thread B: pynput click logger (OS-level mouse events)
    |    -> actions.jsonl (paired card placements: card_id, x_norm, y_norm)
    |
    v
Session directory: recordings/match_YYYYMMDD_HHMMSS/
    |
    v  (offline, post-game)
    |
DatasetBuilder (not yet built)
    |
    +--- screenshots/ -> StateBuilder (YOLO + OCR) -> GameState -> StateEncoder -> obs tensors
    |
    +--- actions.jsonl -> ActionBuilder.clicks_to_actions() -> per-frame action labels
    |
    v
Training data: (obs, action_idx, mask) per frame
```

The key design choice: recording is lightweight. No YOLO inference, no OCR, no state encoding runs during the game. Raw screenshots and raw clicks are saved, and all expensive processing happens offline. This keeps the recording loop fast enough to avoid dropping frames or missing clicks.

---

## Thought Process

### Why separate threads?

Screen capture and click logging have fundamentally different timing requirements:

- **Screen capture** runs on a fixed timer (every 500ms at 2 FPS). It needs to grab a window region, convert pixels, and write a JPEG file. This takes 20-40ms depending on resolution and compression.

- **Click logging** is event-driven. Mouse clicks arrive whenever the human plays, with no predictable timing. They must be captured with sub-millisecond accuracy to avoid missing fast double-clicks (card selection + arena placement can happen within 200ms).

Running both in a single thread would mean either: (a) blocking on screen capture while potentially missing clicks, or (b) polling for clicks between captures and losing timing precision. Two threads avoids both problems. The click logger thread uses pynput's OS-level listener (runs in its own thread internally), while the screen capture thread runs its own timed loop.

### Why mss instead of PyAutoGUI or OpenCV?

Three options for screen capture:

1. **PyAutoGUI** (`pyautogui.screenshot()`): Cross-platform, simple API. But internally calls the OS screenshot API once per capture, which is slow (50-100ms per frame). No region-based capture on all platforms.

2. **OpenCV** (`cv2.VideoCapture`): Designed for webcams and video files, not screen capture. Requires platform-specific backends (DirectShow on Windows, V4L2 on Linux) and does not natively capture arbitrary window regions.

3. **mss** (`mss.mss().grab(region)`): Designed specifically for screen capture. Region-based (captures only the game window, not the full screen). Low overhead (10-20ms per frame). Returns raw BGRA pixel data. Cross-platform.

mss is the clear winner. It is also what KataCR uses for their real-time pipeline.

### Why JPEG instead of PNG?

At 1080x1920 resolution:

| Format | File Size | Encode Time | Quality |
|--------|-----------|-------------|---------|
| PNG | ~3 MB | ~30ms | Lossless |
| JPEG q=85 | ~300 KB | ~5ms | Visually lossless |

Over a 3-minute match at 2 FPS (360 frames):
- PNG: ~1.1 GB, ~11 seconds of total encoding time
- JPEG: ~108 MB, ~1.8 seconds of total encoding time

JPEG at quality=85 is visually indistinguishable from lossless for YOLO inference. The compression artifacts are far smaller than the domain gap between synthetic training data and real gameplay. We tested this during evaluation -- mAP50 does not change between JPEG q=85 and PNG.

### Why 2 FPS?

Clash Royale runs at 30 FPS, but card placements happen roughly every 5-10 seconds. At 2 FPS, we get enough temporal resolution to capture the game state within 250ms of any action. The state does not change fast enough to warrant higher capture rates for BC training.

Higher FPS options:
- **4 FPS**: 2x more data, 2x more storage. Marginal benefit -- consecutive frames are often near-identical.
- **10 FPS**: Wastes disk space and post-processing time. The perception pipeline (YOLO + OCR) takes ~100ms per frame anyway, so processing 10 FPS offline would be 10x slower than real-time.

Lower FPS:
- **1 FPS**: Risks missing transient game states (spell effects, fast troop spawns). The window between a card placement and the next state capture could be up to 1 second, during which the game state changes significantly.

2 FPS is the standard for KataCR's data collection.

### Why re-read window geometry each frame?

The game window might move during recording. If the player accidentally drags the title bar, or if Windows rearranges windows, the capture region would be wrong if we cached it. Re-reading `window.left`, `window.top`, `window.width`, `window.height` each frame adds negligible cost (a property lookup, not a system call) and prevents silently capturing the wrong screen region.

### Why a daemon thread?

If the main process crashes (uncaught exception, Ctrl+C), daemon threads are killed automatically by Python. Without daemon mode, the capture thread would keep running after the main thread exits, producing an orphaned process that silently writes files until manually killed. The `try/finally` in `record_bc.py` is the primary crash safety mechanism, but daemon mode is the backstop.

### Why not run StateBuilder during recording?

We considered a "heavy" recording mode that runs YOLO inference on every captured frame and saves GameState objects directly. This was rejected for several reasons:

1. **YOLO inference takes 65ms on M1 Pro** (more on CPU). At 2 FPS, that is 13% of each 500ms interval consumed by just detection. Adding OCR (~30ms) and state encoding (~1ms) brings it to ~20%. This is workable, but leaves less margin for jitter.

2. **The perception pipeline is still evolving.** If we save raw screenshots, we can re-run the pipeline later with an improved YOLO model, better OCR, or the card classifier wired in. If we only save processed GameState objects, we are locked to whatever pipeline version was running at recording time.

3. **Raw screenshots are more debuggable.** If the BC model behaves unexpectedly, we can visually inspect what the agent "saw" at any training frame. With processed state, we would need to reverse-engineer what the screenshot looked like.

The lightweight approach trades disk space (~100MB per match) for flexibility and simplicity.

---

## Recording a Match

### Prerequisites

1. Google Play Games running with Clash Royale open
2. Window title matches `WINDOW_TITLE` in `record_bc.py` (default: `"Clash Royale - UnwontedTemper73"`)
3. Python environment with `mss`, `Pillow`, `pygetwindow`, and `pynput` installed

### Step-by-step

1. **Open the game** and navigate to the match screen (before tapping "Battle")

2. **Run the recorder:**
   ```bash
   cd docs/josh/click_logger/
   python record_bc.py
   ```
   If the window title is wrong, you get an immediate error:
   ```
   RuntimeError: Window not found: 'Clash Royale - UnwontedTemper73'
   ```

3. **Start the match** in-game. Recording begins immediately.

4. **Play normally**, following the recording rules below.

5. **Press Enter** in the terminal when the match ends. Both threads stop and metadata is written.

### Recording rules

These rules ensure clean action data:

- **Do not drag cards.** Always use two discrete clicks: (1) click a card slot, (2) click an arena position. Dragging produces a single mouse-down + mouse-up event that the ClickLogger would record, but the ActionBuilder's state machine expects two separate clicks from ClickEvent pairs.
- **Do not spam clicks.** Wait for the card placement to register before clicking again. Rapid clicks can produce ambiguous pairings.
- **Use precise clicks.** Click near the center of card slots. The ClickLogger uses a center-point + radius threshold (0.035 normalized), so clicks near the edge of a card might not register.
- **Only play cards using the 4 visible slots.** Do not use the "next card" area or drag from the deck. Only slots 0-3 are tracked.

### Output

After stopping, the session directory contains:

```
recordings/match_YYYYMMDD_HHMMSS/
  screenshots/
    frame_000000.jpg       # JPEG at configured quality
    frame_000001.jpg
    ...
  actions.jsonl            # One line per card placement
  frames.jsonl             # One line per captured screenshot
  metadata.json            # Session summary
```

---

## Output Formats

### actions.jsonl

One JSON line per card placement, written by ClickLogger:

```json
{"timestamp": 1740123460.789, "card_id": 2, "x_norm": 0.45, "y_norm": 0.55}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | float | `time.time()` at mouse release in arena |
| `card_id` | int | Card slot index (0-3) from the preceding card-slot click |
| `x_norm` | float | Normalized arena X position (0=left, 1=right of window) |
| `y_norm` | float | Normalized arena Y position (0=top, 1=bottom of window) |

**Important difference from ActionBuilder:** The ClickLogger's `actions.jsonl` is pre-paired. Each line is already a complete card placement (card_id + arena position). The ActionBuilder, by contrast, takes unpaired ClickEvents and performs its own pairing via a state machine.

When using ClickLogger output with ActionBuilder downstream, you have two options:
1. Feed the raw `actions.jsonl` directly into `ActionBuilder.build_action_timeline()` (which expects ClickEvents, not pre-paired events) -- requires adapter code
2. Convert `actions.jsonl` entries to `ActionEvent` objects directly using `norm_to_cell()` and `placement_to_action()` -- simpler, bypasses ActionBuilder's pairing logic

### frames.jsonl

One JSON line per captured screenshot, written by ScreenCapture:

```json
{"frame_idx": 0, "timestamp": 1740123456.123, "filename": "frame_000000.jpg", "width": 1080, "height": 1920}
```

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | int | Sequential frame counter (0-indexed) |
| `timestamp` | float | `time.time()` at the start of capture |
| `filename` | str | JPEG filename in `screenshots/` subdirectory |
| `width` | int | Window width in pixels at capture time |
| `height` | int | Window height in pixels at capture time |

Width and height are recorded per-frame because the window could be resized during recording.

### metadata.json

Written once when recording stops:

```json
{
  "window_title": "Clash Royale - UnwontedTemper73",
  "window_geometry": {"left": 0, "top": 0, "width": 1080, "height": 1920},
  "fps": 2.0,
  "jpeg_quality": 85,
  "start_time": 1740123456.0,
  "stop_time": 1740123636.0,
  "duration_seconds": 180.0,
  "frame_count": 360,
  "action_count": 22,
  "card_positions": {"0": [0.439, 0.889], "1": [0.494, 0.889], "2": [0.559, 0.889], "3": [0.639, 0.889]},
  "arena_bounds": [0.05, 0.15, 0.95, 0.80]
}
```

The `window_geometry` captures the window position at init time. The `card_positions` and `arena_bounds` record the calibration values used for this session, so downstream processing knows what thresholds were active.

---

## ClickLogger vs ActionBuilder

The ClickLogger and ActionBuilder both handle card placement pairing, but they serve different roles and use different mechanisms:

| Aspect | ClickLogger | ActionBuilder |
|--------|-------------|---------------|
| **When it runs** | During gameplay (real-time) | Offline (post-game processing) |
| **Input** | OS-level mouse events via pynput | List of ClickEvent objects |
| **Card detection** | Center-point + radius (0.035) | Bounding box regions from screen_regions.py |
| **Pairing trigger** | Mouse DOWN on card + mouse RELEASE in arena | Two discrete click events |
| **Output** | Pre-paired JSONL: `{card_id, x_norm, y_norm}` | ActionEvent objects with grid coords + action_idx |
| **State machine** | Two states: idle / card selected | Two states: idle / card_selected |

The ClickLogger uses center-point detection because it was calibrated empirically for Josh's Google Play Games window. The ActionBuilder uses bounding-box regions derived from `screen_regions.py` which generalize better across resolutions.

**Drag behavior difference:** The ClickLogger pairs on mouse DOWN (card) + RELEASE (arena), meaning a drag from a card slot to the arena would register as a valid action. The ActionBuilder pairs on two separate click events, so dragging would not work. The recording rules instruct players not to drag to avoid this inconsistency.

---

## Calibration

### Card slot positions

The ClickLogger uses center-point + radius matching for card detection:

| Slot | Center (x_norm, y_norm) | Threshold |
|------|------------------------|-----------|
| 0 | (0.439, 0.889) | 0.035 |
| 1 | (0.494, 0.889) | 0.035 |
| 2 | (0.559, 0.889) | 0.035 |
| 3 | (0.639, 0.889) | 0.035 |

These are calibrated for Josh's Google Play Games window. To recalibrate:

1. Open Clash Royale in Google Play Games
2. Take a screenshot and note the window dimensions
3. Click the center of each card slot and record the pixel coordinates
4. Convert to normalized: `x_norm = (x_pixel - window_left) / window_width`
5. Update `CARD_POSITIONS` in `record_bc.py`

The threshold (0.035) means a click within ~3.5% of the window width/height from the center counts as hitting that slot. Increase if clicks are being missed; decrease if neighboring slots overlap.

### Arena bounds

The arena region is defined as:

```python
ARENA_BOUNDS = (0.05, 0.15, 0.95, 0.80)  # (x_min, y_min, x_max, y_max)
```

This excludes:
- The top 15% of the screen (timer bar, score display)
- The bottom 20% of the screen (card bar, elixir bar)
- 5% margins on left and right (window chrome)

The ClickLogger only records a card placement when the mouse release lands within these bounds.

---

## Thread Safety

### mss context isolation

mss screen capture objects are not thread-safe across threads. The ScreenCapture class creates its mss context inside the capture thread:

```python
def _capture_loop(self):
    with mss.mss() as sct:  # created inside the thread
        while self._running:
            screenshot = sct.grab(region)
```

If the mss context were created in `__init__` (main thread) and used in the capture thread, you would get segfaults or corrupted images on some platforms.

### File handle isolation

Each component owns its own file handle:
- ClickLogger opens and writes to `actions.jsonl`
- ScreenCapture opens and writes to `frames.jsonl`

No file handle is shared across threads. Both flush after every write for crash safety.

### ClickLogger threading

pynput's `mouse.Listener` runs in its own thread internally. The `_on_click` callback executes in the listener's thread. The callback writes to `self._file` (the actions JSONL), which is safe because only one thread ever writes to it. The `_selected_slot` state variable is accessed from the listener thread only (set on mouse down, read on mouse release), so no lock is strictly needed for the current code. A `threading.Lock` is declared but not used in the critical path.

### Stopping order

`MatchRecorder.stop()` stops ScreenCapture first, then ClickLogger. This ensures no screenshots are captured without the click logger running. The reverse order (stopping clicks before screenshots) could produce frames at the end of the session with no corresponding click coverage.

---

## Assumptions and Limitations

### 1. Single window, stable geometry

The recorder assumes one game window found by exact title match. If multiple windows match (unlikely), it uses the first one. Window geometry is re-read per frame for ScreenCapture, but the ClickLogger reads geometry on click events from the cached window object. If the window moves significantly between a click and the next geometry read, normalized coordinates could be slightly off.

### 2. No frame-action synchronization

Frames and actions have independent timestamps but no explicit sync mechanism. A card placement at t=100.2 might fall between frame t=100.0 and frame t=100.5. The DatasetBuilder (downstream) must handle this timestamp-based assignment. The `frames.jsonl` and `actions.jsonl` timestamps both use `time.time()` from the same system clock, so they are comparable.

### 3. Card positions are for Josh's setup

The normalized card positions (0.439, 0.494, 0.559, 0.639) and arena bounds are calibrated for one specific Google Play Games window layout. Different monitors, DPI settings, or window sizes may shift these positions. Always recalibrate before recording on a new machine.

### 4. No validation of card selection success

The ClickLogger records a card-slot click based purely on position matching. It does not verify that the game actually selected the card (e.g., the card might be on cooldown, or the player might not have enough elixir). Invalid placements are recorded and will appear in `actions.jsonl`. The downstream pipeline must handle or filter these.

### 5. JPEG artifacts at low quality

At quality=85 (default), JPEG artifacts are invisible to YOLO. At quality < 60, compression artifacts could interfere with small object detection (e.g., ice-spirit, skeleton). Do not reduce below 70.

### 6. Crash recovery is partial

The `try/finally` block ensures `stop()` is called on keyboard interrupt or exception, which flushes and closes all files. However, if the process is killed (SIGKILL, system crash), the last few manifest entries and actions might be lost (buffered but not flushed). The per-write flush mitigates this -- at most one entry per file could be lost.

### 7. No duplicate frame detection

If the game is paused or in a menu, ScreenCapture still captures frames. These duplicate/irrelevant frames waste storage and would need to be filtered in post-processing.

---

## File Reference

| File | Purpose |
|------|---------|
| `click_logger.py` | OS-level mouse capture and card-arena pairing via pynput |
| `screen_capture.py` | Threaded mss screen grabber, JPEG saving, frame manifest |
| `match_recorder.py` | Orchestrator: wires ClickLogger + ScreenCapture, writes metadata |
| `record_bc.py` | Entry point script with configuration constants |
| `CLAUDE.md` | Technical reference (output formats, click classification, state machine) |

### Dependencies

```
click_logger.py
  -> pygetwindow (window detection by title)
  -> pynput (OS-level mouse listener)

screen_capture.py
  -> mss (fast region-based screen capture)
  -> Pillow (BGRA->RGB conversion, JPEG encoding)

match_recorder.py
  -> click_logger.ClickLogger
  -> screen_capture.ScreenCapture
  -> pygetwindow (window resolution for fail-fast validation)
```

Install all dependencies:
```bash
pip install mss Pillow pygetwindow pynput
```

### What depends on this module

- **DatasetBuilder** (not yet built) - reads `frames.jsonl` timestamps and `actions.jsonl` events for per-frame action labeling
- **StateBuilder** (offline) - processes saved screenshots through YOLO + OCR to produce GameState per frame
- **ActionBuilder** (offline) - converts click data to Discrete(2305) action indices for training

---

## Common Questions

**Q: Can I record at higher FPS?**
Yes. Change `FPS = 2.0` to `FPS = 4.0` in `record_bc.py`. At 4 FPS on a 1080x1920 window, each interval is 250ms and capture takes ~20-40ms, so there is margin. At 10+ FPS, you may start dropping frames if JPEG encoding cannot keep up. Monitor the frame_count in metadata.json against expected_frames (fps * duration) to check for drops.

**Q: Can I change the window title?**
Yes. Update `WINDOW_TITLE` in `record_bc.py` to match your Google Play Games window exactly. The title must be an exact match -- pygetwindow does substring matching via `getWindowsWithTitle()`, but the code takes the first match, so use the full title to avoid ambiguity.

**Q: Why does ClickLogger use center+radius instead of bounding boxes like ActionBuilder?**
Historical reasons. The ClickLogger was written first with a simpler detection method. The ActionBuilder was written later with access to `screen_regions.py` bounding box definitions. Both work correctly for their use case. The center+radius approach is slightly more forgiving for off-center clicks but requires manual calibration.

**Q: What if the game window is covered by another window?**
mss captures from the screen buffer, not from the window directly. If another window covers the game, mss captures the covering window's pixels, not the game. Keep the game window visible and unobstructed during recording.

**Q: How much disk space does one match use?**
A 3-minute match at 2 FPS, 1080x1920, JPEG q=85 produces ~360 frames at ~300KB each = ~108MB. The JSONL files add negligible size (<100KB total). Budget ~120MB per match.

**Q: Can I record multiple matches in sequence?**
Each run of `record_bc.py` creates a new session directory. Run the script, play a match, press Enter. Run it again for the next match. Each match gets its own `match_YYYYMMDD_HHMMSS/` directory.

**Q: What happens if I press Enter before the match ends?**
Recording stops immediately. The partial match data is still valid -- `metadata.json` records the actual duration, and all frames/actions captured up to that point are preserved.

**Q: Why does ClickLogger pair on mouse DOWN + RELEASE instead of two discrete clicks?**
pynput delivers both press and release events for every click. The ClickLogger uses press (on card slot) to register card selection and release (in arena) to register placement. For normal discrete clicking (press-release on card, then press-release on arena), this produces the same result as pairing two discrete clicks. The difference only matters if someone drags from a card slot to the arena (which the recording rules prohibit).

**Q: How do I know if my card positions are calibrated correctly?**
Run `record_bc.py`, click each card slot deliberately, then click somewhere in the arena for each. Check `actions.jsonl` -- you should see 4 entries with `card_id` values 0, 1, 2, 3. If any slot consistently fails to register, adjust the corresponding position in `CARD_POSITIONS` or increase `slot_threshold`.
