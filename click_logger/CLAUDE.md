# click_logger/ - BC Match Recorder (Screenshots + Click Actions)

Records Clash Royale matches for behavior cloning by running two concurrent threads: an mss screen capture loop (Thread A) and a pynput click logger (Thread B). Produces a self-contained session directory with JPEG screenshots, paired click actions, a frame manifest, and session metadata.

## Files

**click_logger.py** - OS-level mouse click capture and card-arena pairing.
- `ClickLogger(window_title, card_positions, arena_bounds, output_path, slot_threshold)`: Listens for mouse events via `pynput.mouse.Listener`.
  - `start()`: Resolves window via pygetwindow, starts listener thread.
  - `stop()`: Stops listener, flushes output file.
  - State machine: mouse DOWN on card slot -> select card, mouse RELEASE in arena -> emit paired action to JSONL.
  - Uses card center positions + radius threshold (`slot_threshold=0.035`) for slot detection.
  - Only captures left-button clicks within the game window.

**screen_capture.py** - Threaded mss screen grabber.
- `ScreenCapture(window, output_dir, fps=2.0, jpeg_quality=85)`: Takes a pygetwindow window object.
  - `start()`: Launches daemon capture thread.
  - `stop()`: Stops thread, flushes frame manifest, reports count.
  - `frame_count`: Number of frames captured so far.
  - Saves JPEG via Pillow (BGRA->RGB conversion from mss raw pixels).
  - Re-reads window geometry each frame to track window movement.
  - Writes `frames.jsonl` manifest: `{frame_idx, timestamp, filename, width, height}` per frame.

**match_recorder.py** - Orchestrator.
- `MatchRecorder(window_title, card_positions, arena_bounds, output_root, fps, jpeg_quality, slot_threshold)`: Creates session directory, wires ClickLogger + ScreenCapture.
  - `start()`: Starts both components.
  - `stop()`: Stops both, writes `metadata.json` with session summary.
  - `session_dir`: Path to the current recording session.
  - Fails fast if window title is not found.

**record_bc.py** - Entry point script.
- Configures window title, card positions, arena bounds, FPS.
- Creates MatchRecorder, starts recording, waits for Enter, stops.
- Wrapped in try/finally for crash safety.

## Output Structure

```
recordings/match_YYYYMMDD_HHMMSS/
  screenshots/
    frame_000000.jpg       JPEG at configured quality
    frame_000001.jpg
    ...
  actions.jsonl            Paired card placements from ClickLogger
  frames.jsonl             Timestamp-to-filename manifest from ScreenCapture
  metadata.json            Session info (written on stop)
```

## Output Formats

### actions.jsonl (one line per card placement)

```json
{"timestamp": 1740123460.789, "card_id": 2, "x_norm": 0.45, "y_norm": 0.55}
```

- `timestamp`: `time.time()` at the moment the arena click (mouse release) occurred
- `card_id`: Card slot index (0-3) from the preceding card-slot click
- `x_norm`, `y_norm`: Normalized arena position where the card was placed (0-1)

### frames.jsonl (one line per captured screenshot)

```json
{"frame_idx": 0, "timestamp": 1740123456.123, "filename": "frame_000000.jpg", "width": 1080, "height": 1920}
```

- `timestamp`: `time.time()` at the start of the capture
- `width`, `height`: Window dimensions in pixels at capture time

### metadata.json (written once on stop)

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
  "card_positions": {"0": [0.439, 0.889], "1": [0.494, 0.889], ...},
  "arena_bounds": [0.05, 0.15, 0.95, 0.80]
}
```

## Click Classification

ClickLogger uses center-point + radius matching (not bounding boxes):

| Slot | Center (x_norm, y_norm) | Threshold |
|------|------------------------|-----------|
| 0 | (0.439, 0.889) | 0.035 |
| 1 | (0.494, 0.889) | 0.035 |
| 2 | (0.559, 0.889) | 0.035 |
| 3 | (0.639, 0.889) | 0.035 |

A click matches a card slot if `abs(x - center_x) < threshold AND abs(y - center_y) < threshold`. Arena bounds: `(0.05, 0.15, 0.95, 0.80)` as `(x_min, y_min, x_max, y_max)`.

These values are calibrated for Josh's Google Play Games window. Recalibrate if the window layout differs.

## Click Pairing State Machine

```
MOUSE DOWN on card slot   -> select card_id, store in _selected_slot
MOUSE DOWN elsewhere      -> no effect

MOUSE RELEASE in arena (with _selected_slot set) -> emit action, clear _selected_slot
MOUSE RELEASE elsewhere   -> no effect
```

Key difference from ActionBuilder: ClickLogger pairs on mouse DOWN (card) + RELEASE (arena), not two discrete clicks. This means a single press-drag-release that starts on a card slot and ends in the arena would be logged as a valid action. However, recording rules instruct players not to drag.

## Configuration

All configuration lives in `record_bc.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `WINDOW_TITLE` | `"Clash Royale - UnwontedTemper73"` | pygetwindow title (must match exactly) |
| `CARD_POSITIONS` | `{0: (0.439, 0.889), ...}` | Card slot centers, normalized |
| `ARENA_BOUNDS` | `(0.05, 0.15, 0.95, 0.80)` | Arena click region, normalized |
| `FPS` | `2.0` | Screenshot capture rate |
| `JPEG_QUALITY` | `85` | JPEG compression quality |

## Dependencies

```
click_logger.py
  -> pygetwindow (window detection)
  -> pynput (mouse listener)

screen_capture.py
  -> mss (screen capture)
  -> Pillow (BGRA->RGB conversion, JPEG encoding)

match_recorder.py
  -> click_logger.ClickLogger
  -> screen_capture.ScreenCapture
  -> pygetwindow (window resolution)
```

## Downstream Usage

After recording, the session data feeds into the DatasetBuilder pipeline:

```
frames.jsonl -> frame timestamps for ActionBuilder.build_action_timeline()
actions.jsonl -> action events for per-frame labeling
screenshots/ -> StateBuilder (YOLO + OCR) -> GameState -> StateEncoder -> obs tensors
```

The ClickLogger's paired output (`{card_id, x_norm, y_norm}`) maps directly to ActionBuilder's action encoding: `norm_to_cell(x_norm, y_norm) -> (col, row)`, then `placement_to_action(card_id, col, row) -> action_idx`.
