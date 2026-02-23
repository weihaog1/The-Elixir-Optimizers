# Live Game Inference Guide

Run a trained BC model against a live Clash Royale game, predicting and executing card placements in real time.

---

## 1. Prerequisites

### Required

| Package | Install | Used By |
|---------|---------|---------|
| `torch` | `pip install torch` | BC policy inference |
| `numpy` | `pip install numpy` | Observation tensors |
| `opencv-python` | `pip install opencv-python` | Frame color conversion |
| `mss` | `pip install mss` | Screen capture |

### Required for Click Execution (non-dry-run)

| Package | Install | Used By |
|---------|---------|---------|
| `pyautogui` | `pip install pyautogui` | Mouse click execution |

### Optional

| Package | Install | Used By |
|---------|---------|---------|
| `pygetwindow` | `pip install pygetwindow` | Auto-detect game window by title |
| `ultralytics` | `pip install ultralytics` | YOLO detection (Tier 2 perception) |

### Model Files

- **BC checkpoint**: `models/bc/best_bc.pt` (from BC training)
- **YOLO detector**: `models/best_yolov8s_50epochs_fixed_pregen_set.pt` (YOLOv8s, 155 classes)
- **Card classifier**: `models/card_classifier.pt` (MiniResNet, 8 deck card classes)

---

## 2. Quick Start

### Dry Run (safe, no clicks)

```bash
python bc_model_module/run_live.py \
    --model-path models/bc/best_bc.pt \
    --capture-region 0,0,540,960 \
    --dry-run
```

### Live Execution

```bash
python bc_model_module/run_live.py \
    --model-path models/bc/best_bc.pt \
    --capture-region 0,0,540,960 \
    --confidence 1.0 \
    --cooldown 0.5
```

### Auto-Detect Window

```bash
python bc_model_module/run_live.py \
    --model-path models/bc/best_bc.pt \
    --window-title "Clash Royale - UnwontedTemper73" \
    --dry-run
```

---

## 3. CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Path to `best_bc.pt` checkpoint |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--window-title` | `""` | Window title for auto-detection |
| `--capture-region` | `""` | Manual region: `left,top,width,height` |
| `--fps` | `2.0` | Capture frames per second |
| `--no-perception` | off | Disable YOLO, use zero-filled observations |
| `--confidence` | `0.0` | Min logit score to act (0 = always) |
| `--cooldown` | `0.5` | Seconds between card plays |
| `--max-apm` | `20` | Max actions per minute |
| `--dry-run` | off | Log only, no mouse clicks |
| `--log-dir` | `logs/live` | Session log directory |
| `--quiet` | off | Suppress per-frame console output |

---

## 4. Architecture

```
run_live.py (CLI entry point + namespace package setup)
       |
       v
LiveInferenceEngine (main loop at ~2 FPS)
  |
  |-- GameCapture          mss screen grab
  |       |
  |       v
  |-- PerceptionAdapter    CRDetector + CardPredictor → obs tensors (or zero-fill)
  |       |
  |       v
  |-- BCPolicy             predict_action(obs, mask) → action_idx
  |       |
  |       v
  |-- ActionDispatcher     decode action → PyAutoGUI clicks (with window offset)
```

### Pipeline Per Frame (~350ms total at 2 FPS)

| Step | Component | Time |
|------|-----------|------|
| 1 | `GameCapture.capture()` → BGR frame | ~5ms |
| 2 | `CRDetector.detect()` → unit detections + arena grid | ~65ms |
| 3 | `CardPredictor.predict()` × 4 slots → card features | ~20ms |
| 4 | `BCPolicy.forward()` → logits → action | ~5ms |
| 5 | `ActionDispatcher.execute()` → clicks | ~100ms (if card played) |

---

## 5. Perception Tiers

The script degrades gracefully based on available dependencies:

### Tier 2: CRDetector + CardPredictor

**Requires:** `ultralytics` + `models/best_yolov8s_50epochs_fixed_pregen_set.pt` + `models/card_classifier.pt`

Uses `CRDetector` (YOLOv8 wrapper) with real `CLASS_NAME_TO_ID` mapping from `label_list.py` (155 classes). Arena grid uses proper 6-channel per-cell identity encoding (class_id, belonging, mask, tower HP, spell). `CardPredictor` (MiniResNet) classifies 4 card slot crops and populates vector features [11-22] (card present, class index, elixir cost). Belonging uses Y-position heuristic (top half = enemy, bottom half = ally).

**Limitations:** No OCR for elixir/timer (hardcoded defaults). No action mask elixir check. Belonging heuristic fails when troops cross river.

### Tier 3: Zero-Filled Observations

**Requires:** Nothing beyond numpy + torch.

Returns an empty arena grid and a mid-game default vector (5 elixir, 120s remaining, all towers alive, all cards present). All actions are unmasked.

The model relies entirely on its learned biases. It will predict whatever action it most commonly produced during training for "empty" inputs.

---

## 6. Window Offset (Important)

PyAutoGUI clicks use **absolute screen coordinates**, but the game may not start at (0, 0). The capture region's top-left corner is used as the window offset:

```
absolute_x = int(x_norm * frame_w) + window_left
absolute_y = int(y_norm * frame_h) + window_top
```

**How to find your game window position:**
1. Open the game/emulator
2. Note the window's top-left corner position on screen
3. Pass as `--capture-region left,top,width,height`

Example: If the game window is at screen position (100, 50) and is 540x960:
```bash
--capture-region 100,50,540,960
```

---

## 7. Safety Measures

| Measure | Default | Description |
|---------|---------|-------------|
| Confidence threshold | 0.0 | Skip actions with low model confidence |
| Action cooldown | 0.5s | Prevents rapid-fire card plays |
| Rate limit | 20/min | Hard cap on actions per minute |
| Dry run | off | Log everything, click nothing |
| Ctrl+C | always | Clean shutdown with summary |

**Recommended for first use:**
```bash
--dry-run --confidence 2.0
```
This logs what the model *would* do without clicking, and only for high-confidence predictions.

---

## 8. Log Format

Each session produces a JSONL file at `logs/live/session_YYYYMMDD_HHMMSS.jsonl`.

### Per-Frame Entry

```json
{
  "frame": 42,
  "timestamp": 1740000021.234,
  "step_time_ms": 82.3,
  "action_idx": 847,
  "logit_score": 3.42,
  "executed": true,
  "reason": "played",
  "card_id": 1,
  "col": 7,
  "row": 15,
  "x_norm": 0.4167,
  "y_norm": 0.3946,
  "detection_count": 14,
  "perception_active": true
}
```

### Possible `reason` Values

| Reason | Meaning |
|--------|---------|
| `noop` | Model predicted no-op (action 2304) |
| `played` | Card was played (clicks executed) |
| `below_confidence` | Logit below `--confidence` threshold |
| `cooldown` | Too soon since last action |
| `rate_limited` | Hit `--max-apm` cap |
| `dry_run` | Would have played, but `--dry-run` is active |

### Session Summary (Last Line)

```json
{
  "summary": {
    "duration_seconds": 180.5,
    "total_frames": 361,
    "actions_executed": 22,
    "noops": 310,
    "avg_fps": 2.0
  }
}
```

### Analyzing Logs

```python
import pandas as pd

df = pd.read_json("logs/live/session_20260222_143000.jsonl", lines=True)
frames = df[df["frame"].notna()]  # Exclude summary row

print(f"Actions played: {(frames.reason == 'played').sum()}")
print(f"Noops: {(frames.reason == 'noop').sum()}")
print(f"Avg step time: {frames.step_time_ms.mean():.1f}ms")
```

---

## 9. Assumptions and Current Limitations

### Assumptions

1. **540x960 base resolution** — Card slot positions and arena bounds are calibrated for this resolution (Google Play Games default). If your window is a different size, ensure `--capture-region` width and height match the actual window.

2. **Royal Hogs / Royal Recruits deck** — The model was trained on a specific 8-card deck. Playing with a different deck will produce poor predictions.

3. **Single monitor** — If no capture region or window title is provided, captures the primary monitor.

### Current Limitations

1. **No elixir/timer detection** — Without OCR, elixir and timer use hardcoded defaults (5 elixir, 120s remaining). The action mask allows all actions, so the model may try to play cards the player can't afford.

2. **Belonging heuristic** — Unit side assignment uses Y-position (top half = enemy, bottom = ally). This fails when troops cross the river. KataCR's 7-column belonging model would fix this but requires retraining.

3. **Single-frame observation** — No temporal context. The model can't see troop movement direction or anticipate opponent plays.

4. **Distribution shift** — If the model makes one bad play, the resulting game state will be unlike anything in the training data, leading to cascading errors. This is the fundamental limitation of behavior cloning, addressed by PPO fine-tuning.

5. **Static window position** — The capture region is determined at startup. If the game window moves, clicks will land in wrong positions.

6. **No pause/resume** — Only Ctrl+C stop is available (pynput keyboard listener not installed).

7. **No card drag support** — The bot uses a two-click sequence (click card slot, then click arena). Some players prefer drag-and-drop, which is not supported.

---

## 10. Troubleshooting

### "mss is required: pip install mss"

Install the screen capture library:
```bash
pip install mss
```

### "pyautogui not installed. Forcing dry-run mode."

Install PyAutoGUI for click execution:
```bash
pip install pyautogui
```

### "Cannot find window '...' and no --capture-region specified"

Either install `pygetwindow` (`pip install pygetwindow`) for auto-detection, or manually specify the capture region.

### "Monkeypatching src.generation.label_list"

This is expected. The `label_list` module (155 unit class names) doesn't exist locally. A mock is injected so `encoder_constants.py` can load. This means class ID mappings are dummy values — the model's arena embeddings won't match training.

To fix properly: obtain `label_list.py` from the KataCR project or create it with the correct 155 unit names.

### Model always predicts no-op

- **With zero-fill perception:** Expected. The model sees an empty battlefield and defaults to "wait."
- **With YOLO perception:** Class IDs are placeholders, so the arena embedding branch produces garbage. Try lowering `--confidence` to 0.

### Clicks land in wrong position

Verify your `--capture-region` matches the actual game window position and size. The capture region's `left,top` values are used as the offset for all clicks.

---

## 11. Future Work

When the missing dependencies become available:

1. **Install `gymnasium`** and provide `src.generation.label_list` → enables `StateEncoder` import
2. **Provide `src.pipeline.game_state`** → enables full `GameState` construction
3. **Add OCR** (PaddleOCR) → real elixir and timer detection
4. **Add CardPredictor** → real card hand detection with proper action masking
5. **Implement Tier 1 perception** in `PerceptionAdapter._process_full()`
6. **Add keyboard pause/resume** when `pynput` is available
7. **PPO fine-tuning** → addresses distribution shift limitation
