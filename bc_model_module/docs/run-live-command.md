# Running the BC Model on a Live Game

## Quick Command

**Dry run (safe, logs only, no clicks):**

```bash
python bc_model_module/run_live.py --model-path models/bc/best_bc.pt --window-title "Clash Royale - thegoodpersonplayer2" --dry-run
```

**Live execution (plays cards via mouse clicks):**

```bash
python bc_model_module/run_live.py --model-path models/bc/best_bc.pt --window-title "Clash Royale - thegoodpersonplayer2"
```

---

## What This Does

1. Finds the Google Play Games window by title
2. Captures the game screen at 2 FPS via mss
3. Runs YOLO detection (155 Clash Royale unit classes) to identify troops, towers, and spells on the arena
4. Classifies the 4 cards in hand via CardPredictor (MiniResNet, 8-card deck)
5. Encodes detections into a 6-channel arena grid (32x18) and 23-dim vector
6. Feeds observations into BCPolicy (hierarchical play/card/position heads)
7. Executes card placements via PyAutoGUI two-click sequence (card slot then arena)

---

## Models Used

| Model | Path | Purpose |
|-------|------|---------|
| BC Policy | `models/bc/best_bc.pt` | Card placement prediction (304K params) |
| YOLO Detector | `models/best_yolov8s_50epochs_fixed_pregen_set.pt` | Unit/tower detection (YOLOv8s, 155 classes) |
| Card Classifier | `models/card_classifier.pt` | Hand card classification (MiniResNet, 8 deck cards) |

All three models are loaded automatically. The BC policy is specified via `--model-path`. The YOLO detector and card classifier use their default paths.

---

## Prerequisites

Install required packages:

```bash
pip install torch numpy opencv-python mss pyautogui pygetwindow ultralytics
```

---

## Recommended First-Time Usage

Start with dry-run to verify the pipeline works before enabling clicks:

```bash
python bc_model_module/run_live.py --model-path models/bc/best_bc.pt --window-title "Clash Royale - thegoodpersonplayer2" --dry-run
```

Check the console output. You should see:
- `[Perception] Loaded CRDetector: ...` (YOLO loaded)
- `[Perception] Loaded CardPredictor: ...` (card classifier loaded)
- `[Engine] Starting live inference loop` with `Perception: True`
- Per-frame logs showing detection counts and predicted actions

Once satisfied, run live:

```bash
python bc_model_module/run_live.py --model-path models/bc/best_bc.pt --window-title "Clash Royale - thegoodpersonplayer2" --temperature 1.5 --noop-frames 3 --repeat-penalty 2.0
```

Press **Ctrl+C** to stop at any time.

---

## All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Path to `best_bc.pt` |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--window-title` | `""` | Game window title for auto-detection |
| `--capture-region` | `""` | Manual region: `left,top,width,height` |
| `--fps` | `2.0` | Capture FPS |
| `--no-perception` | off | Disable YOLO (zero-filled observations) |
| `--card-classifier` | `models/card_classifier.pt` | Card classifier weights |
| `--detector-model` | (auto) | YOLO detector weights |
| `--confidence` | `0.0` | Min logit to execute action |
| `--cooldown` | `0.5` | Seconds between card plays |
| `--max-apm` | `20` | Max actions per minute |
| `--temperature` | `1.5` | Sampling temperature (>1 = diverse, <1 = greedy) |
| `--noop-frames` | `3` | Force noop for N frames after each card play |
| `--repeat-penalty` | `2.0` | Logit penalty for recently-used actions |
| `--repeat-memory` | `5` | Number of recent actions to penalize |
| `--dry-run` | off | Log only, no mouse clicks |
| `--log-dir` | `logs/live` | Session log directory |
| `--quiet` | off | Suppress per-frame output |

---

## Logs

Session logs are saved as JSONL files in `logs/live/`. Each file contains per-frame entries and a session summary at the end.

```bash
# View the latest session log
ls -t logs/live/*.jsonl | head -1
```
