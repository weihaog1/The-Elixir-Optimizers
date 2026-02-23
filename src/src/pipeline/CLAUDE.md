# src/pipeline/ - Game State Extraction

Combines object detection and OCR to build a structured `GameState` from gameplay screenshots or video frames. This is the bridge between raw perception (YOLOv8 detections) and the RL agent's decision-making.

## Files

**game_state.py** (282 lines) - Data model for structured game state.
- `Tower` dataclass: tower_type (king/princess), position (left/right/center), belonging (0=player, 1=enemy), hp, max_hp, bbox, confidence. Properties: is_destroyed, hp_percentage.
- `Unit` dataclass: class_name, belonging, bbox, confidence, hp. Properties: center, is_player, is_enemy.
- `Card` dataclass: slot (0-3 or -1 for next card), class_name, elixir_cost, confidence.
- `GameState` dataclass: Full game state with timestamp, time_remaining, is_overtime, elixir, towers (list), units (list), cards (list), frame shape/source. Methods: to_dict(), to_json(), from_dict(), summary(). Properties: player_towers, enemy_towers, player_units, enemy_units.
- `TOWER_MAX_HP` constant: Level-based HP values for king/princess towers.

**state_builder.py** (536 lines) - Main pipeline orchestrator.
- `StateBuilder`: Takes a CRDetector and optional GameTextExtractor.
  - `build_state(image)` -> GameState: Runs detection, classifies towers and units, assigns ally/enemy side, optionally runs OCR.
  - `_infer_unit_belonging(detection)`: Uses Y-coordinate heuristic -- units above arena midpoint (~47% of frame height) are enemy, below are ally.
  - `process_video(video_path, target_fps)`: Extract GameState from every frame at target FPS.
  - `visualize(image, state)`: Draw game state overlay with colored bounding boxes (blue=ally, red=enemy) and HUD info.

## Side Classification (Belonging)

Units are assigned to ally (0) or enemy (1) based on vertical position:
```
arena_mid_y = frame_height * 0.42
side = 0 if detection.center_y > arena_mid_y else 1  # below=ally, above=enemy
```

**This is a temporary heuristic.** It fails when troops cross the river (e.g., hog-rider pushing into enemy territory). KataCR's model outputs a 7th belonging column via a binary classification head, which is more accurate. Custom NMS for 7-column output is ported in `src/yolov8_custom/custom_utils.py` but unused because the current model was not trained with belonging labels. Retraining with belonging labels is a priority improvement.

## Dependencies

```
state_builder.py
  -> src.detection.model (CRDetector, Detection)
  -> src.ocr.text_extractor (GameTextExtractor) -- optional, PaddleOCR for timer/elixir/HP
  -> game_state (GameState, Tower, Unit, Card)
  -> src.classification.card_classifier (CardPredictor) -- EXISTS but NOT wired in yet
```

## Card Hand Detection (NOT YET INTEGRATED)

`src/classification/card_classifier.py` contains a trained MiniResNet (~25K params) that classifies card crops into 8 deck classes. `CardPredictor.predict(bgr_crop)` returns `(class_name, confidence)`. However, StateBuilder does not call it, so `GameState.cards` is always an empty list.

Card slot crop regions at 1080x1920: slots at (242,1595)-(430,1830), (445,1595)-(633,1830), (648,1595)-(836,1830), (851,1595)-(1039,1830). Card size 188x235 px.

## RL Agent Readiness (v12 eval findings)

### What the pipeline can reliably provide
- Tower positions and types (AP50 > 0.99 for all tower classes)
- Most troop positions for common meta decks (85 unique classes detected on gameplay video)
- Our own deck cards (royal-hog 0.995, hog-rider 0.869, musketeer 0.979, fireball 0.977)
- Clock/timer, elixir indicators

### What the pipeline struggles with
- Cycle cards: skeleton (AP50=0.670), ice-spirit (0.644) -- affects elixir counting
- Spells: zap (0.245), arrows (0.190) -- can't react to opponent spells via vision alone
- Miner (0.237) -- invisible when it pops up near towers
- Barbarian-barrel (0.396) -- very common defensive card, nearly invisible

### Implications for RL
- Good enough for basic decision-making (troop placement, tower targeting)
- Elixir tracking may be inaccurate due to missed small units
- Spell detection should use temporal/damage-based heuristics rather than visual detection
- Consider class-specific confidence thresholds (high for towers, low for small troops)

## Downstream: StateEncoder

The `GameState` produced by `StateBuilder` is consumed by `src/encoder/StateEncoder`, which converts it to SB3-compatible observation tensors:
- Arena grid: (32, 18, 7) float32 -- spatial unit/tower positions on 18x32 grid
- Vector: (23,) float32 -- elixir, time, tower HP fracs, card hand
- Action mask: (2305,) bool -- valid card placements given current hand and elixir

See `src/encoder/CLAUDE.md` for full observation and action space documentation.
