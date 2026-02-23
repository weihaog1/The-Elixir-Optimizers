# src/dataset/ - DatasetBuilder (Recording Sessions -> BC Training Data)

Converts click_logger session directories (screenshots + paired actions) into training-ready .npz files for behavior cloning. Processes raw recordings through the perception pipeline (EnhancedStateBuilder with YOLO + OCR + card classifier) and the StateEncoder to produce (obs, action, mask) tuples compatible with SB3 MaskableMultiInputPolicy. Handles timestamp-based action assignment and no-op downsampling to address the ~90% no-op class imbalance.

## Files

**dataset_builder.py** - Core processing pipeline.
- `DatasetStats`: Dataclass summarizing processing results.
  - `total_frames`: Number of frames in the session
  - `total_actions`: Number of card placements from actions.jsonl
  - `noop_frames`: Frames with no action (NOOP_ACTION = 2304)
  - `action_frames`: Frames with a card placement action
  - `kept_after_downsample`: Final frame count after no-op filtering
  - `session_dir`: Input session directory path
  - `output_path`: Path to the saved .npz file
- `DatasetBuilder(enhanced_state_builder, state_encoder)`: Main processor.
  - `build_dataset(session_dir, output_dir, noop_keep_ratio=0.15) -> DatasetStats`: Process one session
  - `build_from_multiple(session_dirs, output_dir, noop_keep_ratio=0.15) -> List[DatasetStats]`: Process multiple sessions
  - `_load_session(session_dir) -> (frames, actions, metadata)`: Parse JSONL and JSON files
  - `_convert_actions(actions) -> List[(timestamp, action_idx)]`: Pre-paired actions to Discrete(2305) indices
  - `_assign_actions_to_frames(action_events, frame_timestamps) -> List[int]`: Nearest-timestamp assignment
  - `_process_frame(image_path, timestamp) -> (obs_dict, mask)`: YOLO + OCR + encode single frame
  - `_downsample_noops(indices, action_labels, keep_ratio) -> List[int]`: Random subsample of no-op frames

**card_integration.py** - EnhancedStateBuilder wrapper.
- `EnhancedStateBuilder(state_builder, card_predictor=None)`: Wraps StateBuilder + CardPredictor.
  - `build_state(image, timestamp, **kwargs) -> GameState`: Delegates to base StateBuilder, then crops 4 card slots and classifies each via CardPredictor
  - `_extract_cards(image, frame_w, frame_h) -> List[Card]`: Crops card regions using ScreenConfig.scale_to_resolution(), classifies with CardPredictor

**__init__.py** - Module exports: `DatasetBuilder`, `DatasetStats`, `EnhancedStateBuilder`.

## Input Format

### Session Directory Structure

```
recordings/match_YYYYMMDD_HHMMSS/
  screenshots/
    frame_000000.jpg
    frame_000001.jpg
    ...
  actions.jsonl
  frames.jsonl
  metadata.json
```

### frames.jsonl (one line per captured screenshot)

```json
{"frame_idx": 0, "timestamp": 1740123456.123, "filename": "frame_000000.jpg", "width": 1080, "height": 1920}
```

### actions.jsonl (one line per card placement, pre-paired by ClickLogger)

```json
{"timestamp": 1740123460.789, "card_id": 2, "x_norm": 0.45, "y_norm": 0.55}
```

### metadata.json (session summary)

```json
{
  "window_title": "Clash Royale - UnwontedTemper73",
  "fps": 2.0,
  "duration_seconds": 180.0,
  "frame_count": 360,
  "action_count": 22
}
```

## Output Format

### .npz File Contents

| Array Key | Shape | Dtype | Description |
|-----------|-------|-------|-------------|
| `obs_arena` | `(N, 32, 18, 6)` | float32 | Spatial arena grid per frame |
| `obs_vector` | `(N, 23)` | float32 | Scalar features per frame |
| `actions` | `(N,)` | int64 | Discrete(2305) action index per frame |
| `masks` | `(N, 2305)` | bool | Valid action mask per frame |
| `timestamps` | `(N,)` | float64 | Frame timestamps (for debugging) |

Where N = `kept_after_downsample` (all action frames + downsampled no-op frames).

## Action Assignment Algorithm

1. Initialize all frame labels to NOOP_ACTION (2304)
2. For each action in actions.jsonl:
   a. Convert `(card_id, x_norm, y_norm)` to action index via `norm_to_cell()` + `placement_to_action()`
   b. Find the frame with the closest timestamp (linear scan)
   c. Assign the action index to that frame
3. If multiple actions map to the same frame, the later action overwrites

## No-op Downsampling

- All action frames (action_labels[i] != NOOP_ACTION) are always kept
- No-op frames are randomly subsampled at `noop_keep_ratio` (default 0.15)
- At least 1 no-op frame is always kept (`max(1, int(count * ratio))`)
- Random sampling uses `random.sample()` for uniform selection
- Frame order is preserved (indices are sorted after selection)

## Dependencies

```
dataset_builder.py
  -> cv2 (image loading)
  -> numpy (array operations)
  -> src.encoder.coord_utils (norm_to_cell, placement_to_action)
  -> src.encoder.encoder_constants (NOOP_ACTION)
  -> src.encoder.state_encoder.StateEncoder (lazy import if not provided)
  -> src.pipeline.game_state.GameState (lazy import for fallback)

card_integration.py
  -> numpy
  -> src.data.screen_regions.ScreenConfig (card slot regions)
  -> src.pipeline.game_state (Card, GameState)
```

## Downstream Usage

The .npz output feeds directly into BC training:

```python
import numpy as np

data = np.load("data/bc_training/match_20260222_143000.npz")
obs_arena = data["obs_arena"]    # (N, 32, 18, 6)
obs_vector = data["obs_vector"]  # (N, 23)
actions = data["actions"]        # (N,)
masks = data["masks"]            # (N, 2305)
```

Compatible with SB3 MaskableMultiInputPolicy via a custom Dataset class that returns `{"arena": tensor, "vector": tensor}` observation dicts with corresponding action labels and masks.
