# BC Model Module

Implements the Behavior Cloning neural network for Clash Royale card placement prediction.

**Status:** Implemented.

## Architecture

CRFeatureExtractor (SB3-compatible) with full embedding approach:
- **Arena branch:** `nn.Embedding(156, 8)` for class IDs + 5 remaining channels -> 3-layer CNN -> 128 features
- **Vector branch:** `nn.Embedding(9, 8)` for card classes + 19 scalar features -> 2-layer MLP -> 64 features
- **Output:** Concatenated 192-dim feature vector

BCPolicy uses **hierarchical action decomposition** with three heads from a shared 256-dim trunk:
- **Play head:** `Linear(256, 2)` - binary play/noop decision
- **Card head:** `Linear(256, 4)` - which card slot
- **Position head:** `Linear(256, 576)` - which grid cell

Total parameters: 304,366

This decomposition is critical because the flat 2305-way softmax collapses to always-noop with 652 actions spread across 2304 classes. The hierarchical approach gives each head a tractable classification problem.

## Key Files

| File | Description |
|------|-------------|
| `src/bc/__init__.py` | Package exports |
| `src/bc/feature_extractor.py` | CRFeatureExtractor - arena embedding + CNN, card embedding + MLP, 192-dim output |
| `src/bc/bc_policy.py` | BCPolicy - extractor + hierarchical play/card/position heads, predict_action |
| `src/bc/bc_dataset.py` | BCDataset + load_datasets() - loads .npz files, file-level 80/20 split, class weight computation |
| `src/bc/train_bc.py` | BCTrainer + TrainConfig - hierarchical loss, F1 early stopping, cosine annealing, CLI |
| `src/bc/live_inference.py` | LiveConfig + LiveInferenceEngine - real-time game inference with capture, perception, and action execution |
| `src/bc/CLAUDE.md` | Package technical reference |
| `train_model.py` | CLI entry point for training (with namespace setup) |
| `run_live.py` | CLI entry point for live inference |

## Training Approach

- Custom PyTorch training loop with hierarchical decomposed loss
- Three separate CE losses: play (weighted), card (action frames only), position (action frames only)
- `play_weight=8.0` to address 85:15 noop:action imbalance
- Cosine annealing LR starting at `1e-4`
- 80/20 file-level train/val split (game-level, not frame-level)
- Early stopping on **action F1 score** with `patience=25`
- Gradient clipping at `1.0`

## Training Results (28 games, 4429 frames)

| Metric | Value |
|--------|-------|
| Best Action F1 | 0.324 |
| Action Recall | 69.5% |
| Action Precision | 21.2% |
| Card Accuracy | 34.2% (above 25% random) |
| Best Epoch | 19 |

## PPO Transition

Feature extractor weights saved separately as `bc_feature_extractor.pt`. Load into MaskablePPO via `policy_kwargs`. Optionally freeze extractor during initial PPO training, then unfreeze with lower LR.

## Output Files

| File | Description |
|------|-------------|
| `best_bc.pt` | Full BC policy checkpoint |
| `bc_feature_extractor.pt` | Feature extractor weights only (for PPO) |
| `training_log.json` | Training metrics per epoch |

## Live Inference

Real-time game inference loop: capture screen → CRDetector + CardPredictor → BC policy → PyAutoGUI clicks.

**Perception pipeline:**
- **CRDetector** (`models/best_yolov8s_50epochs_fixed_pregen_set.pt`) — YOLOv8s with real `CLASS_NAME_TO_ID` mapping, proper 6-channel arena encoding
- **CardPredictor** (`models/card_classifier.pt`) — MiniResNet card hand classification, populates vector features [11-22]
- **Fallback** — zero-filled observations if models unavailable

Features: confidence threshold, action cooldown, rate limiting, dry-run mode, JSONL logging.

```bash
python bc_model_module/run_live.py --model-path models/bc/best_bc.pt --capture-region 0,0,540,960 --dry-run
```

## CLI Usage

```bash
# Training
python bc_model_module/train_model.py --data_dir data/bc_training/ --output_dir models/bc/

# Training with tuned hyperparameters
python bc_model_module/train_model.py --data_dir data/bc_training/ --output_dir models/bc/ --lr 1e-4 --batch_size 32 --patience 25 --play_weight 8.0

# Live inference
python bc_model_module/run_live.py --model-path models/bc/best_bc.pt --capture-region 0,0,540,960 --dry-run
```

## Documentation

- `docs/bc-training-guide.md` - Step-by-step training documentation, results, and analysis
- `docs/bc-analysis.md` - 964-line comprehensive analysis and decision guide
- `docs/bc-model-docs.md` - Usage guide, live game testing, PPO transition
- `docs/live-inference-guide.md` - Live inference setup, usage, assumptions, limitations

## Dependencies

torch, numpy, `src.encoder.encoder_constants`

Live inference additionally uses: mss, cv2, pyautogui (optional), ultralytics (optional), pygetwindow (optional), `src.detection.model` (CRDetector), `src.classification.card_classifier` (CardPredictor), `src.generation.label_list` (class mappings)

## Depends On

- `dataset_builder_module` - produces the `.npz` files that BCDataset loads
- `state_encoder_module` - encoder_constants for action space, grid dimensions, CLASS_NAME_TO_ID, UNIT_TYPE_MAP
- `action_builder_module` - action_constants for card slot regions (reference)
- `src/src/detection/` - CRDetector YOLO wrapper for live inference
- `src/src/classification/` - CardPredictor for card hand classification
- `src/src/generation/` - label_list.py for real 155-class name mappings
