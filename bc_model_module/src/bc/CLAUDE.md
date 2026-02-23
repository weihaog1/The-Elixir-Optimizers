# bc Package - Technical Reference

BC model package: neural network, dataset loader, and training loop for behavior cloning.

**Status:** Implemented.

## Files

| File | Description |
|------|-------------|
| `feature_extractor.py` | CRFeatureExtractor - arena embedding + CNN, card embedding + MLP, 192-dim output |
| `bc_policy.py` | BCPolicy - extractor + action head, save/load, predict_action with masking |
| `bc_dataset.py` | BCDataset + load_datasets() - .npz loading, file-level 80/20 split, class weights |
| `train_bc.py` | BCTrainer + TrainConfig - custom PyTorch loop, weighted CE, cosine annealing, early stopping |
| `live_inference.py` | LiveConfig + GameCapture + PerceptionAdapter + ActionDispatcher + LiveInferenceEngine |
| `__init__.py` | Package exports |

## CRFeatureExtractor

Subclasses `nn.Module` (SB3 `BaseFeaturesExtractor` compatible).

**Arena branch:**
1. Extract `CH_CLASS_ID` channel -> denormalize to int -> `nn.Embedding(156, 8)`
2. Concat with 5 remaining channels -> `(B, 13, 32, 18)`
3. `Conv2d(13, 32, 3, pad=1)` + BN + ReLU + MaxPool
4. `Conv2d(32, 64, 3, pad=1)` + BN + ReLU + MaxPool
5. `Conv2d(64, 128, 3, pad=1)` + BN + ReLU + AdaptiveAvgPool(1)
6. Output: 128 features

**Vector branch:**
1. Extract card class floats `[15:19]` -> denormalize -> `nn.Embedding(9, 8)`
2. Concat with 19 scalar features -> 51-dim input
3. `Linear(51, 64)` + ReLU -> `Linear(64, 64)` + ReLU
4. Output: 64 features

**Output:** `(B, 192)` concatenated features

**Constants:**
- `_ARENA_EMBED_ENTRIES=156`, `_ARENA_EMBED_DIM=8`
- `_CARD_EMBED_ENTRIES=9`, `_CARD_EMBED_DIM=8`
- `_ARENA_CNN_IN=13`, `_VECTOR_MLP_IN=51`
- `_DEFAULT_FEATURES_DIM=192`

## BCPolicy

```python
__init__(features_dim=192, hidden_dim=256, dropout=0.2)
forward(obs) -> (B, 2305)  # logits
predict_action(obs, mask) -> int  # masked argmax, no_grad
save(path)
load(path)
get_feature_extractor_state() -> dict  # for PPO transfer
```

Action head: `Linear(192, 256)` + ReLU + Dropout(0.2) + `Linear(256, 2305)`

## BCDataset

```python
__init__(npz_paths: list[Path])  # loads and concatenates .npz files
__getitem__(idx) -> {"arena": float32, "vector": float32, "action": long, "mask": bool}
action_class_counts() -> (noop_count, action_count)
compute_class_weights(noop_weight=0.3, action_weight=3.0) -> (2305,) tensor
load_datasets(paths, val_ratio=0.2, seed=42) -> (train_dataset, val_dataset)  # file-level split
```

## BCTrainer

```python
@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 10
    val_ratio: float = 0.2
    noop_weight: float = 0.3
    action_weight: float = 3.0
    grad_clip: float = 1.0
    seed: int = 42

train(npz_paths, output_dir) -> dict
    # Returns: train_losses, val_losses, val_accuracies, best_val_loss, best_epoch
```

**Saves:** `best_bc.pt`, `bc_feature_extractor.pt`, `training_log.json`

**Training loop:** Weighted cross-entropy loss, AdamW optimizer, cosine annealing scheduler, gradient clipping, early stopping.

## PPO Transition

```python
# Extract weights from trained BC policy
state_dict = policy.get_feature_extractor_state()
torch.save(state_dict, "bc_feature_extractor.pt")

# Load into MaskablePPO
ppo.policy.features_extractor.load_state_dict(
    torch.load("bc_feature_extractor.pt")
)

# Optionally: freeze extractor, train PPO, then unfreeze with lower LR
```

## LiveInferenceEngine

Real-time game inference: capture → perceive → predict → act at ~2 FPS.

```python
LiveConfig(model_path, capture_region, dry_run, confidence_threshold, card_classifier_path, ...)
LiveInferenceEngine(config, project_root)
  .run()  # Main loop, Ctrl+C to stop
```

**Components:**
- `GameCapture` - mss screen capture with rate limiting
- `PerceptionAdapter` - CRDetector (real class IDs via CLASS_NAME_TO_ID) + CardPredictor (card hand classification) → 6-channel arena + 23-dim vector obs tensors. Falls back to zero-fill if models unavailable.
- `ActionDispatcher` - PyAutoGUI clicks with window offset correction
- Constants inlined from encoder_constants/action_constants to avoid import chain issues

**Entry point:** `python bc_model_module/run_live.py --model-path best_bc.pt --capture-region 0,0,540,960 --dry-run`

## Dependencies

torch, numpy, `src.encoder.encoder_constants`

Live inference additionally: mss, cv2, pyautogui (optional), ultralytics (optional), `src.detection.model` (CRDetector), `src.classification.card_classifier` (CardPredictor)
