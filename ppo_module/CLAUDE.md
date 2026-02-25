# PPO Module

PPO fine-tuning pipeline for the Clash Royale BC agent. Wraps the live game as a Gymnasium environment, computes rewards from observation deltas, and trains with sb3_contrib MaskablePPO.

**Status:** Implemented.

## Architecture

```
GameCapture (mss) -> PerceptionAdapter (YOLO) -> ClashRoyaleEnv (Gymnasium)
    -> RewardComputer (obs deltas) -> MaskablePPO (SB3) -> ActionDispatcher (PyAutoGUI)
```

## Key Files

| File | Description |
|------|-------------|
| `src/ppo/sb3_feature_extractor.py` | SB3CRFeatureExtractor - wraps CRFeatureExtractor for MaskablePPO |
| `src/ppo/reward.py` | RewardConfig + RewardComputer - tower count, win/loss, survival, elixir waste |
| `src/ppo/game_detector.py` | GamePhaseDetector - game start/end/outcome detection from raw frames |
| `src/ppo/clash_royale_env.py` | ClashRoyaleEnv - Gymnasium wrapper with action_masks() |
| `src/ppo/callbacks.py` | CRMetricsCallback - per-episode metrics logging |
| `src/ppo/ppo_trainer.py` | PPOTrainer - training orchestrator with freeze/unfreeze schedule |
| `run_ppo.py` | CLI entry point with namespace setup |

## Reward Components

| Component | Signal | Weight |
|-----------|--------|--------|
| Enemy crown | Tower count drops | +10.0 |
| Ally crown lost | Tower count drops | -10.0 |
| Win | Game outcome | +30.0 |
| Loss | Game outcome | -30.0 |
| Draw | Game outcome | -5.0 |
| Survival | Per step | +0.02 |
| Elixir waste | At max elixir | -0.05/step |

## Training Phases

**Phase 1 (frozen extractor, ~15 games):** Only train policy + value heads. LR=1e-4.
**Phase 2 (unfrozen, ~25+ games):** Fine-tune everything. LR=3e-5.

## CLI Usage

```bash
# Phase 1
python ppo_module/run_ppo.py --bc-weights models/bc/bc_feature_extractor.pt --window-title "Clash Royale" --num-episodes 15 --freeze-extractor

# Phase 2
python ppo_module/run_ppo.py --resume models/ppo/latest_ppo.zip --window-title "Clash Royale" --num-episodes 25 --lr 3e-5

# Dry run
python ppo_module/run_ppo.py --bc-weights models/bc/bc_feature_extractor.pt --capture-region 0,0,540,960 --dry-run
```

## Tests

19 tests: `python -m pytest ppo_module/tests/ -v`

## Dependencies

torch, numpy, gymnasium, cv2, stable-baselines3, sb3-contrib, tensorboard

Plus existing: src.bc (feature_extractor, live_inference), src.encoder (encoder_constants)

## Depends On

- `bc_model_module` - CRFeatureExtractor, GameCapture, PerceptionAdapter, ActionDispatcher
- `state_encoder_module` - encoder_constants
- `src/src/generation/` - label_list for class mappings
- `src/src/detection/` - CRDetector for perception
- `src/src/classification/` - CardPredictor for card hand classification
