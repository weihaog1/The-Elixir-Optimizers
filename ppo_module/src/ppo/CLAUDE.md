# src/ppo/ - PPO Training Package

## Files

| File | Description |
|------|-------------|
| `sb3_feature_extractor.py` | `SB3CRFeatureExtractor(BaseFeaturesExtractor)` - wraps CRFeatureExtractor, has freeze/unfreeze/load_bc_weights |
| `reward.py` | `RewardConfig` + `RewardComputer` - stateful reward from obs vector deltas |
| `game_detector.py` | `GamePhaseDetector` + `Phase` enum - pixel heuristics for game state detection |
| `clash_royale_env.py` | `ClashRoyaleEnv(gymnasium.Env)` + `EnvConfig` - full Gym wrapper |
| `callbacks.py` | `CRMetricsCallback(BaseCallback)` - episode metrics for TensorBoard + JSONL |
| `ppo_trainer.py` | `PPOTrainer` + `PPOConfig` - training orchestrator |

## RewardComputer

Tracks tower counts between steps. Vector indices: 0=elixir, 9=ally towers, 10=enemy towers.

```python
rc = RewardComputer(RewardConfig())
rc.reset()
reward = rc.compute(prev_obs, curr_obs, terminal_outcome="win")
```

## ClashRoyaleEnv

Reuses GameCapture, PerceptionAdapter, ActionDispatcher from `src.bc.live_inference`.

```python
env = ClashRoyaleEnv(EnvConfig(capture_region=(0,0,540,960), dry_run=True))
obs, info = env.reset()  # blocks until game starts
obs, reward, terminated, truncated, info = env.step(action)
mask = env.action_masks()  # for MaskablePPO
```

## MaskablePPO Setup

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO("MultiInputPolicy", env, policy_kwargs={
    "features_extractor_class": SB3CRFeatureExtractor,
    "features_extractor_kwargs": {"features_dim": 192},
    "net_arch": dict(pi=[128, 64], vf=[128, 64]),
})
```
