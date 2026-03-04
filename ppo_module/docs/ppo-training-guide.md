# PPO Training Guide

Complete guide for running PPO fine-tuning on a live Clash Royale game. The agent plays autonomously during matches while a human operator queues games between episodes.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Phase 0: Dry Run Verification](#3-phase-0-dry-run-verification)
4. [Phase 1: Frozen Feature Extractor](#4-phase-1-frozen-feature-extractor)
5. [Phase 2: Full Fine-Tuning](#5-phase-2-full-fine-tuning)
6. [Phase 3: Evaluation](#6-phase-3-evaluation)
7. [Operator Workflow](#7-operator-workflow)
8. [Reward Function Reference](#8-reward-function-reference)
9. [Hyperparameter Reference](#9-hyperparameter-reference)
10. [CLI Reference](#10-cli-reference)
11. [Monitoring Training](#11-monitoring-training)
12. [Episode Boundary Safety](#12-episode-boundary-safety)
13. [Troubleshooting](#13-troubleshooting)
14. [Output Files](#14-output-files)

---

## 1. Overview

### BC-to-PPO Transfer Learning Pipeline

```
BC Training (offline)                    PPO Training (live)
  .npz replay data                        Live Clash Royale game
       |                                       |
  BCPolicy (n_frames=3)                  Screen capture (mss, 2 FPS)
  CRFeatureExtractor                          |
       |                                  YOLO + CardPredictor
  bc_feature_extractor.pt -----> SB3CRFeatureExtractor (frozen, then unfrozen)
                                       |              |
                                  Policy head     Value head
                                  [256, 128]      [256, 128]
                                       |
                                  MaskablePPO -> PyAutoGUI clicks
```

The BC feature extractor learns card/unit spatial representations from 28 games of human gameplay. PPO fine-tunes this foundation through live self-play, adding RL reward signals that BC never had (crown scoring, elixir management, win/loss outcomes).

### Semi-Automated Training Loop

Training is semi-automated: the agent plays each match autonomously, but a human operator queues new games between episodes. Each episode:

1. `env.reset()` blocks until the operator clicks "Battle"
2. Agent plays the entire match (~3-5 minutes) autonomously
3. Game end is auto-detected; model checkpoint is saved
4. Operator clicks "Battle" again for the next episode

---

## 2. Prerequisites

### Required Models

| File | Description | How to Obtain |
|------|-------------|---------------|
| `models/bc/bc_feature_extractor.pt` | BC-pretrained feature extractor weights (n_frames=3) | Output of BC training with `--n_frames 3` |
| `models/bc/best_bc.pt` | Full BC policy (optional, for KL penalty) | Output of BC training |
| `models/best_yolov8s_50epochs_fixed_pregen_set.pt` | YOLOv8 unit detection model | Provided in `models/` |
| `models/card_classifier.pt` | Card hand classifier (MiniResNet, 8-class) | Provided in `models/` |

### Optional Files

| File | Description |
|------|-------------|
| `ppo_module/templates/victory.png` | Screenshot of the "Victory" banner for outcome detection |
| `ppo_module/templates/defeat.png` | Screenshot of the "Defeat" banner for outcome detection |

If templates are not provided, outcome detection falls back to color heuristics (golden tones = win, blue/dark tones = loss).

### BC Training (if not already done)

```bash
python bc_model_module/train_model.py \
    --data_dir data/bc_training/ \
    --output_dir models/bc/ \
    --n_frames 3 \
    --epochs 100 \
    --batch_size 64
```

This produces `best_bc.pt` (full policy) and `bc_feature_extractor.pt` (feature extractor only). The feature extractor must be trained with `n_frames=3` to match the PPO architecture.

### Software

- **Python 3.10+** with packages: `torch`, `numpy`, `gymnasium`, `opencv-python`, `stable-baselines3`, `sb3-contrib`, `tensorboard`, `mss`, `pyautogui`
- **Google Play Games** emulator with Clash Royale installed and running
- Game window must be **visible on screen** (not minimized)

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Not required (CPU works) | CUDA-capable GPU |
| RAM | 8 GB | 16 GB |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |

---

## 3. Phase 0: Dry Run Verification

Before live training, verify the full pipeline works without executing clicks.

```bash
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --capture-region 0,0,540,960 \
    --num-episodes 1 \
    --n-frames 3 \
    --dry-run
```

### What to Check

- Console prints `[PPOTrainer] Loaded BC weights from ...`
- Observations have correct shapes (arena: `(32, 18, 18)`, vector: `(69,)` for n_frames=3)
- Actions are logged but no mouse clicks executed
- Game phase detection transitions correctly (UNKNOWN -> LOADING -> IN_GAME)
- No crashes or import errors

---

## 4. Phase 1: Frozen Feature Extractor

**Goal:** Train only the policy and value heads while keeping the BC-learned feature representations frozen. This prevents catastrophic forgetting and lets the new heads learn to use BC features.

### Command

```bash
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --window-title "Clash Royale" \
    --num-episodes 15 \
    --freeze-extractor \
    --n-frames 3 \
    --lr 1e-4 \
    --n-steps 700 \
    --ent-start 0.02 \
    --ent-end 0.005 \
    --reward-scale 0.1 \
    --templates-dir ppo_module/templates
```

### Configuration Summary

| Setting | Value | Rationale |
|---------|-------|-----------|
| `--freeze-extractor` | on | Only train pi/vf heads on top of BC features |
| `--lr 1e-4` | 1e-4 | Standard PPO learning rate, cosine annealed to 1e-5 |
| `--n-steps 700` | 700 | Matches one full game (~300s at 2 FPS + overtime margin) |
| `--n-frames 3` | 3 | 3-frame temporal context (matches BC training) |
| `--ent-start/end` | 0.02 -> 0.005 | Linear entropy annealing: explore early, exploit later |
| `--reward-scale 0.1` | 0.1 | Scale all rewards by 0.1 for value function stability |
| `--clip-range` | 0.1 (default) | Conservative PPO clipping |
| `--n-epochs` | 10 (default) | SGD epochs per rollout |
| `--batch-size` | 64 (default) | Mini-batch size |

### What Trains vs What's Frozen

| Component | Parameters | Status |
|-----------|-----------|--------|
| CRFeatureExtractor (arena CNN + vector MLP) | ~100K | **Frozen** |
| Policy head `[256, 128]` | ~74K | Training |
| Value head `[256, 128]` | ~74K | Training |

### Expected Duration

- ~15 games x 3-5 minutes/game = 45-75 minutes
- Plus operator time between episodes

### Expected Metrics

- `cards_played > 0` (agent is making decisions, not always no-op)
- `episode_length` 200-600 (reasonable game durations)
- Increasing `episode_reward` trend over 15 games
- `anomaly_count` should be 0 or very low

---

## 5. Phase 2: Full Fine-Tuning

**Goal:** Unfreeze the feature extractor and fine-tune the entire network with a lower learning rate. Optionally use a KL penalty to prevent drifting too far from BC behavior.

### Command (without KL penalty)

```bash
python ppo_module/run_ppo.py \
    --resume models/ppo/latest_ppo.zip \
    --window-title "Clash Royale" \
    --num-episodes 25 \
    --lr 3e-5 \
    --n-frames 3 \
    --ent-start 0.01 \
    --ent-end 0.005 \
    --reward-scale 0.1 \
    --templates-dir ppo_module/templates
```

### Command (with KL penalty)

```bash
python ppo_module/run_ppo.py \
    --resume models/ppo/latest_ppo.zip \
    --window-title "Clash Royale" \
    --num-episodes 25 \
    --lr 3e-5 \
    --n-frames 3 \
    --ent-start 0.01 \
    --ent-end 0.005 \
    --reward-scale 0.1 \
    --bc-policy models/bc/best_bc.pt \
    --kl-coef 0.1 \
    --templates-dir ppo_module/templates
```

### Configuration Summary

| Setting | Value | Rationale |
|---------|-------|-----------|
| `--resume` | Phase 1 checkpoint | Continue from frozen training |
| `--lr 3e-5` | 3e-5 | 3x lower than Phase 1 to protect BC features |
| No `--freeze-extractor` | Unfrozen | Feature extractor now adapts to RL rewards |
| `--ent-start/end` | 0.01 -> 0.005 | Lower entropy since policy is already learned |
| `--bc-policy` (optional) | `best_bc.pt` | Full BC policy for KL reference |
| `--kl-coef` (optional) | 0.1 | KL(PPO \|\| BC) penalty coefficient |

### KL Penalty Explained

The `BCReferenceCallback` loads the full BC policy and at each step computes:

```
KL(PPO || BC) = sum(ppo_probs * (log_ppo_probs - log_bc_probs))
```

This penalty is subtracted from the reward, encouraging the PPO policy to stay near the BC distribution. Set `--kl-coef 0.0` (default) to disable.

### Expected Duration

- ~25 games x 3-5 minutes/game = 75-125 minutes
- Can continue beyond 25 games if metrics are still improving

---

## 6. Phase 3: Evaluation

### When to Stop Training

Stop training when:
- Win rate plateaus (e.g., no improvement over 10 consecutive episodes)
- `cards_played` stabilizes at a reasonable level (10-25 per game)
- `episode_reward` is consistently positive

### Key Metrics to Review

| Metric | Good Sign | Warning Sign |
|--------|-----------|--------------|
| `cr/win_rate` | Trending upward | Flat at 0% after 10+ games |
| `cr/cards_played` | 10-25 per game | 0-3 (passive) or 40+ (spam) |
| `cr/episode_reward` | Positive trend | Consistently < -30 (always losing) |
| `cr/ent_coef` | Decreasing 0.02 -> 0.005 | Stuck (callback not firing) |
| `cr/bc_kl` (if enabled) | < 5.0 | > 20 (policy diverged from BC) |
| `cr/anomaly_count` | 0 | Growing (game boundary issues) |

### TensorBoard

```bash
tensorboard --logdir logs/ppo/
# Open http://localhost:6006
```

### JSONL Log

Per-episode metrics in `logs/ppo/training_log.jsonl`:

```json
{"episode": 1, "outcome": "loss", "reward": -2.85, "cards_played": 12, "episode_length": 245, "win_rate": 0.0, "timestep": 700}
{"episode": 2, "outcome": "win", "reward": 4.52, "cards_played": 18, "episode_length": 312, "win_rate": 0.5, "timestep": 1400}
```

Note: rewards are scaled by `reward_scale=0.1`, so a win with survival looks like `+3.0` not `+30.0`.

---

## 7. Operator Workflow

### During Training

```
Start script
    |
    v
[Waiting for game start...]    <-- Operator clicks "Battle" in Clash Royale
    |
    v
[Agent plays autonomously]     <-- Do NOT touch mouse/keyboard
    |                               Agent captures screen at 2 FPS
    |                               Agent executes card placements via PyAutoGUI
    v
[Game ends - auto-detected]
    |
    v
[Checkpoint saved]
[Press Enter to continue...]   <-- Operator presses Enter, then clicks "Battle"
    |
    v
    ... (repeat for num_episodes)
```

### Operator Responsibilities

1. **Before training:** Open Clash Royale in Google Play Games. Navigate to the main menu.
2. **At each episode start:** Click "Battle" to queue a match. The script prints `[Env] Waiting for game start...` when ready.
3. **During gameplay:** Do not move the mouse, click, or switch windows.
4. **After game ends:** Press Enter when prompted (if `--no-pause` is not set), then click "Battle" again.
5. **To stop early:** Press `Ctrl+C`. The model will be saved before exiting.

### Timing

| Event | Duration |
|-------|----------|
| Game start timeout | 120 seconds (2 minutes to click Battle) |
| Typical game length | 3-5 minutes (~300-600 steps at 2 FPS) |
| Max episode length | 700 steps (~350 seconds) |
| Frame capture rate | 2 FPS (500ms per step) |
| Perception latency | ~200-300ms per frame |

### Unattended Mode

Use `--no-pause` to skip the Enter prompt between episodes. The agent will auto-continue, but the operator must still click "Battle" in the game.

---

## 8. Reward Function Reference

All rewards are multiplied by `reward_scale` (default 0.1) before being passed to the agent. The values below show **pre-scale** weights and **post-scale** effective values.

### Reward Components

| Component | Signal | Pre-Scale Weight | Post-Scale (x0.1) | When It Fires |
|-----------|--------|------------------|--------------------|---------------|
| Enemy crown | Enemy tower count drops | +10.0 | **+1.0** | We destroy an enemy tower |
| Ally crown lost | Ally tower count drops | -10.0 | **-1.0** | Enemy destroys our tower |
| Win | Game outcome = win | +30.0 | **+3.0** | Terminal step |
| Loss | Game outcome = loss | -30.0 | **-3.0** | Terminal step |
| Draw | Game outcome = draw | -5.0 | **-0.5** | Terminal step |
| Survival | Every step | +0.02 | **+0.002** | Every step |
| Elixir waste (full) | Elixir >= 9.5/10 | -0.10 | **-0.01** | Sitting at near-max elixir |
| Elixir waste (high) | Elixir >= 8.0/10 | -0.02 | **-0.002** | Mild penalty for high elixir |
| Unit advantage | Ally - enemy unit count | +0.01/unit | **+0.001/unit** | Per-step from arena grid |

### Graduated Elixir Penalty

The elixir penalty has two tiers to encourage spending without being too punitive:

- **8.0+ elixir** (normalized >= 0.8): Mild penalty of -0.02 per step. Signals "you should probably play a card."
- **9.5+ elixir** (normalized >= 0.95): Full penalty of -0.10 per step. At near-max, every moment wastes generated elixir.

### Unit Count Advantage

Computed from the arena observation grid each step:
- Count cells where `CH_ARENA_MASK > 0.5` and `CH_BELONGING < 0` (ally units)
- Count cells where `CH_ARENA_MASK > 0.5` and `CH_BELONGING > 0` (enemy units)
- Reward = `unit_advantage_weight * (ally_count - enemy_count)`

This provides a dense per-step signal that rewards building troop advantage.

### Reward Clamping

Non-terminal rewards are clamped to `[-15.0, +15.0]` per step **before** terminal rewards are added. This prevents extreme reward spikes from corrupting the value function while preserving the strong win/loss signal.

### New Game Anomaly Detection

Tower counts only decrease during a game (towers get destroyed, never rebuilt). If both ally and enemy tower counts jump up by more than 0.15, a new game started mid-episode. When detected:

- Crown deltas are skipped (prevents false rewards)
- Only survival bonus is returned for that step
- The environment terminates the episode

---

## 9. Hyperparameter Reference

### PPOConfig Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-4` | Base LR, cosine annealed to `1e-5` |
| `clip_range` | `0.1` | PPO clipping epsilon |
| `n_steps` | `700` | Timesteps per rollout buffer (~1 game) |
| `batch_size` | `64` | Mini-batch size |
| `n_epochs` | `10` | SGD epochs per rollout |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE lambda |
| `ent_coef` | `0.02` | Starting entropy coefficient |
| `ent_coef_start` | `0.02` | Entropy annealing start |
| `ent_coef_end` | `0.005` | Entropy annealing end |
| `vf_coef` | `0.5` | Value function loss coefficient |
| `max_grad_norm` | `0.5` | Gradient clipping |
| `features_dim` | `192` | Feature extractor output size |
| `n_frames` | `3` | Frame stack depth |
| `freeze_extractor` | `False` | Freeze BC feature extractor |
| `pi_layers` | `[256, 128]` | Policy head architecture |
| `vf_layers` | `[256, 128]` | Value head architecture |
| `bc_policy_path` | `""` | Path to BC policy for KL penalty |
| `kl_coef` | `0.1` | KL penalty coefficient (0 = disabled) |

### Cosine LR Schedule

The learning rate follows a cosine annealing schedule:

```
lr(t) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * (1 - progress)))
```

Where `min_lr = base_lr * 0.1`. For `base_lr=1e-4`, this anneals from `1e-4` down to `1e-5`.

### Entropy Annealing Schedule

Entropy coefficient is linearly annealed per episode via `EntropyScheduleCallback`:

```
ent_coef(ep) = start + (end - start) * min(ep / total_episodes, 1.0)
```

Default: `0.02 -> 0.005` over `num_episodes`. Higher entropy early encourages exploration; lower entropy later encourages exploitation.

### RewardConfig Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enemy_crown_reward` | `10.0` | Reward for destroying enemy tower |
| `ally_crown_penalty` | `-10.0` | Penalty for losing a tower |
| `win_reward` | `30.0` | Terminal win reward |
| `loss_penalty` | `-30.0` | Terminal loss penalty |
| `draw_penalty` | `-5.0` | Terminal draw penalty |
| `survival_bonus` | `0.02` | Per-step survival reward |
| `elixir_waste_penalty` | `-0.1` | Full penalty at >= 9.5 elixir |
| `elixir_waste_threshold` | `0.95` | Full elixir penalty threshold |
| `elixir_high_penalty` | `-0.02` | Mild penalty at >= 8.0 elixir |
| `elixir_high_threshold` | `0.8` | High elixir threshold |
| `unit_advantage_weight` | `0.01` | Per-unit advantage reward |
| `reward_clamp` | `15.0` | Non-terminal reward clamp |
| `reward_scale` | `0.1` | Global reward scaling factor |
| `tower_jump_threshold` | `0.15` | Anomaly detection threshold |

### EnvConfig Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capture_fps` | `2.0` | Frame capture rate |
| `frame_w` | `540` | Base frame width |
| `frame_h` | `960` | Base frame height |
| `use_perception` | `True` | Enable YOLO detection |
| `dry_run` | `False` | Disable mouse clicks |
| `max_episode_steps` | `700` | Max steps before truncation |
| `min_episode_steps` | `60` | Ignore END_SCREEN before this |
| `step_timeout` | `5.0` | Max seconds to wait for frame |
| `identical_frame_limit` | `5` | Truncate after N identical frames |
| `game_start_timeout` | `120.0` | Max seconds to wait for game start |
| `n_frames` | `3` | Frame stack depth |
| `pause_between_episodes` | `True` | Wait for Enter between episodes |

---

## 10. CLI Reference

```
python ppo_module/run_ppo.py [OPTIONS]
```

### Model Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--bc-weights` | path | `""` | Path to `bc_feature_extractor.pt` |
| `--resume` | path | `""` | Path to saved PPO model (`.zip`) to resume from |
| `--device` | choice | `cpu` | `cpu` or `cuda` |

### Capture Arguments (Mutually Exclusive)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--window-title` | string | `""` | Game window title for auto-detection |
| `--capture-region` | ints | `""` | Manual region: `LEFT,TOP,WIDTH,HEIGHT` |

### Training Hyperparameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-episodes` | int | `15` | Number of games to train on |
| `--lr` | float | `1e-4` | Learning rate (cosine annealed to lr*0.1) |
| `--clip-range` | float | `0.1` | PPO clip range (epsilon) |
| `--n-steps` | int | `700` | Timesteps per rollout buffer |
| `--n-epochs` | int | `10` | SGD epochs per rollout |
| `--batch-size` | int | `64` | Mini-batch size |
| `--ent-coef` | float | `0.02` | Starting entropy coefficient |
| `--ent-start` | float | `0.02` | Entropy annealing start |
| `--ent-end` | float | `0.005` | Entropy annealing end |
| `--n-frames` | int | `3` | Frame stack depth |
| `--reward-scale` | float | `0.1` | Reward scaling factor |
| `--freeze-extractor` | flag | off | Freeze BC feature extractor |

### KL Penalty Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--bc-policy` | path | `""` | Path to full BC policy for KL reference |
| `--kl-coef` | float | `0.0` | KL penalty coefficient (0 = disabled) |

### Safety Flags

| Flag | Description |
|------|-------------|
| `--dry-run` | Log actions but do not execute mouse clicks |
| `--no-perception` | Disable YOLO detection (zero-filled observations) |
| `--no-pause` | Skip Enter prompt between episodes |

### Output Paths

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | path | `models/ppo/` | Directory for model checkpoints |
| `--log-dir` | path | `logs/ppo/` | Directory for TensorBoard + JSONL logs |
| `--templates-dir` | path | `""` | Directory with `victory.png` / `defeat.png` |

### Visualization

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--visualize` | flag | off | Show live observation tensor heatmaps |
| `--vis-save-dir` | path | `""` | Save visualization frames to this directory |

---

## 11. Monitoring Training

### Console Output

```
============================================================
[PPOTrainer] Starting PPO training: 15 episodes
  BC weights: models/bc/bc_feature_extractor.pt
  Frozen extractor: True
  LR: 0.0001 (cosine annealing)
  Entropy: 0.02 -> 0.005
  Clip range: 0.1
  n_steps: 700
  n_epochs: 10
  n_frames: 3
  Network: pi=[256, 128], vf=[256, 128]
  Device: cpu
  Output: models/ppo/
============================================================

--- Episode 1/15 ---
[Episode 1] LOSS | reward=-2.8 | cards=12 | steps=245 | win_rate=0% (last 1)
[PPOTrainer] Saved checkpoint: models/ppo/ppo_ep1.zip

--- Episode 2/15 ---
[Episode 2] WIN | reward=+4.5 | cards=18 | steps=312 | win_rate=50% (last 2)
...
```

### TensorBoard Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `cr/win_rate` | Rolling win rate (last 10) | Increasing over time |
| `cr/episode_reward` | Total (scaled) reward per episode | Wins ~+3.0, losses ~-3.0 |
| `cr/avg_reward` | Rolling average reward | Trending upward |
| `cr/cards_played` | Card actions per episode | 10-25 |
| `cr/episode_length` | Steps per episode | 200-600 |
| `cr/ent_coef` | Current entropy coefficient | Decreasing 0.02 -> 0.005 |
| `cr/bc_kl` | KL divergence from BC (if enabled) | < 5.0 |
| `cr/anomaly_count` | Cumulative mid-episode anomalies | 0 |
| `cr/truncation_count` | Cumulative truncations | Low |

---

## 12. Episode Boundary Safety

Four layers prevent multiple games from bleeding into one episode:

### Layer 1: Max Episode Length

Episodes truncate after **700 steps** (`max_episode_steps`). A standard game is ~300s = ~600 steps at 2 FPS, with margin for overtime.

### Layer 2: Observation Anomaly Detection

Tower counts only decrease during a game. If **both** ally and enemy tower counts jump up by > 0.15, a new game started mid-episode. The episode is terminated with only a survival reward for that step.

### Layer 3: Phase Transition Detection

`GamePhaseDetector` uses debounced phase detection (3 stable frames). If IN_GAME transitions to LOADING, it returns END_SCREEN, terminating the episode. The END_SCREEN phase latches until `reset()`.

### Layer 4: Reward Clamping

Non-terminal rewards are clamped to `[-15.0, +15.0]` per step. Even if anomaly detection fails, per-step reward impact is bounded.

---

## 13. Troubleshooting

### Game Window Not Found

```
Error: Could not find window "Clash Royale"
```

Use `--capture-region` as a fallback:
```bash
python ppo_module/run_ppo.py ... --capture-region LEFT,TOP,WIDTH,HEIGHT
```

### Agent Plays Too Passively (0 Cards Played)

- Increase entropy: `--ent-start 0.05`
- Check that BC weights loaded: look for `[PPOTrainer] Loaded BC weights` in console
- Verify elixir penalty is working (TensorBoard rewards should show small negative signals at high elixir)

### Agent Spams Cards Randomly

- Ensure `--freeze-extractor` is used for Phase 1
- Lower entropy: `--ent-start 0.01`
- Check BC weights are from n_frames=3 training (architecture mismatch causes garbage features)

### Every Episode Truncated at 700 Steps

Game end detection is failing. Fixes:
- Provide templates: `--templates-dir ppo_module/templates`
- Ensure game window is fully visible and not obscured
- Verify with `--dry-run` first

### High Anomaly Count

Game phase detector is missing end screens. Fixes:
- Provide victory/defeat templates
- Ensure capture region matches the actual game area
- Check game window position hasn't shifted

### CUDA Out of Memory

```bash
python ppo_module/run_ppo.py ... --device cpu --batch-size 32
```

---

## 14. Output Files

### Model Checkpoints

```
models/ppo/
    latest_ppo.zip     # Most recent (overwritten each episode)
    ppo_ep1.zip        # Episode 1 snapshot
    ppo_ep2.zip        # Episode 2 snapshot
    ...
    final_ppo.zip      # Final model after all episodes
```

Each `.zip` contains the full MaskablePPO model (policy weights, optimizer state, training metadata).

### Logs

```
logs/ppo/
    training_log.jsonl            # Per-episode metrics (append-only)
    MaskablePPO_1/                # TensorBoard event files
        events.out.tfevents.*
```

### Loading a Trained Model

```python
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("models/ppo/final_ppo.zip")

obs, info = env.reset()
while True:
    action_masks = env.action_masks()
    action, _ = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Full Training Recipe

```bash
# Step 0: Dry run to verify setup
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --window-title "Clash Royale" \
    --num-episodes 1 \
    --n-frames 3 \
    --dry-run

# Step 1: Phase 1 - frozen extractor (15 games)
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --window-title "Clash Royale" \
    --num-episodes 15 \
    --freeze-extractor \
    --n-frames 3 \
    --lr 1e-4 \
    --n-steps 700 \
    --ent-start 0.02 \
    --ent-end 0.005 \
    --reward-scale 0.1 \
    --templates-dir ppo_module/templates

# Step 2: Phase 2 - full fine-tuning (25 games)
python ppo_module/run_ppo.py \
    --resume models/ppo/latest_ppo.zip \
    --window-title "Clash Royale" \
    --num-episodes 25 \
    --lr 3e-5 \
    --n-frames 3 \
    --ent-start 0.01 \
    --ent-end 0.005 \
    --reward-scale 0.1 \
    --templates-dir ppo_module/templates

# Step 3: Monitor with TensorBoard
tensorboard --logdir logs/ppo/
```
