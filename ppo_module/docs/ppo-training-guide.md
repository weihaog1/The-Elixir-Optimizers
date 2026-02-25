# Live PPO Training Guide

Step-by-step guide for running PPO fine-tuning on a live Clash Royale game. The agent plays autonomously during matches while a human operator queues games between episodes.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start](#2-quick-start)
3. [Training Phases](#3-training-phases)
4. [CLI Reference](#4-cli-reference)
5. [Operator Workflow](#5-operator-workflow)
6. [Monitoring Training](#6-monitoring-training)
7. [Reward System](#7-reward-system)
8. [Episode Boundary Safety](#8-episode-boundary-safety)
9. [Configuration Reference](#9-configuration-reference)
10. [Troubleshooting](#10-troubleshooting)
11. [Output Files](#11-output-files)

---

## 1. Prerequisites

### Required Files

| File | Description | How to Obtain |
|------|-------------|---------------|
| `bc_feature_extractor.pt` | BC-pretrained feature extractor weights | Output of BC training (`BCTrainer`) |
| `best_yolov8s_50epochs_fixed_pregen_set.pt` | YOLOv8 unit detection model | Provided in `models/` |
| `card_classifier.pt` | Card hand classifier | Provided in `models/` |

### Optional Files

| File | Description |
|------|-------------|
| `templates/victory.png` | Screenshot of the "Victory" banner for outcome detection |
| `templates/defeat.png` | Screenshot of the "Defeat" banner for outcome detection |

If templates are not provided, outcome detection falls back to color heuristics (golden tones = win, blue/dark tones = loss).

### Software

- **Python 3.10+** with packages: `torch`, `numpy`, `gymnasium`, `opencv-python`, `stable-baselines3`, `sb3-contrib`, `tensorboard`, `mss`, `pyautogui`
- **Google Play Games** emulator with Clash Royale installed and running
- The game window must be **visible on screen** (not minimized)

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Not required | CUDA-capable GPU |
| RAM | 8 GB | 16 GB |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |

The agent needs mouse control via PyAutoGUI. Do not move the mouse or interact with the computer during active gameplay steps.

---

## 2. Quick Start

### Dry Run (Test Without Clicking)

Verify the pipeline works before live training. No mouse clicks are executed.

```bash
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --capture-region 0,0,540,960 \
    --num-episodes 1 \
    --dry-run
```

### Phase 1: Train Policy Heads Only

Freeze the BC feature extractor. Only the policy and value heads learn.

```bash
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --window-title "Clash Royale" \
    --num-episodes 15 \
    --freeze-extractor
```

### Phase 2: Fine-Tune Everything

Resume from Phase 1 checkpoint. Unfreeze the feature extractor with a lower learning rate.

```bash
python ppo_module/run_ppo.py \
    --resume models/ppo/latest_ppo.zip \
    --window-title "Clash Royale" \
    --num-episodes 25 \
    --lr 3e-5
```

---

## 3. Training Phases

PPO training uses a two-phase curriculum to avoid catastrophic forgetting of BC-learned features.

### Phase 1: Frozen Feature Extractor

| Setting | Value |
|---------|-------|
| Flag | `--freeze-extractor` |
| Learning Rate | `1e-4` (default) |
| Episodes | ~15 games |
| Duration | ~45-75 minutes |
| What trains | Policy head (pi) + Value head (vf) only |
| What's frozen | CRFeatureExtractor (arena CNN + vector MLP) |

**Purpose:** Let the new PPO policy and value heads learn to use the BC feature representations without corrupting them. The 192-dim feature extractor output stays identical to what BC learned.

### Phase 2: Full Fine-Tuning

| Setting | Value |
|---------|-------|
| Flag | Omit `--freeze-extractor` |
| Learning Rate | `3e-5` (set via `--lr 3e-5`) |
| Episodes | ~25+ games |
| Duration | ~75-125 minutes |
| What trains | Everything (extractor + heads) |

**Purpose:** Allow the feature extractor to adapt its representations to the RL reward signal. The lower learning rate prevents large weight updates that would destroy BC knowledge.

### Resuming Training

Any checkpoint can be resumed:

```bash
# Resume from a specific episode checkpoint
python ppo_module/run_ppo.py \
    --resume models/ppo/ppo_ep10.zip \
    --window-title "Clash Royale" \
    --num-episodes 10
```

The `--resume` flag loads the full model (weights + optimizer state). You do **not** need `--bc-weights` when resuming.

---

## 4. CLI Reference

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

Use `--window-title "Clash Royale"` for auto-detection. Use `--capture-region 0,0,540,960` for manual coordinates.

### Training Hyperparameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-episodes` | int | `15` | Number of games to train on |
| `--lr` | float | `1e-4` | Learning rate |
| `--clip-range` | float | `0.1` | PPO clip range (epsilon) |
| `--n-steps` | int | `512` | Timesteps per rollout buffer |
| `--n-epochs` | int | `10` | SGD epochs per rollout |
| `--batch-size` | int | `64` | Mini-batch size |
| `--ent-coef` | float | `0.01` | Entropy coefficient |
| `--freeze-extractor` | flag | off | Freeze BC feature extractor |

### Safety Flags

| Flag | Description |
|------|-------------|
| `--dry-run` | Log actions but do not execute mouse clicks |
| `--no-perception` | Disable YOLO detection (zero-filled observations) |

### Output Paths

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | path | `models/ppo/` | Directory for model checkpoints |
| `--log-dir` | path | `logs/ppo/` | Directory for TensorBoard + JSONL logs |
| `--templates-dir` | path | `""` | Directory with `victory.png` / `defeat.png` |

---

## 5. Operator Workflow

PPO training is **semi-automated**. The agent plays each match autonomously, but a human operator must queue new games between episodes.

### During Training

```
Start script
    |
    v
[Waiting for game start...]  <-- Operator clicks "Battle" in Clash Royale
    |
    v
[Agent plays autonomously]   <-- Do NOT touch mouse/keyboard
    |                             Agent captures screen at 2 FPS
    |                             Agent executes card placements via PyAutoGUI
    |                             Each step: capture -> perceive -> decide -> act
    v
[Game ends - detected automatically]
    |
    v
[Checkpoint saved]
    |
    v
[Waiting for game start...]  <-- Operator clicks "Battle" again
    |
    ... (repeat for num_episodes)
    |
    v
[Training complete - final model saved]
```

### Operator Responsibilities

1. **Before training:** Open Clash Royale in Google Play Games. Navigate to the main menu.
2. **At each episode start:** Click "Battle" to queue a match. The script prints `[Env] Waiting for game start...` when ready.
3. **During gameplay:** Do not move the mouse, click, or switch windows. The agent controls the mouse via PyAutoGUI.
4. **After game ends:** The script automatically detects the end screen, saves a checkpoint, and prompts for the next match.
5. **If a game freezes:** The env will auto-truncate after 5 identical frames or 700 steps (max episode length).

### Timing

| Event | Duration |
|-------|----------|
| Game start timeout | 120 seconds (2 minutes to click Battle) |
| Typical game length | 3-5 minutes (~300-600 steps at 2 FPS) |
| Max episode length | 700 steps (~350 seconds) |
| Frame capture rate | 2 FPS (500ms per step) |
| Perception latency | ~200-300ms per frame |

---

## 6. Monitoring Training

### TensorBoard

Launch TensorBoard to visualize training metrics in real-time:

```bash
tensorboard --logdir logs/ppo/
```

Then open `http://localhost:6006` in a browser.

**Key metrics to watch:**

| Metric | What It Means | Healthy Range |
|--------|---------------|---------------|
| `cr/win_rate` | Rolling win rate (last 10 episodes) | Increasing over time |
| `cr/episode_reward` | Total reward per episode | Increasing; wins ~+30, losses ~-30 |
| `cr/avg_reward` | Rolling average reward | Trending upward |
| `cr/cards_played` | Card actions per episode | 10-25 (too low = passive, too high = wasteful) |
| `cr/episode_length` | Steps per episode | 200-600 (normal game length) |
| `cr/anomaly_count` | Cumulative mid-episode anomalies | Should be 0 or very low |
| `cr/truncation_count` | Cumulative truncations | Should be low |

### JSONL Log

Episode metrics are also written to `logs/ppo/training_log.jsonl`:

```json
{"episode": 1, "outcome": "loss", "reward": -28.5, "cards_played": 12, "episode_length": 245, "win_rate": 0.0, "avg_reward": -28.5, "timestep": 512, "anomaly_detected": false, "truncation_reason": ""}
{"episode": 2, "outcome": "win", "reward": 45.2, "cards_played": 18, "episode_length": 312, "win_rate": 0.5, "avg_reward": 8.35, "timestep": 1024, "anomaly_detected": false, "truncation_reason": ""}
```

### Console Output

The script prints a summary after each episode:

```
[Episode 1] LOSS | reward=-28.5 | cards=12 | steps=245 | win_rate=0% (last 1)
[Episode 2] WIN | reward=+45.2 | cards=18 | steps=312 | win_rate=50% (last 2)
[Episode 3] WIN | reward=+38.7 | cards=20 | steps=298 | win_rate=67% (last 3)
```

And a final summary:

```
==================================================
Training Summary: 15 episodes
  Win rate (last 10): 7/10 = 70%
  Avg reward: 12.3
  Avg cards/episode: 16.2
==================================================
```

---

## 7. Reward System

The reward function uses observation vector deltas -- no OCR required. All signals come from the 23-element observation vector.

### Reward Components

| Component | Signal | Default Weight | When It Fires |
|-----------|--------|----------------|---------------|
| Enemy crown | Enemy tower count drops | `+10.0` | We destroy an enemy tower |
| Ally crown lost | Ally tower count drops | `-10.0` | Enemy destroys our tower |
| Win | Game outcome = win | `+30.0` | Terminal step (game over) |
| Loss | Game outcome = loss | `-30.0` | Terminal step (game over) |
| Draw | Game outcome = draw | `-5.0` | Terminal step (game over) |
| Survival | Every step | `+0.02` | Every step (encourages longer play) |
| Elixir waste | Elixir >= 9.5/10 | `-0.05/step` | Sitting at max elixir |

### Reward Clamping

Non-terminal rewards are clamped to `[-15.0, +15.0]` per step. Terminal rewards (win/loss/draw) are added **after** clamping, preserving the strongest learning signal.

**Examples:**
- Normal step: `+0.02` (survival only)
- Score a crown: `+10.02` (crown + survival, within clamp)
- Lose a crown: `-9.98` (penalty + survival, within clamp)
- Score crown + win: `min(15, 10.02) + 30 = 40.02`
- Normal step + win: `0.02 + 30 = 30.02`
- Normal step + loss: `0.02 - 30 = -29.98`

### Customizing Rewards

Override defaults via `RewardConfig` in code, or adjust via the defaults in `reward.py`:

```python
from src.ppo.reward import RewardConfig

custom_rewards = RewardConfig(
    enemy_crown_reward=15.0,   # Increase crown incentive
    survival_bonus=0.01,       # Reduce survival bonus
    elixir_waste_penalty=-0.1, # Stronger elixir waste penalty
    reward_clamp=20.0,         # Wider clamp range
)
```

---

## 8. Episode Boundary Safety

Four layers of defense prevent multiple games from bleeding into one episode:

### Layer 1: Max Episode Length

Episodes truncate after **700 steps** (configurable via `EnvConfig.max_episode_steps`). A standard Clash Royale game is ~300 seconds = ~600 steps at 2 FPS. The 700-step limit provides margin for overtime.

### Layer 2: Observation Anomaly Detection

Tower counts only decrease during a game (towers get destroyed, never rebuilt). If **both** ally and enemy tower counts jump up by more than `0.15`, a new game has started mid-episode.

When detected:
- `RewardComputer.new_game_detected` is set to `True`
- Crown deltas are skipped (prevents false +/-10 rewards)
- The environment terminates the episode

### Layer 3: Phase Transition Detection

`GamePhaseDetector` uses debounced phase detection with a stability threshold of 3 consecutive frames. If the detector was ever `IN_GAME` and transitions to `LOADING` (3 stable frames), it returns `END_SCREEN` -- interpreting the transition as a missed end screen.

The `END_SCREEN` phase latches until `reset()` is called.

### Layer 4: Reward Clamping

Non-terminal rewards are clamped to `[-15.0, +15.0]`. Even if anomaly detection misses a game boundary and a false crown event fires, the per-step reward impact is bounded.

---

## 9. Configuration Reference

### PPOConfig (Training Orchestrator)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-4` | SGD learning rate |
| `clip_range` | `0.1` | PPO clipping epsilon |
| `n_steps` | `512` | Timesteps per rollout buffer |
| `batch_size` | `64` | Mini-batch size |
| `n_epochs` | `10` | Epochs per rollout |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE lambda |
| `ent_coef` | `0.01` | Entropy coefficient |
| `vf_coef` | `0.5` | Value function loss coefficient |
| `max_grad_norm` | `0.5` | Gradient clipping |
| `features_dim` | `192` | Feature extractor output size |
| `freeze_extractor` | `False` | Freeze BC feature extractor |
| `pi_layers` | `[128, 64]` | Policy head architecture |
| `vf_layers` | `[128, 64]` | Value head architecture |

### EnvConfig (Environment)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capture_fps` | `2.0` | Frame capture rate |
| `frame_w` | `540` | Base frame width |
| `frame_h` | `960` | Base frame height |
| `use_perception` | `True` | Enable YOLO detection |
| `dry_run` | `False` | Disable mouse clicks |
| `max_episode_steps` | `700` | Max steps before truncation |
| `step_timeout` | `5.0` | Max seconds to wait for frame |
| `identical_frame_limit` | `5` | Truncate after N identical frames |
| `game_start_timeout` | `120.0` | Max seconds to wait for game start |

### RewardConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enemy_crown_reward` | `10.0` | Reward for destroying enemy tower |
| `ally_crown_penalty` | `-10.0` | Penalty for losing a tower |
| `win_reward` | `30.0` | Terminal win reward |
| `loss_penalty` | `-30.0` | Terminal loss penalty |
| `draw_penalty` | `-5.0` | Terminal draw penalty |
| `survival_bonus` | `0.02` | Per-step survival reward |
| `elixir_waste_penalty` | `-0.05` | Penalty per step at max elixir |
| `elixir_waste_threshold` | `0.95` | Elixir threshold (normalized) |
| `reward_clamp` | `15.0` | Non-terminal reward clamp |
| `tower_jump_threshold` | `0.15` | Anomaly detection threshold |

### DetectorConfig (Game Phase Detection)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `card_bar_intensity_threshold` | `40.0` | Card bar pixel intensity threshold |
| `arena_variance_threshold` | `200.0` | Arena pixel variance threshold |
| `end_screen_arena_intensity_max` | `80.0` | End screen detection threshold |
| `template_match_threshold` | `0.7` | Template match confidence threshold |
| `phase_stability_frames` | `3` | Consecutive frames to confirm phase |

---

## 10. Troubleshooting

### Game Window Not Found

```
Error: Could not find window "Clash Royale"
```

**Fix:** Check the exact window title. Open Task Manager to verify. Use `--capture-region` as a fallback:

```bash
# Find coordinates manually, then:
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --capture-region LEFT,TOP,WIDTH,HEIGHT \
    --num-episodes 15 --freeze-extractor
```

### Game Start Timeout

```
[Env] WARNING: Game start timeout. Proceeding anyway.
```

**Fix:** Click "Battle" in Clash Royale within 120 seconds of seeing `[Env] Waiting for game start...`. If you need more time, the env will proceed with the current frame state.

### Too Many Anomalies

If `cr/anomaly_count` is high in TensorBoard, the game phase detector is missing end screens frequently.

**Fix:**
- Provide victory/defeat templates via `--templates-dir`
- Ensure the game window is not obscured or partially off-screen
- Check that the capture region matches the actual game area

### Episode Always Truncated at 700 Steps

If every episode hits the max step limit, the game end is never being detected.

**Fix:**
- Verify game phase detection with `--dry-run` first
- Lower `phase_stability_frames` in `DetectorConfig` (default 3)
- Provide templates for more reliable outcome detection

### Agent Plays Too Passively (Low Cards Played)

**Fix:**
- Increase `ent_coef` to encourage exploration: `--ent-coef 0.05`
- Increase `elixir_waste_penalty` in `RewardConfig` to punish hoarding

### Agent Spams Cards Randomly

**Fix:**
- Decrease `ent_coef`: `--ent-coef 0.001`
- Ensure `--freeze-extractor` is used for Phase 1
- Check that BC weights loaded successfully (look for `[PPOTrainer] Loaded BC weights` in console)

### CUDA Out of Memory

**Fix:** Use CPU (default) or reduce `batch_size`:

```bash
python ppo_module/run_ppo.py ... --device cpu --batch-size 32
```

---

## 11. Output Files

### Model Checkpoints

```
models/ppo/
    latest_ppo.zip     # Most recent checkpoint (overwritten each episode)
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

### Loading a Trained Model for Inference

```python
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("models/ppo/final_ppo.zip")

# Use in environment
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
# Step 1: Dry run to verify setup
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --window-title "Clash Royale" \
    --num-episodes 1 \
    --dry-run

# Step 2: Phase 1 - frozen extractor (15 games, ~1 hour)
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --window-title "Clash Royale" \
    --num-episodes 15 \
    --freeze-extractor \
    --templates-dir ppo_module/templates

# Step 3: Phase 2 - full fine-tuning (25 games, ~2 hours)
python ppo_module/run_ppo.py \
    --resume models/ppo/latest_ppo.zip \
    --window-title "Clash Royale" \
    --num-episodes 25 \
    --lr 3e-5 \
    --templates-dir ppo_module/templates

# Step 4: Monitor with TensorBoard
tensorboard --logdir logs/ppo/
```
