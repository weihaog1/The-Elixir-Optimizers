# PPO Fine-Tuning Implementation Plan

## Context

The BC model (F1=0.324, Recall=69.5%, Precision=21.2%) trained on 28 games catches most action moments but has low precision — it "plays cards" when it shouldn't. PPO fine-tuning will improve decision quality by learning from actual game outcomes (wins, tower destruction, survival) rather than just imitating recordings. The BC-pretrained feature extractor provides a strong initialization so the RL agent starts with game-state understanding rather than from scratch.

**Constraints**: No simulator — the agent plays real Clash Royale via screen capture. Semi-automated: human queues matches, agent plays autonomously. Initial budget: 1-3 hours (~20-40 games), scaling to 5-10 hours later.

---

## New Module Structure

```
ppo_module/
  CLAUDE.md
  run_ppo.py                       # CLI entry point
  src/
    ppo/
      __init__.py
      CLAUDE.md
      clash_royale_env.py          # Gymnasium wrapper
      reward.py                    # RewardComputer
      game_detector.py             # GamePhaseDetector (start/end detection)
      sb3_feature_extractor.py     # SB3-compatible CRFeatureExtractor wrapper
      ppo_trainer.py               # Training orchestrator
      callbacks.py                 # Custom SB3 callbacks for metrics
  tests/
    conftest.py                    # Namespace setup (same pattern as other modules)
    test_reward.py
    test_env.py
```

---

## Step 1: SB3 Feature Extractor Wrapper

**File**: `ppo_module/src/ppo/sb3_feature_extractor.py`

Wrap `CRFeatureExtractor` from `bc_model_module/src/bc/feature_extractor.py` to subclass SB3's `BaseFeaturesExtractor`:

```python
class SB3CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=192):
        super().__init__(observation_space, features_dim=features_dim)
        self._extractor = CRFeatureExtractor(features_dim=features_dim)

    def forward(self, observations):
        return self._extractor(observations)
```

Load BC weights via: `self._extractor.load_state_dict(torch.load("models/bc/bc_feature_extractor.pt"))`

**Key files to reuse**:
- `bc_model_module/src/bc/feature_extractor.py` — CRFeatureExtractor (192-dim output)
- `models/bc/bc_feature_extractor.pt` — pretrained weights

---

## Step 2: Reward Function

**File**: `ppo_module/src/ppo/reward.py`

Reward based on three signals available from observation vector (no OCR needed):

### Reward Components

| Component | Source | Signal | Weight |
|-----------|--------|--------|--------|
| **Enemy crown** | `vector[10]` (enemy tower count /3) | Crown scored when count drops | +10.0 |
| **Ally crown lost** | `vector[9]` (ally tower count /3) | Crown lost when count drops | -10.0 |
| **Win terminal** | GamePhaseDetector outcome | Game won | +30.0 |
| **Loss terminal** | GamePhaseDetector outcome | Game lost | -30.0 |
| **Draw terminal** | GamePhaseDetector outcome | Game drawn | -5.0 |
| **Survival bonus** | Per step | Alive each frame | +0.02 |
| **Elixir waste** | `vector[0]` (elixir /10) | Sitting at max elixir | -0.05/step |

### RewardComputer class

```python
@dataclass
class RewardConfig:
    enemy_crown_reward: float = 10.0
    ally_crown_penalty: float = -10.0
    win_reward: float = 30.0
    loss_penalty: float = -30.0
    draw_penalty: float = -5.0
    survival_bonus: float = 0.02
    elixir_waste_penalty: float = -0.05
    elixir_waste_threshold: float = 0.95  # 9.5/10 elixir

class RewardComputer:
    def reset(self): ...  # Reset prev_tower_counts
    def compute(self, prev_obs, curr_obs, terminal_outcome=None) -> float: ...
```

Tracks `prev_enemy_tower_count` and `prev_ally_tower_count` between steps. Crown events fire when tower count decreases. Elixir waste fires when `vector[0] >= 0.95`. Terminal reward applied on last step.

**Reward magnitude analysis**: A 3-min game = ~360 steps. Survival bonus total = ~7.2. A single crown = 10.0. A win = 30.0. This means scoring a crown is worth ~140 steps of survival, and winning is worth the whole game — correctly prioritizing outcomes over just existing.

---

## Step 3: Game Phase Detector

**File**: `ppo_module/src/ppo/game_detector.py`

Detects game start, in-progress, and end states from raw screen frames.

### Detection Strategy

| Phase | Detection Method |
|-------|-----------------|
| **Game start** | Card bar appears at bottom of screen (y=770-920 at 540x960). Check for card-like color patterns in that region. Timer appears at top. |
| **In-game** | Card bar present AND arena has varied pixel content (not a loading/menu screen). |
| **Game end** | Card bar disappears AND results overlay appears. Detect via pixel uniformity in arena region or template match for "Victory"/"Defeat" banners. |
| **Win/Loss** | After end detected: check crown region for yellow star pixels. Compare player crowns vs enemy crowns. Alternative: check if "Victory" or "Defeat" text banner is present via template matching. |

### Implementation

```python
class GamePhaseDetector:
    class Phase(Enum):
        UNKNOWN = "unknown"
        LOADING = "loading"
        IN_GAME = "in_game"
        END_SCREEN = "end_screen"

    def detect_phase(self, frame: np.ndarray) -> Phase
    def detect_outcome(self, frame: np.ndarray) -> Optional[str]  # "win"/"loss"/"draw"
    def wait_for_game_start(self, capture, timeout=120.0) -> bool
    def wait_for_game_end(self, capture, timeout=360.0) -> Optional[str]
```

Pre-capture template images for "Victory" and "Defeat" banners during calibration. Store in `ppo_module/templates/`.

---

## Step 4: Gymnasium Environment Wrapper

**File**: `ppo_module/src/ppo/clash_royale_env.py`

Wraps the live game as a standard Gymnasium environment.

### Reuses from `bc_model_module/src/bc/live_inference.py`:
- `GameCapture` (mss screen capture with rate limiting)
- `PerceptionAdapter` (YOLO + CardPredictor -> observation tensors)
- `ActionDispatcher` (action index -> PyAutoGUI clicks)

### ClashRoyaleEnv Interface

```python
class ClashRoyaleEnv(gymnasium.Env):
    observation_space = Dict(arena=Box(32,18,6), vector=Box(23,))
    action_space = Discrete(2305)

    def __init__(self, config: EnvConfig): ...
    def reset(self, seed=None, options=None) -> (obs, info): ...
    def step(self, action) -> (obs, reward, terminated, truncated, info): ...
    def action_masks(self) -> np.ndarray: ...  # For MaskablePPO
    def close(self): ...
```

### step() flow:
1. Execute action via `ActionDispatcher` (noop = skip click)
2. Wait for next frame from `GameCapture` (blocks ~500ms at 2 FPS)
3. Run `PerceptionAdapter.process_frame()` -> new observation + action mask
4. Check game end via `GamePhaseDetector.detect_phase(raw_frame)`
5. Compute reward via `RewardComputer.compute(prev_obs, curr_obs, outcome)`
6. Return `(obs, reward, terminated, truncated, info)`

### reset() flow:
1. Print "Waiting for game start... Press Battle in Clash Royale"
2. Call `GamePhaseDetector.wait_for_game_start()` -- polls at 1 FPS
3. Once game detected, capture first frame, run perception
4. Reset `RewardComputer` state
5. Return initial `(obs, info)`

### Edge cases:
- **Step timeout**: If no valid frame in 5s, return prev obs with 0 reward, set `truncated=True`
- **Window unfocused**: Skip action execution, return prev obs, log warning
- **Identical frames** (5+ in a row): Assume freeze, set `truncated=True`

---

## Step 5: PPO Training Orchestrator

**File**: `ppo_module/src/ppo/ppo_trainer.py`

### MaskablePPO Configuration

```python
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class": SB3CRFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        "activation_fn": torch.nn.ReLU,
    },
    learning_rate=1e-4,
    clip_range=0.1,         # Conservative: limited data
    n_steps=512,            # ~1.4 episodes per update
    batch_size=64,
    n_epochs=10,            # Maximize reuse of scarce data
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="logs/ppo/",
)
```

**Why these values**:
- `n_steps=512` slightly larger than one episode (~360 steps) to capture full games
- `n_epochs=10` squeezes maximum learning from each batch (safe with `clip_range=0.1`)
- `clip_range=0.1` prevents large policy swings with limited data
- `net_arch=dict(pi=[128,64], vf=[128,64])` separate policy/value MLPs on shared 192-dim features

### Training Phases

**Phase 1: Frozen feature extractor (first ~15 games)**
- Freeze BC feature extractor weights (`requires_grad = False`)
- Only train the policy head (pi: 192->128->64->2305) and value head (vf: 192->128->64->1)
- Purpose: calibrate value function and policy head without destroying BC representations
- LR: `1e-4`

**Phase 2: Full fine-tuning (remaining games)**
- Unfreeze feature extractor (`requires_grad = True`)
- Reduce LR to `3e-5` for the feature extractor (use param groups)
- Purpose: adapt the feature extractor to reward-based signal
- Monitor for performance collapse; if win rate drops, revert to phase 1 checkpoint

### Semi-Automated Episode Loop

```python
for episode in range(num_episodes):
    obs, info = env.reset()          # Blocks until game starts
    done = False
    while not done:
        action, _ = model.predict(obs, action_masks=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    log_episode(info)
    model.save("models/ppo/latest_ppo.zip")
    print(f"Episode {episode} done. Result: {info.get('outcome')}. Queue next match.")
    # Operator clicks "Battle" in Clash Royale, then the next reset() picks it up
```

### Checkpointing
- Save `models/ppo/latest_ppo.zip` after every episode
- Save `models/ppo/best_ppo.zip` when rolling win rate is best
- Append episode metrics to `logs/ppo/training_log.jsonl`

---

## Step 6: Custom Callbacks

**File**: `ppo_module/src/ppo/callbacks.py`

Track per-episode metrics via SB3 callback:

```python
class CRMetricsCallback(BaseCallback):
    # Tracks: win_rate (last 10), avg_crowns, avg_episode_reward,
    #         avg_cards_played, avg_episode_length
    # Logs to TensorBoard + prints summary after each episode
```

---

## Step 7: CLI Entry Point

**File**: `ppo_module/run_ppo.py`

Follows the same namespace registration pattern as `bc_model_module/run_live.py` (lines 34-98) to set up `src.bc`, `src.encoder`, `src.generation`, etc.

```bash
# Phase 1: frozen extractor
python ppo_module/run_ppo.py \
  --bc-weights models/bc/bc_feature_extractor.pt \
  --capture-region 0,0,540,960 \
  --num-episodes 15 \
  --freeze-extractor \
  --output-dir models/ppo/

# Phase 2: full fine-tuning (resume from phase 1)
python ppo_module/run_ppo.py \
  --resume models/ppo/latest_ppo.zip \
  --capture-region 0,0,540,960 \
  --num-episodes 25 \
  --lr 3e-5 \
  --output-dir models/ppo/
```

---

## Step 8: Dependencies

```bash
pip install stable-baselines3 sb3-contrib tensorboard
```

All other dependencies (torch, gymnasium, numpy, mss, cv2, pyautogui) already installed.

---

## Architecture Decision: Flat vs Hierarchical Action Space

**Decision: Use flat Discrete(2305) with MaskablePPO.**

The BC hierarchical decomposition (play->card->position) was designed to solve the class imbalance problem in supervised learning (85% noop in training data). PPO doesn't have this problem -- it collects its own data and the reward function directly shapes behavior. MaskablePPO handles the 2305-way categorical natively with action masking to prevent invalid actions.

Only the feature extractor transfers. The BC play/card/position heads are discarded. MaskablePPO creates:
- Policy net: 192 -> 128 -> 64 -> 2305
- Value net: 192 -> 128 -> 64 -> 1

---

## Implementation Order

| # | Task | Depends On | Est. Effort |
|---|------|------------|-------------|
| 1 | `sb3_feature_extractor.py` -- SB3 wrapper + BC weight loading | -- | Small |
| 2 | `reward.py` -- RewardComputer with tower count + win/loss + survival | -- | Small |
| 3 | `game_detector.py` -- Phase detection from raw frames | Template images | Medium |
| 4 | `clash_royale_env.py` -- Gymnasium wrapper composing GameCapture + Perception + Reward + Detector | Steps 1-3 | Large |
| 5 | `callbacks.py` -- Metrics logging callback | -- | Small |
| 6 | `ppo_trainer.py` -- Training orchestrator with freeze/unfreeze | Steps 1-5 | Medium |
| 7 | `run_ppo.py` -- CLI with namespace setup | Step 6 | Small |
| 8 | Tests for reward and env | Steps 2, 4 | Medium |
| 9 | Live calibration -- template images, threshold tuning | Step 3 | Manual |
| 10 | Training run -- Phase 1 (frozen) then Phase 2 (unfrozen) | All above | Wall-clock |

---

## Verification Plan

### Unit Tests
- `test_reward.py`: Verify reward computation for tower count changes, win/loss, elixir waste
- `test_env.py`: Mock-based test of env step/reset with synthetic observations

### Integration Test (dry run)
1. Run `ClashRoyaleEnv` with `dry_run=True` (no actual clicks)
2. Verify observation shapes, action masking, reward computation
3. Verify `MaskablePPO` can `predict()` and `learn()` against the env

### Live Validation
1. Run Phase 1 training for 15 games (~45 min)
2. Compare win rate against BC baseline (run BC for 10 games first)
3. Check TensorBoard for value loss convergence, entropy trends, reward curves
4. If value loss not decreasing -> reward function issue
5. If entropy collapses to 0 -> agent collapsed to single action, increase `ent_coef`
6. If win rate drops vs BC -> feature extractor being damaged, keep it frozen longer

---

## Assumptions & Limitations

1. **No simulator**: Each training step takes ~500ms (real-time). 20 games = 1 hour. Standard PPO wants millions of steps; we'll have ~7,000-20,000. Mitigation: BC pretrain + high n_epochs + conservative clip.
2. **Sparse reward**: Crown events happen 1-5 times per game. Survival bonus provides mild dense signal. The value function must learn to propagate sparse crown/win rewards backward through ~200+ steps.
3. **Perception noise**: YOLO detection is imperfect (mAP50=0.804). OCR is not used. Tower counts come from perception and may have occasional errors. Crown detection must be robust.
4. **No opponent control**: Opponent skill varies wildly on ladder. Win rate is noisy. Need 20+ games for statistically meaningful comparisons.
5. **Single environment**: No parallelization. SB3's vectorized env wrapper won't help. Training is inherently serial.
6. **Game UI interruptions**: Chest rewards, trophy road notifications, and season-end popups can interfere. Semi-automated mode lets the operator dismiss these.
7. **Tower count vs HP**: Using tower counts (crowns) rather than HP deltas means the reward is triggered only on tower destruction, not incremental damage. This is sparser but more reliable without OCR.
