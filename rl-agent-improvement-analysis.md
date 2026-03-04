# RL Agent Improvement Analysis

## Context

The PPO RL agent wraps live Clash Royale gameplay as a Gymnasium environment, using BC-pretrained feature extraction and MaskablePPO from sb3-contrib. Training runs at 2 FPS in real-time (no simulator), making every training step expensive. The agent currently uses a sparse reward signal (crown events + win/loss) with minimal reward shaping.

This document identifies bugs, improvement opportunities, and design decisions for the RL agent.

---

## 1. CRITICAL BUG: Tower Counts Hardcoded During Live Play

**File:** `reward.py:165-184` (docstring confirms it), `clash_royale_env.py:494-503`

The `compute_manual_crowns()` docstring explicitly states:
> "since tower counts in the observation vector are hardcoded and don't change during live play"

This means **crown rewards (+10/-10) never fire automatically** during a game. The vector features at indices 9 (ally tower count) and 10 (enemy tower count) are always 1.0, so `prev_enemy_towers - curr_enemy_towers` is always 0. The only way crowns get rewarded is via manual operator input.

**Impact:** The agent's per-step reward is essentially just `survival_bonus (0.02) + elixir_waste_penalty (-0.05)` for the entire game, with a single terminal reward at game end. This is **extremely sparse** — the agent gets almost no signal about in-game progress.

**Fix Options:**
- A) Wire `PerceptionAdapter` to dynamically count visible towers and populate vector[9]/vector[10]
- B) Add tower HP tracking via pixel analysis of HP bars
- C) Accept the limitation and focus entirely on terminal reward + reward shaping (see Section 3)

---

## 2. ARCHITECTURE IMPROVEMENTS

### 2a. Frame Stacking for Temporal Context

**Current:** Single-frame observations (Markovian). The agent cannot perceive troop movement direction, elixir generation trends, or attack momentum.

**Proposal:** Stack the last 2-3 frames to give temporal context:
- Arena: (32, 18, 6) -> (32, 18, 18) for 3-frame stack
- Vector: (23,) -> (69,) concatenated, or (23,) with delta features appended

**Trade-off:** Increases observation size, requires BC retraining for compatibility. Could add just delta features (difference between current and previous frame) as a lighter alternative.

### 2b. Larger Policy/Value Networks

**Current:** `pi=[128, 64]`, `vf=[128, 64]` — quite small for a 2305-action space.

**Proposal:** Increase to `pi=[256, 128]`, `vf=[256, 128]` or add a third layer. The feature extractor outputs 192 dims, which gets compressed to 64 before the action head — may lose information.

### 2c. Separate Value Function Feature Extractor

**Current:** Policy and value function share the same CRFeatureExtractor.

**Proposal:** Use `share_features_extractor=False` in SB3 policy_kwargs to give the value function its own feature extractor. The value function needs different features (game state evaluation) than the policy (action selection).

---

## 3. REWARD SHAPING IMPROVEMENTS

### 3a. Tower HP Delta Reward (if tower counts remain hardcoded)

Instead of binary crown events, reward continuous tower HP changes:
- `enemy_tower_hp_decreased` -> small positive reward proportional to damage dealt
- `ally_tower_hp_decreased` -> small negative reward proportional to damage taken

This gives much denser signal — every troop hit generates reward.

**Challenge:** Requires tower HP tracking, which is not currently in the observation vector (or is hardcoded).

### 3b. Elixir Efficiency Reward

**Current:** Only penalizes sitting at max elixir (-0.05).

**Improvements:**
- Reward spending elixir when it would otherwise cap (proactive play)
- Penalize playing cards when at low elixir and no threat exists (wasteful spending)
- Track elixir advantage: reward when opponent is likely overcommitted

### 3c. Positional Control Reward

Reward the agent for maintaining troops in strategically valuable positions:
- Troops past the bridge (offensive pressure): small positive
- Troops near ally towers defending: small positive during defense
- No troops on the field while enemy has many: negative (losing control)

**Source:** Can be computed from the arena grid observation — count friendly vs enemy units.

### 3d. Card Play Diversity Reward

**Current:** No signal about which cards are played.

**Proposal:** Small reward for playing different cards (entropy over card usage). Penalize playing the same card repeatedly (may indicate a degenerate policy).

### 3e. Time-Weighted Terminal Rewards

**Current:** Win = +30 regardless of when it happens.

**Proposal:** Scale terminal reward by game efficiency:
- Fast win (under 2 minutes): bonus multiplier
- 3-crown victory: extra bonus
- Late draw: less penalty than early draw

---

## 4. TRAINING EFFICIENCY IMPROVEMENTS

### 4a. n_steps vs Episode Length Mismatch

**Current:** `n_steps=512` but a typical game is 300-600 steps. This means `model.learn(total_timesteps=512)` may not complete a full episode, causing mid-episode policy updates with incomplete reward signals.

**Proposal:** Set `n_steps` to match expected episode length (~600-700) so each policy update sees at least one complete episode with terminal rewards.

### 4b. Increase n_epochs for Data Efficiency

**Current:** `n_epochs=10` — already above typical PPO defaults (3-4).

Since real-time training generates very limited data, this is reasonable. Could push to 15-20 but watch for overfitting (monitor KL divergence via TensorBoard).

### 4c. Experience Replay / Off-Policy Augmentation

PPO is on-policy, but with only ~600 steps per game, we discard data aggressively.

**Proposal:** Consider switching to SAC (Soft Actor-Critic) or adding an experience replay buffer. SAC is more sample-efficient for continuous-control-like problems.

**Counter-argument:** MaskablePPO doesn't have an off-policy equivalent in SB3. Would need custom implementation or a different library.

### 4d. Learning Rate Schedule

**Current:** Flat LR per phase (1e-4 for Phase 1, 3e-5 for Phase 2).

**Proposal:** Use cosine annealing or linear decay within each phase. SB3 supports callable learning rate schedules:
```python
learning_rate=lambda progress: 1e-4 * (1 - 0.9 * progress)
```

### 4e. Entropy Coefficient Annealing

**Current:** Fixed `ent_coef=0.01`.

**Proposal:** Start higher (0.05) for exploration, anneal to 0.005 over training. Early high entropy prevents premature convergence to noop-heavy policies.

---

## 5. ROBUSTNESS IMPROVEMENTS

### 5a. Action Masking Validation

**Current:** Mask comes from PerceptionAdapter each frame. If perception fails, all actions may be masked (or none).

**Proposal:** Add a safety check: if fewer than 2 actions are valid (just noop), log a warning. If all 2305 actions are valid, perception probably failed — fall back to a conservative mask.

### 5b. Reward Normalization

**Current:** Raw rewards range from -30 (loss) to +30 (win), with per-step rewards ~0.02. This is a 1500x scale difference.

**Proposal:** Use SB3's `VecNormalize` wrapper or manually normalize rewards. Large terminal rewards can destabilize the value function if it's calibrated to tiny per-step rewards.

### 5c. Observation Normalization

**Current:** Arena values range [-1, 10], vector values [0, 1]. The arena has much higher variance.

**Proposal:** Normalize arena channels independently. Use running mean/std normalization or clip to a tighter range.

---

## 6. DECISION QUESTIONS

These are open design choices that significantly affect agent behavior:

1. **Fix tower count tracking or accept sparse rewards?**
   - Fixing it gives crown rewards during gameplay (much denser signal)
   - Leaving it means relying on terminal win/loss only + manual operator input
   - This is the single highest-impact change

2. **How many reward shaping signals to add?**
   - More shaping = faster learning but risk of reward hacking
   - Minimal shaping = slower but more robust
   - Recommendation: Start with elixir efficiency + unit count advantage

3. **Should n_steps match episode length?**
   - Current 512 may split episodes across policy updates
   - Setting to 700 ensures each update sees a full game
   - Trade-off: larger rollout buffer = more memory

4. **Separate value function extractor?**
   - More parameters to train with limited data
   - But value function learning won't interfere with policy features
   - Recommendation: Try shared first (current), switch if value loss doesn't converge

5. **Entropy annealing schedule?**
   - High initial entropy risks random play in early games (losing rating/trophies)
   - Low entropy risks noop-heavy degenerate policy
   - Recommendation: Start at 0.02, anneal to 0.005 over 20 games

6. **Frame stacking vs delta features?**
   - Full frame stacking: 3x observation size, requires BC retraining
   - Delta features only: append (curr - prev) as extra channels, cheaper
   - Recommendation: Delta features as a first step

7. **Should we add a KL penalty to stay close to BC policy?**
   - Prevents catastrophic forgetting of BC-learned behaviors
   - But may limit RL improvement ceiling
   - Common in RLHF (PPO + KL from reference policy)

---

## 7. PRIORITIZED RECOMMENDATIONS

| Priority | Change | Impact | Effort | Files |
|----------|--------|--------|--------|-------|
| **P0** | Fix tower count tracking in obs vector | Enables crown rewards (~10x denser signal) | Medium | `live_inference.py` PerceptionAdapter |
| **P1** | Set `n_steps=700` to match episode length | Complete episodes per policy update | Trivial | `ppo_trainer.py:44` |
| **P1** | Add entropy annealing (0.02 -> 0.005) | Better exploration-exploitation | Low | `ppo_trainer.py`, `run_ppo.py` |
| **P2** | Add unit count advantage reward | Dense positional signal from arena grid | Low | `reward.py` |
| **P2** | Reward normalization | Stabilize value function training | Low | `ppo_trainer.py` (VecNormalize) |
| **P2** | Increase network size to [256, 128] | More capacity for 2305 actions | Trivial | `ppo_trainer.py:58-59` |
| **P3** | Delta features (curr - prev observation) | Temporal context without BC retraining | Medium | `clash_royale_env.py`, `sb3_feature_extractor.py` |
| **P3** | KL penalty from BC reference policy | Prevent forgetting during RL fine-tuning | Medium | New file + `ppo_trainer.py` |
| **P3** | LR cosine annealing | Smoother training dynamics | Low | `ppo_trainer.py` |
