# RL Agent Improvement Plan: Addressing Card Spam, Poor Placement, and Training Strategy

## Context

The PPO agent has several behavioral issues after initial training:

1. **Clicks when not in game window** — ActionDispatcher has a focus check but it can fail or race
2. **Plays cards on "nothing"** — arrows on empty ground (likely YOLO detection gaps)
3. **Always plays cheapest cards** — no reward signal differentiates card quality
4. **Doesn't know when to wait** — BC training downsampled 85% of no-op frames, under-representing "waiting is correct"
5. **Places cards on wrong side** — possibly caused by horizontal flip augmentation in BC dataset
6. **General lack of strategic behavior** — reward function is purely outcome-based (towers/win/loss), no signal for which card or where

### Root Cause Analysis

| Problem | Root Cause | Evidence |
|---------|-----------|---------|
| Clicks when unfocused | `ActionDispatcher._is_game_focused()` defaults to `True` on failure; no re-check between card click and arena click | `live_inference.py:815-824`, `873` |
| Plays arrows on nothing | YOLO detection misses (mAP=0.804) — agent "sees" enemy units that aren't there; no verification that target exists | No placement validation in action execution |
| Always cheapest cards | Reward function has ZERO signal for which card is played — only tracks tower counts, win/loss, elixir level, unit count advantage | `reward.py` — no card-specific or placement-specific reward |
| Doesn't wait | BC data keeps only 15% of no-ops (`noop_keep_ratio=0.15`); PPO survival bonus (+0.002/step scaled) is tiny vs card play incentive | `dataset_builder.py:282-310`, `reward.py:37` |
| Wrong-side placement | BC dataset uses horizontal flip augmentation (`augment=True`), which mirrors placements left↔right. If human always defends left, model learns mirrored defense-right equally | `bc_dataset.py:86-102` |
| No strategy | Reward has no per-action shaping — all card plays are equally rewarded as long as they eventually lead to unit advantage | `reward.py` full file — outcome-only signals |

---

## Modifications & Improvements

### Priority 1: Fix Window Focus Safety (Bug Fix)

**File:** `bc_model_module/src/bc/live_inference.py`

**Changes:**
- In `ActionDispatcher._click_card_then_arena()`: add a second focus check AFTER the first click (card slot) and BEFORE the arena click. If focus was lost between clicks, abort and don't click the arena.
- In `ActionDispatcher._is_game_focused()`: log when returning the `True` default (HWND unavailable or exception) so the operator knows the safety check is degraded.

**Why:** The current code checks focus once, then does two clicks 150ms apart. If you alt-tab between clicks, the second click lands on whatever window is now in front.

### Priority 2: Add Reactive Placement Reward (Reward Shaping)

**File:** `ppo_module/src/ppo/reward.py`

The current reward function is purely outcome-based. Add signals that connect the agent's action to what's happening on the field:

**a) Elixir efficiency signal** — penalize playing cheap cards when expensive ones would be more impactful:
- Add `elixir_spent_bonus`: small reward proportional to the elixir cost of the card just played (e.g., `+0.005 * card_cost`). This makes playing a 7-cost card slightly more rewarding than a 1-cost, counteracting the "spam cheap" bias.

**b) Defensive placement reward** — reward placing units near active threats:
- After each action step, compare the placement column to the column(s) with the highest enemy unit density. If the placement is within ±3 columns of enemy concentration, add a small bonus (`+0.03`). This teaches "react to where the enemy is."

**c) Noop when low elixir bonus** — reward waiting when elixir is scarce:
- If `current_elixir < 3` and the agent chooses noop, add a small bonus (`+0.01`). This directly teaches "waiting is correct when you can't afford meaningful cards."

**Implementation:** Add these to `RewardComputer.compute()`. The method already receives `prev_obs` and `curr_obs` (which contain the arena grid and vector). The action taken is NOT currently passed to `compute()` — the env's `step()` will need to pass it so `compute()` can know what card was played and where.

**File:** `ppo_module/src/ppo/clash_royale_env.py`
- Pass `action` to `self._reward_computer.compute()` so it can factor in the card played and placement location.

### Priority 3: Improve BC No-Op Representation (Data/Training Fix)

**File:** `dataset_builder_module/src/dataset/dataset_builder.py`

The current 15% no-op keep ratio severely under-represents "waiting is correct" situations. The model learns from data where 85% of waiting frames are deleted — it sees an action-heavy world.

**Change:** Increase `noop_keep_ratio` from `0.15` to `0.30` and retrain BC. This preserves more "waiting" context while still downsampling the majority of idle frames. The play_weight in training (8.0-10.0) already compensates for class imbalance, so the model will still learn to play — it will just also see more examples of when NOT to play.

**Alternative (no retrain):** This can also be addressed purely through PPO reward shaping (Priority 2c above). More RL training on live games will eventually teach the agent to wait IF the reward function makes waiting beneficial. The BC fix just gives it a better starting point.

### Priority 4: Evaluate Horizontal Flip Augmentation (Data Investigation)

**File:** `bc_model_module/src/bc/bc_dataset.py`

The horizontal flip augmentation doubles training data by mirroring placements. This is standard for games with symmetric fields BUT can be harmful when:
- Human demos consistently favor one side (e.g., always defend left princess tower first)
- Card placement context is directional (placing units to the LEFT of a push is different from the RIGHT)

**Investigation needed:** Before disabling, check if the BC model without augmentation (`augment=False`) produces better directional behavior. This requires retraining BC with augmentation disabled and comparing placement patterns.

**Recommendation:** For now, keep augmentation on (doubles a small 652-action dataset), but add a flag to the PPO training guide documenting this trade-off. The horizontal flip shouldn't cause "always wrong side" — it should cause "50/50 random side." If the agent consistently places on the WRONG side, the issue is more likely the sparse reward signal (Priority 2).

### Priority 5: Add Card Diversity Metrics (Observability)

**File:** `ppo_module/src/ppo/callbacks.py`

Currently `CRMetricsCallback` only tracks `cards_played` (count). Add:
- `card_cost_avg`: average elixir cost of cards played this episode
- `card_distribution`: count per card slot (detect if one card dominates)
- `noop_ratio`: fraction of steps that were noop (detect passivity vs spam)

This won't change behavior, but will let you diagnose problems during training.

**File:** `ppo_module/src/ppo/clash_royale_env.py`
- Track which card was played each step (card_id from action decoding) and pass to callback via `info` dict.

### Priority 6: Continue PPO Training (Training Strategy)

**Should you just train more?** Yes, but with the reward shaping from Priority 2. Without reward shaping, more training will only reinforce "spam cheap cards → occasionally win." With shaping:

- **Phase 1 (frozen extractor, 15 games):** Already done — teaches basic card play
- **Phase 2 (unfrozen, 25+ games):** With reward shaping, this is where strategic behavior should emerge
- **Phase 3 (extended, 50+ games):** If Phase 2 shows improvement (rising card_cost_avg, better noop_ratio), continue training

**Entropy schedule matters:** The current schedule (0.02 → 0.005) encourages exploration early. Consider keeping entropy higher longer (0.02 → 0.01 over 50 games) to prevent premature convergence on cheap-card-spam.

---

## Implementation Order

| Step | Priority | Effort | Impact |
|------|----------|--------|--------|
| 1. Fix focus safety | P1 | Small (10 lines) | Prevents clicking outside game |
| 2. Add reward shaping | P2 | Medium (50-80 lines) | Core fix for card selection and waiting |
| 3. Add card metrics | P5 | Small (20 lines) | Enables diagnosing training progress |
| 4. Retrain PPO with shaping | P6 | Time (hours of gameplay) | Teaches strategic behavior |
| 5. Investigate flip augmentation | P4 | Medium (retrain BC) | May improve directional placement |
| 6. Increase noop ratio & retrain BC | P3 | Medium (retrain BC + PPO) | Better "waiting" foundation |

---

## Files to Modify

| File | Changes |
|------|---------|
| `bc_model_module/src/bc/live_inference.py` | Focus re-check between clicks in `_click_card_then_arena()` |
| `ppo_module/src/ppo/reward.py` | Add `elixir_spent_bonus`, `defensive_placement_bonus`, `low_elixir_noop_bonus`; accept action in `compute()` |
| `ppo_module/src/ppo/clash_royale_env.py` | Pass action + card info to reward computer; track card_id per step for metrics |
| `ppo_module/src/ppo/callbacks.py` | Add `card_cost_avg`, `card_distribution`, `noop_ratio` metrics |

## Verification

1. **Focus fix:** Alt-tab during dry-run — confirm second click is suppressed, logs show `reason=focus_lost_mid_action`
2. **Reward shaping:** Dry-run with `--visualize` — verify reward printouts show new components (elixir bonus, placement bonus, noop bonus)
3. **Metrics:** Check TensorBoard for new `cr/card_cost_avg` and `cr/noop_ratio` scalars
4. **Training:** After 10 games with shaping, `card_cost_avg` should be > 2.0 (not always cheapest) and `noop_ratio` should be 0.3-0.6 (not always playing)
5. **Existing tests:** `python -m pytest ppo_module/tests/ -v` — all pass
