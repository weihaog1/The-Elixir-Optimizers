# PPO Training Debug Guide

Diagnosis and fixes for two common PPO training issues:
1. **Stuck on "Waiting for game start..."** even when a game is in progress
2. **Agent plays cards in some games but does nothing in others**

---

## Root Causes

### Root Cause #1: mss Captures Screen Pixels, Not the Game Window

`GameCapture` uses `mss.grab(monitor)` which captures raw screen pixels at fixed coordinates. It does NOT capture from a window handle. When another window (e.g. VS Code) covers the game:

1. `mss.grab()` captures VS Code pixels instead of the game
2. `GamePhaseDetector` sees low card bar intensity and low arena variance
3. Detector classifies the frame as LOADING or END_SCREEN
4. If END_SCREEN is confirmed for 3 consecutive frames (1.5 seconds), the episode terminates mid-game
5. `reset()` calls `wait_for_game_start()` which keeps seeing non-game pixels → stuck

**Why it's intermittent:** Detection works when the game window is visible. Any brief alt-tab (even 2 seconds) can trigger a false END_SCREEN and terminate the episode.

**Files:** `bc_model_module/src/bc/live_inference.py:252-263`, `ppo_module/src/ppo/game_detector.py:277-300`

### Root Cause #2: Card Mask Bug When CardPredictor is Absent

In `PerceptionAdapter._process_with_detection()`, when `CardPredictor` is `None` or fails:

```python
card_names: list[str] = ["", "", "", ""]
if self._card_predictor is not None:
    card_names = self._populate_card_vector(frame, vector, fw, fh)
else:
    vector[0, 11:15] = 1.0  # assume cards present
    # BUG: card_names stays ["", "", "", ""]

# Mask only unmasks cards with non-empty names
for i in range(4):
    if card_names[i] != "":  # ALL FAIL → all 2304 actions masked
        mask[start:end] = True
```

Result: All card actions are masked, forcing the agent to NOOP every step.

Even when `CardPredictor` is loaded, individual card crop failures are caught silently (`except Exception: continue`), leaving that card slot as `""` and its 576 grid cells masked.

**File:** `bc_model_module/src/bc/live_inference.py:562-576`

### Root Cause #3: Probe Frame Timing at Startup

During env `__init__`, a probe frame is captured to detect game bounds (pillarbox cropping). If the game window isn't visible yet (user is still in VS Code), the probe captures VS Code pixels and computes wrong game bounds for the entire session.

**File:** `ppo_module/src/ppo/clash_royale_env.py:176`

### Root Cause #4: Corrupted Frames → Garbage Observations

When non-game pixels are captured:
- YOLO tries to detect "units" in VS Code UI → zero or spurious detections
- Arena encoding is filled with noise
- Tower count tracking gets corrupted values
- Reward computation produces anomalous data

---

## What Happens With Your Workflow

Based on the described workflow:

```
1. Run run_ppo.py → VS Code has focus
2. See "Initializing environment..." → alt-tab to Clash Royale
3. Probe frame may capture VS Code (race condition) → wrong game bounds
4. wait_for_game_start() polls at 1 Hz
5. Wait ~1 minute, click Battle → game starts
6. If capture coords correct: detector sees IN_GAME → proceeds
7. After 30s of agent inactivity, alt-tab to VS Code to check console
8. mss captures VS Code pixels for ~2-4 seconds
9. 3 consecutive non-game frames → detector confirms END_SCREEN
10. Episode terminates MID-GAME
11. reset() calls wait_for_game_start() → still capturing VS Code
12. Console shows "Waiting for game start..." (this is what you see)
13. Tab back to Clash Royale → game still running
14. Detector sees IN_GAME → starts new "episode" mid-game (corrupted state)
```

This explains:
- "Waiting for game start..." during an active game
- Intermittent performance (episodes that start when watching work; episodes interrupted by alt-tab break)

---

## Fixes Applied

### Fix 1: Card Mask Fallback (live_inference.py)

When `CardPredictor` is absent, `card_names` is now set to `["unknown"] * 4` so the mask correctly unmasks all card slots.

### Fix 2: Window Focus Guard (clash_royale_env.py)

New `_is_game_focused()` method checks if the game window is the foreground window using Windows API.

**In `step()`:** When the game loses focus, the step returns the previous observation with a survival-only reward instead of processing garbage pixels. A warning is logged once when focus is lost.

**In `reset()`:** The `wait_for_game_start` capture function returns a black frame when unfocused, preventing false IN_GAME/END_SCREEN detection from non-game pixels.

### Fix 3: Probe Frame Timing (clash_royale_env.py)

The env `__init__` now waits up to 15 seconds for the game window to be focused before capturing the probe frame. If the window isn't focused in time, it proceeds with a warning.

### Fix 4: Diagnostic Logging (clash_royale_env.py)

- **Mask validation:** Warns when all card actions are masked (only NOOP available)
- **Initial perception state:** Logs detected cards and valid action count at episode start
- **Focus loss warning:** Logs when the game window loses focus during an episode

---

## Operator Best Practices

### The Golden Rule

**The game window must be visible and unobstructed at all times during training.** `mss` captures screen pixels at fixed coordinates — anything covering the game (VS Code, browser, dialog box) produces garbage frames.

### Recommended Setup

| Setup | How |
|-------|-----|
| **Dual monitors** | VS Code on monitor 1, game on monitor 2 |
| **Side-by-side** | VS Code left half, game right half (never overlapping) |
| **Single monitor** | Use `--visualize` and never alt-tab; check logs on phone |

### What NOT To Do

- Do NOT alt-tab to VS Code during a game
- Do NOT minimize the game window
- Do NOT place any window on top of the game
- Do NOT drag windows across the game area

### Using --no-pause Safely

With `--no-pause`, the agent auto-continues between episodes. You must:
1. Keep the game visible at all times
2. Click "Battle" as soon as the previous game ends
3. Monitor the VS Code console from a second device or second monitor

---

## Diagnostic Checklist

If issues persist after the fixes, try these steps:

### 1. Verify Game Detection Thresholds

```bash
python ppo_module/run_ppo.py \
    --bc-weights models/bc/bc_feature_extractor.pt \
    --window-title "Clash Royale" \
    --num-episodes 1 \
    --dry-run
```

Check startup output:
- `[Env] Initialized. Game bounds: (x,y) WxH` — bounds should match game area
- `[Env] Game started. Initial cards: [...]` — should show 4 card names
- `[Env] Initial mask: N valid actions (M card placements)` — M should be > 0

### 2. Test Without Alt-Tabbing

Keep the game visible for an entire episode. If the agent plays cards normally, Root Cause #1 is confirmed. The focus guard should now protect against accidental alt-tabs.

### 3. Test With --no-perception

```bash
python ppo_module/run_ppo.py ... --no-perception --dry-run
```

If the agent tries to play cards (logged but not executed in dry-run), the issue is in the perception/masking pipeline, not the policy.

### 4. Check For Mask Warnings

Look for `[Env] WARNING step N: All card actions masked!` in the console. If this appears frequently, card detection is failing — check:
- CardPredictor model exists at `models/card_classifier.pt`
- YOLO model exists at `models/best_yolov8s_50epochs_fixed_pregen_set.pt`
- Card crop regions align with actual card slots in the game

### 5. Use --capture-region Instead of --window-title

If window auto-detection is unreliable, manually specify coordinates:
```bash
# Find exact game area coordinates first (use screenshot tool)
python ppo_module/run_ppo.py ... --capture-region LEFT,TOP,WIDTH,HEIGHT
```

---

## New Console Messages

After the fixes, you'll see these new messages:

| Message | Meaning |
|---------|---------|
| `[Env] Keep game window visible (don't alt-tab over it).` | Reminder at game start |
| `[Env] Game started. Initial cards: [...]` | Shows detected card names |
| `[Env] Initial mask: N valid actions (M card placements)` | M=0 means all cards masked |
| `[Env] WARNING step N: Game window lost focus` | Alt-tab detected, frames being skipped |
| `[Env] WARNING step N: All card actions masked!` | No cards detected, only NOOP valid |
| `[Env] WARNING: Game window not focused for probe.` | Probe frame may have wrong bounds |

---

## Technical Details

### Game Phase Detection Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `card_bar_intensity` | > 40.0 | Card bar has colored card art (game active) |
| `arena_variance` | > 200.0 | Arena has varied content (troops, spells) |
| `arena_intensity` | < 80.0 | Arena is dark (loading screen) |
| Phase stability | 3 consecutive frames | Debounce: 1.5s at 2 FPS |

### Focus Guard Behavior

When the game window loses focus during `step()`:
- Frame capture still occurs (mss can't be told not to)
- Captured frame is **discarded** (not fed to detector or perception)
- Previous stacked observation is returned unchanged
- Survival-only reward is given (no crown/elixir/unit signals)
- Action is not executed (no mouse clicks)
- The warning is printed once (not every frame)

When focus is restored, normal processing resumes immediately.
