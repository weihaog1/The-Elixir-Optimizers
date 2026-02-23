# Action Space & ActionBuilder Design Analysis

## CS175 - Clash Royale RL Agent
## February 2026

---

## 1. Invalid Action Handling: BC Learning vs Explicit Masking

**Recommendation: Use explicit masking. Do not rely on BC alone.**

### Why masking is necessary

Invalid actions in Clash Royale fall into two categories:

**Hard-invalid (physically impossible):**
- Playing a card not in your hand (empty slot)
- Playing a card you can't afford (insufficient elixir)
- These can never appear in human demonstration data because the game prevents them

**Soft-invalid (technically possible but strategically wrong):**
- Placing troops on the enemy half of the arena (game silently ignores it)
- Placing a building outside its valid zone
- Playing a spell on empty ground

The StateEncoder already implements hard-invalid masking via `action_mask()`. It checks two things:
1. Is the card slot occupied with a real card (not None, not "empty-slot")?
2. Does the player have enough elixir (`state.elixir >= CARD_ELIXIR_COST[card_name]`)?

If either check fails, all 576 grid cells for that card slot are masked False. The no-op (action 2304) is always valid.

### What the mask does NOT check

The current mask does NOT enforce:
- Troop placement only on the player's half (rows 17-31)
- Building placement distance restrictions
- Spell targeting rules (spells can go anywhere, but placing arrows on empty ground is wasteful)

### Why BC alone is insufficient for hard-invalid actions

During BC data collection with a click logger, the human player can only perform valid actions (the game blocks invalid ones). So the BC dataset contains zero examples of hard-invalid actions. This means:

- The BC model has never seen "play card with 0 elixir" and has no learned behavior for it
- Without masking, the model might assign nonzero probability to hard-invalid actions due to generalization noise
- During PPO exploration, the agent WILL try invalid actions if they're not masked
- Each invalid action wastes an environment step and creates a confusing reward signal

**Explicit masking costs nothing and eliminates an entire failure class. Always mask hard-invalid actions.**

### Let BC handle soft-invalid actions

Soft-invalid placement (e.g., troops on enemy half) is a different story. The game either silently ignores the placement or places the unit at the nearest valid position. The human demonstration data naturally shows correct placements. BC learns where to place troops from the distribution of human clicks, and this transfers well.

Adding soft-invalid masking (e.g., restrict troops to rows 17-31 only) is optional and has trade-offs:
- Pro: Reduces effective action space from 576 to ~270 cells per troop card
- Con: Requires encoding game-specific placement rules that differ per card type
- Con: Some cards have subtle rules (buildings have distance restrictions from river/towers)
- Con: Rule errors in the mask are worse than no mask (agent literally cannot take the correct action)

**Recommendation for MVP: Mask hard-invalid only. Let BC + PPO learn soft placement constraints from data.**

### Summary

| Invalid Type | Examples | Handling | Rationale |
|-------------|----------|----------|-----------|
| Hard-invalid | Empty slot, insufficient elixir | Explicit mask (already implemented) | Zero-cost, eliminates impossible actions |
| Soft-invalid | Troops on enemy half, building zones | Learn from BC data | Rules are complex and card-specific; errors in mask are worse than no mask |
| Strategic | Arrows on empty ground, e-spirit into nothing | Learn from BC + PPO | Cannot be rule-defined; depends on game context |

---

## 2. Data Collection Method: Click Logger vs Elixir-Change Detection

### Option A: Click Logger with Live Play (50 new matches)

**How it works:**
- Thread A: `mss.capture()` at 2-4 FPS -> StateBuilder -> GameState + timestamp
- Thread B: OS-level mouse click logger (pynput) -> (x_pixel, y_pixel, timestamp) per click
- Post-game: Match click events to card slot changes to produce (card_id, x_norm, y_norm, timestamp) tuples
- Merge with GameState snapshots by timestamp -> (state, action) training pairs

**Pros:**
- Ground truth actions -- you know exactly what was played and where
- No inference ambiguity for spells (click logging captures the exact placement)
- Works for all card types equally (troops, spells, buildings)
- Clean action labels with high confidence

**Cons:**
- Requires playing 50 new matches manually while the logger runs
- Time cost: 50 matches x ~3 min = ~2.5 hours of gameplay + setup time
- Requires implementing the click logger (pynput + card slot correlation logic)
- CardPredictor must be wired into StateBuilder first (currently `GameState.cards` is always empty -- this is a blocking dependency)

### Option B: Elixir-Change Detection on Pre-Recorded Matches

**How it works:**
- Process existing recorded gameplay videos through StateBuilder at 2-4 FPS
- Detect card placements by observing:
  - Elixir drops between frames (e.g., elixir goes from 7 to 2 = 5-cost card played)
  - Card slot changes (card disappears from hand between frames)
  - New unit appearing on the field
- Infer: which card was played from the elixir delta + card hand change
- Infer: where it was placed from the new unit's bounding box position

**Pros:**
- Uses existing data -- no new gameplay sessions needed
- No click logger implementation required

**Cons:**
- **Spell placement is unrecoverable.** Spells (arrows, barbarian-barrel) do not spawn persistent units. You can detect THAT a spell was played (elixir drop + card disappears) but NOT WHERE it was placed. Our deck has 2 spells (arrows, barbarian-barrel), so ~25% of card placements would have missing placement coordinates
- **Elixir OCR is only ~90% accurate.** A misread (7 -> 6 instead of 7 -> 2) corrupts the elixir delta and misidentifies which card was played
- **Frame rate limitation.** At 2-4 FPS, rapid card plays (double-placing with queuing) may appear as simultaneous in adjacent frames, making it ambiguous which was placed first
- **CardPredictor is still needed** to detect hand changes (same blocking dependency as Option A)
- **New unit position != placement position.** Units move immediately after spawning. Between frame t (pre-placement) and frame t+1 (post-placement), the unit has already moved ~0.5-1 tile from the placement point. This introduces systematic placement noise.

### Pre-recorded data status

Currently only 3 video files exist:
- `gameplay-videos/pigs_lose_0_1_crowns(1).mp4` (45 MB)
- `gameplay-videos/pigs_lose_0_3_crowns(1).mp4` (28 MB)
- `Project-Ralph-test/personal-gameplay-dataset/original-video/2026-01-25_15-52-46.mp4`

If 50 matches are planned but not yet recorded, both options require new recordings. The question becomes: record with or without the click logger running.

### Recommendation: Click Logger (Option A)

**Use the click logger.** The 2.5 hours of gameplay is a one-time cost, and the action labels are perfect. Elixir-change detection sounds appealing ("use existing data") but the spell placement gap is fatal -- 25% of card plays would have corrupted or missing placement coordinates, and these are some of the most strategically important actions (arrows to counter swarm, barbarian-barrel to counter ground push).

**However**, if time pressure makes implementing the click logger infeasible, a hybrid approach works:
1. Use elixir-change detection for troop placements (positions recoverable from spawned unit bbox)
2. Discard spell action frames entirely (treat them as no-ops in the dataset)
3. Accept that the BC model will never learn to place spells correctly
4. Fix spell placement during the PPO phase where the agent learns from rewards

This hybrid is strictly worse but may be pragmatic if you have 50 pre-recorded matches and no time to re-record.

### Blocking dependency for either option

**CardPredictor must be wired into StateBuilder first.** Without card detection, `GameState.cards` is always empty, which means:
- The action mask only allows no-op (all card slots masked out)
- The vector observation has zeros for all card features (indices 11-22)
- Action labels cannot be generated (no way to know which card slot corresponds to which card)

This integration is estimated at 30 minutes of work (crop 4 card regions from screenshot, run `CardPredictor.predict()` on each, populate `GameState.cards`).

---

## 3. Action Space Specifics

### What IS included in the action space

**Discrete(2305) = 4 card slots x 576 grid cells + 1 no-op**

| Action Range | Meaning |
|-------------|---------|
| 0-575 | Card slot 0 placed at one of 576 grid cells |
| 576-1151 | Card slot 1 placed at one of 576 grid cells |
| 1152-1727 | Card slot 2 placed at one of 576 grid cells |
| 1728-2303 | Card slot 3 placed at one of 576 grid cells |
| 2304 | No-op (wait / do nothing) |

**Encoding formula:** `action = card_id * 576 + row * 18 + col`
**Decoding formula:**
```
card_id = action // 576
cell = action % 576
row = cell // 18
col = cell % 18
```

**Grid:** 18 columns (x-axis, full screen width) x 32 rows (y-axis, arena region only -- excludes timer bar and card bar)

### What IS NOT included (and why)

| Excluded Element | Why Excluded | Where It Lives Instead |
|-----------------|-------------|----------------------|
| Next card (5th card preview) | Can't be played -- it's a preview only | Not in action space or observation |
| Card identity in action | Redundant -- slot index maps to identity via GameState.cards | Vector observation indices 15-18 encode card class per slot |
| Continuous (x, y) placement | Discretized to 18x32 grid -- sufficient precision (~2 tiles per cell) | Grid is the action space; coord_utils converts back to screen coords |
| Card rotation / direction | Clash Royale doesn't support directional placement | N/A |
| Emote / chat actions | Strategically irrelevant | Excluded entirely |
| Troop targeting | Not a player action -- troops auto-target | N/A |
| Spell timing (hold vs tap) | All spells deploy immediately on click | N/A |

### What IS separated and encoded elsewhere

| Element | Location | Format |
|---------|----------|--------|
| Card identity per slot | Vector observation indices 15-18 | Normalized class index (0 to 1) |
| Card elixir cost per slot | Vector observation indices 19-22 | cost / 10 |
| Card availability | Vector observation indices 11-14 | Binary (0 or 1) |
| Card playability | Action mask (2305,) | True/False per action |
| Current elixir | Vector observation index 0 | elixir / 10 |
| Unit positions | Arena grid (32, 18, 7) channels 0-3 | Unit count per cell per type |
| Tower HP | Arena grid channels 4-5 + vector indices 3-8 | HP fraction (0-1) |

---

## 4. Action Space and State Space Relationship

### Key relationships for the StateAndActionBuilder

**1. Card hand links action space to state space:**
The 4 card slots in `GameState.cards` determine which 576-cell blocks in the action space are valid. Each `Card(slot=i, class_name, elixir_cost)` maps to actions `i*576` through `(i+1)*576 - 1`.

**2. Elixir constrains action availability:**
`GameState.elixir` combined with `CARD_ELIXIR_COST[card.class_name]` determines the action mask. If `elixir < cost`, all 576 cells for that card are masked out.

**3. Grid cells are shared between arena observation and action targets:**
The same 18x32 grid is used for:
- Arena observation: what's at each cell (unit counts, tower HP, spell effects)
- Action targets: where to place a card

This means the agent sees the board state and selects a placement position using the same coordinate system. This is intentional and critical for learning spatial relationships ("place troops where enemy units are pushing").

**4. Action decodes to screen coordinates for execution:**
```
action_idx -> action_to_placement() -> (card_id, col, row)
(col, row) -> cell_to_norm() -> (x_norm, y_norm)
(x_norm, y_norm) -> (x_pixel, y_pixel) via frame resolution
```
The reverse path converts screen clicks to actions during BC data collection:
```
(x_pixel, y_pixel) -> pixel_to_cell() -> (col, row)
card_slot -> card_id
(card_id, col, row) -> placement_to_action() -> action_idx
```

**5. Action mask is derived entirely from state:**
`StateEncoder.action_mask(state)` is a pure function of `GameState`. No external information is needed. This is important for SB3 compatibility -- the mask is computed fresh each step from the current state.

### Notes for building the StateAndActionBuilder

- The StateAndActionBuilder must produce matched `(obs, action, mask)` triples for each decision timestep
- `obs` comes from `StateEncoder.encode(game_state)`
- `mask` comes from `StateEncoder.action_mask(game_state)`
- `action` comes from the click logger (card_id + grid cell) or the agent's policy output
- All three must correspond to the same game frame (same timestamp)
- The StateEncoder is stateless -- call it fresh each frame, no carry-over between frames

---

## 5. Wait Action Handling

### Are we recording the wait action?

**Yes. The wait (no-op) action must be recorded.** Here's why:

In Clash Royale, choosing NOT to play a card is often the optimal action:
- Waiting for elixir to accumulate (elixir leak is bad, but premature placement is worse)
- Waiting for the opponent to commit first (reactive play)
- Saving a counter card for the right moment (e.g., holding arrows for a swarm push)

If you skip no-op frames entirely, the BC model has no concept of "do nothing" and will compulsively play cards every decision step, leading to elixir waste and bad card cycling.

### How no-op is encoded

No-op is action index 2304 (`NOOP_ACTION`). It is always valid in the action mask. When the agent selects it, no card is played and no mouse click is sent to the game.

**No-op is NOT "something more specific."** There is no "wait for elixir" vs "wait for opponent" distinction. The agent's internal state (weights) learns when to wait from the observation context. A single no-op action is standard practice for this type of game AI.

### Handling excessive no-op capture

At 2-4 FPS capture rate, ~70% of frames are no-ops. This creates severe class imbalance that makes BC training unstable (the model learns to always output no-op because that minimizes cross-entropy on 70% of the data).

**Recommendation: Downsampling during data collection + weighted loss during training.**

**Step 1: Downsample no-ops during DatasetBuilder processing**

```
Raw capture: 100 frames = 30 action frames + 70 no-op frames
After downsampling: 30 action frames + 14 no-op frames (keep 20% of no-ops)
Resulting ratio: ~68% action, ~32% no-op
```

Keep 100% of action frames (card placements). Randomly keep 15-20% of no-op frames. This brings the ratio to roughly 2:1 action:no-op, which is manageable for cross-entropy training.

**Step 2: Weighted cross-entropy loss during BC training**

```python
weights = torch.ones(ACTION_SPACE_SIZE)
weights[NOOP_ACTION] = 0.3      # Downweight no-op
weights[:NOOP_ACTION] = 3.0     # Upweight card placements
loss = F.cross_entropy(logits, target, weight=weights)
```

This tells the model that getting a card placement right is 10x more important than getting a no-op right.

**Why not let BC do ALL the heavy lifting (no downsampling)?**

Pure BC on the raw 70/30 imbalanced distribution will converge to "always no-op" because that achieves ~70% accuracy immediately. The weighted loss helps but cannot fully overcome a 7:3 ratio. Downsampling is cheap and dramatically improves training stability. Use both together.

**Why not downsample more aggressively (e.g., keep 5% of no-ops)?**

The agent needs SOME no-op examples to learn when waiting is correct. If you downsample too aggressively, the model never learns to wait and plays cards recklessly. 15-20% retention is a balanced sweet spot validated in similar imitation learning setups.

---

## 6. Should the ActionBuilder Validate Actions?

### What "validate" means

Action validation has two aspects:
1. **At data collection time:** Is this recorded action actually valid (legal in the game state)?
2. **At execution time (PPO):** Should the ActionBuilder reject or modify the agent's chosen action before sending it to the game?

### Data collection validation

**Yes, validate during data collection.** When processing (state, click) pairs into (obs, action, mask) training samples:

1. Check that the click corresponds to a card slot that has a card
2. Check that the player had enough elixir for that card
3. Check that `mask[action_idx] == True` for the recorded action

If validation fails, the training sample is corrupted (likely a timing mismatch between state snapshot and click event). **Discard these samples rather than training on invalid data.**

This is a data quality filter, not a learned behavior.

### Execution validation (PPO online play)

**No, do not validate at execution time beyond the mask.** The action mask already prevents hard-invalid actions. If the agent selects a masked action (which SB3's `MaskableMultiInputPolicy` prevents by construction), something is broken at the policy level.

For soft-invalid actions (troop on enemy half), the game itself handles rejection silently. The agent experiences the consequence (wasted decision step, no unit spawned) and learns from the negative reward signal. This is how PPO is supposed to work -- the agent learns the rules by interacting with the environment.

**Do NOT add post-hoc action correction** (e.g., "if agent tries to place troop on enemy half, snap to nearest valid cell"). This creates a discrepancy between what the agent thinks it did and what actually happened, which corrupts the policy gradient.

---

## 7. Execution Timing and Delays

### The card placement sequence

Playing a card in Clash Royale is a two-step mouse action:
1. **Click the card** in the card bar (bottom of screen)
2. **Click the arena position** where you want to place it

Between step 1 and step 2, the game enters "placement mode" where a ghost image follows the cursor. The player drags or clicks to place.

### Delay between clicks

**Recommended: 100-200ms between card click and placement click.**

Clash Royale's UI requires a brief delay between selecting the card and placing it. Too fast and the game may not register the placement. Too slow and you waste time.

| Delay | Effect |
|-------|--------|
| < 50ms | Game may not register the card selection; placement fails silently |
| 50-100ms | Works but is superhuman; may trigger anti-bot detection |
| 100-200ms | Mimics fast human play; reliable and natural |
| 200-500ms | Mimics casual human play; safe but slow |
| > 500ms | Unnecessary delay; wastes decision time |

**Recommendation:** Use 150ms base delay + random jitter of +/- 50ms (so 100-200ms range). This replicates human reaction time for the "select card then tap arena" motion.

### Does this replicate human timing?

Human card placement in competitive Clash Royale takes ~200-400ms from decision to placement (includes visual processing, hand movement, and the two taps). Our pipeline:
- Decision: 0ms (policy forward pass is <1ms)
- Card click: 0ms (PyAutoGUI click)
- Delay: 150ms +/- 50ms
- Arena click: 0ms (PyAutoGUI click)
- Total: ~150ms

This is faster than human but within realistic bounds. The 100-200ms delay prevents robotic-feeling play while maintaining competitive speed. The game's server-side tick rate (~16ms per tick) is much faster than our action frequency (500ms per decision), so timing precision is not critical.

### Full action execution pseudocode

```python
import pyautogui
import time
import random

# Card slot pixel centers (1080x1920 resolution)
CARD_SLOT_CENTERS = [
    (336, 1712),   # Slot 0: midpoint of (242,1595)-(430,1830)
    (539, 1712),   # Slot 1: midpoint of (445,1595)-(633,1830)
    (742, 1712),   # Slot 2: midpoint of (648,1595)-(836,1830)
    (945, 1712),   # Slot 3: midpoint of (851,1595)-(1039,1830)
]

def play_card(card_id: int, x_norm: float, y_norm: float,
              screen_w: int = 1080, screen_h: int = 1920):
    # Step 1: Click the card slot
    cx, cy = CARD_SLOT_CENTERS[card_id]
    pyautogui.click(cx, cy)

    # Step 2: Wait (human-like delay)
    delay = 0.15 + random.uniform(-0.05, 0.05)
    time.sleep(delay)

    # Step 3: Click the arena placement position
    px = int(x_norm * screen_w)
    py = int(y_norm * screen_h)
    pyautogui.click(px, py)
```

---

## 8. Repeated Action Selection

### If the same action is selected multiple times in a row

This scenario occurs when the agent repeatedly wants to play the same card at the same grid cell across consecutive decision steps. In practice this is rare because:
- After playing a card, it leaves the hand and a new card cycles in
- The action mask removes the played card until it reappears in hand
- The 500ms decision interval means the card is gone by the next step

### When it can happen

- The agent selects no-op repeatedly (normal -- waiting for elixir)
- The agent wants to place a card but the action is masked (insufficient elixir), and keeps selecting no-op while waiting
- Edge case: the agent selects a card placement, but the game is laggy and the card appears to still be in hand on the next frame

### Recommendation: Execute exactly, no deduplication

**Execute every action the agent selects.** Do not downsample, debounce, or add jitter to repeated actions.

Reasons:
1. The action mask already prevents double-playing a card (once played, the slot empties and gets masked)
2. Repeated no-ops are correct behavior (the agent is choosing to wait each step)
3. Deduplication adds complexity and can suppress legitimate plays (e.g., two different cards placed in rapid succession look like "repeated action" if the deduplication window is too wide)
4. If the same placement action appears valid twice in a row, something is wrong with the state capture (card should be gone), not with the agent

**Exception:** If during PPO online play the agent tries to place the same card within 200ms of a previous successful placement AND the card is still shown in hand (stale observation), skip the second execution. This prevents accidentally double-clicking. Implement as a simple cooldown per card slot:

```python
last_play_time = [0.0] * 4  # per card slot

def execute_action(action_idx):
    result = action_to_placement(action_idx)
    if result is None:
        return  # no-op
    card_id, col, row = result
    now = time.time()
    if now - last_play_time[card_id] < 0.3:
        return  # cooldown, skip
    last_play_time[card_id] = now
    x_norm, y_norm = cell_to_norm(col, row)
    play_card(card_id, x_norm, y_norm)
```

---

## 9. What BC Handles in the Action Space

### What BC learns from demonstration data

| Aspect | BC Responsibility | How It Learns |
|--------|------------------|---------------|
| When to play vs wait | Full | Distribution of no-op vs action frames in training data (after downsampling) |
| Which card to play | Full | Expert's card choices given board state; encoded as card_id in action label |
| Where to place troops | Full | Expert's click positions discretized to grid cells; learns spatial patterns (e.g., hog at bridge) |
| Where to place spells | Full | Same as troops; spell placement is just a grid cell like any other |
| Placement precision | Partial | Limited by 18x32 grid resolution (~2 tiles per cell); fine-grained placement lost |
| Elixir management | Implicit | Learned from expert's wait patterns; expert waits when low on elixir |
| Card cycling | Implicit | Learned from order of card plays; expert cycles to specific cards |
| Reactive vs proactive play | Full | Distribution of expert's timing relative to opponent actions |

### What BC does NOT learn (requires PPO)

| Aspect | Why BC Can't Learn It |
|--------|----------------------|
| Optimal play against unseen strategies | BC memorizes the expert's response; novel opponent behavior causes confusion |
| Recovery from mistakes | Expert data contains no mistakes (or very few); agent has no model of "what to do after a bad play" |
| Long-horizon elixir planning | BC is reactive (current frame -> action); doesn't optimize over 10-20 second horizons |
| Adaptation to different opponents | Expert may only face certain playstyles; PPO explores the full space |
| Tradeoff evaluation | Should I defend or counter-push? BC copies the expert's decision but doesn't learn the underlying value function |

### What the action mask handles (not BC's job)

| Aspect | Mask Responsibility |
|--------|-------------------|
| Preventing play of missing cards | Hard mask -- slot empty -> 576 cells False |
| Preventing play without elixir | Hard mask -- cost > elixir -> 576 cells False |
| No-op always available | Hard mask -- action 2304 always True |

### Interaction between BC and mask during training

During BC training with SB3's `MaskableMultiInputPolicy`:
1. The policy outputs logits over all 2305 actions
2. The mask zeros out logits for invalid actions
3. Softmax is applied to remaining valid actions only
4. Cross-entropy loss is computed against the expert's action
5. The expert's action is guaranteed to be in the valid set (it was recorded from real gameplay)

This means BC never trains on invalid actions and never needs to "learn" that empty slots can't be played -- the mask handles it structurally.

---

## 10. ActionBuilder Inputs and Outputs

### What the ActionBuilder is

The ActionBuilder converts raw action signals (click logger output or agent policy output) into discrete action indices compatible with the StateEncoder's Discrete(2305) action space.

It is NOT a component of the StateEncoder. It sits alongside it:

```
Click Logger                     Agent Policy
(card_id, x_pixel, y_pixel)     (action_idx from model)
        |                               |
        v                               v
  ActionBuilder                   ActionBuilder
  .from_click()                   .to_execution()
        |                               |
        v                               v
  action_idx (int, 0-2304)       play_card(card_id, x_norm, y_norm)
        |
        v
  DatasetBuilder pairs with obs + mask
```

### Inputs

**For BC data collection (from click logger):**
```python
ActionBuilder.from_click(
    card_id: int,           # Which card slot was clicked (0-3)
    x_pixel: int,           # Click x position in pixels
    y_pixel: int,           # Click y position in pixels
    frame_width: int,       # Capture resolution width
    frame_height: int,      # Capture resolution height
) -> int                    # action_idx (0-2304)
```

**For no-op frames (no click detected):**
```python
ActionBuilder.noop() -> int  # Returns 2304
```

**For PPO execution (from agent output):**
```python
ActionBuilder.to_execution(
    action_idx: int,        # Agent's chosen action (0-2304)
    screen_width: int,      # Game window width in pixels
    screen_height: int,     # Game window height in pixels
) -> tuple[int, float, float] | None
    # Returns (card_id, x_norm, y_norm) or None for no-op
```

### Outputs

**For BC data collection:**
- A single integer `action_idx` in range [0, 2304]
- This gets stored alongside `obs` and `mask` as a training sample: `(obs, action_idx, mask)`

**For PPO execution:**
- `None` if no-op (do nothing)
- `(card_id, x_norm, y_norm)` tuple if card placement
- The caller passes this to `play_card()` for PyAutoGUI execution

### Relation to the rest of the pipeline

```
                      BC Data Collection Pipeline
                      ===========================

Screen Capture -----> StateBuilder -----> GameState
     |                                       |
     |                                       v
     |                                 StateEncoder
     |                                   /       \
     |                            obs dict     action mask
     |                           (arena+vec)    (2305 bool)
     |                                |              |
     v                                |              |
Click Logger -----> ActionBuilder     |              |
                    .from_click()     |              |
                         |            |              |
                         v            v              v
                    action_idx  +--------------------------+
                                |     DatasetBuilder       |
                                | (obs, action, mask) per  |
                                | frame, with no-op        |
                                | downsampling             |
                                +--------------------------+
                                          |
                                          v
                                   Training Dataset
                                   (.npz or HDF5)


                      PPO Online Training Pipeline
                      ============================

                    ClashRoyaleEnv(gym.Env)
                    +-----------------------+
Screen Capture ---> | StateBuilder          |
                    | StateEncoder          |
                    |   obs = encode(state) |
                    |   mask = action_mask()|
                    +-----------+-----------+
                                |
                          obs + mask
                                |
                                v
                    +------------------------+
                    | SB3 PPO Policy         |
                    | MaskableMultiInputPolicy|
                    |   action = predict()   |
                    +------------------------+
                                |
                          action_idx
                                |
                                v
                    +------------------------+
                    | ActionBuilder          |
                    | .to_execution()        |
                    | -> (card_id, x, y)     |
                    +------------------------+
                                |
                                v
                    +------------------------+
                    | play_card()            |
                    | PyAutoGUI click card   |
                    | then click arena       |
                    +------------------------+
```

### Validation in the ActionBuilder

The ActionBuilder should perform minimal validation:

**from_click():**
- Assert `card_id` in [0, 3]
- Assert pixel coordinates within screen bounds
- Convert via `pixel_to_cell()` -> `placement_to_action()` and return

**to_execution():**
- If `action_idx == NOOP_ACTION`: return None
- Otherwise: `action_to_placement()` -> `cell_to_norm()` -> return tuple
- Assert `card_id` in [0, 3]
- Enforce per-slot cooldown (300ms) to prevent accidental double-plays

**No game-rule validation.** The ActionBuilder converts coordinates and action indices. It does not check elixir, card availability, or placement legality. That is the action mask's job.

---

## Assumptions

1. **50 matches are planned, not existing.** Only 3 video files currently exist in the repository. The 50-match dataset must be recorded (with or without click logger).

2. **CardPredictor integration is a prerequisite.** Both data collection methods require `GameState.cards` to be populated. This is currently broken (always empty). Wire CardPredictor into StateBuilder before building anything else.

3. **Single deck (Royal Hogs / Royal Recruits).** The entire action space, card registry, and elixir costs are hard-coded for this 8-card deck. Changing decks requires updating `encoder_constants.py` and retraining the CardPredictor.

4. **Google Play Games at 1080x1920.** Card slot positions, arena bounds, and OCR regions are calibrated for this resolution. The StateEncoder uses normalized coordinates internally, but the ActionBuilder's `play_card()` function needs the actual screen resolution.

5. **Decision frequency of 2 Hz (one action every 500ms).** This is `frame_skip=15` at 30 FPS capture. Fast enough for all strategic decisions in Clash Royale.

6. **SB3's MaskableMultiInputPolicy handles masked sampling.** We do not need to implement custom masked softmax. SB3-contrib provides this natively.

7. **Horizontal flip augmentation is free.** Clash Royale's arena is left-right symmetric. Flipping the arena grid and swapping left/right tower labels doubles the dataset with no additional gameplay.

8. **OCR staleness is acceptable for BC.** If OCR returns None for a frame, the last known value is used. For elixir at ~90% accuracy, this means ~10% of frames may have slightly wrong elixir values, but BC is robust to this noise because the expert actions are still correct.

9. **PPO will learn what BC cannot.** BC provides a warm start. It will not learn optimal play, recovery from mistakes, or adaptation to different opponents. These are explicitly deferred to the PPO phase.

10. **No anti-bot detection concerns.** Google Play Games running locally does not monitor for automated input. PyAutoGUI clicks are indistinguishable from real mouse clicks at the OS level.
