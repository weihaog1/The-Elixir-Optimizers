# State Builder & Action Builder - BC + RL Pipeline
## Questions, Concerns & Context - February 2026
### CS175 - Clash Royale RL Agent Project

---

## Current State of What Exists

**Built and working:**
- Single YOLOv8s (155 classes, mAP50=0.804) - no belonging output
- MiniResNet card classifier (~25K params, 8-class deck recognition)
- PaddleOCR pipeline (timer at ~85%, elixir at ~90%, tower HP)
- StateBuilder that fuses YOLOv8 detections + OCR into a `GameState` dataclass
- ScreenConfig with fixed pixel regions for 540x960 resolution
- Synthetic data generator (ported from KataCR, background15, 155 classes)
- Custom NMS code for 7-column belonging output (`custom_utils.py`) - ported but unused because current model was not trained with belonging

**Not built yet:**
- ActionBuilder / ActionSpace
- StateEncoder (GameState -> tensor observation for RL)
- Gym environment wrapper
- Reward function
- BC data collection pipeline
- BC training code
- PPO training code
- Automated game reset / match start

**Critical gap:** The current YOLOv8s only outputs `(x1, y1, x2, y2, conf, cls)` - 6 columns. It does NOT output belonging (ally/enemy). The `_infer_unit_belonging()` in `state_builder.py:181-208` uses a Y-coordinate heuristic (`frame_height * 0.42` midpoint), which fails when enemy troops cross the river into the player's half. The model needs retraining with the belonging-aware loss that KataCR uses. The custom NMS for 7-column output is already ported in `src/yolov8_custom/custom_utils.py`.

---

## Q1 - STATE BUILDER: CV Pipeline

### 1. How do we handle OCR failures mid-episode - last-known-good, interpolation, or a sentinel/NaN channel?

Use last-known-good with a staleness counter. If OCR returns None for timer, keep the previous value and decrement by 1 per second (the frame rate is known). If it returns None for elixir, keep the previous value. Add a `ocr_stale_count: int` to GameState. If stale > 5 consecutive frames, set a `low_confidence` flag that the reward function can check. Never pass NaN to the agent.

### 2. How do we address domain shift (Android to GPC)? Fine-tune on a small GPC-specific set, or train from scratch?

Fine-tune on 30-50 labeled GPC screenshots. Do NOT train from scratch - the synthetic pretraining provides good feature representations. Freeze the backbone, fine-tune only the detection head for 10-20 epochs. This is the single highest-ROI improvement available. If time is tight, just lower the confidence threshold to 0.3 for inference and accept more false positives.

### 3. What is the minimum acceptable mAP before the RL agent can learn? Is 70% enough for the BC phase?

0.70 mAP50 on a GPC test set is sufficient for BC. Towers are at 0.95+. The common meta units are at 0.70-0.98. BC is more forgiving of detection errors than PPO because the expert demonstrations carry the signal - even with noisy observations, the actions are correct. PPO will need higher accuracy (0.80+) because the agent is exploring and bad observations lead to bad credit assignment.

### 4. Should the StateBuilder run asynchronously in a separate thread to decouple CV latency from the RL step frequency?

Yes, run it async. The architecture should be:

```
Thread 1: mss.capture() -> queue
Thread 2: YOLOv8 + OCR -> latest_state (shared)
Main:     env.step() reads latest_state, executes action
```

This decouples CV latency from decision frequency. The agent always gets the freshest available state.

---

## Q2 - ACTION BUILDER: Masking & Spell/Troop Distinction

### 5. Where should spell/troop metadata live - in ActionSpace, a static card registry, or derived from live GameState?

Static card registry. Create a `card_registry.py`:

```python
CARD_INFO = {
    "hog-rider": {"elixir": 4, "type": "troop", "target": "buildings"},
    "fireball": {"elixir": 4, "type": "spell", "radius": 2.5},
    "skeleton": {"elixir": 1, "type": "troop", "count": 3},
    ...
}
```

The ActionSpace queries this registry to build the mask. Don't derive spell/troop from live GameState - it is static knowledge.

### 6. Does the 16x9 resolution lose enough precision to matter for MVP, or is this a documented trade-off we accept now?

The 18x32 grid from KataCR (which maps to their generator's grid) is already fine. That is 576 cells, not 144. If the plan was 16x9=144 cells, use 18x32=576 instead - it matches the generator coordinate system and gives enough precision for bridge spam and split pushes. Each cell is roughly 2 tiles wide in-game, which captures meaningful placement differences.

### 7. Should no-ops be downsampled in BC training (~70% of frames are no-ops), or do we train on the raw imbalanced distribution?

Downsample no-ops. Keep all action frames, randomly keep 15-20% of no-op frames. Additionally, use weighted loss: `weight[no_op] = 0.3`, `weight[card_i] = 3.0`. This is simpler than focal loss and works well for moderate imbalance.

### 8. How do we handle invalid actions during online PPO rollouts - pre-sample masking, post-hoc correction, or a penalty reward?

Pre-sample masking. Before the policy samples, zero out logits for invalid actions (insufficient elixir, troop cards on enemy half, card not in hand). SB3's `MaskableMultiInputPolicy` from `sb3-contrib` supports this natively. Do NOT use post-hoc correction (wastes rollout steps) or penalty reward (introduces spurious negative signal).

---

## Q3 - STATE ENCODER: Observation Completeness

### 9. Should we add unit class identity channels per tile? If so, how many given 110+ card types (one-hot is too wide)?

Don't use one-hot (155-wide is absurd). Use category embeddings compressed to 4-8 channels via a learned lookup table. At each cell, if a unit occupies it, look up its class embedding and write it to the spatial tensor. Alternatively, group the 155 classes into ~10-15 functional categories (tank, ranged DPS, melee DPS, spell, building, swarm, etc.) and use a categorical channel. This is simpler and domain-appropriate.

### 10. Is a static binary arena_mask channel worth implementing, or does action masking alone provide sufficient spatial guidance?

Worth implementing - it is trivial. The `map_ground` and `map_fly` grids already exist in `generation_config.py:174-242`. These are 18x32 binary grids showing valid placements. Add them as 1-2 static channels in the arena observation. It costs nothing and saves the agent from wasting exploration on impossible placements.

### 11. For BC specifically, is single-frame enough, or do we need frame stacking (N=3 or 4) to give implicit motion information?

Start with single-frame. BC learns a reactive policy ("see state, do action") which works for Clash Royale because most decisions are based on current board state, not velocity. If BC performance plateaus, add 2-frame stacking (current + previous, 0.5s apart). Frame stacking adds complexity and memory for marginal gain at the BC phase. PPO may benefit more from temporal context, but cross that bridge later.

### 12. How should the 4 cards in hand be encoded - one-hot (440-dim) or learned embeddings projected to a smaller dimension?

Learned embeddings. Create a card embedding layer: `nn.Embedding(num_cards + 1, embed_dim)` with `embed_dim=16`. Each of the 4 hand slots gets mapped to a 16-dim vector, concatenated to a 64-dim card vector, then projected down to fit the vector observation. This is more expressive than one-hot and scales to different decks.

---

## Q4 - BEHAVIOR CLONING: Data Collection & Action Inference

### 13. Best spell inference strategy: HP delta region tracking, OS-level click logging, or manual labeling of spell subset?

OS-level click logging. This is the most reliable approach by far. Run a keylogger/mouse logger alongside the game that records every click with timestamp and (x, y) position. Then correlate click events with card hand changes to determine exactly which card was placed where. Inferring spell placement from HP deltas is noisy and ambiguous (multiple damage sources). Manual labeling is slow. Click logging is trivially automated and ground truth.

### 14. Should we apply horizontal arena flip augmentation to double dataset size? Any game-specific reason it breaks semantics?

Yes, flip the arena left-right. Clash Royale is symmetric across the vertical center line. Flipping doubles the dataset for free. The only thing that "breaks" is the next-card slot position, which is cosmetically on the left - but the card identities stay the same, just swap "left princess tower" and "right princess tower" labels. Implement this.

### 15. How to handle class imbalance in BC - weighted loss, oversample rare actions, cap no-op ratio, or focal loss?

Combination approach:

1. Downsample no-ops to 20% of their raw frequency during data collection
2. Use weighted cross-entropy with `weight[no_op]=0.3`, `weight[card_i]=3.0`
3. If rare card actions (e.g., spell placement) are still underrepresented, oversample those specific frames 2-3x

Don't use focal loss - it is designed for detection, not action classification, and adds a hyperparameter to tune.

### 16. Minimum viable dataset size for BC to beat a random baseline? Is there a rule of thumb for imitation learning in RTS games?

30-40 games is a reasonable target. At 5 FPS and ~3 minutes per game, that is ~27,000 frames per game, or ~900,000 total. After no-op downsampling, ~200,000 training frames. For a discrete action space of 577, roughly 500-1000 examples per commonly-used action are needed to get basic competence. With 8 card types, 30 games should give 2,000-5,000 placements per card. This should beat a random baseline. 10 games might work but is risky.

---

## Q5 - GYM ENVIRONMENT: Step Loop Timing & Synchronization

### 17. What is the right frame_skip given CV latency? Tune empirically or is there a theoretical optimum for real-time strategy pacing?

Start with `frame_skip=15` at 30 FPS capture (one decision every 0.5 seconds). This gives the CV pipeline 500ms to process, which is tight but feasible if running async. In Clash Royale, the fastest meaningful action cycle is ~1 second (play card, wait for elixir). 0.5s decisions are fast enough for all strategic play. Tune empirically: if the agent misses fast interactions, reduce to `frame_skip=10` (0.33s).

### 18. Should the CV pipeline run async and return the latest available observation, or should env.step() block until processing is done?

Yes, async. CV pipeline runs in a daemon thread/process. `env.step()` reads the latest available `GameState` from shared memory. If the CV is slow, the agent just gets a slightly stale observation - this is fine because Clash Royale changes slowly enough that a 100-200ms stale observation is barely different from real-time.

### 19. Most reliable game-over detection: YOLOv8 king tower destruction alone, or combined with OCR on the VICTORY/DEFEAT banner?

Combined. Primary: OCR on the center-screen region for "VICTORY" / "DEFEAT" text (KataCR does this with `part4` text detection). Secondary: YOLOv8 king tower destruction (king tower present in previous frame, absent in current). Tertiary: timer reaching 0:00 without overtime. Use a voting system - any 2 of 3 signals triggering = game over.

### 20. Mid-episode CV failure strategy: terminate+reset, return last-known-good state, or pause+retry with a timeout?

Return last-known-good state with a `stale` flag. If CV fails for more than 2 consecutive seconds (4 decision steps), terminate the episode with a neutral reward (0). Don't retry in a loop - it blocks the game. Don't reset mid-game - a match cannot be restarted.

---

## Q6 - REWARD FUNCTION: Signal Design & Corruption

### 21. What reward components beyond HP delta are highest-value for CR: elixir advantage, unit count differential, lane pressure score?

For MVP, stick with tower HP delta only. It is clean, directly measurable, and correlates with winning. Additional signals to consider for PPO (not BC):

- Elixir advantage: `+0.01 * (enemy_elixir_spent - our_elixir_spent)` per trade - but this requires tracking elixir spent, which is hard to observe
- Crown reward: `+5.0` per crown taken, `-5.0` per crown lost
- Don't add lane pressure or board control - these are too subjective and add reward noise

### 22. Should we add an explicit win/loss terminal reward (+1000/-1000)? If so, how to normalize it relative to per-step HP rewards?

Yes, add `+10.0` for win, `-10.0` for loss, `0.0` for draw. Scale relative to per-step HP rewards: if average per-step reward magnitude is ~0.01-0.1, a terminal reward of 10.0 creates a strong gradient toward winning. Don't go to 1000 - it overwhelms the HP shaping signal and makes credit assignment harder.

### 23. How to guard against OCR-corrupted reward - clip per-step delta to a max plausible damage value, or a sanity filter?

Clip per-step HP delta to `[-500, +500]`. No single card play can deal more than ~500 damage (Rocket does ~500 at tournament standard). If the OCR reads a delta > 500, it is a misread - clamp it. Also, add a running median filter: if the current HP delta is > 3x the median absolute delta over the last 10 frames, flag it as suspicious and use 0 instead.

### 24. Should reward shaping be added during BC, or kept simple for BC and introduced at the PPO fine-tuning stage?

Keep BC reward-free. BC is pure imitation - it doesn't use reward at all (cross-entropy loss on actions). Introduce reward shaping only at the PPO stage. This is standard practice: BC learns "what to do," PPO learns "why to do it."

---

## Q7 - PPO FINE-TUNING: Architecture, Hyperparameters & Evaluation

### 25. Recommended feature extractor for Dict{arena(18,32,16), vector(20)} in SB3 - CNN+concat, or a shared transformer?

CNN + concat. Use SB3's `CombinedExtractor` pattern:

```python
class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # Arena branch: 3-layer CNN
        # Conv2d(C, 32, 3, padding=1) -> ReLU
        # Conv2d(32, 64, 3, stride=2) -> ReLU
        # Conv2d(64, 64, 3, stride=2) -> ReLU
        # Flatten -> Linear(64*4*8, 128)
        #
        # Vector branch: Linear(N, 64) -> ReLU
        #
        # Concat: 128 + 64 = 192 -> Linear(192, 128)
```

Don't use a transformer - overkill for this observation size. CNN handles the spatial structure well.

### 26. Starting PPO hyperparameters given BC initialization: learning rate, n_steps, batch_size, entropy coeff, clip range?

```python
PPO(
    learning_rate=1e-4,       # 10x lower than default, preserve BC prior
    n_steps=2048,             # ~17 minutes of gameplay per rollout
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,           # Tight clip to prevent catastrophic forgetting
    ent_coef=0.01,            # Mild exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
)
```

Key: `clip_range=0.1` (default is 0.2) and `learning_rate=1e-4` (default is 3e-4). Both are conservative to prevent overwriting the BC initialization.

### 27. Curriculum approach: manually set Trainer AI difficulty, or use BC win rate as an automatic threshold for progression?

Manual. Start with Trainer Cheddar (easiest), graduate to Cori when win rate > 60% over 20 games. Don't automate the threshold - too few games will be run per level to get statistical significance. Play 20-30 games at each level, manually decide to advance.

### 28. How many evaluation games vs Trainer Cori for a statistically meaningful win rate, given high variance in CR outcomes?

At least 30 games. With a true win rate of 60%, 30 games gives a 95% confidence interval of roughly [42%, 76%] (binomial). That is wide but enough to distinguish "better than 50%" from "worse than 50%". For tighter bounds (55-65%), 100+ games would be needed, which is impractical early on. 30 games is the pragmatic minimum.

---

## Specific Design Questions

### How is the StateBuilder handling ResNet, YOLOv8, and OCR inputs?

Right now, it doesn't integrate ResNet at all. The StateBuilder (`state_builder.py:20-117`) only wires two things:

1. `CRDetector.detect(image)` -> `List[Detection]` -> `_process_detections()` maps these to `Tower` and `Unit` objects
2. `GameTextExtractor.extract_game_text(image)` -> `GameOCRResults` -> `_process_ocr()` fills in timer, elixir, tower HP

The `CardPredictor` from `card_classifier.py` is completely separate. It is used in `realtime_demo.py` but never wired into StateBuilder. The `GameState.cards` field exists (list of `Card` objects) but is never populated by `build_state()`.

**What needs to happen:** StateBuilder needs a third input path:

```
image -> crop 4 card slots -> CardPredictor.predict(each) -> GameState.cards
```

The card slot coordinates are already defined in `screen_regions.py` and `CLAUDE.md` (1080x1920: slots at 242-430, 445-633, 648-836, 851-1039, all y=1595-1830).

### What do ResNet, YOLOv8, and OCR inputs look like for StateBuilder and ActionBuilder?

**YOLOv8 output** (per detection): `Detection(class_id=47, class_name="hog-rider", confidence=0.87, bbox=(234, 456, 298, 534), side=-1)`. The `side` field is always -1 with the current model because it was not trained for belonging.

**OCR output**: `GameOCRResults(timer=TimerResult(minutes=2, seconds=15, total_seconds=135, is_overtime=False), elixir=7, player_king_hp=4008, enemy_left_princess_hp=2534, ...)`

**ResNet output** (per card slot): `("hog-rider", 0.97)` - a `(class_name, confidence)` tuple.

**For the ActionBuilder** (not yet built): It would consume consecutive `GameState` frames and diff them. When a card disappears from `GameState.cards` between frame t and t+1, and a new unit appears on the field, that is an action: `Action(card_slot=2, grid_x=9, grid_y=24)`.

### In KataCR, state, action, and reward are combined into one. Are we going to have this kind of implementation?

KataCR uses a `SARBuilder` that orchestrates `StateBuilder`, `ActionBuilder`, and `RewardBuilder` into a single `(state, action, reward)` tuple per frame. Their `OfflineDatasetBuilder` processes gameplay videos and saves sequences of these tuples for offline Decision Transformer training.

**Recommendation: Yes, build a SARBuilder, but simpler.** KataCR's version is overengineered for our needs because they do offline-only RL with JAX transformers. We need:

1. A `SARBuilder` for BC data collection (processing recorded gameplay into (s, a, r) tuples)
2. A `ClashRoyaleEnv(gym.Env)` for PPO online training (live step loop where state, action, reward are computed each step)

These share the same StateBuilder and reward logic but differ in how actions are determined (inferred from video diff for BC, chosen by the agent for PPO).

### How are health bars for enemy troops and towers and friendly troops and towers handled? Is it relevant for behavior cloning?

**Tower HP:** Extracted via OCR from fixed screen regions (`text_extractor.py:228-238`). The OCR reads the actual numeric HP text displayed above each tower. Tower detection tells which towers exist; OCR tells their HP values. `_process_ocr()` at line 231-247 matches OCR results to detected towers.

**Troop HP:** NOT handled at all. The current pipeline has `Unit.hp = None` always. KataCR handles troop HP through bar-item matching - they detect the small HP bar sprites above units and read them as images, not OCR. Our single YOLO model detects `bar` and `bar-level` as separate classes but does not associate them with specific units.

**Is it relevant for BC?** Tower HP is critical - it is how reward is computed (HP delta). Troop HP is nice-to-have but not essential for MVP. KataCR's bar-matching system is complex and fragile. For MVP, just know that a unit exists and where it is. The agent can learn implicitly that damaged units behave differently from the visual features.

### How are we handling excessive wait actions?

**Not handled yet.** This is one of the biggest BC data quality issues. In Clash Royale, ~70% of frames are no-ops (waiting for elixir, waiting for opponent to commit, etc.).

**Recommendation for BC:** Downsample no-ops. During BC data collection, keep ALL action frames (card placement) and randomly subsample no-op frames at a ratio. A good starting point:

- Keep 100% of action frames
- Keep 15-20% of no-op frames
- This brings the ratio to roughly 50/50 action vs no-op

Additionally, use weighted cross-entropy loss where the no-op class gets weight ~0.3 and action classes get weight ~3.0 (inversely proportional to frequency).

### How are we handling potential delays?

**CV pipeline latency:** The current pipeline is synchronous. `build_state()` calls `detector.detect()` then `ocr_extractor.extract_game_text()` sequentially. At imgsz=960 on M1 Pro, YOLOv8 takes ~65ms, OCR takes ~40-80ms per region (8 regions). Total: ~400-700ms per frame. That is 1.4-2 FPS - far too slow for real-time.

**Recommendation for the live PPO environment:**

1. Run CV in a separate thread/process that continuously captures and processes, storing the latest `GameState` in shared memory
2. `env.step()` grabs the latest available state rather than blocking
3. Use `frame_skip=15` (0.5 sec at 30 FPS) - this is enough time for most CV processing to complete
4. Skip OCR on most frames - only run OCR every 3rd or 4th frame. Timer changes by 1 second per second, elixir changes slowly. Use last-known-good values for intermediate frames.

### How are we calculating troop positions using YOLOv8 bounding boxes?

Currently, `Detection.center` returns the pixel center of the bbox: `((x1+x2)//2, (y1+y2)//2)`. The `Unit` dataclass has the same `.center` property at `game_state.py:50-53`.

To convert to grid cells for the RL observation, pixel coordinates need to map to the 18x32 cell grid that KataCR uses. From `generation_config.py`:

```python
# Grid: 18 columns x 32 rows
# Pixel bounds of the grid: (6, 64) to (562, 864) on a 568x896 image
xyxy_grids = (6, 64, 562, 864)
grid_size = (18, 32)

def pixel_to_cell(px, py, frame_w, frame_h):
    # Scale pixel to 568x896 generator space
    gx = px * 568 / frame_w
    gy = py * 896 / frame_h
    # Map to grid cell
    cell_x = int((gx - 6) / (562 - 6) * 18)
    cell_y = int((gy - 64) / (864 - 64) * 32)
    return clamp(cell_x, 0, 17), clamp(cell_y, 0, 31)
```

For the arena observation tensor `arena(18, 32, C)`, each unit's center maps to a cell, and the appropriate channel at that cell position gets incremented.

### Will we have to create a constants file for the specific video training files? Does it have to match actual RL training?

**Yes, a constants file is needed.** The OCR regions, card slot coordinates, arena bounds, and grid mapping all depend on the screen resolution. Currently there are:

- `screen_regions.py`: 540x960 base resolution with scaling
- `text_extractor.py`: 540x960 OCR regions
- `realtime_demo.py`: 1080x1920 card slot coordinates

These are inconsistent. One authoritative constants file is needed that defines:

- Screen resolution for Google Play Games on the target PC
- Arena pixel bounds (where the 18x32 grid maps to)
- Card slot pixel coordinates
- OCR region coordinates
- All expressed at the actual capture resolution

**It must match between video training and live RL.** If BC data is recorded at 1080x1920 but PPO runs at 540x960, the pixel-to-cell mapping will differ and the model will be confused. Lock in one resolution and use it everywhere.

### What will our StateBuilder and ActionBuilder output?

**StateBuilder output** (already built): A `GameState` dataclass containing:

- `time_remaining: int` (seconds)
- `is_overtime: bool`
- `elixir: int` (0-10)
- 6 tower slots with `Tower(tower_type, position, belonging, hp, bbox, confidence)`
- `units: List[Unit]` with `Unit(class_name, belonging, bbox, confidence)`
- `cards: List[Card]` (currently empty - needs ResNet integration)

**ActionBuilder output** (needs to be built): For BC data collection, it should output:

```python
@dataclass
class Action:
    card_idx: int         # 0=no-op, 1-4=card slot played
    grid_x: int           # 0-17 placement column (ignored if no-op)
    grid_y: int           # 0-31 placement row (ignored if no-op)
    discrete_id: int      # Flat index into Discrete(577):
                          #   card_idx * 144 + grid_y * 18 + grid_x,
                          #   or 576 for no-op
```

**StateEncoder output** (needs to be built): The tensor observation for the RL agent:

```python
obs = {
    "arena": np.ndarray(shape=(18, 32, C), dtype=float32),  # Spatial channels
    "vector": np.ndarray(shape=(N,), dtype=float32),         # Global features
}
```

### How are we handling capturing and storing time for states?

Currently, `GameState.timestamp` stores the frame timestamp from video processing (`state_builder.py:398`). For live capture, `time.time()` relative to episode start would be used.

**For BC data:** Store `(timestamp, GameState)` pairs at 5 FPS from recorded video. The timestamp allows computing frame deltas and inferring actions.

**For PPO:** `env.step()` returns the current state at each decision point. Time is implicit in `frame_skip`. `GameState.time_remaining` from OCR gives the in-game clock. Wall-clock time is not needed for training - the game clock is the relevant signal.

**Recommendation:** Add a `game_clock: float` field that tracks the actual in-game time from OCR, and a `wall_clock: float` for profiling. The RL agent sees `game_clock` as a feature; `wall_clock` is for debugging latency.

### Is correctly classifying unit types as ground or flying important for the state space?

**For the observation space:** Moderately important. Ground and flying units have different pathing, can cross the river differently, and interact differently with defenses (anti-air vs ground-target buildings). KataCR's `generation_config.py` already categorizes units into ground (level 1) and flying (level 2) via `level2units`.

**For action masking:** Not important. In Clash Royale, card placement rules don't depend on whether the card spawns ground or flying units - they depend on whether the card is a troop (player's half only) or a spell (anywhere).

**Recommendation:** Don't add separate ground/flying channels for MVP. The unit type identity itself (which the agent can learn from) implicitly encodes ground/flying. If unit-type channels are later added to the observation, a ground/flying bit could be included, but it is not blocking.

---

## Risk Summary

| Area | Risk | Priority |
|------|------|----------|
| OCR accuracy / domain shift | HIGH | HIGH |
| BC action inference (spells) | HIGH | HIGH |
| Spell/troop action masking | MED | HIGH |
| Env step timing / sync | MED | HIGH |
| Reward signal corruption | MED | HIGH |
| Observation completeness | MED | MED |
| PPO architecture / curriculum | LOW | MED |

---

## Implementation Order

### Phase A: Fix the perception layer (1-2 weeks)

1. Retrain YOLOv8s with belonging labels (custom NMS already ported)
2. Wire CardPredictor into StateBuilder (populate `GameState.cards`)
3. Create `constants.py` with authoritative screen coordinates for the GPC resolution
4. Add OCR staleness handling (last-known-good with decrement)

### Phase B: Build the data pipeline (1 week)

5. Build `StateEncoder` (GameState -> arena tensor + vector)
6. Build `ActionBuilder` (diff consecutive GameStates to infer actions)
7. Build `SARBuilder` (orchestrates StateBuilder + ActionBuilder for video processing)
8. Build click logger for BC recording

### Phase C: BC training (1 week)

9. Record 30+ expert games with click logging
10. Process videos through SARBuilder to create training dataset
11. Build BC model (CNN+MLP policy, cross-entropy loss, weighted sampling)
12. Train BC, validate that it beats random baseline

### Phase D: Gym environment + PPO (2 weeks)

13. Build `ClashRoyaleEnv(gym.Env)` with async CV, mss capture, PyAutoGUI execution
14. Build reward function (tower HP delta + terminal win/loss)
15. Build game-over detection (OCR + king tower)
16. Initialize PPO from BC weights, train against Trainer Cheddar

### Quick wins (can do immediately)

- Lower inference conf to 0.35 (instant recall boost, no retraining)
- Wire CardPredictor into StateBuilder (30 minutes of work)
- Create the card registry with spell/troop metadata (1 hour)
- Add `map_ground` as a static arena mask channel (trivial)

### Deferred

- Multi-resolution inference (960 + 1280)
- Frame stacking
- Domain shift fine-tuning on real GPC screenshots
- Curriculum automation
- Elixir advantage reward shaping

---

## Validation Checkpoints

| Milestone | Pass Criteria |
|-----------|---------------|
| Belonging model trained | mAP50 >= 0.75 on validation, belonging accuracy > 85% |
| Card classifier integrated | Correct card identification on 20 test screenshots, >= 90% |
| StateEncoder produces valid tensors | Unit test: known GameState -> expected tensor, no NaN/Inf |
| BC data pipeline | 30 games processed, >= 10,000 action frames extracted |
| BC model trained | Win rate > 10% vs Trainer Cheddar (random is ~2%) |
| Gym env stable | 10 consecutive episodes complete without crash |
| PPO training | Reward curve trending upward over 100k steps |
| PPO evaluation | Win rate > 40% vs Trainer Cheddar over 30 games |
