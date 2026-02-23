# StateEncoder Analysis: Design Questions and Technical Deep-Dive

## CS175 - Clash Royale RL Agent
## February 2026

---

## Q1. "Encoder is the bridge between perception and learning - used identically by behavior cloning and PPO." How do you go from BC to PPO? Can PPO start without BC?

### What "used identically" means

The StateEncoder is a **stateless conversion layer** with two pure functions:

```python
obs  = encoder.encode(game_state)       # {"arena": (32,18,7), "vector": (23,)}
mask = encoder.action_mask(game_state)   # bool (2305,)
```

Both BC and PPO call these exact same functions to convert a `GameState` into tensors. The encoder does not know or care which training paradigm is consuming its output. The observation space `Dict(arena, vector)` and action space `Discrete(2305)` are defined once in the encoder and shared by both.

What changes between BC and PPO is **everything around the encoder** - the data source, the loss function, and the training loop. The encoder itself is untouched.

### The BC-to-PPO workflow

**Phase 1: Behavior Cloning (offline)**
1. Collect 30-50 expert games using click logger + StateBuilder
2. For each frame: `obs = encode(state)`, `mask = action_mask(state)`, `action = from_click_log()`
3. Train an SB3 `MaskableMultiInputPolicy` with cross-entropy loss on the expert's actions
4. BC does NOT use reward. It is pure supervised learning: "given this observation, the expert chose this action"
5. Output: a trained policy network (CNN for arena + MLP for vector + action head)

**Phase 2: PPO Fine-Tuning (online)**
1. **Load the BC-trained weights** into a new PPO agent's policy network
2. Build a `ClashRoyaleEnv(gym.Env)` that calls `encode()` and `action_mask()` each step
3. Train PPO with **conservative hyperparameters** to preserve BC knowledge:
   - `learning_rate=1e-4` (10x lower than default 3e-4)
   - `clip_range=0.1` (tighter than default 0.2)
   - These prevent catastrophic forgetting of BC behavior
4. PPO collects rollouts by playing the actual game, computing reward from tower HP deltas
5. The policy gradually improves beyond what BC learned, discovering strategies the expert didn't use

```
BC Phase                          PPO Phase
--------                          ---------
Recorded gameplay                 Live gameplay
  |                                 |
  v                                 v
StateBuilder -> GameState         StateBuilder -> GameState
  |                                 |
  v                                 v
StateEncoder.encode()    <----  IDENTICAL  ---->   StateEncoder.encode()
StateEncoder.action_mask()                         StateEncoder.action_mask()
  |                                 |
  v                                 v
(obs, mask, expert_action)        (obs, mask, agent_action, reward)
  |                                 |
  v                                 v
Cross-entropy loss                PPO policy gradient + value loss
  |                                 |
  v                                 v
Trained policy weights  ------>  Initialize PPO policy
```

### Can PPO start without BC?

**Technically yes. Practically no.**

PPO from random initialization would:
- Spend thousands of game steps discovering that no-op is sometimes better than random plays
- Waste exploration on trivially bad actions (playing a 7-cost card when you have 2 elixir - though the mask prevents this)
- Take 3-4x longer to reach the same competence level
- Struggle with credit assignment because early games are essentially random

BC provides a warm start. The BC policy already knows:
- When to wait vs play (elixir management)
- Where to place troops (bridge, behind king tower, etc.)
- Which cards to use against common pushes

PPO builds on this foundation. Skipping BC means PPO has to discover all of this from scratch through trial and error against the game AI.

**The CLAUDE.md statement means:** The encoder's output format is designed so that the same policy network architecture works for both BC and PPO. You train BC first, save the weights, load them into PPO, and keep training. The encoder doesn't change - only the training loop changes.

---

## Q2. Why isn't unit identity a channel? How does the BC model differentiate between units?

### Current encoding - what the model sees

The arena has 7 channels. When a unit (say, a Mega Knight) occupies grid cell (5, 20):

```
arena[20, 5, CH_ALLY_GROUND] += 1.0    # Mega Knight: ground, ally
```

When a different unit (say, a Skeleton) also occupies (5, 20):

```
arena[20, 5, CH_ALLY_GROUND] += 1.0    # Now equals 2.0 - two ground allies
```

The model sees "2 allied ground units at (5, 20)". It does NOT see "Mega Knight + Skeleton at (5, 20)". A Mega Knight and a Skeleton are completely indistinguishable in the current encoding. So are a Musketeer and a Witch. So are Arrows and Fireball.

### What information IS available to the model

The model is not completely blind to unit context. It has:

1. **Spatial patterns**: Different units appear in different locations. Hog Rider appears near bridges. Elixir Collector appears behind the king tower. The CNN can learn location-based heuristics.

2. **Unit count**: Multiple skeletons in a cell increment the count. A single Mega Knight is 1.0. This gives rough DPS/threat estimation but not identity.

3. **Card hand** (vector indices 15-18): The model knows which 4 cards it currently holds, encoded as class indices. So it knows "I have arrows in slot 2" even though it can't see "the enemy pushed with skeleton army."

4. **Temporal signal** (if frame stacking is added later): Unit movement speed and spawn patterns differ. But current encoding is single-frame.

### Why this matters for BC

Consider this game situation:
- Enemy plays Skeleton Army (15 small ground units)
- Expert player responds with Arrows (3-cost spell, perfect counter)

What the model sees:
```
arena: CH_ENEMY_GROUND cells spiking (15 units across 3-4 cells)
vector: card slot 0 = "arrows" (3 elixir)
```

What the model learns: "When enemy_ground count is high in a cluster, play the spell in my hand." This is a coarse heuristic that works for Arrows vs Skeleton Army because the count is distinctive (15 units in a cluster is unusual). But it fails to distinguish:
- 15 Skeletons (Arrows is great) vs 6 Royal Recruits (Arrows is terrible - waste of elixir)
- A single PEKKA (needs sustained DPS defense) vs a single Hog Rider (needs immediate distraction)

### Where unit identity SHOULD be handled

Unit identity is a known gap acknowledged in `docs/pipeline-design-qa.md` Q9:

> "Don't use one-hot (155-wide is absurd). Use category embeddings compressed to 4-8 channels via a learned lookup table... Alternatively, group the 155 classes into ~10-15 functional categories and use a categorical channel."

The current 7-channel encoding was an explicit decision to start simple:
1. Build the full pipeline end-to-end
2. Validate that BC works at all with coarse observations
3. Add unit identity channels as a targeted improvement

**It is NOT handled elsewhere.** The vector observation encodes card hand identity (indices 15-18), but arena unit identity is lost at encoding time. The `Unit.class_name` field exists in `GameState` but `_encode_arena()` deliberately discards it.

### Will BC work without unit identity?

**For the MVP: likely yes, with limitations.** BC is imitation learning - it copies the expert's behavior given similar-looking states. Even with coarse observations:

- Expert always uses Arrows against swarm pushes (high unit count) - model learns this
- Expert always places Royal Hogs at the bridge - model learns spatial pattern
- Expert waits when enemy has a tank + support combo - model learns from count + timing

But the model will fail at fine-grained tactical decisions that require knowing WHICH unit is on the field. This is acceptable for MVP. The goal is "better than random" (>10% win rate), not optimal play.

**For PPO: unit identity becomes more important.** The agent needs to learn counter-strategies that depend on knowing whether the threat is a PEKKA (play swarm) vs a Mega Knight (play air units). This is when adding archetype channels or embeddings becomes worthwhile.

---

## Q3. Detection confusion: What happens if royal-hogs are detected as hog or hog-rider?

### The three "hog" classes in the model

The YOLOv8s detection model has 155 classes. Three of them are hog-related:

| Class | label_list.py | UNIT_TYPE_MAP | AP50 | Description |
|-------|---------------|---------------|------|-------------|
| `hog` | ground_unit_list line 113 | "ground" | Not in deck eval | Generic hog unit (summoned by hog-rider card but also from other interactions) |
| `hog-rider` | ground_unit_list line 94 | "ground" | 0.869 | The troop: rider sitting on a hog |
| `royal-hog` | ground_unit_list line 279 | "ground" | 0.995 | One of the 4 hogs from the Royal Hogs card |

All three map to `"ground"` in `UNIT_TYPE_MAP`, so they all end up in the same arena channel: `CH_ALLY_GROUND` (0) or `CH_ENEMY_GROUND` (2).

### What happens with misdetection

**Scenario: 4 royal-hogs on the field, detected as 1 hog-rider**

This is a realistic concern because royal-hog and hog-rider are visually similar (both are hog-shaped units, hog-rider has a rider on top).

| | True State | Misdetected State |
|--|-----------|------------------|
| arena[row, col, CH_ALLY_GROUND] | 4.0 (four royal-hogs) | 1.0 (one hog-rider) |
| Unit count in cell | 4 | 1 |
| Unit type channel | CH_ALLY_GROUND | CH_ALLY_GROUND (same) |
| Action mask | Unaffected | Unaffected |
| Vector observation | Unaffected | Unaffected |

**Impact on the agent:**
- The agent sees 1 ally ground unit instead of 4. This is a significant undercount.
- The agent might think its push is weak and unnecessarily reinforce it (wasting elixir).
- The action mask is unaffected because it only depends on card hand + elixir, not on-field units.
- The card hand still correctly shows "royal-hogs" or its absence (CardPredictor is separate from YOLO detection).

**Scenario: Enemy's hog-rider detected as hog or royal-hog**

| | True State | Misdetected State |
|--|-----------|------------------|
| arena[row, col, CH_ENEMY_GROUND] | 1.0 (one hog-rider) | 1.0 (same - still one ground unit) |

Under the **current 7-channel encoding**, this misdetection is invisible. A hog-rider and a royal-hog both increment `CH_ENEMY_GROUND` by 1.0. The model cannot distinguish them anyway, so class confusion within the same type (ground) has zero effect on the observation tensor.

### When does it matter?

Class confusion matters in the current encoding **only when it changes the unit type category or count**:

| Confusion | Type Change | Impact |
|-----------|------------|--------|
| royal-hog -> hog-rider | ground -> ground | **None** (same channel) |
| royal-hog -> hog | ground -> ground | **None** (same channel) |
| flying-machine -> baby-dragon | flying -> flying | **None** (same channel) |
| electro-spirit -> hog-rider | ground -> ground | **None** (same channel, same count) |
| skeleton -> bat | ground -> flying | **Channel flip** (incorrect type) |
| arrows -> fireball | spell -> spell (but fireball is also flying!) | Depends on UNIT_TYPE_MAP priority |

The real risk is **count errors**, not class errors. If YOLO detects only 1 of 4 royal-hogs (misses 3), the count drops from 4.0 to 1.0 regardless of what class name the detection gets. This is a recall problem, not a confusion problem.

### Actual detection quality for our deck

From the v12 evaluation:
- `royal-hog`: AP50 = 0.995 (near-perfect, Tier 1)
- `hog-rider`: AP50 = 0.869 (adequate, Tier 2)

These are NOT in a confusion cluster per the evaluation report. The model distinguishes them well. The actual weak spots in our deck are:
- `skeleton`: AP50 = 0.670 (small, hard to detect)
- `ice-spirit`: AP50 = 0.644 (very small, easily missed)

### Assumption stated

**Assumption:** I'm assuming that YOLO misclassification between hog classes (hog, hog-rider, royal-hog) is infrequent based on the evaluation metrics. However, the per-class AP50 measures detection, not class confusion specifically. A dedicated confusion matrix analysis on real gameplay frames would give a more precise answer. The evaluation did identify confusion clusters (wizard family, barbarian family, spirit family), and the hog family was NOT among them.

---

## Q4. Archetype channels vs learned embeddings vs current encoding

### Option A: Current 7-channel encoding (what we have)

```
Channels: ally_ground, ally_flying, enemy_ground, enemy_flying,
          ally_tower_hp, enemy_tower_hp, spell
Total: 7 channels
Info: Unit count by side + type. No identity.
```

**Pros:**
- Already implemented and tested
- Simplest possible encoding
- Minimal parameters in CNN feature extractor
- Works for coarse tactical decisions (push strength, defense needed)

**Cons:**
- Cannot distinguish between units of the same type
- BC model cannot learn counter-plays (e.g., "play arrows when you see skeleton army")
- Spell channel is aggregated (all spells look identical)

### Option B: Archetype channels (4-5 functional groups) -- RECOMMENDED FOR BC

Add 4-5 archetype channels to the arena, each representing a functional role. Define archetypes in the constants file:

```python
UNIT_ARCHETYPES = {
    # Tanks: high HP, damage sponges
    "tank": ["giant", "golem", "golemite", "lava-hound", "mega-knight",
             "pekka", "electro-giant", "royal-giant", "royal-giant-evolution",
             "knight", "knight-evolution", "ice-golem", "goblin-giant",
             "elixir-golem-big", "elixir-golem-mid", "elixir-golem-small",
             "giant-skeleton", "goblin-cage", "barbarian-hut", ...],

    # Ranged DPS: ranged attackers
    "ranged": ["musketeer", "wizard", "ice-wizard", "electro-wizard",
               "archer", "archer-evolution", "princess", "dart-goblin",
               "magic-archer", "firecracker", "firecracker-evolution",
               "executioner", "hunter", "witch", "mother-witch",
               "flying-machine", "baby-dragon", "inferno-dragon",
               "mega-minion", ...],

    # Melee DPS: single-target melee threats
    "melee": ["mini-pekka", "prince", "dark-prince", "lumberjack",
              "bandit", "valkyrie", "valkyrie-evolution", "elite-barbarian",
              "golden-knight", "mighty-miner", "ram-rider",
              "royal-ghost", "little-prince", ...],

    # Swarm: many small units
    "swarm": ["skeleton", "skeleton-evolution", "bat", "bat-evolution",
              "barbarian", "barbarian-evolution", "goblin", "spear-goblin",
              "rascal-boy", "rascal-girl", "royal-recruit",
              "royal-recruit-evolution", "royal-hog",
              "guard", "minion", "wall-breaker", "wall-breaker-evolution",
              "fire-spirit", "electro-spirit", "ice-spirit",
              "ice-spirit-evolution", "heal-spirit", ...],

    # Building: defensive structures
    "building": ["inferno-tower", "bomb-tower", "cannon", "tesla",
                 "tesla-evolution", "mortar", "mortar-evolution",
                 "x-bow", "furnace", "tombstone", "elixir-collector",
                 "goblin-drill", "goblin-hut", ...],
}
```

New arena encoding: **(32, 18, 12)** -- 7 existing + 5 archetype channels

| Channel | Name | Description |
|---------|------|-------------|
| 0-6 | (existing) | ally_ground, ally_flying, enemy_ground, enemy_flying, ally_tower, enemy_tower, spell |
| 7 | tank | Unit count of tanks in cell (both sides combined, or split ally/enemy) |
| 8 | ranged | Unit count of ranged DPS |
| 9 | melee | Unit count of melee DPS |
| 10 | swarm | Unit count of swarm units |
| 11 | building | Unit count of buildings |

**Alternatively, split by side:** (32, 18, 17) -- 7 existing + 5 archetypes x 2 sides = 17 total channels. This is more expressive but makes the CNN feature extractor larger.

**Pros:**
- The BC model can now distinguish "3 swarm units in a cluster" (use spell) from "1 tank pushing" (use swarm/melee counter)
- Archetype definitions are domain knowledge that helps the model learn faster
- Simple to implement (add a lookup dict, add 5 `+= 1.0` operations in `_encode_arena`)
- No learned parameters needed (pure engineering, no retraining)
- Interpretable: you can inspect the archetype channels and understand what the agent sees
- Minimal computational overhead

**Cons:**
- Manual archetype assignment requires Clash Royale domain knowledge
- Some units don't fit cleanly (Hog Rider is melee but targets buildings; Cannon Cart is both building and ranged)
- Misclassifying an archetype is worse than not having archetypes (misinformation > no information)
- Adding archetypes later requires retraining BC from scratch (observation space changed)

**Recommendation: Use archetypes for BC.** They provide the right level of abstraction for imitation learning. The model doesn't need to know "this is a Musketeer" - it needs to know "this is a ranged threat" so it can learn "play Goblin Cage to distract ranged threats." Five archetypes capture 80% of strategic decision-making in Clash Royale.

### Option C: Learned embeddings (KataCR approach)

Store a `unit_class_id` integer per cell. Add a learned `nn.Embedding(155, embed_dim)` layer at the start of the network. The embedding layer converts class IDs to dense vectors, which become additional arena channels.

**Implementation sketch:**

```python
class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)

        # Arena observation now includes a class_id channel
        # shape: (32, 18, 8) -- 7 original + 1 class_id (integer)
        self.unit_embedding = nn.Embedding(156, 8)  # 155 classes + 1 for "empty"

        # After embedding lookup, arena becomes (32, 18, 7 + 8) = (32, 18, 15)
        self.arena_cnn = nn.Sequential(
            nn.Conv2d(15, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),
        )
        # ...rest of feature extractor
```

**The problem:** Multiple units can occupy the same cell. The current encoding handles this by incrementing counts (`+= 1.0`). But an embedding represents a single unit's identity. If two different units are in the same cell, which class_id do you embed?

**Solutions to the multi-unit-per-cell problem:**
1. **Sum embeddings**: Embed each unit separately, sum the vectors per cell. Preserves multi-unit info but loses count.
2. **Most-dangerous-unit**: Only embed the highest-threat unit per cell. Loses minor units.
3. **Separate layers per unit**: Track up to K units per cell with K embedding channels. Complex, memory-intensive.

**Pros:**
- Most expressive: learned features that can capture any unit relationship
- Scales to different decks without manual archetype design
- Embeddings can learn nuances (PEKKA is a tank but also high single-target DPS)

**Cons:**
- Requires a **custom SB3 feature extractor** (cannot use `MaskableMultiInputPolicy` out of the box with embedding layers)
- Multi-unit-per-cell problem adds complexity
- Embedding layer adds 155 x 8 = 1,240 new parameters (small, but they need to converge)
- Requires more data to train well (embeddings for rare units may never converge with 30-50 games)
- **Overkill for BC with 30-50 games.** BC is supervised learning on a small dataset. Learned embeddings shine when you have large datasets and long training (PPO's thousands of episodes).

**Is it too costly for our simple BC model?** The parameter cost is trivial (1,240 params vs ~100K+ in the CNN). The real costs are:

1. **Engineering cost**: Writing a custom `CRFeatureExtractor` instead of using SB3 defaults. This is 2-4 hours of work.
2. **Data efficiency cost**: With only 30-50 games (~200K frames after downsampling), rare unit embeddings won't see enough examples. If Mega Knight appears in 5 of 50 games, the embedding for class "mega-knight" trains on ~500 frames - not enough for reliable learning.
3. **Debugging cost**: If BC underperforms, is it the embedding layer or the training? Archetypes are transparent; embeddings are opaque.

**Can the model be modified to allow embeddings?** Yes. SB3 supports custom feature extractors via `policy_kwargs`:

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=dict(
        features_extractor_class=CRFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    ),
)
```

This works for both BC and PPO. The custom extractor handles the embedding lookup internally.

### Recommendation summary

| Approach | Complexity | Data Efficiency | BC Phase | PPO Phase |
|----------|------------|-----------------|----------|-----------|
| Current 7-channel | Low | High | Sufficient for MVP | Insufficient |
| Archetype channels (4-5 groups) | Low | High | **Recommended** | Good |
| Learned embeddings (4-8 dim) | Medium | Low-Medium | Overkill | **Recommended** |

**Path forward:**
1. Add archetype channels now (for BC training)
2. If BC reaches >10% win rate, move to PPO with the same archetypes
3. If PPO plateaus, upgrade to learned embeddings (initialize from archetype-informed values)

This gives incremental improvement without committing to the most complex approach upfront.

---

## Q5. Does pixel_to_cell account for the arena's X bounds? Does it need to?

### Current implementation

From `src/encoder/coord_utils.py` lines 53-70:

```python
def norm_to_cell(x_norm: float, y_norm: float) -> tuple[int, int]:
    # Y: accounts for arena bounds
    arena_y_frac = (y_norm - ARENA_Y_START_FRAC) / _ARENA_Y_SPAN

    # X: maps full screen width directly to columns
    col = _clamp(int(x_norm * GRID_COLS), 0, GRID_COLS - 1)
    row = _clamp(int(arena_y_frac * GRID_ROWS), 0, GRID_ROWS - 1)
    return col, row
```

**Y-axis:** Correctly applies `ARENA_Y_START_FRAC` (50/960 = 0.0521) and `ARENA_Y_END_FRAC` (750/960 = 0.7813) to map only the arena portion of the screen to grid rows.

**X-axis:** Maps full normalized range [0, 1] directly to columns [0, 17] with NO X offset or margin.

### Does the arena have X bounds?

**From `screen_regions.py` (540x960 base resolution):**
```python
arena: Region = Region(0, 50, 540, 750, "arena")
#                      x1  y1   x2   y2
```

The arena spans **x = 0 to x = 540** -- the **full screen width**. There is no horizontal margin.

**From `generation_config.py` (synthetic data, 568x896):**
```python
background_size = (568, 896)
xyxy_grids = (6, 64, 562, 864)  # the grid in synthetic image coords
```

The synthetic data generator uses a 568-wide canvas with 6px margins on each side (grid X spans [6, 562]). But this is the generator's coordinate system, NOT the real game's.

### The two coordinate systems

| System | Width | Grid X Bounds | Used For |
|--------|-------|---------------|----------|
| Real game (540x960 or 1080x1920) | 540 / 1080 | Full width [0, 540] / [0, 1080] | StateEncoder, pixel_to_cell |
| Synthetic generator (568x896) | 568 | [6, 562] with margins | YOLO training data generation |

**These are separate systems.** The synthetic generator creates images at 568x896 for YOLO training. The encoder processes real gameplay screenshots at 540x960 (or 1080x1920 scaled). They never share coordinates. YOLO detections are output as pixel bounding boxes relative to the **input image's** resolution, regardless of what the model was trained on.

### Does pixel_to_cell need X bounds?

**No. Here is why:**

1. In the real game, the arena spans the full screen width. There is no left/right margin between the arena and the screen edge. Units can be detected anywhere from x=0 to x=540.

2. The 18-column grid should map to the full arena width (= full screen width). Column 0 is the far left of the arena, column 17 is the far right.

3. The current formula `col = clamp(int(x_norm * 18), 0, 17)` correctly maps:
   - x_norm = 0.0 (left edge) -> col 0
   - x_norm = 0.5 (center) -> col 9
   - x_norm = 1.0 (right edge) -> col 17

### Will it affect states?

**No.** Since the arena spans the full screen width, there is no misalignment. A unit detected at x=270 (center of 540-wide screen) maps to col 9, which is the center of the 18-column grid. This is correct.

If the arena had horizontal margins (say, x=[30, 510] on a 540-wide screen), then `pixel_to_cell` would incorrectly map units near the edges. But the Clash Royale arena fills the entire width of the viewport in Google Play Games.

### Will it affect training?

**No.** For BC training, the encoder processes real gameplay screenshots. The YOLO model detects units and returns pixel coordinates relative to the input image. `pixel_to_cell` converts these to grid cells. Since the arena spans the full width, the conversion is accurate.

The 6px margin in `generation_config.py` only affects how synthetic training images are laid out. It does NOT affect how the encoder interprets real game detections.

### Edge case: units at screen edges

In Clash Royale, units can walk along the very edge of the arena. A unit at x=5 (nearly at the left edge) maps to:
- x_norm = 5 / 540 = 0.0093
- col = clamp(int(0.0093 * 18), 0, 17) = clamp(0, 0, 17) = 0

A unit at x=535 (nearly at the right edge) maps to:
- x_norm = 535 / 540 = 0.9907
- col = clamp(int(0.9907 * 18), 0, 17) = clamp(17, 0, 17) = 17

Both are correctly handled by the clamp. No X-bounds adjustment is needed.

### Subtle inconsistency with the generator (documented, not a bug)

The generator places units on its 18x32 grid using X bounds [6, 562] on a 568-wide image. This means synthetic unit positions have a slightly different spatial distribution than real game positions. Specifically:

- Generator: column 0 maps to x=6 (not x=0)
- Real game: column 0 maps to x=0

This is a ~1% difference at the edges (6/568 = 1.06%). It introduces a tiny domain gap in unit positioning between synthetic training data and real gameplay. This is negligible compared to the 4.7x classification domain gap already documented.

**Conclusion:** `pixel_to_cell` is correct for real game screenshots. No X-bounds adjustment is needed.

---

## Assumptions

1. **Arena spans full screen width.** Confirmed from `screen_regions.py`: arena Region(0, 50, 540, 750) has x1=0, x2=540 (full width). If the game window has decorative borders or padding, this assumption breaks. Google Play Games on PC does not add borders to the game viewport.

2. **YOLO detection coordinates are in input image space.** Standard YOLO behavior: output bboxes are in pixel coordinates of the input image, not the training image. Ultralytics confirms this.

3. **Hog family classes are NOT frequently confused.** Based on v12 evaluation (royal-hog AP50=0.995, hog-rider AP50=0.869), these are well-detected and not in a confusion cluster. However, a per-class confusion matrix on real gameplay has NOT been generated. The AP50 metric measures detection quality, not confusion rates specifically. Cross-class confusion could exist at lower confidence thresholds.

4. **50 games is planned, not existing.** Only 3 video files exist in the repository. The 50-match dataset must be recorded.

5. **BC can work with coarse observations.** This is the assumption behind the 7-channel encoding. BC copies expert behavior given "similar" states. If the observation is too coarse (many game states that require different actions map to the same tensor), BC performance degrades. The 7-channel encoding is the minimum viable representation. Adding archetype channels improves this.

6. **Archetype assignments are approximate.** Some units don't fit cleanly into one category (Hog Rider is a fast single-target melee unit that only attacks buildings - is it "melee" or should it be its own category?). The archetype groups proposed above are informed by Clash Royale gameplay knowledge but may need iteration.

7. **Frame resolution at inference time.** The analysis assumes screenshots are captured at 540x960 or 1080x1920 (matching `screen_regions.py`). If the game window is resized or a different resolution is used, all coordinate conversions need recalibration.

8. **CardPredictor integration is still pending.** The analysis of "what the model sees" assumes `GameState.cards` will be populated (card hand visible in vector observation). Currently `GameState.cards` is always empty, which means the model sees zero card information - a much worse situation than described above. This is a critical blocker that must be fixed before any training.
