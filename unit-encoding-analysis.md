# Unit Encoding Deep Analysis: 7-Channel vs Archetypes vs Learned Embeddings

## CS175 - Clash Royale RL Agent
## February 2026

---

## Executive Summary

The state space cannot distinguish between unit types. A Mega Knight and a Skeleton both map to `CH_ALLY_GROUND += 1.0`. This document analyzes three approaches to fix this, including a deep investigation of how KataCR actually solves the problem (it is not what our previous analysis assumed).

**Key finding from this analysis:** KataCR does NOT use sum-embeddings. It uses a **PositionFinder collision avoidance** algorithm that guarantees exactly one unit per cell, then stores a single integer class ID per cell. This fundamentally changes the comparison.

**Recommendation:** Use archetype channels for BC. The embedding approach is not "too costly" in parameters (only 1,240 extra), but it requires solving the multi-unit-per-cell problem, which KataCR solves with collision avoidance. Adopting KataCR's collision avoidance in our encoder is more work than adding archetype channels, and archetypes provide sufficient signal for BC-level imitation learning.

---

## The Problem

Current `_encode_arena()` in `state_encoder.py:169-224`:

```python
for unit in state.units:
    cx, cy = unit.center
    col, row = pixel_to_cell(cx, cy, fw, fh)
    unit_type = UNIT_TYPE_MAP.get(unit.class_name, "ground")
    is_ally = unit.belonging == 0

    if unit_type == "spell":
        arena[row, col, CH_SPELL] += 1.0
    elif unit_type == "flying":
        ch = CH_ALLY_FLYING if is_ally else CH_ENEMY_FLYING
        arena[row, col, ch] += 1.0
    else:  # ground
        ch = CH_ALLY_GROUND if is_ally else CH_ENEMY_GROUND
        arena[row, col, ch] += 1.0
```

Every unit's `class_name` (e.g., "mega-knight", "skeleton", "musketeer") is reduced to one of 5 categories via `UNIT_TYPE_MAP`. The information lost:

| Game State | What Model Sees | Correct Response | Model's Response |
|------------|----------------|------------------|-----------------|
| Enemy PEKKA pushing | 1.0 at CH_ENEMY_GROUND | Swarm defense (skeleton army) | "A ground unit is coming" |
| Enemy Skeleton Army pushing | 15.0 at CH_ENEMY_GROUND | Splash spell (arrows) | "Many ground units coming" |
| Enemy Balloon flying | 1.0 at CH_ENEMY_FLYING | Anti-air defense (musketeer) | "A flying unit is coming" |
| Enemy Hog Rider at bridge | 1.0 at CH_ENEMY_GROUND | Building to distract (goblin cage) | Same as PEKKA |

The model CAN distinguish "many small units" from "one large unit" via count. But it cannot distinguish "one PEKKA" from "one Hog Rider" -- both are 1.0 in the same channel. This matters because the correct defense is completely different.

---

## Option A: Current 7-Channel Encoding (Baseline)

### What it is

```
Arena shape: (32, 18, 7)
Channels: ally_ground, ally_flying, enemy_ground, enemy_flying,
          ally_tower_hp, enemy_tower_hp, spell
Info per cell: unit count (float) by side and movement type
```

### Parameter cost in a 3-layer CNN

```
Conv2d(7, 32, 3x3):    7 * 9 * 32 + 32      =   2,048
Conv2d(32, 64, 3x3):   32 * 9 * 64 + 64     =  18,496
Conv2d(64, 64, 3x3):   64 * 9 * 64 + 64     =  36,928
                                        Total: ~57,472 CNN params
```

### What it can and cannot learn

**Can learn:**
- Push strength (high count = strong push)
- Push location (which lane, which side)
- Tower presence and health
- Ground vs flying composition
- "Something is happening at the bridge"

**Cannot learn:**
- Counter-play decisions (what EXACTLY to play against what)
- Tank vs DPS vs swarm distinction
- When to save a specific spell for a specific threat
- Building targeting units (hog, balloon) vs everything else

### When is this sufficient?

For a BC model that needs to beat a random baseline (>10% win rate against Trainer Cheddar), this may be enough. The expert demonstration data carries strong signal even with coarse observations:
- Expert always plays arrows when count is high in a cluster
- Expert always plays goblin cage when a single unit is at the bridge
- These spatial+count heuristics get basic behavior right

**Assumption:** This has NOT been validated empirically. The 10% win rate claim is a reasonable estimate based on BC literature, not a measured result. If BC with 7 channels fails to beat random, unit identity is likely the bottleneck.

---

## Option B: Archetype Channels (Recommended for BC)

### What it is

Add 4-6 new channels to the arena that encode functional role instead of unit identity. Each unit maps to exactly one archetype via a lookup table.

### Proposed archetype groups

Based on Clash Royale gameplay mechanics, not arbitrary groupings. Each archetype represents a strategic role that requires a specific counter-strategy:

**1. Tank** - High HP units that absorb damage. Counter: sustained DPS or swarm.
```python
"tank": [
    "giant", "golem", "golemite", "lava-hound", "mega-knight",
    "pekka", "electro-giant", "royal-giant", "royal-giant-evolution",
    "ice-golem", "goblin-giant", "giant-skeleton",
    "elixir-golem-big", "elixir-golem-mid", "elixir-golem-small",
    "golden-knight", "monk",
]
```

**2. Ranged DPS** - Ranged attackers that provide support behind a tank. Counter: spell or melee rush.
```python
"ranged": [
    "musketeer", "wizard", "ice-wizard", "electro-wizard",
    "archer", "archer-evolution", "princess", "dart-goblin",
    "magic-archer", "firecracker", "firecracker-evolution",
    "executioner", "hunter", "witch", "mother-witch",
    "flying-machine", "baby-dragon", "inferno-dragon",
    "mega-minion", "sparky", "bowler",
]
```

**3. Swarm** - Many cheap units. Counter: splash damage or spell.
```python
"swarm": [
    "skeleton", "skeleton-evolution", "bat", "bat-evolution",
    "goblin", "spear-goblin", "minion", "barbarian",
    "barbarian-evolution", "rascal-boy", "rascal-girl",
    "guard", "royal-recruit", "royal-recruit-evolution",
    "royal-hog", "wall-breaker", "wall-breaker-evolution",
    "fire-spirit", "electro-spirit", "ice-spirit",
    "ice-spirit-evolution", "heal-spirit", "skeleton-barrel",
    "goblin-barrel",
]
```

**4. Win Condition** - Units that directly threaten towers. Counter: building or direct counter.
```python
"win_con": [
    "hog", "hog-rider", "balloon", "miner",
    "ram-rider", "battle-ram", "battle-ram-evolution",
    "lumberjack", "bandit", "royal-ghost",
    "goblin-drill", "graveyard",
]
```

**5. Building** - Defensive structures. Counter: spell or ignore.
```python
"building": [
    "cannon", "cannon-cart", "tesla", "tesla-evolution",
    "inferno-tower", "bomb-tower", "mortar", "mortar-evolution",
    "x-bow", "furnace", "tombstone", "goblin-cage",
    "barbarian-hut", "goblin-hut", "elixir-collector",
]
```

### Arena encoding with archetypes

Two encoding strategies:

**Strategy 1: Separate ally/enemy per archetype (10 new channels)**
```
Arena shape: (32, 18, 17)
Channels 0-6:  existing (ally/enemy ground/flying, tower HP, spell)
Channel 7:   ally_tank count
Channel 8:   enemy_tank count
Channel 9:   ally_ranged count
Channel 10:  enemy_ranged count
Channel 11:  ally_swarm count
Channel 12:  enemy_swarm count
Channel 13:  ally_win_con count
Channel 14:  enemy_win_con count
Channel 15:  ally_building count
Channel 16:  enemy_building count
```

**Strategy 2: Combined (no ally/enemy split, 5 new channels)**
```
Arena shape: (32, 18, 12)
Channels 0-6:  existing
Channel 7:   tank count (both sides - already have ally/enemy from ch 0-3)
Channel 8:   ranged count
Channel 9:   swarm count
Channel 10:  win_con count
Channel 11:  building count
```

**Recommendation: Strategy 2 (12 channels).** The ally/enemy split is already captured in channels 0-3. Adding archetype info on top doesn't need a second ally/enemy split -- the model can combine "enemy_ground = high" + "swarm = high" to infer "enemy ground swarm push."

### Parameter cost (12 channels)

```
Conv2d(12, 32, 3x3):   12 * 9 * 32 + 32     =   3,488  (was 2,048)
Conv2d(32, 64, 3x3):   32 * 9 * 64 + 64     =  18,496  (same)
Conv2d(64, 64, 3x3):   64 * 9 * 64 + 64     =  36,928  (same)
                                         Total: ~58,912 CNN params
                                         Delta: +1,440 params (+2.5%)
```

Adding 5 archetype channels increases total CNN parameters by **2.5%**. This is negligible.

### Implementation effort

```python
# encoder_constants.py -- add this block
UNIT_ARCHETYPE_MAP: dict[str, str] = {}
_archetype_defs = {
    "tank": ["giant", "golem", "golemite", ...],
    "ranged": ["musketeer", "wizard", ...],
    "swarm": ["skeleton", "bat", ...],
    "win_con": ["hog", "hog-rider", "balloon", ...],
    "building": ["cannon", "tesla", ...],
}
for arch, units in _archetype_defs.items():
    for unit_name in units:
        UNIT_ARCHETYPE_MAP[unit_name] = arch

# state_encoder.py -- add to _encode_arena()
ARCHETYPE_TO_CHANNEL = {
    "tank": 7, "ranged": 8, "swarm": 9, "win_con": 10, "building": 11,
}
archetype = UNIT_ARCHETYPE_MAP.get(unit.class_name)
if archetype:
    arena[row, col, ARCHETYPE_TO_CHANNEL[archetype]] += 1.0
```

Total implementation: ~30 lines of code changes. No new dependencies. No model architecture changes. No retraining of YOLO. Works with the existing SB3 CombinedExtractor out of the box.

### Units that don't fit cleanly

| Unit | Ambiguity | Proposed Assignment | Rationale |
|------|-----------|-------------------|-----------|
| Hog Rider | Win condition + melee | win_con | Primary threat is tower damage |
| Valkyrie | Splash + melee DPS | swarm (counter) or ranged | Assign to ranged (she is used as splash defense) |
| Dark Prince | Melee DPS + splash | ranged | Splash capability makes him a support unit |
| Cannon Cart | Building + ranged | building | Spawns as building first |
| Princess | Ranged + building (behind bridge) | ranged | Primary role is splash DPS from range |
| Goblin Barrel | Swarm + win condition | win_con | It targets towers directly |

**Assumption:** Some assignments are judgment calls. Changing a unit's archetype after BC training requires retraining. Choose assignments that match the counter-play logic most relevant to your deck (Royal Hogs / Royal Recruits).

### What archetypes enable that 7-channel doesn't

| Situation | 7-Channel Model's View | 12-Channel Model's View |
|-----------|----------------------|------------------------|
| Enemy PEKKA at bridge | enemy_ground=1 | enemy_ground=1, tank=1 |
| Enemy Skeleton Army | enemy_ground=15 | enemy_ground=15, swarm=15 |
| Enemy Hog at bridge | enemy_ground=1 | enemy_ground=1, win_con=1 |
| Enemy Inferno Tower | enemy_ground=1 | enemy_ground=1, building=1 |

Now the BC model can learn:
- "tank=1 at bridge" -> play swarm units (royal recruits) to surround
- "swarm=high at bridge" -> play arrows
- "win_con=1 at bridge" -> play goblin cage to distract
- "building=1 behind their tower" -> spell it or ignore it

These distinctions are impossible with 7 channels alone.

---

## Option C: Learned Embeddings (KataCR Approach)

### What we previously assumed KataCR does (WRONG)

Previous analysis suggested:
- Store unit_class_id per cell
- Use sum-embeddings when multiple units share a cell
- Add nn.Embedding(155, 8) layer

**This is not what KataCR actually does.**

### What KataCR actually does

From `KataCR/katacr/policy/offline/dataset.py:143-226`:

**Step 1: PositionFinder collision avoidance**

```python
class PositionFinder:
    def __init__(self, r=32, c=18):
        self.used = np.zeros((r, c), np.bool_)

    def find_near_pos(self, xy):
        y, x = np.clip(int(xy[1]), 0, 31), np.clip(int(xy[0]), 0, 17)
        if self.used[y, x]:
            # Cell occupied -- find nearest FREE cell
            avail_center = self.center[~self.used]
            map_index = np.argwhere(~self.used)
            dis = scipy.spatial.distance.cdist(...)
            y, x = map_index[np.argmin(dis)]
        self.used[y, x] = True
        return np.array((x, y))
```

KataCR **never puts two units in the same cell**. If the target cell is occupied, it finds the nearest unoccupied cell using spatial distance. Every unit gets its own cell.

**Step 2: Per-cell feature vector (386 dimensions)**

```python
arena = np.zeros((32, 18, 386), np.int32)
for info in state['unit_infos']:
    xy = pos_finder.find_near_pos(info['xy'])
    pos = arena[xy[1], xy[0]]
    pos[0] = info['cls']   # Integer class ID (0-154)
    pos[1] = info['bel']   # Belonging (-1 or 1)
    pos[2:194] = info['bar1']   # Health bar 1 (24x8 px grayscale)
    pos[194:386] = info['bar2'] # Health bar 2
```

Each cell stores: 1 class ID integer + 1 belonging flag + 384 health bar pixels.

**Step 3: Decision Transformer with learned embeddings**

From `KataCR/katacr/policy/offline/dt.py:120-186`:

```python
cls, bel = arena[..., 0], arena[..., 1]
cls = Embed(n_unit + 1, 8)(cls)    # (B, T, 32, 18, 8) -- learned embedding
bel = bel[..., None]                # (B, T, 32, 18, 1)
bar1 = CNN(bar_cfg)(bar1)           # (B, T, 32, 18, 3) -- CNN on health bar
bar2 = CNN(bar_cfg)(bar2)           # (B, T, 32, 18, 3)
arena = concatenate([cls, bel, bar1, bar2], -1)  # (B, T, 32, 18, 15)
```

The final arena representation is **(32, 18, 15)** per timestep: 8-dim class embedding + 1 belonging + 6 health bar features.

### Why KataCR's approach works for them but not us

| Factor | KataCR | Our Project |
|--------|--------|-------------|
| Multi-unit cells | Avoided via PositionFinder | Common (we use count channels) |
| Health bars | Detected and stored as raw pixels | Not detected (Unit.hp = None always) |
| Model architecture | Decision Transformer (JAX) | SB3 PPO/BC (PyTorch) |
| Training regime | Offline RL on replay data | Online RL (PPO) + offline imitation (BC) |
| Data volume | Large replay dataset | 30-50 games |
| YOLO output | 7-column (cls + belonging) | 6-column (no belonging) |

**The collision avoidance assumption is the critical difference.** Our count-based encoding (`+= 1.0`) handles multiple units per cell naturally. KataCR's integer-ID encoding requires exactly one unit per cell. We'd need to adopt PositionFinder to use KataCR's approach, which changes the fundamental nature of our arena representation.

### If we adopted collision avoidance + embeddings

**Arena representation change:**

Before:
```
Cell (5, 20) with 3 allied ground units:
  arena[20, 5, CH_ALLY_GROUND] = 3.0   # Count
```

After (KataCR style):
```
Cell (5, 20): unit A (e.g., royal-recruit, class_id=97)
Cell (5, 21): unit B (displaced, nearest free cell)
Cell (6, 20): unit C (displaced, nearest free cell)
```

**Problems with collision avoidance in our pipeline:**

1. **Spatial noise**: Displacing units to nearby cells introduces positional error. A cluster of 5 skeletons that should be in one cell gets spread across 5 cells, each showing 1 unit. The spatial pattern no longer reflects reality.

2. **Depends on processing order**: If unit A occupies cell (5,20), unit B gets displaced to (5,21). If unit B was processed first, it gets (5,20) and unit A gets displaced. The representation is order-dependent.

3. **Density information lost**: Our count channel tells the model "3 units are clumped here." Collision avoidance says "1 unit here, 1 unit there, 1 unit there." The model must learn to reconstruct density from spatial patterns, which is harder.

4. **Incompatible with our action mask**: Our action mask operates on the 18x32 grid as a placement space. If units are displaced from their true positions, the model's spatial understanding of "where to place cards" is corrupted.

### Embedding approach WITHOUT collision avoidance (sum-embeddings)

If we keep count-based encoding but add embeddings via summation:

```python
# For each unit in the cell, look up embedding and sum
for unit in state.units:
    col, row = pixel_to_cell(unit.center[0], unit.center[1], fw, fh)
    class_id = CLASS_NAME_TO_ID[unit.class_name]
    embedding = embedding_table[class_id]  # (8,) vector
    arena[row, col, 7:15] += embedding     # Sum into channels 7-14
```

**Failure modes of sum-embeddings:**

| Failure | Example | Impact |
|---------|---------|--------|
| **Non-uniqueness** | E[skeleton] * 3 = E[mega-knight]? Sum can accidentally alias | Model confuses unit compositions |
| **Scale collapse** | 10 skeletons: sum of 10 embeddings has large magnitude. 1 PEKKA: small magnitude. Model may learn to only use magnitude, ignoring direction | Loses identity, becomes a count proxy |
| **Order invariance is correct** | A+B = B+A. This is actually fine for CR (no ordering in same cell) | Not a problem |
| **Dense battle instability** | Late-game: 8-12 units in 3-4 cells. Sum magnitudes spike | Training instability, gradient issues |

**Mitigation: Mean instead of sum**

```python
# Track count per cell
count = np.zeros((32, 18), dtype=np.float32)
for unit in state.units:
    col, row = pixel_to_cell(...)
    arena[row, col, 7:15] += embedding_table[class_id]
    count[row, col] += 1
# Normalize
mask = count > 0
arena[:, :, 7:15][mask] /= count[mask, None]
```

Mean-embedding is more stable than sum but still loses composition information ("1 PEKKA + 1 skeleton" averages to a mid-range embedding that doesn't represent either unit well).

### SB3 custom feature extractor for embeddings

This is feasible but requires replacing SB3's default CombinedExtractor:

```python
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 192):
        super().__init__(observation_space, features_dim=features_dim)

        arena_shape = observation_space["arena"].shape
        n_channels = arena_shape[-1]  # 7 or 12 or 15

        # Arena: 3-layer CNN
        self.arena_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with th.no_grad():
            sample = th.zeros(1, n_channels, 32, 18)
            cnn_out = self.arena_cnn(sample).shape[1]

        self.arena_linear = nn.Sequential(
            nn.Linear(cnn_out, 128), nn.ReLU()
        )

        # Vector: simple MLP
        vec_dim = observation_space["vector"].shape[0]
        self.vector_mlp = nn.Sequential(
            nn.Linear(vec_dim, 64), nn.ReLU()
        )

        self._features_dim = 128 + 64  # 192

    def forward(self, obs: dict) -> th.Tensor:
        # Arena: permute (B, H, W, C) -> (B, C, H, W) for Conv2d
        arena = obs["arena"].permute(0, 3, 1, 2)
        arena_feat = self.arena_linear(self.arena_cnn(arena))

        vec_feat = self.vector_mlp(obs["vector"])
        return th.cat([arena_feat, vec_feat], dim=1)
```

Usage with MaskablePPO:

```python
from sb3_contrib import MaskablePPO

model = MaskablePPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=dict(
        features_extractor_class=CRFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=192),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    ),
)
```

**Parameter cost with embeddings:**

```
Conv2d(15, 32, 3x3):    15 * 9 * 32 + 32    =   4,352
Conv2d(32, 64, 3x3):    32 * 9 * 64 + 64    =  18,496
Conv2d(64, 64, 3x3):    64 * 9 * 64 + 64    =  36,928
Linear(cnn_out, 128):   depends on pooling    =  ~8,320
Embedding(155, 8):       155 * 8              =   1,240
Vector MLP:              23 * 64 + 64         =   1,536
                                          Total: ~70,872 CNN params
                                          Delta vs 7-ch: +13,400 (+23%)
```

The 23% increase comes primarily from the first Conv2d layer processing more input channels (15 vs 7), not from the embedding itself. The embedding (1,240 params) is 1.7% of total.

---

## Head-to-Head Comparison

### Parameter costs

| Approach | Arena Channels | CNN Params (first layer) | Embedding Params | Total Policy Params | Delta from Baseline |
|----------|---------------|-------------------------|-----------------|--------------------|--------------------|
| 7-channel (baseline) | 7 | 2,048 | 0 | ~500K | -- |
| 12-channel (archetypes) | 12 | 3,488 | 0 | ~501K | +0.3% |
| 17-channel (split archetypes) | 17 | 4,928 | 0 | ~503K | +0.6% |
| 15-channel (embeddings, no split) | 15 | 4,352 | 1,240 | ~504K | +0.8% |

**All approaches have negligible parameter overhead.** The policy head (Linear(features_dim, 2305)) dominates at ~450K params regardless. The encoding choice is about information quality, not computational cost.

### Data efficiency

| Approach | Info Per Unit | Frames Needed for Convergence | Rationale |
|----------|-------------|-------------------------------|-----------|
| 7-channel | Type only | ~50K | Simple features, fast learning |
| 12-channel archetypes | Type + role | ~75K | Slightly more channels to correlate |
| 15-channel embeddings | Type + identity | ~150K+ | Embeddings for rare units need many examples |

**Assumption:** These convergence estimates are rough. Real convergence depends on BC loss, training hyperparameters, and data quality. The relative ordering (7ch < archetype < embedding) is confident.

With 50 games at 2 FPS over 3 minutes, you get ~18,000 frames per game, or ~900,000 raw frames. After no-op downsampling to 20%, that's ~180,000 training frames. This is sufficient for archetypes. It is borderline for embeddings -- rare units (Mega Knight appears in maybe 5 of 50 games) will have embeddings trained on very few examples.

### Implementation effort

| Approach | Code Changes | New Dependencies | Architecture Changes | Time Estimate |
|----------|-------------|-----------------|---------------------|--------------|
| 7-channel | None | None | None | 0 |
| 12-channel archetypes | ~30 lines (constants + encode) | None | None | 1-2 hours |
| 15-channel embeddings (sum) | ~80 lines + custom extractor | None (PyTorch built-in) | Custom feature extractor | 4-6 hours |
| Full KataCR (collision avoidance + embeddings) | ~200 lines + custom extractor + PositionFinder | scipy (for spatial distance) | Major encoder rewrite | 8-12 hours |

### Information quality

| Scenario | 7-ch | 12-ch Archetype | 15-ch Embedding |
|----------|------|----------------|-----------------|
| Distinguish PEKKA from Skeleton | NO | YES (tank vs swarm) | YES (unique embedding) |
| Distinguish PEKKA from Golem | NO | NO (both tank) | YES (different embeddings) |
| Distinguish Hog from Mega Knight | NO | YES (win_con vs tank) | YES |
| Distinguish Musketeer from Wizard | NO | NO (both ranged) | YES |
| Distinguish Arrows from Zap | NO | NO (both spell) | YES |
| Count units in a cell | YES | YES | Partial (sum collapses count) |
| Position accuracy | Exact | Exact | Exact (sum) or displaced (collision avoidance) |

**Key insight:** Archetypes lose within-archetype distinctions (PEKKA vs Golem are both "tank"). But these within-archetype distinctions rarely affect counter-play for our deck. Against both PEKKA and Golem, the correct response is similar: deploy swarm + DPS behind.

Embeddings theoretically capture everything, but sum-embeddings degrade the signal when multiple units share a cell, which is common during battles. The theoretical advantage of embeddings is partially canceled by the practical limitation of sum aggregation.

---

## The Real Tradeoffs

### Cost is NOT the differentiator

All three approaches have nearly identical parameter counts (~500K total). "Too costly" was a mischaracterization. The real costs are:

1. **Implementation complexity**: Archetypes require a lookup table (trivial). Embeddings require a custom SB3 feature extractor (moderate). KataCR-style requires PositionFinder + custom extractor (significant).

2. **Data efficiency**: Archetypes work with 50 games. Embeddings need more data for rare units.

3. **Debugging transparency**: If BC fails with archetypes, you can inspect which archetype channels fired and verify correctness. With embeddings, you have opaque 8-dim vectors that require visualization tools (t-SNE, PCA) to understand.

### The actual decision matrix

| | Implementation | Data Needs | Debuggability | BC Phase | PPO Phase |
|--|---------------|-----------|--------------|----------|-----------|
| 7-channel | Already done | Low | Full | Minimum viable | Insufficient |
| Archetypes | 1-2 hours | Low | Full (inspect channels) | **Strong** | Good |
| Embeddings | 4-6 hours | Medium-High | Low (opaque) | Marginal improvement over archetypes | **Strong** |

### Archetype groups are sufficient for BC counter-play

The strategic decisions that matter for our Royal Hogs / Royal Recruits deck:

| Enemy Situation | Correct Response | Archetype Signal | Embedding Signal |
|----------------|-----------------|-----------------|-----------------|
| Tank + support push | Royal Recruits (surround the tank) | tank=1, ranged=1 | Knows exact tank type (marginal benefit) |
| Swarm push | Arrows or Barbarian Barrel | swarm=high | Knows exact swarm type (marginal benefit) |
| Win condition at bridge | Goblin Cage to distract | win_con=1 | Knows exact win con (marginal benefit) |
| Building placed | Zappies to outrange or ignore | building=1 | Knows exact building (marginal benefit) |
| Splash unit behind tank | Fly over with Flying Machine | ranged=1 (splash is a subcategory of ranged) | Knows it's splash specifically (real benefit) |

In 4 of 5 cases, the archetype signal provides the same actionable information as the full embedding. Only for splash-vs-ranged-within-ranged does the embedding provide genuinely different guidance. This is not worth the complexity trade-off for BC.

---

## Recommended Implementation Path

### Phase 1: BC with archetypes (now)

1. Add `UNIT_ARCHETYPE_MAP` to `encoder_constants.py`
2. Add 5 archetype channels to `_encode_arena()` in `state_encoder.py`
3. Update `NUM_ARENA_CHANNELS` from 7 to 12
4. Update `StateEncoder.__init__()` observation space shape
5. Update `encoder/CLAUDE.md` documentation
6. Run existing smoke tests (they should catch shape mismatches)

**Estimated effort: 1-2 hours**

### Phase 2: PPO with archetypes (after BC validates)

If BC achieves >10% win rate with archetypes, continue to PPO with the same encoding. The archetype channels provide enough signal for PPO exploration to learn counter-strategies.

### Phase 3: Embeddings (if PPO plateaus)

If PPO win rate plateaus below 40%, upgrade to learned embeddings:
1. Write `CRFeatureExtractor` custom class
2. Switch from count channels to mean-embedding channels
3. Initialize embeddings by archetype (same initial vector for all "tank" units)
4. Fine-tune from PPO checkpoint

This preserves the archetype-level knowledge while adding within-archetype discrimination.

---

## What about units not in any archetype?

The label_list has 155 classes. Not all of them appear in real games with our deck. Units that don't fit archetypes:

| Category | Units | Handling |
|----------|-------|---------|
| UI elements | bar, clock, emote, bar-level, etc. (11 units) | Already filtered out by UNIT_TYPE_MAP -> "other" |
| Towers | king-tower, queen-tower, etc. (4 units) | Already handled by tower channels |
| Spells | arrows, fireball, zap, etc. (16 units) | Keep in existing CH_SPELL (no archetype needed) |
| Evolution variants | archer-evolution, barbarian-evolution, etc. | Same archetype as base unit |
| Summoned sub-units | golemite, lava-pup, phoenix-egg, etc. | Same archetype as parent |

**Units without clear archetype assignment:**

```python
# These default to the most conservative archetype
_unassigned = {
    "bomb": "swarm",          # Projectile from bomber, treat as temporary threat
    "axe": "ranged",          # Executioner's thrown axe
    "dirt": "swarm",          # Visual effect
    "clone": "swarm",         # Clone spell effect creates copies
    "rage": "swarm",          # Rage spell visual (doesn't need archetype)
    "freeze": "swarm",        # Freeze spell visual
    "poison": "swarm",        # Poison spell visual
    "lightning": "swarm",     # Lightning spell visual
    "goblin-ball": "win_con", # From goblin barrel
    "skeleton-dragon": "ranged", # Flying ranged unit
}
```

**Assumption:** Some of these assignments are debatable. The impact is minimal because these units are rarely the primary threat in a game state. Getting tank/swarm/win_con right is 80% of the counter-play value. Edge cases in spell visual effects or sub-units contribute very little.

---

## Assumptions Summary

1. **BC can learn counter-play from archetype signals.** Not empirically validated. Based on the reasoning that expert demonstrations paired with archetype-level observations provide sufficient signal for imitation learning. If BC fails with archetypes, unit identity may not be the bottleneck (it could be card hand detection, belonging accuracy, or observation staleness).

2. **50 games is enough data for archetypes but borderline for embeddings.** Based on the calculation that rare units appear in ~5 of 50 games. With mean-embeddings, the embedding for a rare unit trains on ~500 frames (5 games x 100 post-downsample frames with that unit). 500 frames is thin for learning a useful 8-dim representation. Archetypes don't have this problem because "tank" appears in nearly every game.

3. **Sum-embeddings degrade when cells are dense.** Based on the theoretical analysis that sum/mean of multiple embeddings loses composition information. Not empirically tested in our pipeline. The severity depends on how often 3+ units share a cell (common during battle engagements, rare during setup phases).

4. **Archetype assignments are correct.** Based on Clash Royale gameplay knowledge. Some assignments are judgment calls (Valkyrie as ranged vs melee, Dark Prince as melee vs splash). Incorrect assignments mislead the BC model but can be corrected by updating the lookup table and retraining.

5. **KataCR's PositionFinder approach is incompatible with our count-based encoding.** This is a design choice, not a hard constraint. We COULD switch to collision avoidance, but it would require rewriting `_encode_arena()`, changing the observation space semantics, and validating that displaced unit positions don't confuse the model.

6. **Parameter cost is not the real bottleneck.** All three approaches have ~500K total policy parameters. The bottleneck is data quality (perception accuracy, belonging heuristic, card hand detection) and data quantity (50 games for BC).

7. **The custom SB3 feature extractor code provided has NOT been tested.** It is based on SB3 documentation and source code patterns. Integration testing is required before training.
