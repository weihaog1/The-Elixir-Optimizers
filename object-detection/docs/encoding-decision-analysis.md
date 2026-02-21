# Unit Encoding Decision: Full-Pipeline Analysis

## CS175 - The Elixir Optimizers
## February 2026

---

## 1. Project Context

This analysis informs a single decision: **how to encode unit identity in the arena observation tensor**. The decision is critical because it locks in the observation space for the entire training pipeline - BC, PPO, and beyond.

### 1.1 Project Goals (from proposal)

| Tier | Goal | Implication for Encoding |
|------|------|--------------------------|
| **Minimum** | End-to-end pipeline with BC agent showing basic competence (contextually reasonable card placements) | Encoding must be simple enough to work with limited BC data (~50 games) |
| **Realistic** | BC warm-start + PPO fine-tuning, >60% win rate against Trainer AI | Encoding must carry through from BC to PPO without observation space changes |
| **Moonshot** | Competitive at higher difficulty, transfer learning across deck archetypes | Encoding must generalize beyond the current 8-card deck |

### 1.2 Training Pipeline

```
Phase 1: Behavior Cloning (offline, ~50 games)
  - Supervised learning on expert demonstrations
  - Cross-entropy loss: "given this observation, the expert chose this action"
  - Data: ~180K frames after no-op downsampling
  - Output: trained policy weights
        |
        | Load BC weights into PPO
        v
Phase 2: PPO Fine-Tuning (online, unlimited games)
  - Model-free on-policy RL
  - Agent plays live games, collects reward from tower HP deltas
  - Conservative hyperparameters (lr=1e-4, clip_range=0.1) to preserve BC knowledge
  - Output: improved policy
        |
        | (Moonshot) Retrain with different deck
        v
Phase 3: Transfer Learning (aspirational)
  - Same architecture, different deck cards
  - Encoding must represent units in a deck-agnostic way
```

### 1.3 The Critical Constraint

**The observation space cannot change between phases.** If BC trains on `(32, 18, 7)` arena and PPO trains on `(32, 18, 12)` arena, the CNN weights from BC cannot initialize PPO. The first convolutional layer expects a specific number of input channels.

This means the encoding we choose now must serve all three phases. Choosing "good enough for BC" and planning to upgrade later means throwing away BC weights at the transition.

### 1.4 Current State

| Component | Status |
|-----------|--------|
| YOLOv8s detection model | Trained, mAP50=0.804. 155 classes, 30.2 detections/frame |
| StateBuilder | Functional. Extracts GameState from detections + OCR. CardPredictor NOT wired in (blocking) |
| StateEncoder | Implemented. Arena (32, 18, 7) + vector (23,). Action space Discrete(2305) |
| CardPredictor | Trained (~25K params, 8-class deck). Not integrated into StateBuilder |
| Click logger | Designed, not implemented |
| DatasetBuilder | Designed, not implemented |
| BC training | Not started |
| PPO training | Not started |

---

## 2. The Problem

### 2.1 What the Current Encoder Loses

`state_encoder.py:182-203` reduces every unit to one of 5 categories via `UNIT_TYPE_MAP`:

```python
for unit in state.units:
    unit_type = UNIT_TYPE_MAP.get(unit.class_name, "ground")  # "ground", "flying", "spell", "tower", "other"
    is_ally = unit.belonging == 0
    if unit_type == "ground":
        arena[row, col, CH_ALLY_GROUND if is_ally else CH_ENEMY_GROUND] += 1.0
```

The `class_name` string (e.g., "mega-knight", "skeleton", "musketeer") is discarded. After encoding, the model has no access to unit identity.

### 2.2 Concrete Failures

| Game Situation | What Actually Matters | What the Model Sees | Correct Response | Model Can Distinguish? |
|---------------|----------------------|--------------------|-----------------|-----------------------|
| Enemy PEKKA at bridge | High HP tank, needs sustained DPS | enemy_ground = 1 | Deploy swarm (royal recruits) | NO - same as any single ground unit |
| Enemy Skeleton Army | Low HP swarm, needs splash | enemy_ground = 15 | Arrows or barbarian barrel | YES - high count signals swarm |
| Enemy Hog Rider at bridge | Fast win-condition, needs immediate distraction | enemy_ground = 1 | Goblin cage to pull | NO - identical to PEKKA |
| Enemy Balloon crossing river | Air win-condition, needs anti-air | enemy_flying = 1 | Flying machine to intercept | NO - same as any single flying unit |
| Enemy Inferno Tower | Building, counters tanks | enemy_ground = 1 | Zappies to outrange or ignore | NO - identical to PEKKA |
| Enemy Royal Recruits push | Multi-unit ground, needs splash or surround | enemy_ground = 6 | Arrows or barbarian barrel | PARTIAL - count helps but same signal as 6 skeletons |

**In 4 of 6 common game situations, the model cannot choose the correct response because it cannot distinguish between unit types with the same count.**

### 2.3 Why This Matters Differently for BC vs PPO

**BC (Phase 1):** The model copies expert behavior. If the expert always plays arrows against high enemy_ground counts, the model learns "high count = arrows" regardless of whether it's skeletons or royal recruits. This heuristic works ~60% of the time because swarms are the most common high-count scenario. BC can achieve basic competence without unit identity.

**PPO (Phase 2):** The agent must learn counter-strategies through trial and error. If it cannot distinguish PEKKA from Hog Rider, it cannot learn that PEKKA requires sustained DPS while Hog Rider requires immediate distraction. PPO exploration cannot discover correct counter-play when the observation conflates fundamentally different threats. The agent will plateau at a level determined by how often the coarse observation happens to suggest the correct response.

**Transfer (Phase 3/Moonshot):** Deck-specific behavior collapses entirely. A new deck has different cards with different roles. Without unit identity in the observation, the model has no representation of "what the opponent is playing" that transfers across decks. A model trained against Hog Rider decks has no learned representation to reuse when facing Golem decks if both look like "enemy_ground = 1".

---

## 3. The Three Encoding Options

### 3.1 Option A: Current 7-Channel Encoding (Baseline)

```
Arena shape: (32, 18, 7)
Channels: CH_ALLY_GROUND, CH_ALLY_FLYING, CH_ENEMY_GROUND, CH_ENEMY_FLYING,
          CH_ALLY_TOWER_HP, CH_ENEMY_TOWER_HP, CH_SPELL
Per-cell info: unit count (float) by side and movement type
```

**What it encodes:** How many ground/flying units are in each cell, on which side, plus tower HP and spell presence.

**What it discards:** All unit identity. A Mega Knight (7 elixir, 350 DPS, 5000 HP) and a Skeleton (1/15 of 3 elixir, 100 DPS, 80 HP) both produce `+= 1.0` in the same channel.

**CNN parameter cost (3-layer):**
```
Conv2d(7, 32, 3x3):    7 * 9 * 32 + 32    =  2,048
Conv2d(32, 64, 3x3):  32 * 9 * 64 + 64    = 18,496
Conv2d(64, 64, 3x3):  64 * 9 * 64 + 64    = 36,928
                                      Total: 57,472
```

**Suitability per phase:**
| Phase | Rating | Reasoning |
|-------|--------|-----------|
| BC | Marginal | Basic heuristics work (count-based decisions) but cannot learn counter-play |
| PPO | Insufficient | Cannot discover counter-strategies when threats are indistinguishable |
| Transfer | Impossible | No deck-agnostic unit representation |

### 3.2 Option B: Archetype Channels

Add 5 channels encoding the functional role of each unit. Every unit in the 155-class label list is assigned to exactly one archetype via a static lookup table. Archetypes are based on Clash Royale strategic roles:

| Archetype | Strategic Role | Counter-Strategy | Example Units |
|-----------|---------------|-----------------|---------------|
| **Tank** | High HP, absorbs damage | Sustained DPS or swarm surround | giant, golem, pekka, mega-knight, lava-hound |
| **Ranged** | Ranged attacker, support behind tank | Spell or rush the support | musketeer, wizard, archer, baby-dragon, sparky |
| **Swarm** | Many cheap units, overwhelm by numbers | Splash damage or area spell | skeleton, goblin, minion, barbarian, bat |
| **Win Condition** | Directly threatens towers | Building to distract, or direct counter | hog-rider, balloon, miner, ram-rider, goblin-drill |
| **Building** | Defensive structure, pulls or damages | Spell it or ignore | cannon, tesla, inferno-tower, bomb-tower, x-bow |

**Two sub-options for channel layout:**

**Strategy 2 (recommended): 5 archetype channels, no ally/enemy split. Arena shape: (32, 18, 12)**

```
Channels 0-6:   existing (ally_ground, ally_flying, enemy_ground, enemy_flying,
                          ally_tower_hp, enemy_tower_hp, spell)
Channel 7:      tank count in cell (both sides)
Channel 8:      ranged count in cell (both sides)
Channel 9:      swarm count in cell (both sides)
Channel 10:     win_con count in cell (both sides)
Channel 11:     building count in cell (both sides)
```

The ally/enemy split is already captured by channels 0-3. The CNN can combine "enemy_ground = 1" + "tank = 1" to infer "enemy ground tank" without redundant ally/enemy archetype channels.

**Strategy 1 (alternative): 10 archetype channels, ally/enemy split. Arena shape: (32, 18, 17)**

Separate ally/enemy per archetype. More expressive but doubles the archetype parameter cost with marginal benefit since channels 0-3 already encode side.

**CNN parameter cost (Strategy 2, 12 channels):**
```
Conv2d(12, 32, 3x3):  12 * 9 * 32 + 32    =  3,488
Conv2d(32, 64, 3x3):  32 * 9 * 64 + 64    = 18,496
Conv2d(64, 64, 3x3):  64 * 9 * 64 + 64    = 36,928
                                       Total: 58,912
                                       Delta: +1,440 (+2.5% over baseline)
```

**What it enables:**

| Situation | 7-Channel View | 12-Channel View |
|-----------|---------------|-----------------|
| Enemy PEKKA at bridge | enemy_ground=1 | enemy_ground=1, tank=1 -> deploy swarm |
| Enemy Skeleton Army | enemy_ground=15 | enemy_ground=15, swarm=15 -> play arrows |
| Enemy Hog at bridge | enemy_ground=1 | enemy_ground=1, win_con=1 -> play goblin cage |
| Enemy Inferno Tower | enemy_ground=1 | enemy_ground=1, building=1 -> spell or ignore |
| Enemy Wizard behind Giant | enemy_ground=2 | enemy_ground=2, tank=1, ranged=1 -> deal with support |

**Suitability per phase:**
| Phase | Rating | Reasoning |
|-------|--------|-----------|
| BC | Strong | Archetypes provide the right abstraction level for imitation learning |
| PPO | Good | PPO can learn archetype-specific counter-strategies through exploration |
| Transfer | Good | Archetypes are deck-agnostic ("tank" means "tank" regardless of which deck contains it) |

**Limitations:**
- Cannot distinguish within-archetype units (PEKKA vs Golem are both "tank")
- Requires manual archetype assignment (domain knowledge needed)
- Misassigning an archetype is worse than no archetype (misinformation > no information)

### 3.3 Option C: Learned Embeddings

Store a class_id per cell and use a learned `nn.Embedding(155, dim)` layer in a custom SB3 feature extractor to convert class IDs to dense vectors.

**The multi-unit-per-cell problem:**

Our encoder uses count channels (`+= 1.0`) that naturally handle multiple units in one cell. An embedding represents a single unit's identity. When two different units share a cell, there is no clean way to embed both.

**Three sub-approaches:**

| Approach | Method | Problem |
|----------|--------|---------|
| **Sum-embedding** | Embed each unit, sum vectors per cell | Sum of 3 skeletons may equal sum of 1 mega-knight. Non-unique. Scale collapse with many units |
| **Mean-embedding** | Embed each unit, average per cell | Loses composition info. "PEKKA + Skeleton" averages to a vector that represents neither |
| **Collision avoidance (KataCR)** | Guarantee one unit per cell via PositionFinder | Displaces units from true positions. Order-dependent. Loses density information |

### 3.3.1 What KataCR Actually Does

From `KataCR/katacr/policy/offline/dataset.py:143-226`:

**Step 1: PositionFinder** guarantees exactly one unit per cell. If the target cell is occupied, it finds the nearest free cell using spatial distance (scipy). This is not an approximation - every unit gets its own cell.

**Step 2: Per-cell feature vector (386 dimensions)**
```python
arena = np.zeros((32, 18, 386), np.int32)
for info in state['unit_infos']:
    xy = pos_finder.find_near_pos(info['xy'])
    pos = arena[xy[1], xy[0]]
    pos[0] = info['cls']         # Integer class ID (0-154)
    pos[1] = info['bel']         # Belonging (-1 or 1)
    pos[2:194] = info['bar1']    # Health bar 1 (24x8 px grayscale)
    pos[194:386] = info['bar2']  # Health bar 2
```

**Step 3: Decision Transformer with learned embeddings**
```python
cls = Embed(n_unit + 1, 8)(cls)    # (B, T, 32, 18, 8)
bel = bel[..., None]                # (B, T, 32, 18, 1)
bar1 = CNN(bar_cfg)(bar1)           # (B, T, 32, 18, 3)
bar2 = CNN(bar_cfg)(bar2)           # (B, T, 32, 18, 3)
arena = concatenate([cls, bel, bar1, bar2], -1)  # (B, T, 32, 18, 15)
```

Final arena representation: **(32, 18, 15)** per timestep.

### 3.3.2 Why We Cannot Copy KataCR's Approach

| Factor | KataCR | Our Project |
|--------|--------|-------------|
| Multi-unit cells | Avoided via PositionFinder collision avoidance | Common (we use count channels) |
| Health bars | Detected and stored as raw pixel data (384 dims) | Not detected (Unit.hp is always None for non-towers) |
| Model architecture | Decision Transformer (JAX, Flax) | SB3 PPO/BC (PyTorch) |
| YOLO output | 7-column (cls + belonging bit) | 6-column (no belonging) |
| Training regime | Offline RL on large replay dataset | Online RL (PPO) + small offline dataset (BC) |
| Data volume | Large replay corpus | 50 games for BC, unlimited for PPO |

**The collision avoidance assumption is the critical difference.** Our count-based encoding handles overlapping units naturally. Switching to collision avoidance would:
1. Introduce spatial noise (units displaced from true positions)
2. Create order-dependent representations (processing order changes the grid)
3. Lose density information (cluster of 5 skeletons spread across 5 cells)
4. Corrupt the shared action/observation grid (the 18x32 grid serves both as "what's here" and "where to place")

### 3.3.3 Sum/Mean Embedding Without Collision Avoidance

If we keep count-based encoding but add embeddings via summation/averaging:

```python
for unit in state.units:
    col, row = pixel_to_cell(unit.center[0], unit.center[1], fw, fh)
    class_id = CLASS_NAME_TO_ID[unit.class_name]
    embedding = embedding_table[class_id]  # (8,) vector
    arena[row, col, 7:15] += embedding     # Sum into channels 7-14
```

**Sum-embedding failure modes:**

| Failure Mode | Example | Severity |
|-------------|---------|----------|
| Non-uniqueness | E[skeleton]*3 could alias to E[mega-knight]*1 | High - model confuses compositions |
| Scale collapse | 10 skeletons = large magnitude sum; 1 PEKKA = small magnitude. Model learns to use magnitude as a count proxy, ignoring direction | Medium - reduces to count channel with noise |
| Dense battle instability | Late-game: 8-12 units in 3-4 cells. Sum magnitudes spike unpredictably | Medium - training instability |
| Rare unit under-training | Mega Knight appears in ~5 of 50 BC games. Its embedding trains on ~500 frames | High for BC - embedding may not converge |

**Mean-embedding is more stable** but loses composition info ("PEKKA + Skeleton" averages to a mid-range vector representing neither unit).

### 3.3.4 SB3 Custom Feature Extractor

Learned embeddings require replacing SB3's default `CombinedExtractor`:

```python
class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 192):
        super().__init__(observation_space, features_dim=features_dim)

        # Learned embedding for unit class IDs
        self.unit_embedding = nn.Embedding(156, 8)  # 155 classes + 1 "empty"

        arena_shape = observation_space["arena"].shape
        n_base_channels = arena_shape[-1]  # 7 or 12

        # Arena CNN processes base channels + embedded channels
        total_channels = n_base_channels + 8  # +8 from embedding
        self.arena_cnn = nn.Sequential(
            nn.Conv2d(total_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, total_channels, 32, 18)
            cnn_out_dim = self.arena_cnn(sample).shape[1]

        self.arena_linear = nn.Sequential(nn.Linear(cnn_out_dim, 128), nn.ReLU())

        vec_dim = observation_space["vector"].shape[0]
        self.vector_mlp = nn.Sequential(nn.Linear(vec_dim, 64), nn.ReLU())

        self._features_dim = 128 + 64  # 192

    def forward(self, obs: dict) -> th.Tensor:
        arena = obs["arena"].permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
        # Split base channels and class_id channel
        base = arena[:, :7, :, :]
        class_ids = arena[:, 7, :, :].long()  # (B, 32, 18) integer
        embedded = self.unit_embedding(class_ids)  # (B, 32, 18, 8)
        embedded = embedded.permute(0, 3, 1, 2)    # (B, 8, 32, 18)
        arena_input = th.cat([base, embedded], dim=1)

        arena_feat = self.arena_linear(self.arena_cnn(arena_input))
        vec_feat = self.vector_mlp(obs["vector"])
        return th.cat([arena_feat, vec_feat], dim=1)
```

**CNN parameter cost (with embeddings, 15 input channels):**
```
Conv2d(15, 32, 3x3):  15 * 9 * 32 + 32    =  4,352
Conv2d(32, 64, 3x3):  32 * 9 * 64 + 64    = 18,496
Conv2d(64, 64, 3x3):  64 * 9 * 64 + 64    = 36,928
Embedding(155, 8):     155 * 8              =  1,240
                                       Total: 61,016
                                       Delta: +3,544 (+6.2% over baseline)
```

**Suitability per phase:**
| Phase | Rating | Reasoning |
|-------|--------|-----------|
| BC | Risky | Embeddings for rare units won't converge with ~500 frames per unit. Random embeddings add noise |
| PPO | Strong | Unlimited online data. Embeddings converge fully. Can learn nuanced counter-play |
| Transfer | Strong | Embeddings are class-specific, not deck-specific. A "golem" embedding learned against one deck transfers to another |

---

## 4. Head-to-Head Comparison

### 4.1 Parameter Costs (all negligible)

| Approach | Arena Channels | First Conv2d Params | Embedding Params | Total Policy (~) | Delta |
|----------|---------------|--------------------|-----------------|--------------------|-------|
| 7-channel baseline | 7 | 2,048 | 0 | ~500K | -- |
| 12-channel archetypes | 12 | 3,488 | 0 | ~501K | +0.3% |
| 17-channel split archetypes | 17 | 4,928 | 0 | ~503K | +0.6% |
| 15-channel embeddings | 15 | 4,352 | 1,240 | ~504K | +0.8% |

**All approaches have negligible parameter overhead.** The policy head (`Linear(features_dim, 2305)`) dominates at ~450K params regardless. The encoding choice is about information quality, not computation.

### 4.2 Data Efficiency

| Approach | Frames Needed for Convergence | Available for BC (50 games) | Sufficient? |
|----------|------------------------------|----------------------------|-------------|
| 7-channel | ~50K | ~180K (after no-op downsampling) | Yes |
| 12-channel archetypes | ~75K | ~180K | Yes |
| 15-channel embeddings | ~150K+ (rare units need many examples) | ~180K total, but ~500 per rare unit | Borderline - common units converge, rare units don't |

With 50 games at 2 FPS over ~3 minutes: ~18K raw frames/game, ~900K total, ~180K after 20% no-op retention. This is sufficient for archetypes. It is borderline for embeddings because rare units (Mega Knight appears in ~5 of 50 games) have embeddings trained on very few examples.

**PPO data is unlimited** - the agent generates its own training data by playing. Embedding convergence is not a concern for PPO.

### 4.3 Implementation Effort

| Approach | Code Changes | Architecture Changes | New Dependencies | Estimated Time |
|----------|-------------|---------------------|-----------------|---------------|
| 7-channel | None (current) | None | None | 0 |
| 12-channel archetypes | ~40 lines (constants + encode loop) | None (SB3 default works) | None | 1-2 hours |
| Embeddings (sum/mean) | ~100 lines + custom extractor | Custom `CRFeatureExtractor` | None (PyTorch built-in) | 4-6 hours |
| Full KataCR (collision avoidance) | ~250 lines + custom extractor + PositionFinder | Major encoder rewrite | scipy | 8-12 hours |

### 4.4 Debuggability

| Approach | Can You Inspect the Encoding? | If Training Fails, Can You Diagnose? |
|----------|------------------------------|--------------------------------------|
| 7-channel | Yes: print channel values per cell | Limited: can't distinguish "wrong unit identity" from other problems |
| 12-channel archetypes | Yes: print archetype channels, verify assignments | Strong: check if archetype assignment is wrong, if channel is firing correctly |
| Embeddings | No: opaque 8-dim vectors per cell | Weak: need t-SNE/PCA visualization to inspect embedding space. Training failure could be embedding convergence, multi-unit aliasing, or something else entirely |

### 4.5 Weight Transfer (BC -> PPO)

| Approach | Observation Space | BC -> PPO Weight Transfer |
|----------|------------------|--------------------------|
| 7-channel | (32, 18, 7) | Clean transfer (same obs space) |
| 12-channel archetypes | (32, 18, 12) | Clean transfer (same obs space) |
| Embeddings | (32, 18, 8) base + embedding lookup | Clean transfer IF same feature extractor class. Embedding weights transfer and continue training |
| Archetypes now, embeddings later | (32, 18, 12) then (32, 18, 15+) | **BREAKS** - first Conv2d expects 12 channels, gets 15+. Cannot reuse BC CNN weights |

**This is the key insight.** If you start with archetypes and later switch to embeddings, you lose all BC-trained weights. If you start with embeddings, BC might underperform but PPO can refine everything from the BC checkpoint.

### 4.6 Information Quality

| Scenario | 7-ch | 12-ch Archetype | 15-ch Embedding |
|----------|------|----------------|-----------------|
| PEKKA vs Skeleton | NO | YES (tank vs swarm) | YES (unique embeddings) |
| PEKKA vs Golem | NO | NO (both "tank") | YES |
| Hog Rider vs Mega Knight | NO | YES (win_con vs tank) | YES |
| Musketeer vs Wizard | NO | NO (both "ranged") | YES |
| Arrows vs Zap | NO | NO (both in spell channel) | YES |
| Count of units in cell | YES | YES | PARTIAL (sum/mean degrades count) |
| Spatial accuracy | EXACT | EXACT | EXACT (sum/mean) or DISPLACED (collision avoidance) |

**Archetypes capture the strategically important distinctions** (tank vs swarm vs win_con) while missing within-archetype differences (PEKKA vs Golem). In practice, within-archetype counter-play is usually similar: both PEKKA and Golem require swarm + DPS response.

---

## 5. Archetype Group Definitions

### 5.1 Proposed Groups (5 archetypes)

These are based on Clash Royale strategic roles - each archetype requires a distinct counter-strategy.

**Tank** - High HP units that absorb damage. Counter: sustained DPS or swarm surround.
```python
"tank": [
    "giant", "golem", "golemite", "lava-hound", "lava-pup",
    "mega-knight", "pekka", "electro-giant",
    "royal-giant", "royal-giant-evolution",
    "ice-golem", "goblin-giant", "giant-skeleton",
    "elixir-golem-big", "elixir-golem-mid", "elixir-golem-small",
    "knight", "knight-evolution",
    "golden-knight", "monk", "skeleton-king",
    "battle-healer",
]
```

**Ranged** - Ranged attackers that provide support behind a tank. Counter: spell or rush.
```python
"ranged": [
    "musketeer", "wizard", "ice-wizard", "electro-wizard",
    "archer", "archer-evolution",
    "princess", "dart-goblin", "magic-archer",
    "firecracker", "firecracker-evolution",
    "executioner", "hunter", "witch", "mother-witch", "night-witch",
    "baby-dragon", "inferno-dragon", "skeleton-dragon", "electro-dragon",
    "mega-minion", "sparky", "bowler",
    "flying-machine",
    "phoenix-big", "phoenix-small",
    "bomber", "bomber-evolution",
    "fisherman",
]
```

**Swarm** - Many cheap units. Counter: splash damage or area spell.
```python
"swarm": [
    "skeleton", "skeleton-evolution",
    "bat", "bat-evolution",
    "goblin", "spear-goblin",
    "minion", "barbarian", "barbarian-evolution",
    "rascal-boy", "rascal-girl",
    "royal-recruit", "royal-recruit-evolution",
    "royal-hog",
    "guard",
    "wall-breaker", "wall-breaker-evolution",
    "fire-spirit", "electro-spirit", "ice-spirit", "ice-spirit-evolution", "heal-spirit",
    "elite-barbarian",
    "zappy",
    "little-prince",
    "valkyrie", "valkyrie-evolution",
    "dark-prince",
    "mini-pekka", "prince",
    "lumberjack", "bandit", "royal-ghost",
    "mighty-miner",
    "goblin-brawler",
]
```

Note: This group is broad. It includes melee DPS units (mini-pekka, prince, lumberjack) alongside true swarms (skeleton, goblin). An alternative is to split into "swarm" and "melee_dps" for 6 archetypes total. See Section 5.2 for the 6-group variant.

**Win Condition** - Units that directly threaten towers. Counter: building or direct counter.
```python
"win_con": [
    "hog", "hog-rider", "balloon", "miner",
    "ram-rider", "battle-ram", "battle-ram-evolution",
    "goblin-drill",
    "royal-guardian",
]
```

**Building** - Defensive structures. Counter: spell or ignore.
```python
"building": [
    "cannon", "cannon-cart",
    "tesla", "tesla-evolution",
    "inferno-tower", "bomb-tower",
    "mortar", "mortar-evolution",
    "x-bow", "furnace", "tombstone",
    "goblin-cage", "goblin-hut", "barbarian-hut",
    "elixir-collector",
]
```

### 5.2 Six-Group Alternative (More Specific)

512_owl's suggestion to be "more specific" is addressed by splitting melee DPS from swarm:

| 5-Group | 6-Group | What Changes |
|---------|---------|-------------|
| swarm (broad) | **swarm** (pure multi-unit) | skeleton, bat, goblin, minion, barbarian, spirit, wall-breaker, royal-hog, royal-recruit, zappy |
| | **melee_dps** (single-target damage) | mini-pekka, prince, dark-prince, lumberjack, bandit, valkyrie, elite-barbarian, mighty-miner, royal-ghost, little-prince, goblin-brawler |
| tank | tank (unchanged) | |
| ranged | ranged (unchanged) | |
| win_con | win_con (unchanged) | |
| building | building (unchanged) | |

**Arena shape with 6 groups:** (32, 18, 13) -- 7 base + 6 archetype channels.

**Trade-off:** 6 groups is more precise but makes the "swarm" group narrower. The model can now distinguish "single melee DPS at bridge" (melee_dps=1) from "many small units at bridge" (swarm=many). This distinction matters for spell targeting: arrows against swarm, not against mini-pekka.

**Recommendation: Start with 5 groups.** If BC training reveals that the model confuses single melee DPS for swarm, upgrade to 6.

### 5.3 Units That Fall Outside Archetypes

| Category | Units | Handling |
|----------|-------|---------|
| UI elements | bar, clock, emote, bar-level, etc. (12 units) | Already filtered by UNIT_TYPE_MAP -> "other" -> skipped |
| Towers | king-tower, queen-tower, etc. (4 units) | Already handled by tower HP channels (4-5) |
| Spells | arrows, fireball, zap, etc. (16 units) | Stay in existing CH_SPELL (no archetype needed) |
| Projectiles / visuals | bomb, axe, dirt, goblin-ball | Skip (not real units) |
| Evolution symbols | evolution-symbol, ice-spirit-evolution-symbol | Skip (UI element) |
| Graveyard | graveyard | win_con (spawns skeletons on enemy tower) |

Units with no archetype assignment get no archetype channel increment. They still appear in the base channels (ground/flying count), so they are not invisible.

### 5.4 Ambiguous Assignments

| Unit | Ambiguity | Proposed | Rationale |
|------|-----------|----------|-----------|
| Hog Rider | Win-con + melee | win_con | Primary threat is tower damage, not lane DPS |
| Valkyrie | Splash melee + tank-ish | swarm (5-group) or melee_dps (6-group) | Used as splash defense, not as a tank |
| Cannon Cart | Building + ranged DPS | building | Spawns as building first, transitions to ranged unit |
| Goblin Barrel | Swarm delivery + win-con | win_con | Targets towers directly |
| Dark Prince | Melee DPS + splash | swarm (5-group) or melee_dps (6-group) | Splash capability makes him a support/counter unit |
| Knight | Tank-ish + melee | tank | Used as mini-tank to absorb damage, not for DPS |

---

## 6. Evaluation Against Project Goals

### 6.1 Minimum Goal (BC showing basic competence)

| Approach | Achieves Minimum? | Reasoning |
|----------|------------------|-----------|
| 7-channel | Probably | Count-based heuristics (high count = swarm = spell) get basic behavior. Expert demonstrations carry strong signal even with coarse observations |
| Archetypes | Yes | Everything 7-channel does plus archetype-specific counter-play decisions |
| Embeddings | Yes (but risky) | Should work, but if rare-unit embeddings are noisy, may perform worse than archetypes in BC |

### 6.2 Realistic Goal (BC + PPO, >60% win rate)

| Approach | Achieves Realistic? | Reasoning |
|----------|---------------------|-----------|
| 7-channel | Unlikely | PPO cannot discover counter-strategies when threats are indistinguishable. Will plateau early |
| Archetypes | Plausible | PPO learns archetype-specific responses through exploration. "tank at bridge = deploy swarm" is learnable. Within-archetype distinctions (PEKKA vs Golem) don't matter much for this win rate target |
| Embeddings | Plausible | PPO refines embeddings with unlimited data. Can learn nuanced responses. Higher ceiling but same floor as archetypes |

### 6.3 Moonshot Goal (transfer across decks)

| Approach | Supports Transfer? | Reasoning |
|----------|-------------------|-----------|
| 7-channel | No | No deck-agnostic unit representation at all |
| Archetypes | Yes | "Tank" means "tank" regardless of deck. Transfer works at the archetype level. Agent knows how to counter tanks even with a new deck |
| Embeddings | Yes (best) | Per-unit embeddings transfer directly. Agent has learned representations for all 155 units. Switching decks only changes the card hand vector |

### 6.4 Win Rate Estimates (Speculative)

| Approach | BC Win Rate (vs Trainer) | PPO Win Rate (vs Trainer) | Reasoning |
|----------|-------------------------|--------------------------|-----------|
| 7-channel | 10-25% | 30-45% | BC copies expert heuristics; PPO improves timing/elixir but cannot learn counter-play |
| Archetypes | 20-35% | 45-65% | BC learns archetype counter-play; PPO refines strategy and discovers new patterns |
| Embeddings | 15-30% | 50-70% | BC slightly worse than archetypes (noisy rare embeddings); PPO compensates with refined embeddings and nuanced play |

**These are rough estimates, not empirical results.** The relative ordering is based on reasoning about information quality, not measured performance.

---

## 7. The Weight Transfer Problem

This is the deciding factor between "archetypes now, embeddings later" vs "pick one and commit."

### 7.1 Scenario: Start with Archetypes, Switch to Embeddings for PPO

```
BC phase:    obs_space = Dict(arena=(32,18,12), vector=(23,))
             Conv2d(12, 32, 3x3) learns feature detectors for 12-channel input

PPO phase:   obs_space = Dict(arena=(32,18,15), vector=(23,))
             Conv2d(15, 32, 3x3) expects 15-channel input

Problem:     The BC-trained Conv2d has weight shape (32, 12, 3, 3).
             PPO needs weight shape (32, 15, 3, 3).
             Cannot load BC weights into PPO.
```

**Solutions:**
1. **Pad the weight tensor:** Initialize the extra 3 input channels with zeros. BC-trained features for channels 0-11 are preserved. Channels 12-14 (embedding) start from scratch. This works but the embedding channels have no learned features - PPO must learn them online.
2. **Retrain from scratch:** Discard BC weights entirely. PPO starts from random initialization.
3. **Don't switch:** Use the same encoding for both phases.

### 7.2 Scenario: Start with Embeddings for Both BC and PPO

```
BC phase:    obs_space = Dict(arena=(32,18,12), vector=(23,))  # 7 base + 5 archetype
             OR obs_space = Dict(arena=(32,18,8), vector=(23,))  # 7 base + class_id
             Custom CRFeatureExtractor with Embedding(155, 8)
             Conv2d(15, 32, 3x3) processes 7 base + 8 embedded channels

PPO phase:   SAME obs_space, SAME feature extractor
             Load BC weights directly. Continue training.

Benefit:     Clean weight transfer. BC embedding weights (partially converged)
             serve as initialization for PPO.
```

### 7.3 Scenario: Archetypes for Both BC and PPO

```
BC phase:    obs_space = Dict(arena=(32,18,12), vector=(23,))
             SB3 default CombinedExtractor

PPO phase:   SAME obs_space, SAME extractor
             Load BC weights directly. Continue training.

Benefit:     Clean weight transfer. No custom code needed.
             PPO has the same archetype information as BC.

Limitation:  PPO ceiling is determined by archetype granularity.
             Cannot learn within-archetype distinctions.
```

---

## 8. Recommendation

### 8.1 Decision: Archetypes for Both BC and PPO

**Use 5 archetype channels (12-channel arena) for the entire pipeline.**

**Rationale:**

1. **Weight transfer is clean.** BC trains on (32, 18, 12), PPO loads those exact weights and continues. No observation space changes, no weight padding, no custom extractors.

2. **Implementation is minimal.** ~40 lines of code. No custom SB3 feature extractor. Works with default `MaskableMultiInputPolicy` out of the box.

3. **Data efficiency is proven.** Archetypes work with 50 games of BC data. No convergence concerns for rare units.

4. **Debuggability is full.** Every channel is interpretable. If BC fails, you can inspect which archetype channels fired and verify correctness.

5. **PPO ceiling is sufficient for the realistic goal.** The proposal targets >60% win rate against Trainer AI. Archetype-level counter-play (tank -> swarm, swarm -> spell, win_con -> building) is the primary strategic axis. Within-archetype distinctions (PEKKA vs Golem) have minimal impact against Trainer AI opponents.

6. **Moonshot transfer works.** Archetypes are deck-agnostic. "Tank" is "tank" regardless of which deck contains it.

7. **Leaves room for future upgrade.** If PPO plateaus, embeddings can be added as additional channels on top of archetypes (pad Conv2d weights for the new channels). The archetype channels provide a warm-start floor that the embedding channels improve upon.

### 8.2 Why Not Embeddings From the Start?

Embeddings are theoretically superior for PPO (higher information ceiling). But:

- They require a custom `CRFeatureExtractor` (4-6 hours of implementation, testing, debugging)
- Sum/mean-embedding aggregation degrades signal quality for multi-unit cells
- BC performance may be worse due to noisy rare-unit embeddings
- Debugging is harder (opaque embedding vectors vs interpretable archetype channels)
- The marginal PPO benefit (50-70% vs 45-65%) is speculative and may not materialize
- For a course project with limited time, engineering simplicity has high value

If this were a production system with unlimited engineering time, embeddings from the start would be the correct choice. For a 10-week course project where BC + PPO must both work, archetypes are the pragmatic choice.

### 8.3 Upgrade Path If Needed

If PPO with archetypes plateaus below the 60% target:

```
Phase 1: BC + PPO with archetypes (12 channels)
         Conv2d(12, 32, 3x3) weight shape: (32, 12, 3, 3)

Phase 2: Add embedding channels (12 + 8 = 20 channels)
         Pad Conv2d weight: (32, 12, 3, 3) -> (32, 20, 3, 3)
         Channels 0-11: loaded from PPO checkpoint (trained)
         Channels 12-19: initialized to zero (untrained embedding features)
         PPO resumes training. New channels learn from online data.
         Archetype channels provide the floor; embeddings add refinement.
```

This preserves all learned features while adding new capacity. The archetype channels continue to provide the primary strategic signal; the embedding channels add within-archetype discrimination that PPO can learn from unlimited online data.

### 8.4 Implementation Steps

1. Add `UNIT_ARCHETYPE_MAP` lookup table to `encoder_constants.py`
2. Add 5 archetype channel constants (`CH_TANK`, `CH_RANGED`, etc.) to `encoder_constants.py`
3. Update `NUM_ARENA_CHANNELS` from 7 to 12
4. Update `_encode_arena()` in `state_encoder.py` to write archetype channels
5. Update `StateEncoder.__init__()` observation space shape
6. Update `encoder/CLAUDE.md` documentation with new channel layout
7. Run smoke test to verify shapes and values

---

## 9. Assumptions

1. **BC can learn counter-play from archetype signals.** Not empirically validated. Based on reasoning that expert demonstrations paired with archetype-level observations provide sufficient signal for imitation learning.

2. **50 games is enough data for archetypes but borderline for embeddings.** Based on calculation that rare units appear in ~5 of 50 games, giving ~500 post-downsample frames per rare unit.

3. **Sum-embeddings degrade when cells are dense.** Based on theoretical analysis of sum aliasing and scale collapse. Not empirically tested in this pipeline.

4. **Archetype assignments are approximately correct.** Based on Clash Royale gameplay knowledge. Some assignments are judgment calls. Incorrect assignments can be fixed by updating the lookup table and retraining.

5. **KataCR's PositionFinder approach is incompatible with our count-based encoding.** Design choice, not a hard constraint. We could switch, but it changes the fundamental nature of our arena representation.

6. **PPO with archetypes can reach >60% win rate.** Speculative. Depends on reward shaping, training stability, and whether Trainer AI opponents are predictable enough for archetype-level counter-play.

7. **Within-archetype distinctions (PEKKA vs Golem) have minimal impact at the target win rate.** Both require similar counter-strategies (swarm + DPS). At higher competitive levels, this distinction becomes more important.

8. **Custom SB3 feature extractor code has NOT been tested.** The code in Section 3.3.4 is based on SB3 documentation patterns and requires integration testing.

9. **Win rate estimates are speculative.** No empirical validation. Used for directional comparison only.

10. **CardPredictor integration remains a blocking dependency.** Without card hand detection, the action mask only allows no-op and all card features in the vector observation are zero. This must be fixed regardless of which encoding approach is chosen.
