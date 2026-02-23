# KataCR Unit Encoding Analysis

## Comprehensive comparison of KataCR's per-cell feature encoding vs our additive-count approach

**Date:** 2026-02-20
**Scope:** PositionFinder, per-cell features, learned embeddings, pipeline impact, data requirements
**Researchers:** codebase-researcher (our pipeline), katacr-researcher (KataCR codebase + paper)

---

## Table of Contents

1. [Q1: PositionFinder](#q1-positionfinder)
2. [Q2: Compatibility and Affected Modules](#q2-compatibility-and-affected-modules)
3. [Q3: Swarm/Horde Recognition](#q3-swarmhorde-recognition)
4. [Q4: Per-Cell Feature Vector Implementation](#q4-per-cell-feature-vector-implementation)
5. [Q5: Learned Embeddings in BC Network](#q5-learned-embeddings-in-bc-network)
6. [Q6: Pipeline Impact](#q6-pipeline-impact)
7. [Q7: Data Requirements](#q7-data-requirements)
8. [Summary of Recommendations](#summary-of-recommendations)

---

## Q1: PositionFinder

### What is PositionFinder?

PositionFinder is a collision-resolution algorithm used by KataCR during **offline data preprocessing** (not real-time inference). Its purpose is to enforce a constraint that each cell in the 32x18 arena grid contains at most one unit, enabling per-cell feature vectors with fixed structure.

**Location in KataCR's pipeline:** `katacr/policy/offline/dataset.py`, lines 143-159. It runs inside `build_feature()`, which is called during dataset construction -- after state extraction and before tensor serialization. It is NOT part of the real-time StateBuilder or the model itself.

### Algorithm

```python
class PositionFinder:
    def __init__(self, r=32, c=18):
        self.used = np.zeros((r, c), np.bool_)
        self.center = np.swapaxes(
            np.array(np.meshgrid(np.arange(r), np.arange(c))), 0, -1
        ) + 0.5  # (r, c, 2) -- cell centers at +0.5 offset

    def find_near_pos(self, xy):
        yx = np.array(xy)[::-1]          # Convert (x, y) -> (y, x)
        y, x = yx.astype(np.int32)
        y = np.clip(y, 0, 31)
        x = np.clip(x, 0, 17)
        if self.used[y, x]:              # Cell already occupied
            avail_center = self.center[~self.used]
            map_index = np.argwhere(~self.used)
            dis = scipy.spatial.distance.cdist(yx.reshape(1, 2), avail_center)
            y, x = map_index[np.argmin(dis)]
        self.used[y, x] = True
        return np.array((x, y), np.int32)
```

**Step by step:**
1. A fresh `PositionFinder` is created per frame (the `used` grid resets each frame).
2. For each unit, convert its continuous cell coordinate to an integer (y, x).
3. If that cell is unoccupied, place the unit there and mark it used.
4. If occupied, compute Euclidean distance from the unit's position to ALL unoccupied cell centers using `scipy.spatial.distance.cdist`.
5. Place the unit in the nearest unoccupied cell.

### Pipeline location: Where would it go in our system?

```
CRDetector.detect()
    |
    v
StateBuilder._process_detections()  -->  GameState (units with pixel coords)
    |
    v
[PositionFinder]  <-- NEW: runs here, per-frame, before encoding
    |
    v
StateEncoder.encode()  -->  arena tensor (32, 18, C)
```

It sits between StateBuilder output and StateEncoder input. Specifically, PositionFinder would operate on the list of `Unit` objects in `GameState.units`, converting each unit's pixel-based center to a grid cell and resolving collisions. The output would be a mapping from grid cells to unit info (class, belonging, health bars).

### Performance with swarm units

**Problem:** Swarm cards deploy many units in a small area. Examples:

| Card | Unit Count | Typical Cluster Size |
|------|-----------|---------------------|
| Skeleton Army | 15 | 3-4 cells |
| Minion Horde | 6 | 2-3 cells |
| Royal Recruits | 6 | 3-4 cells (split) |
| Royal Hogs | 4 | 2-3 cells |
| Barbarians | 5 | 2-3 cells |

When 15 skeletons map to 3-4 natural cells, PositionFinder displaces 11-12 of them to neighboring cells. The `scipy.spatial.distance.cdist` call computes a distance matrix of shape `(1, N_available)` where `N_available` can be up to 576. This is fast for a single query (microseconds), but repeated 15 times with shrinking available cells causes cascading displacement.

**Computational cost:** For a frame with K units total, PositionFinder makes K calls. Each displaced call computes `cdist` over the remaining available cells. Worst case with 30+ units on screen: still well under 1ms total on modern hardware. **Performance is not a practical concern.**

### Positional error from displacement

**This is the real concern.** When a skeleton at cell (5, 20) gets displaced to (6, 21) because (5, 20) is occupied, the agent "sees" a unit at (6, 21) that is not actually there. The magnitude of error depends on:

1. **Cluster density:** Higher density = more displacement. 15 skeletons in 4 cells means some get pushed 2-3 cells away.
2. **Processing order:** The first unit processed claims the "correct" cell. Later units in the same cluster get progressively worse positions.
3. **Cell size:** Each cell represents ~30x25 pixels in the 568x896 generator space (~2 in-game tiles). A 1-cell displacement is ~2 tiles of positional error.

**Quantitative estimate:** For a Skeleton Army (15 units, ~4 natural cells), the average displacement would be approximately 1.5 cells for the 11 displaced skeletons. The cluster expands from a 2x2 region to roughly a 4x4 region. The "shape" of the swarm is preserved (still a cluster) but its footprint is artificially inflated.

### Order dependency

Units are iterated in the order they appear in the `unit_infos` list. KataCR sorts units by `(belonging descending, top-y descending)` before processing -- enemy units first, then friendly, bottom-to-top within each faction. This means:

- Enemy units get their "correct" cells first
- If an enemy and ally unit overlap, the enemy keeps the true position
- Within a faction, units closer to the bottom (closer to the river) get priority

**This introduces a systematic bias:** enemy units have slightly more accurate positions than ally units. However, KataCR's paper argues that the Transformer architecture provides permutation invariance within spatial patches, mitigating the impact of processing order on the final learned representation.

### KataCR's fix for positional error

KataCR does NOT explicitly fix the positional error. Their mitigation is architectural:

1. **2x2 patch tokenization:** The arena (32, 18, 15) is split into 2x2 patches, yielding 144 tokens of 60 dimensions each (2x2x15). This means displaced units within a 2-cell radius still fall within the same or adjacent patch, softening the impact.
2. **Transformer attention:** The StARformer processes these patches with self-attention, which can learn to aggregate information across nearby patches regardless of exact cell placement.
3. **Temporal context:** StARformer uses 3-timestep sequences, so momentary displacement artifacts are smoothed over time.

### What would our fix be?

We have several options, with different trade-offs:

| Approach | Pros | Cons |
|----------|------|------|
| **A: Use PositionFinder as-is** | Matches KataCR, enables per-cell features | Positional error for swarms, order dependency |
| **B: PositionFinder + shuffle order** | Reduces systematic bias | Randomness between frames may confuse temporal learning |
| **C: Multi-unit cells (up to K=3)** | No displacement needed, preserves true positions | Variable-length per-cell features, more complex encoding |
| **D: Keep additive counting, add class channel** | Simplest change, no PositionFinder needed | Loses per-unit detail in multi-unit cells |

**Recommendation:** Option A (use PositionFinder as-is) for initial implementation. The positional error is bounded (1-3 cells for worst-case swarms) and the CNN+MLP architecture we plan to use can learn to handle this. If swarm recognition becomes a problem during training, Option C is the cleanest upgrade path.

**Assumption:** We assume SB3's MaskablePPO with a CNN feature extractor can tolerate the positional error introduced by PositionFinder. KataCR's StARformer has stronger architectural mitigation (attention over patches), but our simpler model should still benefit from per-cell unit identity even with displacement artifacts.

---

## Q2: Compatibility and Affected Modules

### Action mask incompatibility

The current action mask (`state_encoder.py:120-163`) is **not incompatible** with switching to per-cell feature encoding. The action mask operates on card availability and elixir, not on the arena encoding. Changing how units are represented in the arena tensor does not affect the action mask at all.

However, there is a **pre-existing gap** in the action mask that becomes more relevant with a richer encoding: the mask does not enforce spatial placement rules. `PLAYER_HALF_ROW_START = 17` and `CARD_IS_SPELL` exist in `encoder_constants.py` but are never used. With per-cell features giving the model better spatial understanding, the lack of spatial masking becomes a larger source of wasted exploration.

### Modules affected by adopting KataCR's per-cell approach

| Module | File | Impact | Changes Needed |
|--------|------|--------|---------------|
| **StateEncoder** | `src/encoder/state_encoder.py` | **Major** | Rewrite `_encode_arena()` to produce per-cell feature vectors instead of additive counts. Change arena shape from (32, 18, 7) to (32, 18, C) where C depends on feature design. |
| **encoder_constants** | `src/encoder/encoder_constants.py` | **Moderate** | Add new channel constants, remove CH_ALLY_GROUND etc., add class-to-ID mapping for all 155 detection classes. |
| **observation_space** | `state_encoder.py:91-104` | **Moderate** | Update Box shape and bounds for new arena channels. |
| **BC network** | Not yet built | **Moderate** | Feature extractor must handle new arena shape. If using learned embeddings, need an `nn.Embedding` layer. |
| **GameState** | `src/pipeline/game_state.py` | **Minor** | Unit dataclass may need `health_bar_image` field if we include bar data. |
| **StateBuilder** | `src/pipeline/state_builder.py` | **Minor** | Needs to crop and store health bar images per unit (if including bars). |
| **coord_utils** | `src/encoder/coord_utils.py` | **None** | `pixel_to_cell()` and action functions are unchanged. |
| **Action mask** | `state_encoder.py:120-163` | **None** | Unaffected -- operates on cards and elixir, not arena encoding. |
| **Action space** | `encoder_constants.py` | **None** | Discrete(2305) is unchanged. Action encoding is separate from observation encoding. |
| **DatasetBuilder** | Not yet built | **None** | Will consume StateEncoder output; adapts automatically to new shapes. |

### Key finding: The action space and action builder are NOT affected

The action space `Discrete(2305) = 4 cards x 576 cells + 1 no-op` defines where cards can be placed, not how the board is perceived. Changing the observation encoding from additive counts to per-cell features is purely an observation-side change. The action encoding, action mask, `coord_utils.py`, and `placement_to_action()`/`action_to_placement()` functions all remain identical.

**Analytical question:** If we add spatial masking to the action mask (enforcing troop placement on player's half only), should that be done simultaneously with the encoding change? It is an independent improvement, but the combination gives the agent better observations AND a tighter action space, which could accelerate learning.

---

## Q3: Swarm/Horde Recognition

### Can BC/RL recognize swarms with per-cell encoding?

**Yes, but differently than with additive counting.** The two approaches provide different signals:

**Current approach (additive counting):**
- Cell (5, 20) has `CH_ENEMY_GROUND = 6.0` -- the model knows "6 enemy ground units are here"
- Density information is explicit and immediate
- No unit identity -- the model cannot distinguish 6 skeletons from 6 barbarians

**Per-cell approach (KataCR-style):**
- 6 skeletons in a 2x2 area get spread to a ~4x4 area by PositionFinder
- Each cell contains `(class_id=skeleton, belonging=enemy)`
- The model sees "skeleton, skeleton, skeleton..." in nearby cells
- Density must be inferred from the spatial pattern of occupied cells

### Why is density harder to learn through spatial patterns?

There are three reasons:

**1. Indirect signal.** With additive counting, a count of 15 at one cell is an unambiguous density signal. With per-cell encoding, the model must learn that "many adjacent cells with the same class" means "swarm." This requires the model to develop receptive fields large enough to see the cluster, AND to learn the correspondence between cluster size and actual unit count.

**2. PositionFinder distortion.** The artificial spreading of swarm units creates an enlarged footprint that does not match the actual in-game cluster. A Skeleton Army occupying a 2x2 area in reality might appear as a 4x4 cluster after displacement. The model may learn to associate "skeleton cluster = threat" but will misjudge the cluster's actual spatial extent.

**3. Variable cluster shapes.** The same swarm card can produce different displacement patterns depending on:
- Which other units are already on the grid
- Processing order within the swarm
- How tightly the YOLO bounding boxes overlap

This means the same game state can produce different per-cell representations, adding noise to the observation.

### Can the model still recognize swarms effectively?

**Yes, with caveats.** The model can learn:
- "Multiple skeleton class IDs in nearby cells" = Skeleton Army
- "4 royal-hog class IDs in a row" = Royal Hogs push
- The specific class identity tells the model what card was played, even if it cannot count exactly

What the model loses:
- Precise count of units in a cluster
- Exact spatial extent of the swarm
- Ability to distinguish "6 barbarians" from "3 barbarians" at a glance (the latter just has fewer occupied cells)

**Mitigating factor for our project:** Our 8-card deck does not include Skeleton Army (the worst case with 15 units). The most swarm-heavy card is Royal Recruits (6 units), which naturally split into two groups of 3 and rarely all occupy the same cell. Royal Hogs (4 units) also spread naturally. The worst-case displacement scenario is unlikely with our specific deck.

**Analytical question:** Should we consider a hybrid approach where the arena tensor has BOTH per-cell class features AND a density count channel? This would give the model the best of both worlds -- unit identity per cell plus explicit count information. The cost is additional channels.

---

## Q4: Per-Cell Feature Vector Implementation

### KataCR's per-cell structure (for reference)

```
arena[y, x, 0]       = cls    (int class ID, or -1 if empty)
arena[y, x, 1]       = bel    (-1 friendly, +1 enemy)
arena[y, x, 2:194]   = bar1   (24x8 grayscale HP bar image, flattened = 192 values)
arena[y, x, 194:386] = bar2   (24x8 grayscale secondary bar, flattened = 192 values)
```

Raw per-cell: 386 dimensions. After the StARformer's embedding/CNN layers: 15 dimensions (8 cls_embed + 1 bel + 3 bar1_cnn + 3 bar2_cnn).

Plus `arena_mask`: shape (32, 18), boolean True where a unit exists.

### Our recommended implementation

We should NOT copy KataCR's 386-dim raw representation. Our architecture (SB3's CNN+MLP, not a Transformer) does not have per-cell embedding layers. Instead, we should encode features that are directly usable by a CNN.

**Recommended per-cell feature vector (Option A - Minimal):**

```
Channel 0:   class_id (int, normalized to [0, 1] by dividing by num_classes)
Channel 1:   belonging (-1 = ally, +1 = enemy, 0 = empty)
Channel 2:   arena_mask (1 if unit present, 0 if empty)
```

Shape: (32, 18, 3). Simple, fast, and gives the CNN access to unit identity and side.

**Recommended per-cell feature vector (Option B - With functional groups):**

```
Channel 0:   class_id (normalized)
Channel 1:   belonging (-1 / +1 / 0)
Channel 2:   arena_mask (binary)
Channel 3:   unit_type (0=empty, 1=ground, 2=flying, 3=spell, 4=building)
Channel 4:   is_swarm_unit (binary -- 1 if class is skeleton, minion, bat, etc.)
```

Shape: (32, 18, 5). Adds functional information that helps the CNN without requiring it to learn class-to-type mappings.

**Recommended per-cell feature vector (Option C - Full):**

```
Channel 0:     class_id (normalized)
Channel 1:     belonging (-1 / +1 / 0)
Channel 2:     arena_mask (binary)
Channel 3:     unit_type (categorical)
Channel 4:     has_health_bar (binary)
Channel 5:     health_bar_fraction (0-1, estimated from bar width if available)
Channels 6-8:  tower HP channels (ally_tower_hp, enemy_tower_hp, spell -- kept from current)
```

Shape: (32, 18, 9).

### Should we include health bar values?

**Not for MVP.** Health bar extraction requires:
1. Detecting bar sprites above units (our YOLO detects `bar` class but does not associate bars with specific units)
2. Cropping and resizing bar images (KataCR's bar-body matching algorithm)
3. Either storing raw bar images (192 values per bar, needs CNN processing) or estimating HP fraction from bar width

KataCR's approach (storing 24x8 grayscale bar images) is tightly coupled to their Transformer architecture which has per-cell CNN processing. Our CNN-based SB3 policy would need to process these differently.

**Recommendation:** Start without health bars. Add `has_health_bar` as a binary flag if bar detection is available. Defer bar-to-HP estimation to a later iteration.

### Would we need to update channels if we include belonging?

Yes. The current 7-channel layout would be entirely replaced:

**Current channels (to be removed):**
```
CH_ALLY_GROUND  = 0   # Additive count
CH_ALLY_FLYING  = 1   # Additive count
CH_ENEMY_GROUND = 2   # Additive count
CH_ENEMY_FLYING = 3   # Additive count
CH_ALLY_TOWER_HP = 4  # Fraction
CH_ENEMY_TOWER_HP = 5 # Fraction
CH_SPELL = 6          # Additive count
```

**New channels (replacement):**
```
CH_CLASS_ID     = 0   # Normalized class ID
CH_BELONGING    = 1   # -1/0/+1
CH_ARENA_MASK   = 2   # Binary
CH_ALLY_TOWER_HP = 3  # Fraction (kept)
CH_ENEMY_TOWER_HP = 4 # Fraction (kept)
```

The ally/enemy distinction that was previously handled by separate ground/flying channels is now handled by the belonging channel. The ground/flying distinction that was previously handled by separate channels is now implicit in the class ID (the network learns which class IDs are flying).

Tower HP channels are kept as-is because towers occupy fixed, known cells and their HP is a continuous value (not a class ID).

### Pseudocode for the new `_encode_arena()`

```python
def _encode_arena(self, state: GameState) -> np.ndarray:
    arena = np.zeros((GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS), dtype=np.float32)
    fw = state.frame_width or 540
    fh = state.frame_height or 960

    # Build PositionFinder for this frame
    pf = PositionFinder(r=GRID_ROWS, c=GRID_COLS)

    # Sort units: enemy first, then ally, bottom-to-top within each group
    sorted_units = sorted(
        state.units,
        key=lambda u: (-u.belonging, -u.center[1])
    )

    for unit in sorted_units:
        cx, cy = unit.center
        # Convert pixel center to continuous cell coordinate
        col_f, row_f = pixel_to_cell_float(cx, cy, fw, fh)
        # Resolve collisions
        col, row = pf.find_near_pos((col_f, row_f))

        unit_type = UNIT_TYPE_MAP.get(unit.class_name, "ground")
        if unit_type in ("tower", "other"):
            continue

        class_idx = CLASS_NAME_TO_ID.get(unit.class_name, 0)
        bel = -1.0 if unit.belonging == 0 else 1.0

        arena[row, col, CH_CLASS_ID] = class_idx / NUM_CLASSES
        arena[row, col, CH_BELONGING] = bel
        arena[row, col, CH_ARENA_MASK] = 1.0

    # Towers (same as before)
    for tower, is_ally in tower_slots:
        if tower is None or tower.bbox is None:
            continue
        hp_frac = self._get_tower_hp_frac(tower)
        cx = (tower.bbox[0] + tower.bbox[2]) // 2
        cy = (tower.bbox[1] + tower.bbox[3]) // 2
        col, row = pixel_to_cell(cx, cy, fw, fh)
        ch = CH_ALLY_TOWER_HP if is_ally else CH_ENEMY_TOWER_HP
        arena[row, col, ch] = hp_frac

    return arena
```

---

## Q5: Learned Embeddings in BC Network

### What are learned embeddings and why use them?

A normalized class ID (0.0 to 1.0) implies a linear ordering between classes -- the network might incorrectly learn that "skeleton" (id=0.1) is "closer" to "knight" (id=0.12) than to "minion" (id=0.8). Learned embeddings map each class ID to a trainable vector, allowing the network to discover its own notion of similarity between unit types.

### Implementation in our BC network

KataCR uses `Embed(n_unit+1, 8)` in the StARformer -- a JAX/Flax embedding that maps each class integer to an 8-dimensional vector. For our PyTorch/SB3 setup:

```python
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=8, features_dim=128):
        super().__init__(observation_space, features_dim=features_dim)

        num_classes = 156  # 155 classes + 1 for "empty" (index 0)

        # Learned embedding for class IDs
        self.class_embed = nn.Embedding(num_classes, embed_dim)

        # Per-cell feature assembly:
        #   embed_dim (class) + 1 (belonging) + 1 (arena_mask)
        #   + 2 (tower HP channels)
        per_cell_dim = embed_dim + 1 + 1 + 2  # = 12

        # Arena CNN branch
        self.arena_cnn = nn.Sequential(
            nn.Conv2d(per_cell_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 3, 128),  # After two stride-2 convs: 32->15->7, 18->8->3
            nn.ReLU(),
        )

        # Vector branch
        self.vector_mlp = nn.Sequential(
            nn.Linear(23, 64),
            nn.ReLU(),
        )

        # Combined
        self.combined = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        arena_raw = observations["arena"]  # (B, 32, 18, C)
        vector = observations["vector"]    # (B, 23)

        # Extract class ID channel and convert to integer indices
        class_ids = arena_raw[:, :, :, 0].long()  # (B, 32, 18)
        # Denormalize: class_ids were stored as class_idx / NUM_CLASSES
        # We need integer indices for the embedding lookup
        class_ids = (class_ids * NUM_CLASSES).clamp(0, NUM_CLASSES - 1).long()

        # Look up embeddings
        class_embeds = self.class_embed(class_ids)  # (B, 32, 18, embed_dim)

        # Concatenate with other channels
        other_channels = arena_raw[:, :, :, 1:]  # (B, 32, 18, C-1)
        arena_features = torch.cat([class_embeds, other_channels], dim=-1)
        # Shape: (B, 32, 18, embed_dim + C - 1)

        # Reshape for Conv2d: (B, C, H, W)
        arena_features = arena_features.permute(0, 3, 1, 2)

        arena_out = self.arena_cnn(arena_features)
        vector_out = self.vector_mlp(vector)

        return self.combined(torch.cat([arena_out, vector_out], dim=1))
```

**Important design note:** The class ID must be stored as an integer in the arena tensor for embedding lookup, NOT as a normalized float. This means the observation space `Box` should store the class channel as an integer (or we round-trip through float and cast back). An alternative is to store class IDs in a separate integer tensor, but SB3's `Dict` observation space supports mixing `Box` and `MultiBinary`/`MultiDiscrete` spaces.

**Cleaner approach - separate the class ID from continuous features:**

```python
observation_space = gym.spaces.Dict({
    "arena_class": gym.spaces.Box(0, 155, shape=(32, 18), dtype=np.int32),
    "arena_features": gym.spaces.Box(-1, 1, shape=(32, 18, 4), dtype=np.float32),
    # 4 channels: belonging, arena_mask, ally_tower_hp, enemy_tower_hp
    "vector": gym.spaces.Box(0, 1, shape=(23,), dtype=np.float32),
})
```

### Final arena representation shape

With the cleaner approach:
- `arena_class`: (32, 18) integer class IDs -> embedded to (32, 18, 8) by `nn.Embedding`
- `arena_features`: (32, 18, 4) continuous features
- Concatenated: (32, 18, 12) -> Conv2d input after permute to (12, 32, 18)

After the CNN branch: 128-dim vector. Combined with the 64-dim vector branch: 192-dim, projected to 128-dim features for the policy/value heads.

### Comparison: Raw class ID vs Learned embedding vs Functional categories

| Approach | Arena Shape | Pros | Cons |
|----------|-------------|------|------|
| Normalized class ID (float) | (32, 18, 5) | Simple, no extra parameters | Implies false ordering between classes |
| Learned embedding (8-dim) | (32, 18, 12) | Learns semantic similarity, most expressive | More parameters (~1.2K), needs integer obs |
| Functional categories (10 groups) | (32, 18, 5) with one-hot group | Domain-interpretable, no learned params | Loses fine-grained identity (e.g., knight vs PEKKA) |

**Recommendation:** Use learned embeddings. The extra 1.2K parameters (156 classes x 8 dims) are negligible relative to the CNN and MLP layers. The embedding learns which units are strategically similar (e.g., tanks together, swarm units together) without us manually defining categories.

---

## Q6: Pipeline Impact

### How the pipeline changes

The full pipeline with per-cell encoding:

```
Current Pipeline:
  CRDetector.detect()
      |
      v
  StateBuilder._process_detections()  -->  GameState
      |                                      |
      v                                      v
  StateBuilder._process_ocr()          StateEncoder.encode()
                                            |
                                            v
                                       obs = {"arena": (32,18,7), "vector": (23,)}


Modified Pipeline:
  CRDetector.detect()
      |
      v
  StateBuilder._process_detections()  -->  GameState (units with pixel coords)
      |                                      |
      v                                      v
  StateBuilder._process_ocr()          PositionFinder (collision resolution)
                                            |
                                            v
                                       StateEncoder.encode()
                                            |
                                            v
                                       obs = {
                                         "arena_class": (32,18),     # int
                                         "arena_features": (32,18,4), # float
                                         "vector": (23,)             # float
                                       }
```

### Changes to each component

**StateBuilder** (`state_builder.py`): Minimal changes. It continues to produce `GameState` with `List[Unit]`. No structural change needed unless we add health bar image storage.

**StateEncoder** (`state_encoder.py`): Major rewrite of `_encode_arena()`. The PositionFinder is instantiated per-call inside `encode()`. New observation space definition. See pseudocode in Q4.

**encoder_constants** (`encoder_constants.py`):
- Remove: `CH_ALLY_GROUND`, `CH_ALLY_FLYING`, `CH_ENEMY_GROUND`, `CH_ENEMY_FLYING`, `CH_SPELL`
- Add: `CH_CLASS_ID`, `CH_BELONGING`, `CH_ARENA_MASK`, `CLASS_NAME_TO_ID` (mapping 155 class names to integer IDs)
- Keep: `CH_ALLY_TOWER_HP`, `CH_ENEMY_TOWER_HP` (renumbered)
- Update: `NUM_ARENA_CHANNELS`

**coord_utils** (`coord_utils.py`): Add a `pixel_to_cell_float()` function that returns continuous (col, row) coordinates (not integer-clamped) for PositionFinder input. The existing `pixel_to_cell()` can remain for backward compatibility.

### Action space/builder impact

**No impact.** The action space `Discrete(2305)` is defined by:
- 4 card slots x 576 grid cells + 1 no-op

This is an output encoding, not an input encoding. Changing how we represent the board (observation) does not change how we encode actions. The `action_to_placement()` and `placement_to_action()` functions in `coord_utils.py` are entirely unaffected.

The action mask logic is also unaffected -- it checks card availability and elixir, neither of which depends on the arena encoding method.

### BC data collection impact

The BC click logger design (Thread A: capture + StateBuilder, Thread B: mouse logger, post-game merge) is also unaffected. The DatasetBuilder will call `StateEncoder.encode()` to produce observations from `GameState` objects. Whether the encoder uses additive counts or per-cell features is transparent to the data collection pipeline.

**One caveat:** If we store raw `GameState` objects (recommended) and re-encode with `StateEncoder` during training, we can freely experiment with different encoding approaches without re-collecting data. This is a strong argument for storing the intermediate `GameState` format rather than pre-encoded tensors.

---

## Q7: Data Requirements

### How much footage/matches do we need?

The data requirements for BC do NOT change significantly with per-cell encoding vs additive counting. The bottleneck is action coverage, not observation complexity.

**Target: 30-40 games** (unchanged from `pipeline-design-qa.md` recommendation)

Calculation:
- 3 minutes per game average, 2-4 FPS capture
- ~360-720 frames per game, ~12,000-28,800 total frames
- ~70% are no-ops, leaving ~3,600-8,640 action frames
- 8 card types, ~450-1,080 placements per card type
- After no-op downsampling to 20%: ~5,000-12,000 training frames

This gives sufficient coverage for the 8 cards in our deck. The per-cell encoding does not require more data -- it just provides richer observations for the same state-action pairs.

### Rare troop handling: What if we have no footage with certain enemy units?

**This is a real concern with per-cell encoding.** With additive counting, all enemy ground units look the same (a count in CH_ENEMY_GROUND). The model never sees individual class IDs, so it does not matter if rare troops are absent from training data.

With per-cell encoding, the model learns an embedding for each class ID. If a class never appears in training data, its embedding is randomly initialized and never updated. When that class appears at test time, the model receives a meaningless embedding vector.

**Severity assessment:**

| Scenario | Additive Counting | Per-Cell Encoding |
|----------|-------------------|-------------------|
| Common enemy (knight, archer) | Treated as "ground unit" | Learned embedding |
| Rare enemy (miner, lava hound) | Treated as "ground unit" (same) | Random embedding (PROBLEM) |
| Unseen enemy (class never in data) | Treated as "ground unit" (same) | Random embedding (PROBLEM) |

**Mitigations:**

1. **Default embedding for unseen classes.** During encoding, if a class ID was not seen during training, map it to a generic "unknown_ground" or "unknown_flying" class ID. This requires tracking which classes were seen during training.

2. **Pre-train embeddings from class attributes.** Initialize the embedding table using known attributes (elixir cost, HP, damage, speed, ground/flying) so that even unseen classes start with meaningful embeddings. This is a form of zero-shot transfer.

3. **Augment training data with synthetic enemy placements.** During BC data collection, we only see enemies that the Trainer AI plays. We can augment by injecting synthetic enemy unit observations into training frames. This is straightforward because we control the GameState-to-tensor pipeline.

4. **Functional category fallback.** Maintain a mapping from class_name to functional category (tank, ranged DPS, swarm, building, spell). If the embedding is untrained, fall back to the category embedding. This gives a coarse but meaningful representation.

**Recommendation:** Use mitigation #1 (default embedding for unseen classes) as the simplest approach. Map all unseen ground units to a single `unknown_ground` ID and all unseen flying units to `unknown_flying`. This reduces the problem to the same behavior as additive counting for rare classes while preserving learned identity for common ones.

### Would PPO eventually handle rare troop issues?

**Yes, with important caveats.**

PPO explores by playing live games against opponents. Over many games, it will encounter rare troops and update their embeddings through gradient flow. The question is how quickly.

**Favorable factors:**
- PPO sees whatever the opponent plays, not just our deck
- Over hundreds of games, even rare troops appear multiple times
- The embedding only needs a few dozen examples to become meaningful (not thousands)

**Unfavorable factors:**
- PPO updates are noisy -- a rare troop appearing once in 50 games gets very few gradient updates
- If the initial BC policy avoids situations where rare troops appear (e.g., never pushes against a specific defense), PPO may not explore those states
- The embedding table is updated globally, but rare classes get proportionally fewer updates

**Estimated timeline:** With 100-200 PPO training games (each ~3 minutes, so 5-10 hours of gameplay), the agent should encounter most common and semi-rare troops enough times (5-20 encounters each) for the embeddings to converge to useful representations. Truly rare troops (miner, lava hound, sparky in specific archetypes) may need 500+ games.

**Practical recommendation:** Train BC on 30-40 games. Then run PPO for 100-200 games. After PPO training, evaluate which embeddings are still under-trained by checking gradient magnitude or embedding norm. If specific classes have low-norm embeddings, either:
- Continue PPO training (the embeddings will improve over time)
- Add synthetic observations for those classes to the BC dataset and fine-tune

---

## Summary of Recommendations

### Recommended approach: Per-cell encoding with learned embeddings

| Design Choice | Recommendation | Rationale |
|--------------|----------------|-----------|
| Grid collision handling | PositionFinder (same as KataCR) | Enables one-unit-per-cell, simple implementation |
| Per-cell feature vector | class_id (int) + belonging + arena_mask + tower_hp | Minimal viable features, add health bars later |
| Class representation | Learned embeddings (8-dim) in BC network | Avoids false ordering, learns semantic similarity |
| Arena observation shape | `arena_class (32,18)` int + `arena_features (32,18,4)` float | Clean separation of integer and continuous data |
| PositionFinder sort order | Enemy first, then ally, bottom-to-top | Matches KataCR, gives priority to enemy positions |
| Rare class handling | Default "unknown" embedding for unseen classes | Degrades gracefully to additive-counting behavior |
| Health bars | Defer to later iteration | Complex extraction, bar-body matching not built |
| Action space | No change (Discrete 2305) | Action encoding is independent of observation |
| Data volume | 30-40 games (unchanged) | Observation encoding does not change action coverage needs |

### Implementation order

1. Add `CLASS_NAME_TO_ID` mapping to `encoder_constants.py`
2. Implement `PositionFinder` class (can be in `state_encoder.py` or a new `position_finder.py`)
3. Rewrite `_encode_arena()` in `state_encoder.py`
4. Update `observation_space` definition
5. Build `CRFeatureExtractor` with `nn.Embedding` in the BC network
6. Add unit tests comparing old vs new encoding on known GameState inputs

### Open questions for the team

1. **Hybrid approach?** Should we keep a density count channel alongside per-cell features to give the model both unit identity and explicit count information?

2. **Embedding dimension.** KataCR uses 8 dimensions. Should we experiment with 4 or 16? Lower dims are faster but less expressive. With only 155 classes, 8 is likely sufficient.

3. **Belonging source.** Per-cell encoding makes belonging more important (it is a per-unit field, not a channel-level split). Should we prioritize retraining YOLOv8 with belonging labels before switching encodings? Or use the Y-heuristic initially?

4. **Tower encoding.** Should towers also get class_id + belonging per-cell treatment, or keep the current HP-fraction channels? Towers occupy fixed known positions, so per-cell features add less value for them.

5. **Spell encoding.** Current approach has no ally/enemy distinction for spells (CH_SPELL). Per-cell encoding can include belonging for spells, but our YOLO model does not detect spell belonging. Should spells keep a separate encoding path?

---

## Appendix: Code Reference

### Files to modify

| File | Path | Change Type |
|------|------|-------------|
| state_encoder.py | `src/encoder/state_encoder.py` | Major rewrite of `_encode_arena()`, `observation_space` |
| encoder_constants.py | `src/encoder/encoder_constants.py` | Add CLASS_NAME_TO_ID, update channel constants |
| coord_utils.py | `src/encoder/coord_utils.py` | Add `pixel_to_cell_float()` |
| (new) position_finder.py | `src/encoder/position_finder.py` | PositionFinder class |

### Files unchanged

| File | Path | Why |
|------|------|-----|
| state_builder.py | `src/pipeline/state_builder.py` | Produces GameState, not tensors |
| game_state.py | `src/pipeline/game_state.py` | Data model is encoding-agnostic |
| coord_utils.py (action funcs) | `src/encoder/coord_utils.py` | Action encoding is separate |
| Action mask logic | `state_encoder.py:120-163` | Operates on cards/elixir, not arena |
