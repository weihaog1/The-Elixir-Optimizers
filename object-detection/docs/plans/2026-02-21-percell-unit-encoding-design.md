# Per-Cell Unit Encoding Design Document

## KataCR-style state encoder rewrite for the CS175 Clash Royale RL pipeline

**Date:** 2026-02-21
**Author:** Alan Guo, with analysis from KataCR research team
**Status:** Design approved, pending implementation
**Scope:** `src/encoder/` module rewrite + new `position_finder.py` + documentation

---

## Table of Contents

1. [Motivation and Background](#1-motivation-and-background)
2. [Current System (Before)](#2-current-system-before)
3. [New System (After)](#3-new-system-after)
4. [PositionFinder: Collision Resolution](#4-positionfinder-collision-resolution)
5. [Arena Channel Layout](#5-arena-channel-layout)
6. [CLASS_NAME_TO_ID Mapping](#6-class_name_to_id-mapping)
7. [Observation Space Definition](#7-observation-space-definition)
8. [Encoding Algorithm (_encode_arena)](#8-encoding-algorithm-_encode_arena)
9. [Unchanged Components](#9-unchanged-components)
10. [File-by-File Change Manifest](#10-file-by-file-change-manifest)
11. [Modifications Outside the Encoder Module](#11-modifications-outside-the-encoder-module)
12. [Pipeline Usage: Inputs and Outputs](#12-pipeline-usage-inputs-and-outputs)
13. [BC Network Integration (Future)](#13-bc-network-integration-future)
14. [Design Decisions and Rationale](#14-design-decisions-and-rationale)
15. [Known Limitations and Trade-offs](#15-known-limitations-and-trade-offs)
16. [Testing Strategy](#16-testing-strategy)
17. [Next Steps After Implementation](#17-next-steps-after-implementation)
18. [Appendix A: PositionFinder Displacement Analysis](#appendix-a-positionfinder-displacement-analysis)
19. [Appendix B: Full Class ID Table (Excerpt)](#appendix-b-full-class-id-table-excerpt)

---

## 1. Motivation and Background

### Problem with additive counting

The current `StateEncoder` uses a 7-channel arena grid where each channel holds an
**additive count** of units in that cell. For example, if 3 enemy ground units occupy
cell (5, 20), then `arena[20, 5, CH_ENEMY_GROUND] = 3.0`. This gives the agent
explicit density information but **no unit identity** -- it cannot distinguish 3
skeletons from 3 barbarians, even though the correct response differs dramatically.

### KataCR's per-cell approach

KataCR uses a fundamentally different encoding: each cell holds **at most one unit**
with a rich per-cell feature vector (class ID, belonging flag, health bar images).
A `PositionFinder` algorithm resolves collisions when multiple units map to the same
cell by displacing units to the nearest unoccupied cell.

This gives the agent **unit identity** -- it knows exactly which troop is at each
position -- at the cost of losing explicit density counts (the agent must infer
density from spatial patterns of same-class cells).

### Why switch now

The BC/PPO pipeline is about to be built. The observation encoding is the foundation
that everything downstream depends on. Switching after BC data is collected would
require re-encoding or re-collecting data. Switching now, before data collection
begins, has zero cost.

The per-cell approach is better for our use case because:
- Knowing WHICH enemy troop is approaching (mini-pekka vs skeleton) determines
  whether to commit 5+ elixir to defend or use a cheap card
- Our 8-card deck contains cards with very different strategic responses to
  different enemy compositions
- The BC network can learn unit-to-unit matchup knowledge via embeddings
- KataCR demonstrated this encoding works for Clash Royale RL specifically

---

## 2. Current System (Before)

### Arena tensor: (32, 18, 7) float32

```
Channel 0: CH_ALLY_GROUND    - Count of allied ground units in cell (0, 1, 2, ...)
Channel 1: CH_ALLY_FLYING    - Count of allied flying units in cell
Channel 2: CH_ENEMY_GROUND   - Count of enemy ground units in cell
Channel 3: CH_ENEMY_FLYING   - Count of enemy flying units in cell
Channel 4: CH_ALLY_TOWER_HP  - Allied tower HP fraction (0-1)
Channel 5: CH_ENEMY_TOWER_HP - Enemy tower HP fraction (0-1)
Channel 6: CH_SPELL          - Count of spell effects in cell
```

### Encoding logic (current `_encode_arena`)

```python
for unit in state.units:
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

### What the agent sees

For a cell with 3 enemy skeletons and 1 enemy barbarian:
```
arena[row, col] = [0, 0, 4.0, 0, 0, 0, 0]
                          ^--- 4 enemy ground units (identity unknown)
```

### What the agent does NOT see

- Which unit types are at each position
- Whether a cluster is a swarm card (Skeleton Army) or independent units
- Strategic threat level differences between unit types

---

## 3. New System (After)

### Arena tensor: (32, 18, 6) float32

```
Channel 0: CH_CLASS_ID       - Normalized class ID (class_idx / NUM_CLASSES), 0.0 = empty
Channel 1: CH_BELONGING      - -1.0 = ally, +1.0 = enemy, 0.0 = empty
Channel 2: CH_ARENA_MASK     - 1.0 if unit present, 0.0 if empty
Channel 3: CH_ALLY_TOWER_HP  - Allied tower HP fraction (0-1) [kept from current]
Channel 4: CH_ENEMY_TOWER_HP - Enemy tower HP fraction (0-1) [kept from current]
Channel 5: CH_SPELL          - Count of spell effects in cell [kept from current]
```

### What the agent now sees

For a cell with 1 enemy skeleton (after PositionFinder displaces the other 3 to
nearby cells):
```
arena[row, col] = [0.168, 1.0, 1.0, 0, 0, 0]
                   ^       ^    ^
                   |       |    +--- unit present
                   |       +--- enemy
                   +--- skeleton (class_idx=26, 26/155=0.168)
```

And 3 adjacent cells also have skeleton entries, forming a spatial cluster.

### Key structural change

**Before:** Multiple units per cell, no identity. Unit counts are explicit.
**After:** One unit per cell (enforced by PositionFinder), with identity. Counts are
implicit (cluster size = count).

---

## 4. PositionFinder: Collision Resolution

### Purpose

Enforce the one-unit-per-cell invariant. When multiple units map to the same grid
cell, displace later units to the nearest unoccupied cell.

### Algorithm

```python
class PositionFinder:
    """Resolve grid cell collisions by displacing units to nearest free cell.

    KataCR's approach: katacr/policy/offline/dataset.py lines 143-159.
    Fresh instance per frame -- the used-cell grid resets each frame.
    """

    def __init__(self, rows: int = 32, cols: int = 18):
        self.rows = rows
        self.cols = cols
        self.used = np.zeros((rows, cols), dtype=np.bool_)
        # Pre-compute cell centers for distance calculations
        row_coords, col_coords = np.meshgrid(
            np.arange(rows) + 0.5, np.arange(cols) + 0.5, indexing="ij"
        )
        self.centers = np.stack([row_coords, col_coords], axis=-1)  # (R, C, 2)

    def find_position(self, col_f: float, row_f: float) -> tuple[int, int]:
        """Find the best cell for a unit at continuous position (col_f, row_f).

        If the natural cell is free, return it. Otherwise, find the nearest
        unoccupied cell by Euclidean distance to cell centers.

        Args:
            col_f: Continuous column coordinate (0.0 to 17.999...).
            row_f: Continuous row coordinate (0.0 to 31.999...).

        Returns:
            (col, row) integer cell coordinates.
        """
        row = int(np.clip(row_f, 0, self.rows - 1))
        col = int(np.clip(col_f, 0, self.cols - 1))

        if not self.used[row, col]:
            self.used[row, col] = True
            return col, row

        # Cell occupied -- find nearest free cell
        available_mask = ~self.used
        available_centers = self.centers[available_mask]  # (N_free, 2)

        if len(available_centers) == 0:
            # Extremely unlikely: all 576 cells occupied
            return col, row

        query = np.array([[row_f, col_f]])
        distances = scipy.spatial.distance.cdist(query, available_centers)
        nearest_idx = np.argmin(distances)
        available_indices = np.argwhere(available_mask)
        row, col = available_indices[nearest_idx]

        self.used[row, col] = True
        return int(col), int(row)
```

### Processing order

Units are sorted before PositionFinder processes them:

```python
sorted_units = sorted(
    non_spell_non_tower_units,
    key=lambda u: (-u.belonging, -u.center[1])
)
```

This means:
1. **Enemy units first** (belonging=1 sorts before belonging=0 with negative key)
2. **Bottom-to-top within each faction** (higher center_y = closer to player side, processed first)

**Rationale:** Enemy positions are more strategically important for defensive
decisions. Processing enemies first gives them accurate positions; allies (which
we control) get slightly displaced but our agent knows where it placed them.

### Performance characteristics

- Per-frame cost: K units x distance computation against (576 - occupied) cells
- Typical frame: 15-30 units, each `cdist` call is O(576) at worst
- Total: well under 1ms. **Not a performance concern.**
- `scipy.spatial.distance.cdist` is the expensive call but operates on tiny arrays

### Displacement analysis

See [Appendix A](#appendix-a-positionfinder-displacement-analysis) for detailed
analysis of positional error introduced by displacement. Summary: 1-3 cell error
for worst-case swarms in our deck (Royal Recruits = 6 units, Royal Hogs = 4 units).
The 15-skeleton Skeleton Army worst case does not appear in our deck.

---

## 5. Arena Channel Layout

### New constants (encoder_constants.py)

```python
# Per-cell unit identity channels
CH_CLASS_ID = 0        # Normalized class ID: class_idx / NUM_CLASSES (0.0 = empty)
CH_BELONGING = 1       # -1.0 = ally, +1.0 = enemy, 0.0 = empty cell
CH_ARENA_MASK = 2      # 1.0 = unit present, 0.0 = empty

# Continuous channels (unchanged purpose, renumbered)
CH_ALLY_TOWER_HP = 3   # Tower HP fraction 0-1
CH_ENEMY_TOWER_HP = 4  # Tower HP fraction 0-1
CH_SPELL = 5           # Additive count of spell effects

NUM_ARENA_CHANNELS = 6
```

### Removed constants

```python
# These are REMOVED:
CH_ALLY_GROUND = 0
CH_ALLY_FLYING = 1
CH_ENEMY_GROUND = 2
CH_ENEMY_FLYING = 3
```

### Why arena_mask is needed

Channel 2 (`CH_ARENA_MASK`) distinguishes "empty cell" from "class_id=0 unit."
Without it, the network cannot tell if a cell with `[0.0, 0.0, ...]` is empty
or contains class 0 with unknown belonging. The mask is a clean binary signal
for "there is something here."

### Why spells keep a separate channel

Spells are transient visual effects (arrows rain, fireball impact, zap flash).
They:
1. Do NOT have meaningful belonging in our YOLO model output
2. Often overlap with troop positions (a fireball landing on enemy troops)
3. Should NOT consume a cell and displace a troop from its true position
4. Are better represented as additive effects (multiple spells can hit one cell)

Spells bypass PositionFinder entirely and accumulate in CH_SPELL as before.

---

## 6. CLASS_NAME_TO_ID Mapping

### Source

Built from `src/generation/label_list.unit_list`, which contains 155 class names
in a defined order (indices 0-154 in KataCR's convention). We add 1 to each index
so that **class ID 0 is reserved for "empty cell."**

```python
# In encoder_constants.py:
from src.generation.label_list import unit_list, unit2idx

NUM_CLASSES = len(unit_list)  # 155

# CLASS_NAME_TO_ID maps detection class_name -> integer ID (1-indexed)
# ID 0 is reserved for empty cells
CLASS_NAME_TO_ID: dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(unit_list)
}
```

### Coverage

All 155 detection classes are mapped, including:
- Towers (king-tower, queen-tower, etc.) -- mapped but towers use HP channels, not per-cell
- Other/UI elements (bar, clock, emote, etc.) -- mapped but filtered out during encoding
- Spells -- mapped but routed to CH_SPELL instead of per-cell

Only "ground" and "flying" type units actually get placed via PositionFinder.

### Normalization

Class IDs are stored normalized: `class_idx / NUM_CLASSES` in the arena tensor.
This keeps values in [0, 1] range consistent with other channels. The BC network's
`CRFeatureExtractor` will denormalize back to integers for embedding lookup.

---

## 7. Observation Space Definition

### New observation space

```python
self.observation_space = gym.spaces.Dict({
    "arena": gym.spaces.Box(
        low=-1.0,   # CH_BELONGING can be -1.0 (ally)
        high=10.0,  # CH_SPELL can accumulate (kept from before)
        shape=(GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),  # (32, 18, 6)
        dtype=np.float32,
    ),
    "vector": gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(NUM_VECTOR_FEATURES,),  # (23,) -- UNCHANGED
        dtype=np.float32,
    ),
})
```

### Changes from current

| Property | Before | After |
|----------|--------|-------|
| Arena shape | (32, 18, 7) | (32, 18, 6) |
| Arena low | 0.0 | -1.0 |
| Arena high | 10.0 | 10.0 |
| Vector shape | (23,) | (23,) -- no change |
| Action space | Discrete(2305) | Discrete(2305) -- no change |

### Per-channel value ranges

| Channel | Name | Min | Max | Type |
|---------|------|-----|-----|------|
| 0 | class_id | 0.0 | 1.0 | Normalized (0 = empty, >0 = unit class) |
| 1 | belonging | -1.0 | 1.0 | Categorical (-1=ally, 0=empty, +1=enemy) |
| 2 | arena_mask | 0.0 | 1.0 | Binary |
| 3 | ally_tower_hp | 0.0 | 1.0 | Continuous fraction |
| 4 | enemy_tower_hp | 0.0 | 1.0 | Continuous fraction |
| 5 | spell | 0.0 | ~5.0 | Additive count |

---

## 8. Encoding Algorithm (_encode_arena)

### Complete pseudocode

```python
def _encode_arena(self, state: GameState) -> np.ndarray:
    arena = np.zeros(
        (GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS), dtype=np.float32
    )
    fw = state.frame_width or 540
    fh = state.frame_height or 960

    # --- Per-cell units via PositionFinder ---
    pf = PositionFinder(rows=GRID_ROWS, cols=GRID_COLS)

    # Separate units by type
    placeable_units = []
    for unit in state.units:
        unit_type = UNIT_TYPE_MAP.get(unit.class_name, "ground")
        if unit_type in ("tower", "other"):
            continue  # Towers handled below; UI elements skipped
        if unit_type == "spell":
            # Spells go to CH_SPELL (additive, no PositionFinder)
            cx, cy = unit.center
            col, row = pixel_to_cell(cx, cy, fw, fh)
            arena[row, col, CH_SPELL] += 1.0
            continue
        placeable_units.append(unit)

    # Sort: enemy first (belonging=1), then ally (belonging=0)
    # Within each faction: bottom-to-top (higher center_y first)
    placeable_units.sort(key=lambda u: (-u.belonging, -u.center[1]))

    for unit in placeable_units:
        cx, cy = unit.center
        col_f, row_f = pixel_to_cell_float(cx, cy, fw, fh)
        col, row = pf.find_position(col_f, row_f)

        class_idx = CLASS_NAME_TO_ID.get(unit.class_name, 0)
        if class_idx == 0:
            continue  # Unknown class, skip

        arena[row, col, CH_CLASS_ID] = class_idx / NUM_CLASSES
        arena[row, col, CH_BELONGING] = -1.0 if unit.belonging == 0 else 1.0
        arena[row, col, CH_ARENA_MASK] = 1.0

    # --- Towers (same as current, different channel indices) ---
    tower_slots = [
        (state.player_king_tower, True),
        (state.player_left_princess, True),
        (state.player_right_princess, True),
        (state.enemy_king_tower, False),
        (state.enemy_left_princess, False),
        (state.enemy_right_princess, False),
    ]
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

### `pixel_to_cell_float` (new function in coord_utils.py)

```python
def pixel_to_cell_float(
    cx: float, cy: float, frame_w: int, frame_h: int
) -> tuple[float, float]:
    """Convert pixel bbox center to continuous grid cell coordinates.

    Unlike pixel_to_cell() which returns clamped integers, this returns
    float coordinates for PositionFinder input. The fractional part
    captures sub-cell positioning.

    Args:
        cx: Center x in pixels.
        cy: Center y in pixels.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        (col_f, row_f) as continuous floats. NOT clamped -- caller
        (PositionFinder) handles clamping.
    """
    x_norm = cx / frame_w
    y_norm = cy / frame_h
    arena_y_frac = (y_norm - ARENA_Y_START_FRAC) / _ARENA_Y_SPAN
    col_f = x_norm * GRID_COLS
    row_f = arena_y_frac * GRID_ROWS
    return col_f, row_f
```

---

## 9. Unchanged Components

These components are explicitly **not modified** by this change:

### Action space: Discrete(2305)

The action space encodes where to place cards, not how the board is perceived.
`Discrete(2305) = 4 cards x 576 cells + 1 no-op` remains identical.

Functions in `coord_utils.py` that are unchanged:
- `action_to_placement(action_idx) -> (card_id, col, row) | None`
- `placement_to_action(card_id, col, row) -> int`
- `cell_to_norm(col, row) -> (x_norm, y_norm)`
- `norm_to_cell(x_norm, y_norm) -> (col, row)`
- `pixel_to_cell(cx, cy, fw, fh) -> (col, row)` (kept for tower encoding and spells)

### Action mask logic (state_encoder.py:120-163)

The action mask checks card availability and elixir cost. It does not reference
any arena channels. Completely unaffected.

### Vector encoding (state_encoder.py:226-274)

`_encode_vector()` produces the (23,) scalar feature vector. It reads from
`GameState` fields (elixir, time, towers, cards), not from the arena tensor.
Completely unaffected.

### GameState dataclass (game_state.py)

The `GameState`, `Tower`, `Unit`, and `Card` dataclasses are encoding-agnostic.
They store raw detection data. No changes needed.

### StateBuilder (state_builder.py)

`StateBuilder.build_state()` produces `GameState` from YOLO detections + OCR.
It does not produce tensors. No changes needed.

The Y-position heuristic for belonging (`_infer_unit_belonging` at line 181)
continues to work exactly as before. The belonging value it writes to `Unit.belonging`
is consumed by the new `_encode_arena()` method identically to how the old one
consumed it.

### Deck configuration

`DECK_CARDS`, `CARD_ELIXIR_COST`, `CARD_IS_SPELL`, `DECK_CARD_TO_IDX` are all
unchanged. They are used by `_encode_vector()` and `action_mask()`, not by
`_encode_arena()`.

---

## 10. File-by-File Change Manifest

### New files

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `src/encoder/position_finder.py` | PositionFinder class | ~60 |

### Modified files

| File | Change Type | What Changes |
|------|-------------|-------------|
| `src/encoder/encoder_constants.py` | Moderate | Remove CH_ALLY_GROUND/FLYING/ENEMY_GROUND/FLYING. Add CH_CLASS_ID/BELONGING/ARENA_MASK. Add CLASS_NAME_TO_ID, NUM_CLASSES. Renumber CH_ALLY_TOWER_HP, CH_ENEMY_TOWER_HP, CH_SPELL. Update NUM_ARENA_CHANNELS to 6. |
| `src/encoder/state_encoder.py` | Major | Rewrite `_encode_arena()`. Update observation_space bounds and shape. Update module docstring. Import PositionFinder and new constants. |
| `src/encoder/coord_utils.py` | Minor | Add `pixel_to_cell_float()` function (~20 lines). |
| `src/encoder/__init__.py` | Minor | Export `PositionFinder`. |
| `src/encoder/CLAUDE.md` | Rewrite | Update observation space docs to reflect new channels. |

### Unchanged files

| File | Why Unchanged |
|------|--------------|
| `src/pipeline/game_state.py` | Data model is encoding-agnostic |
| `src/pipeline/state_builder.py` | Produces GameState, not tensors |
| `src/encoder/coord_utils.py` (action funcs) | Action encoding is independent |
| `src/generation/label_list.py` | Read-only source for class names |
| All files outside `src/encoder/` | Observation encoding is self-contained |

---

## 11. Modifications Outside the Encoder Module

**None.** This change is entirely contained within `src/encoder/`. No files in
`src/pipeline/`, `src/generation/`, `src/detection/`, `src/classification/`,
`src/ocr/`, `scripts/`, or `configs/` need modification.

The only external dependency is the **read-only import** of `unit_list` and
`unit2idx` from `src/generation/label_list.py` to build the `CLASS_NAME_TO_ID`
mapping. This import already exists in `encoder_constants.py` (it currently imports
`ground_unit_list`, `flying_unit_list`, etc.).

### Impact on downstream consumers

Any code that currently reads the `StateEncoder.observation_space` or calls
`StateEncoder.encode()` will see:
- Arena shape changed from (32, 18, 7) to (32, 18, 6)
- Arena channel semantics changed entirely
- Arena value range now includes -1.0 (belonging channel)

**Currently, no downstream consumers exist.** The BC model and Gym environment
have not been built yet. This is precisely why we are making this change now --
before any downstream code depends on the old encoding.

---

## 12. Pipeline Usage: Inputs and Outputs

### Full pipeline flow with new encoder

```
Screen Capture (mss)
       |
       v
CRDetector.detect(frame)
       |
       v
List[Detection(x1,y1,x2,y2,conf,cls)]
       |
       v
StateBuilder.build_state(frame)
  |--- _process_detections() --> Units with Y-heuristic belonging
  |--- _process_ocr()        --> Timer, elixir, tower HP
       |
       v
GameState
  .units: List[Unit]   -- class_name, belonging (0/1), bbox, confidence
  .cards: List[Card]   -- slot, class_name, elixir_cost (currently empty)
  .elixir: Optional[int]
  .time_remaining: Optional[int]
  .towers: 6 Optional[Tower] slots
       |
       v
StateEncoder.encode(game_state)
  |--- _encode_arena(state)
  |     |--- PositionFinder (collision resolution)
  |     |--- Per-cell: class_id, belonging, arena_mask
  |     |--- Towers: HP fractions (fixed cells)
  |     |--- Spells: additive count (bypass PositionFinder)
  |     v
  |   arena: np.float32 (32, 18, 6)
  |
  |--- _encode_vector(state)  [UNCHANGED]
  |     v
  |   vector: np.float32 (23,)
  |
  v
obs = {"arena": (32,18,6), "vector": (23,)}

StateEncoder.action_mask(game_state)  [UNCHANGED]
  v
mask: np.bool_ (2305,)
```

### Input requirements

`StateEncoder.encode()` requires a `GameState` object with:

| Field | Required | Used By | Notes |
|-------|----------|---------|-------|
| `units` | Yes | `_encode_arena` | List of Unit with class_name, belonging, bbox |
| `frame_width` | Yes | `_encode_arena` | For pixel-to-cell conversion (default 540) |
| `frame_height` | Yes | `_encode_arena` | For pixel-to-cell conversion (default 960) |
| `player_king_tower` | Optional | `_encode_arena`, `_encode_vector` | Tower HP encoding |
| `player_left_princess` | Optional | `_encode_arena`, `_encode_vector` | Tower HP encoding |
| `player_right_princess` | Optional | `_encode_arena`, `_encode_vector` | Tower HP encoding |
| `enemy_king_tower` | Optional | `_encode_arena`, `_encode_vector` | Tower HP encoding |
| `enemy_left_princess` | Optional | `_encode_arena`, `_encode_vector` | Tower HP encoding |
| `enemy_right_princess` | Optional | `_encode_arena`, `_encode_vector` | Tower HP encoding |
| `elixir` | Optional | `_encode_vector`, `action_mask` | Normalized to /10 |
| `time_remaining` | Optional | `_encode_vector` | Normalized to /300 |
| `is_overtime` | Optional | `_encode_vector` | Binary flag |
| `cards` | Optional | `_encode_vector`, `action_mask` | Card hand (currently empty) |

### Output specification

**`encode(state) -> dict`:**

```python
{
    "arena": np.ndarray,   # shape (32, 18, 6), dtype float32
                           # Channel 0: class_id / 155 (0.0 = empty)
                           # Channel 1: belonging (-1.0/0.0/+1.0)
                           # Channel 2: arena_mask (0.0/1.0)
                           # Channel 3: ally tower HP fraction
                           # Channel 4: enemy tower HP fraction
                           # Channel 5: spell count
    "vector": np.ndarray,  # shape (23,), dtype float32 [UNCHANGED]
}
```

**`action_mask(state) -> np.ndarray`:**

```python
np.ndarray  # shape (2305,), dtype bool [UNCHANGED]
            # True = action is valid, False = masked out
```

### Usage example (updated)

```python
from src.encoder import StateEncoder
from src.pipeline.game_state import GameState

encoder = StateEncoder()

# Encode a game state
obs = encoder.encode(game_state)
mask = encoder.action_mask(game_state)

# Inspect arena encoding
arena = obs["arena"]  # (32, 18, 6)

# Find all occupied cells
occupied = arena[:, :, 2] > 0  # CH_ARENA_MASK
print(f"Occupied cells: {occupied.sum()}")

# Find all enemy units
enemy_cells = arena[:, :, 1] > 0  # CH_BELONGING > 0 means enemy
print(f"Enemy unit cells: {enemy_cells.sum()}")

# Get class ID at a specific cell
class_id_norm = arena[20, 5, 0]  # CH_CLASS_ID
if class_id_norm > 0:
    class_idx = round(class_id_norm * 155)
    print(f"Unit at (5, 20): class index {class_idx}")

# For SB3 policy definition:
policy_kwargs = dict(
    features_extractor_class=CRFeatureExtractor,  # future: learned embeddings
    features_extractor_kwargs=dict(features_dim=128),
)
```

---

## 13. BC Network Integration (Future)

This section describes how the BC network will consume the new encoding. This is
**not implemented now** -- it is documented here for planning purposes.

### CRFeatureExtractor with learned embeddings

The class_id channel stores a normalized float. For the BC/PPO network, a
custom `BaseFeaturesExtractor` will:

1. Extract channel 0, denormalize to integer: `class_ids = (arena[:,:,0] * NUM_CLASSES).round().long()`
2. Feed through `nn.Embedding(NUM_CLASSES + 1, 8)` to get 8-dim vectors per cell
3. Concatenate with channels 1-5 (belonging, mask, tower HP, spell)
4. Feed the (32, 18, 13) tensor through a CNN

```python
class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=8, features_dim=128):
        super().__init__(observation_space, features_dim=features_dim)
        self.class_embed = nn.Embedding(NUM_CLASSES + 1, embed_dim)
        # embed_dim + 5 remaining channels = 13 input channels to CNN
        self.arena_cnn = nn.Sequential(
            nn.Conv2d(embed_dim + 5, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # ... vector branch + combination layers
```

### Why embeddings matter

A normalized float class ID implies false ordering: skeleton (26/155=0.168) appears
"closer" to electro-spirit (30/155=0.194) than to minion (52/155=0.335). Learned
embeddings let the network discover strategic similarities (e.g., swarm units cluster
together, tanks cluster together) without imposed ordering.

With 155 classes and 8 dimensions, the embedding table is only 1,240 parameters --
negligible relative to CNN weights.

### Embedding dimension: 8

As specified, using 8-dimensional embeddings (matching KataCR's choice). Sufficient
for 155 classes. Lower (4) loses expressiveness; higher (16) adds unnecessary
parameters. Can be tuned later without changing the encoder.

---

## 14. Design Decisions and Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Collision resolution | PositionFinder (KataCR's algorithm) | Proven approach. Simple to implement. Bounded displacement error. |
| Belonging source | Y-position heuristic (current) | As specified. Retraining YOLO with belonging labels is separate work. Y-heuristic is imperfect but functional. |
| Embedding dimension | 8 (deferred to BC network) | As specified. Encoder stores normalized float; embedding is network-side. |
| Tower encoding | HP-fraction channels (current) | As specified. Towers occupy fixed known positions. HP is continuous, not categorical. Per-cell class_id adds no value for towers. |
| Spell encoding | Separate additive channel, no PositionFinder | As specified. Spells are transient, overlap with troops, have no reliable belonging. Keeping additive channel preserves information without cell displacement. |
| Health bars | Not included | Deferred. Bar-body matching not built. Would require StateBuilder changes and adds complexity with unclear benefit for MVP. |
| Sort order | Enemy first, bottom-to-top | KataCR convention. Enemies get accurate positions because defensive decisions are more strategically critical. |
| class_id=0 reserved for empty | Yes | Clean separation. arena_mask=0 AND class_id=0 means empty. Any class_id>0 means a real unit. |
| Normalization of class_id | Divide by NUM_CLASSES | Keeps channel 0 in [0, 1] range. BC network denormalizes for embedding lookup. |
| unknown belonging default | +1.0 (enemy) | Units with belonging=-1 (unknown) are treated as enemy for conservative defensive play. |

---

## 15. Known Limitations and Trade-offs

### Loss of explicit density information

The most significant trade-off. With additive counting, `arena[r,c,CH_ENEMY_GROUND]=6`
immediately tells the agent "6 enemies here." With per-cell encoding, the agent
must learn to count adjacent same-class cells. This is harder to learn but still
learnable -- a CNN with sufficient receptive field can detect clusters.

**Mitigating factor:** Our deck does not contain Skeleton Army (15 units). The
worst swarm case is Royal Recruits (6 units, naturally split into 2 groups of 3)
and Royal Hogs (4 units). Displacement is modest (1-2 cells).

### PositionFinder displacement introduces positional error

Units displaced to neighboring cells appear at slightly wrong positions. A unit at
true position (5, 20) might appear at (6, 21) if its cell was occupied. The error
magnitude is bounded by the cluster density.

**Mitigating factor:** The BC/PPO network's CNN has multi-cell receptive fields. A
1-2 cell error is within the receptive field of a 3x3 conv kernel. The network can
learn to aggregate nearby cells.

### Order dependency in PositionFinder

The first unit processed claims its true cell; later units get displaced. Enemy
units are processed first (getting accurate positions), allies second. This means
ally unit positions are systematically less accurate than enemy positions.

**Mitigating factor:** The agent controls ally placement (it chose where to deploy),
so exact ally positions are less critical for decision-making.

### Y-heuristic belonging errors

The Y-position heuristic (`arena_mid = frame_height * 0.42`) incorrectly classifies
units that have crossed the river. An allied hog-rider in enemy territory gets
classified as enemy (belonging=1). This pre-existing issue becomes more impactful
with per-cell encoding because belonging is now a per-unit field rather than a
channel-level split.

**Not new:** This limitation exists in the current system. The fix is retraining
YOLO with belonging labels (separate work, not part of this change).

### Normalized float class_id loses precision for embedding lookup

Storing class_id as `class_idx / 155` then recovering via `round(val * 155)` can
introduce rounding errors. For class indices 0-155, the maximum float32 round-trip
error is less than 1e-6, which rounds correctly. Not a practical concern.

---

## 16. Testing Strategy

### Unit tests for PositionFinder

```python
def test_position_finder_empty_grid():
    """First unit always gets its natural cell."""
    pf = PositionFinder()
    col, row = pf.find_position(5.3, 20.7)
    assert col == 5 and row == 20

def test_position_finder_collision():
    """Second unit at same cell gets displaced to nearest free."""
    pf = PositionFinder()
    pf.find_position(5.0, 20.0)  # Claims (5, 20)
    col, row = pf.find_position(5.0, 20.0)  # Must displace
    assert (col, row) != (5, 20)
    # Should be adjacent (distance = 1 cell)
    assert abs(col - 5) <= 1 and abs(row - 20) <= 1

def test_position_finder_swarm():
    """15 units at same cell: all get unique positions."""
    pf = PositionFinder()
    positions = set()
    for _ in range(15):
        col, row = pf.find_position(9.0, 16.0)
        positions.add((col, row))
    assert len(positions) == 15  # All unique
```

### Unit tests for new _encode_arena

```python
def test_encode_arena_single_unit():
    """Single unit produces correct per-cell features."""
    state = GameState(frame_width=540, frame_height=960)
    state.units = [Unit(class_name="knight", belonging=0, bbox=(270,480,290,500), confidence=0.9)]
    encoder = StateEncoder()
    obs = encoder.encode(state)
    arena = obs["arena"]

    # Find the occupied cell
    occupied = arena[:, :, CH_ARENA_MASK] > 0
    assert occupied.sum() == 1

    row, col = np.argwhere(occupied)[0]
    assert arena[row, col, CH_CLASS_ID] > 0  # Has a class
    assert arena[row, col, CH_BELONGING] == -1.0  # Ally
    assert arena[row, col, CH_ARENA_MASK] == 1.0

def test_encode_arena_spell_bypasses_position_finder():
    """Spells go to CH_SPELL, not per-cell channels."""
    state = GameState(frame_width=540, frame_height=960)
    state.units = [Unit(class_name="fireball", belonging=0, bbox=(270,480,290,500), confidence=0.9)]
    encoder = StateEncoder()
    obs = encoder.encode(state)
    arena = obs["arena"]

    # No per-cell unit
    assert arena[:, :, CH_ARENA_MASK].sum() == 0
    # Spell channel has a count
    assert arena[:, :, CH_SPELL].sum() > 0

def test_encode_arena_shape():
    """Output shape is (32, 18, 6)."""
    encoder = StateEncoder()
    state = GameState()
    obs = encoder.encode(state)
    assert obs["arena"].shape == (32, 18, 6)
    assert obs["vector"].shape == (23,)

def test_action_mask_unchanged():
    """Action mask behavior is identical to before."""
    encoder = StateEncoder()
    state = GameState(elixir=5)
    # With no cards, only no-op is valid
    mask = encoder.action_mask(state)
    assert mask[NOOP_ACTION] == True
    assert mask[:NOOP_ACTION].sum() == 0
```

### Regression test: vector encoding unchanged

```python
def test_vector_encoding_unchanged():
    """Vector encoding produces identical output to before."""
    # Compare against known-good output from current encoder
    state = GameState(elixir=5, time_remaining=120)
    encoder = StateEncoder()
    vec = encoder.encode(state)["vector"]
    assert vec[0] == 0.5  # 5/10
    assert vec[1] == 0.4  # 120/300
```

---

## 17. Next Steps After Implementation

After the per-cell encoder is implemented and tested, the pipeline progression is:

### Immediate next: Wire CardPredictor into StateBuilder

`GameState.cards` is currently always empty because `StateBuilder` does not call
`CardPredictor`. This means `_encode_vector()` indices 11-22 are always zero and
the action mask allows no card placements. Wiring in CardPredictor is a prerequisite
for meaningful BC data collection.

**Files to modify:** `src/pipeline/state_builder.py` (add CardPredictor call),
`src/classification/card_classifier.py` (confirm API compatibility).

### Then: Build BC data collection pipeline

1. **Click logger** -- OS-level mouse click recording during expert gameplay
2. **DatasetBuilder** -- Merge state snapshots + click events by timestamp
3. **BC training script** -- SB3 MaskableMultiInputPolicy with CRFeatureExtractor

### Then: BC model training

1. Record 30-40 expert games
2. Process through DatasetBuilder
3. Train BC model with weighted cross-entropy (downweight no-ops)
4. Evaluate: compare against random baseline

### Then: PPO fine-tuning

1. Build ClashRoyaleEnv (Gym wrapper with mss capture + PyAutoGUI)
2. Reward function (tower HP delta + win/loss terminal)
3. PPO training from BC-initialized weights
4. Game-over detection

### Phase order summary

```
[CURRENT] Per-cell encoder     -- this document
    v
Wire CardPredictor into StateBuilder
    v
Build click logger + DatasetBuilder
    v
Train BC model (30-40 games)
    v
Build Gym environment wrapper
    v
PPO fine-tuning from BC weights
```

---

## Appendix A: PositionFinder Displacement Analysis

### Worst-case analysis for our deck

| Card | Units Spawned | Typical Cluster | Max Displacement |
|------|---------------|-----------------|-----------------|
| Royal Recruits | 6 | 3+3 split | 1-2 cells (3 in each cluster) |
| Royal Hogs | 4 | 2x2 | 1 cell (mild overlap) |
| Zappies | 3 | 1x3 | 0-1 cells (linear spawn) |
| Goblin Cage | 1 (building) | 1x1 | 0 cells (no collision) |
| Flying Machine | 1 | 1x1 | 0 cells |
| Electro Spirit | 1 | 1x1 | 0 cells |

### Enemy deck worst case (not our deck)

| Card | Units | Max Displacement | Notes |
|------|-------|-----------------|-------|
| Skeleton Army | 15 | 3-4 cells | Worst case, not in our deck |
| Minion Horde | 6 | 2 cells | Naturally spread |
| Barbarians | 5 | 1-2 cells | Tight cluster |
| Three Musketeers | 3 | 0-1 cells | Split spawn |

### Impact on learning

A 1-2 cell displacement corresponds to roughly 30-60 pixels at 540x960 resolution
(about 1-2 in-game tile widths). For strategic decision-making (whether to defend,
which card to play), this level of positional imprecision is acceptable. The agent
cares more about "enemy push in left lane" than "enemy at exact cell (5, 20) vs
(5, 21)."

---

## Appendix B: Full Class ID Table (Excerpt)

Class IDs are derived from `label_list.unit_list` order + 1 offset:

| class_idx | class_name | Type | Notes |
|-----------|------------|------|-------|
| 0 | (empty) | -- | Reserved for empty cells |
| 1 | king-tower | tower | Handled by tower HP channels |
| 2 | queen-tower | tower | Handled by tower HP channels |
| ... | ... | ... | ... |
| 17 | bar | other | Filtered out during encoding |
| ... | ... | ... | ... |
| 26 | skeleton | ground | Common enemy unit |
| 27 | skeleton-evolution | ground | |
| 28 | electro-spirit | ground | In our deck |
| 29 | fire-spirit | ground | |
| 30 | ice-spirit | ground | |
| ... | ... | ... | ... |
| 48 | knight | ground | Common enemy unit |
| ... | ... | ... | ... |
| 82 | mini-pekka | ground | Common enemy unit |
| 83 | musketeer | ground | |
| 84 | goblin-cage | ground | In our deck |
| ... | ... | ... | ... |
| 91 | flying-machine | flying | In our deck |
| 92 | hog-rider | ground | |
| ... | ... | ... | ... |
| 111 | royal-hog | ground | In our deck |
| ... | ... | ... | ... |

The full table has 155 entries. The exact mapping is deterministic from
`label_list.unit_list` and is generated at import time in `encoder_constants.py`.
