# Per-Cell Unit Encoding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the StateEncoder to use KataCR-style per-cell unit identity encoding with PositionFinder collision resolution, producing a (32, 18, 6) arena tensor instead of the current (32, 18, 7) additive-count tensor.

**Architecture:** A new `PositionFinder` class resolves grid cell collisions (one unit per cell). The rewritten `_encode_arena()` sorts units by belonging/position, runs them through PositionFinder, and writes per-cell features (class_id, belonging, arena_mask) plus unchanged tower HP and spell channels. Action space, action mask, vector encoding, GameState, and StateBuilder are all untouched.

**Tech Stack:** Python 3.11+, numpy 1.26, scipy 1.13 (for `cdist`), gymnasium 1.2, pytest

**Target directory:** All implementation goes in `docs/josh/state_encoder_module/` (parallel copy for Josh to review). Tests go in `docs/josh/state_encoder_module/tests/`.

**Design doc:** `docs/plans/2026-02-21-percell-unit-encoding-design.md` (already written, contains full rationale)

---

## Context for the Implementer

### What you are building

The StateEncoder converts `GameState` (Python dataclasses with unit lists, towers, cards) into fixed-shape numpy tensors for Stable Baselines 3. You are rewriting how units are encoded on the arena grid.

**Before (additive counting):** Multiple units can share a cell. Each cell holds counts per faction/type. No unit identity.
```
arena[row, col] = [ally_ground_count, ally_flying_count, enemy_ground_count, enemy_flying_count, ally_tower_hp, enemy_tower_hp, spell_count]
```

**After (per-cell identity):** Each cell holds at most one unit with its class ID and belonging. A PositionFinder resolves collisions.
```
arena[row, col] = [class_id_normalized, belonging, arena_mask, ally_tower_hp, enemy_tower_hp, spell_count]
```

### Key files you will touch

All paths relative to `docs/josh/state_encoder_module/`:

| File | Action | Purpose |
|------|--------|---------|
| `src/encoder/position_finder.py` | CREATE | PositionFinder collision resolution |
| `src/encoder/encoder_constants.py` | REWRITE | New channel constants, CLASS_NAME_TO_ID mapping |
| `src/encoder/coord_utils.py` | MODIFY | Add `pixel_to_cell_float()` |
| `src/encoder/state_encoder.py` | REWRITE | New `_encode_arena()`, updated obs space |
| `src/encoder/__init__.py` | MODIFY | Export PositionFinder |
| `src/encoder/CLAUDE.md` | REWRITE | Updated observation/action docs |
| `docs/state-encoder-docs.md` | REWRITE | Updated developer documentation |
| `tests/test_position_finder.py` | CREATE | PositionFinder unit tests |
| `tests/test_state_encoder.py` | CREATE | Encoder integration tests |
| `tests/test_coord_utils.py` | CREATE | pixel_to_cell_float tests |
| `tests/__init__.py` | CREATE | Package init |

### What you must NOT touch

- Action space `Discrete(2305)` -- unchanged
- Action mask logic -- unchanged (checks cards/elixir, not arena)
- `_encode_vector()` -- unchanged (reads GameState scalars)
- `action_to_placement()`, `placement_to_action()`, `cell_to_norm()`, `norm_to_cell()`, `pixel_to_cell()` -- unchanged
- Any file outside `src/encoder/` -- no pipeline/detection/generation changes

### Dependencies you can import

```python
import numpy as np
import scipy.spatial.distance  # for cdist in PositionFinder
import gymnasium as gym
```

The code imports `from src.pipeline.game_state import GameState, Tower, Unit, Card` and `from src.generation.label_list import ...`. These exist in the real codebase at `/Users/alanguo/Codin/CS175/Project/cr-object-detection/`. For testing, you will create mock GameState objects directly since the imports resolve at the repo root.

---

## Task 1: Create PositionFinder

**Files:**
- Create: `src/encoder/position_finder.py`
- Create: `tests/__init__.py`
- Create: `tests/test_position_finder.py`

### Step 1: Write the failing tests

Create `tests/__init__.py` (empty file) and `tests/test_position_finder.py`:

```python
"""Tests for PositionFinder collision resolution."""

import numpy as np
import pytest

from src.encoder.position_finder import PositionFinder


class TestPositionFinderBasic:
    """Basic placement: no collisions."""

    def test_first_unit_gets_natural_cell(self):
        """A unit placed on an empty grid gets its natural cell."""
        pf = PositionFinder()
        col, row = pf.find_position(5.3, 20.7)
        assert col == 5
        assert row == 20

    def test_boundary_top_left(self):
        """Coordinate (0.0, 0.0) maps to cell (0, 0)."""
        pf = PositionFinder()
        col, row = pf.find_position(0.0, 0.0)
        assert col == 0
        assert row == 0

    def test_boundary_bottom_right(self):
        """Coordinate near (17.9, 31.9) maps to cell (17, 31)."""
        pf = PositionFinder()
        col, row = pf.find_position(17.9, 31.9)
        assert col == 17
        assert row == 31

    def test_negative_coords_clamped(self):
        """Negative coordinates are clamped to 0."""
        pf = PositionFinder()
        col, row = pf.find_position(-1.0, -5.0)
        assert col == 0
        assert row == 0

    def test_overflow_coords_clamped(self):
        """Coordinates beyond grid are clamped to max."""
        pf = PositionFinder()
        col, row = pf.find_position(25.0, 40.0)
        assert col == 17
        assert row == 31


class TestPositionFinderCollision:
    """Collision resolution: displacing to nearest free cell."""

    def test_second_unit_same_cell_displaced(self):
        """Second unit at the same cell gets displaced."""
        pf = PositionFinder()
        col1, row1 = pf.find_position(5.0, 20.0)
        col2, row2 = pf.find_position(5.0, 20.0)
        assert (col1, row1) == (5, 20)
        assert (col2, row2) != (5, 20)

    def test_displaced_unit_is_adjacent(self):
        """Displaced unit lands within 1-2 cells of target."""
        pf = PositionFinder()
        pf.find_position(9.0, 16.0)
        col, row = pf.find_position(9.0, 16.0)
        dist = abs(col - 9) + abs(row - 16)  # Manhattan distance
        assert dist <= 2, f"Displaced to ({col}, {row}), too far from (9, 16)"

    def test_three_units_same_cell_all_unique(self):
        """Three units at the same position get three unique cells."""
        pf = PositionFinder()
        positions = set()
        for _ in range(3):
            col, row = pf.find_position(9.0, 16.0)
            positions.add((col, row))
        assert len(positions) == 3

    def test_swarm_fifteen_units_all_unique(self):
        """15 units (Skeleton Army worst case) all get unique cells."""
        pf = PositionFinder()
        positions = set()
        for _ in range(15):
            col, row = pf.find_position(9.0, 16.0)
            positions.add((col, row))
        assert len(positions) == 15


class TestPositionFinderReset:
    """Per-frame reset behavior."""

    def test_separate_instances_independent(self):
        """Two PositionFinder instances don't share state."""
        pf1 = PositionFinder()
        pf1.find_position(5.0, 20.0)

        pf2 = PositionFinder()
        col, row = pf2.find_position(5.0, 20.0)
        assert (col, row) == (5, 20), "New instance should have empty grid"

    def test_used_count_tracks_placements(self):
        """Internal used grid tracks placed units."""
        pf = PositionFinder()
        assert pf.used.sum() == 0
        pf.find_position(3.0, 10.0)
        assert pf.used.sum() == 1
        pf.find_position(3.0, 10.0)
        assert pf.used.sum() == 2
```

### Step 2: Run tests to verify they fail

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. pytest docs/josh/state_encoder_module/tests/test_position_finder.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.encoder.position_finder'`

### Step 3: Implement PositionFinder

Create `src/encoder/position_finder.py`:

```python
"""PositionFinder -- resolves grid cell collisions for per-cell unit encoding.

When multiple units map to the same arena grid cell, PositionFinder displaces
later units to the nearest unoccupied cell. This enforces a one-unit-per-cell
invariant that enables per-cell feature vectors (class ID, belonging, etc.).

Based on KataCR's approach (katacr/policy/offline/dataset.py lines 143-159).
A fresh PositionFinder instance should be created per frame.

Usage:
    pf = PositionFinder()
    for unit in sorted_units:
        col_f, row_f = pixel_to_cell_float(cx, cy, fw, fh)
        col, row = pf.find_position(col_f, row_f)
        arena[row, col, CH_CLASS_ID] = class_idx / NUM_CLASSES
"""

import numpy as np
import scipy.spatial.distance


class PositionFinder:
    """Resolve grid cell collisions by displacing units to nearest free cell.

    Attributes:
        rows: Number of grid rows (default 32).
        cols: Number of grid columns (default 18).
        used: Boolean array (rows, cols) tracking occupied cells.
    """

    def __init__(self, rows: int = 32, cols: int = 18):
        self.rows = rows
        self.cols = cols
        self.used = np.zeros((rows, cols), dtype=np.bool_)

        # Pre-compute cell centers for distance calculations.
        # Shape: (rows, cols, 2) where last dim is (row_center, col_center).
        row_coords, col_coords = np.meshgrid(
            np.arange(rows, dtype=np.float64) + 0.5,
            np.arange(cols, dtype=np.float64) + 0.5,
            indexing="ij",
        )
        self._centers = np.stack([row_coords, col_coords], axis=-1)

    def find_position(self, col_f: float, row_f: float) -> tuple[int, int]:
        """Find the best grid cell for a unit at continuous coordinates.

        If the natural cell (floor of coordinates) is free, returns it.
        Otherwise finds the nearest unoccupied cell by Euclidean distance.

        Args:
            col_f: Continuous column coordinate (can be outside [0, cols-1]).
            row_f: Continuous row coordinate (can be outside [0, rows-1]).

        Returns:
            (col, row) as integer cell indices, clamped to valid range.
        """
        row = int(np.clip(int(row_f), 0, self.rows - 1))
        col = int(np.clip(int(col_f), 0, self.cols - 1))

        if not self.used[row, col]:
            self.used[row, col] = True
            return col, row

        # Cell occupied -- find nearest free cell by Euclidean distance
        available_mask = ~self.used
        if not available_mask.any():
            # All 576 cells occupied (extremely unlikely). Return clamped pos.
            return col, row

        available_indices = np.argwhere(available_mask)  # (N, 2) as (row, col)
        available_centers = self._centers[available_mask]  # (N, 2)

        query = np.array([[row_f, col_f]], dtype=np.float64)
        distances = scipy.spatial.distance.cdist(query, available_centers)
        nearest_idx = int(np.argmin(distances))
        row, col = available_indices[nearest_idx]

        self.used[row, col] = True
        return int(col), int(row)
```

### Step 4: Run tests to verify they pass

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. pytest docs/josh/state_encoder_module/tests/test_position_finder.py -v
```

Expected: All 10 tests PASS.

### Step 5: Commit

```bash
git add docs/josh/state_encoder_module/src/encoder/position_finder.py docs/josh/state_encoder_module/tests/
git commit -m "feat: add PositionFinder for per-cell collision resolution"
```

---

## Task 2: Update encoder_constants.py

**Files:**
- Modify: `src/encoder/encoder_constants.py` (full rewrite)

### Step 1: Rewrite encoder_constants.py

Replace the file contents entirely:

```python
"""Constants for the per-cell StateEncoder and action space.

Single source of truth for grid dimensions, action encoding, unit type
classification, class-to-ID mapping, and deck configuration.

Used by StateEncoder, PositionFinder, coord_utils, and the future
DatasetBuilder / Gym env.

Change log:
    - Per-cell encoding: replaced additive-count channels (CH_ALLY_GROUND,
      CH_ALLY_FLYING, CH_ENEMY_GROUND, CH_ENEMY_FLYING) with per-cell
      identity channels (CH_CLASS_ID, CH_BELONGING, CH_ARENA_MASK).
      CH_ALLY_TOWER_HP, CH_ENEMY_TOWER_HP, and CH_SPELL retained.
"""

from src.generation.label_list import (
    unit_list,
    ground_unit_list,
    flying_unit_list,
    spell_unit_list,
    tower_unit_list,
    other_unit_list,
)

# ---------------------------------------------------------------------------
# Grid dimensions (from generation_config.py)
# ---------------------------------------------------------------------------
GRID_COLS = 18
GRID_ROWS = 32
GRID_CELLS = GRID_COLS * GRID_ROWS  # 576

# ---------------------------------------------------------------------------
# Action space: Discrete(2305)
#   action 0..2303 = card_id * 576 + row * 18 + col
#   action 2304    = no-op (wait)
# ---------------------------------------------------------------------------
NUM_CARD_SLOTS = 4
ACTION_SPACE_SIZE = NUM_CARD_SLOTS * GRID_CELLS + 1  # 2305
NOOP_ACTION = ACTION_SPACE_SIZE - 1  # 2304

# ---------------------------------------------------------------------------
# Arena bounds as fractions of full screen height
# Derived from screen_regions.py ScreenConfig at 540x960 base resolution:
#   arena region = (0, 50, 540, 750)
# ---------------------------------------------------------------------------
ARENA_Y_START_FRAC = 50.0 / 960.0   # ~0.0521
ARENA_Y_END_FRAC = 750.0 / 960.0    # ~0.7813

# ---------------------------------------------------------------------------
# Normalization ceilings
# ---------------------------------------------------------------------------
MAX_ELIXIR = 10
MAX_TIME_SECONDS = 300  # 5 min max (regular + overtime)

# ---------------------------------------------------------------------------
# Arena grid channel indices -- PER-CELL ENCODING
#
# Channels 0-2 are per-cell unit identity features (one unit per cell,
# enforced by PositionFinder). Channels 3-5 are continuous/additive
# features that do not participate in PositionFinder.
# ---------------------------------------------------------------------------
CH_CLASS_ID = 0        # Normalized class ID: class_idx / NUM_CLASSES (0.0 = empty)
CH_BELONGING = 1       # -1.0 = ally, +1.0 = enemy, 0.0 = empty cell
CH_ARENA_MASK = 2      # 1.0 = unit present, 0.0 = empty

CH_ALLY_TOWER_HP = 3   # Tower HP fraction 0-1
CH_ENEMY_TOWER_HP = 4  # Tower HP fraction 0-1
CH_SPELL = 5           # Additive count of spell effects

NUM_ARENA_CHANNELS = 6

# ---------------------------------------------------------------------------
# Class name to integer ID mapping
#
# Built from label_list.unit_list (155 classes in KataCR order).
# ID 0 is reserved for empty cells. Actual classes are 1-indexed.
# ---------------------------------------------------------------------------
NUM_CLASSES = len(unit_list)  # 155

CLASS_NAME_TO_ID: dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(unit_list)
}

# ---------------------------------------------------------------------------
# Vector feature count
# ---------------------------------------------------------------------------
NUM_VECTOR_FEATURES = 23

# ---------------------------------------------------------------------------
# Unit type classification lookup
# Maps detection class_name -> "ground" | "flying" | "spell" | "tower" | "other"
#
# Spell units overlap with ground/flying lists. We check spell first so that
# spell effects (zap, arrows, fireball, etc.) go into the spell channel rather
# than ground/flying.
# ---------------------------------------------------------------------------
UNIT_TYPE_MAP: dict[str, str] = {}

_spell_set = set(spell_unit_list)
_flying_set = set(flying_unit_list)
_ground_set = set(ground_unit_list)
_tower_set = set(tower_unit_list)

# Priority: tower > spell > flying > ground > other
for name in tower_unit_list:
    UNIT_TYPE_MAP[name] = "tower"
for name in spell_unit_list:
    if name not in UNIT_TYPE_MAP:
        UNIT_TYPE_MAP[name] = "spell"
for name in flying_unit_list:
    if name not in UNIT_TYPE_MAP:
        UNIT_TYPE_MAP[name] = "flying"
for name in ground_unit_list:
    if name not in UNIT_TYPE_MAP:
        UNIT_TYPE_MAP[name] = "ground"
for name in other_unit_list:
    if name not in UNIT_TYPE_MAP:
        UNIT_TYPE_MAP[name] = "other"

# ---------------------------------------------------------------------------
# Deck configuration (Royal Hogs / Royal Recruits deck)
#
# Class names match CardPredictor output, which uses filename stems from
# data/deck-card-crops/frames/*.png as class names. Note: "eletro-spirit"
# is a filename typo (missing 'c') preserved here for compatibility.
# ---------------------------------------------------------------------------
DECK_CARDS = [
    "arrows",
    "barbarian-barrel",
    "eletro-spirit",
    "flying-machine",
    "goblin-cage",
    "royal-hogs",
    "royal-recruits",
    "zappies",
]
NUM_DECK_CARDS = len(DECK_CARDS)
DECK_CARD_TO_IDX = {name: i for i, name in enumerate(DECK_CARDS)}

CARD_ELIXIR_COST = {
    "arrows": 3,
    "barbarian-barrel": 2,
    "eletro-spirit": 1,
    "flying-machine": 4,
    "goblin-cage": 4,
    "royal-hogs": 5,
    "royal-recruits": 7,
    "zappies": 4,
    "empty-slot": 0,
}

# Card type for placement rules (spells can target anywhere)
CARD_IS_SPELL = {
    "arrows": True,
    "barbarian-barrel": True,
    "eletro-spirit": False,
    "flying-machine": False,
    "goblin-cage": False,
    "royal-hogs": False,
    "royal-recruits": False,
    "zappies": False,
}

# Player's deployable half of the arena (bottom half)
# In the 32-row grid, rows 0-15 are enemy side, rows 16-31 are player side
# (row 15-16 is the river)
PLAYER_HALF_ROW_START = 17

# Default tower HP (level 14 -- standard tournament level)
DEFAULT_KING_MAX_HP = 6408
DEFAULT_PRINCESS_MAX_HP = 4032
```

### Step 2: Verify import works

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. python -c "
from src.encoder.encoder_constants import (
    CH_CLASS_ID, CH_BELONGING, CH_ARENA_MASK,
    CH_ALLY_TOWER_HP, CH_ENEMY_TOWER_HP, CH_SPELL,
    NUM_ARENA_CHANNELS, NUM_CLASSES, CLASS_NAME_TO_ID,
)
print(f'NUM_ARENA_CHANNELS = {NUM_ARENA_CHANNELS}')
print(f'NUM_CLASSES = {NUM_CLASSES}')
print(f'knight ID = {CLASS_NAME_TO_ID[\"knight\"]}')
print(f'skeleton ID = {CLASS_NAME_TO_ID[\"skeleton\"]}')
"
```

Expected:
```
NUM_ARENA_CHANNELS = 6
NUM_CLASSES = 155
knight ID = 49
skeleton ID = 27
```

### Step 3: Commit

```bash
git add docs/josh/state_encoder_module/src/encoder/encoder_constants.py
git commit -m "feat: update encoder_constants for per-cell encoding with CLASS_NAME_TO_ID"
```

---

## Task 3: Add pixel_to_cell_float to coord_utils.py

**Files:**
- Modify: `src/encoder/coord_utils.py` (add one function)
- Create: `tests/test_coord_utils.py`

### Step 1: Write the failing tests

Create `tests/test_coord_utils.py`:

```python
"""Tests for pixel_to_cell_float coordinate conversion."""

import pytest

from src.encoder.coord_utils import pixel_to_cell, pixel_to_cell_float


class TestPixelToCellFloat:
    """pixel_to_cell_float returns continuous coordinates for PositionFinder."""

    def test_center_of_screen(self):
        """Center of a 540x960 frame gives roughly center of grid."""
        col_f, row_f = pixel_to_cell_float(270, 400, 540, 960)
        # x=270/540=0.5, col=0.5*18=9.0
        assert abs(col_f - 9.0) < 0.01
        # row depends on arena Y mapping
        assert 0 < row_f < 32

    def test_returns_float_not_int(self):
        """Output should be float, not clamped integer."""
        col_f, row_f = pixel_to_cell_float(150, 300, 540, 960)
        assert isinstance(col_f, float)
        assert isinstance(row_f, float)

    def test_consistent_with_pixel_to_cell(self):
        """Integer truncation of float output matches pixel_to_cell."""
        for cx, cy in [(100, 200), (270, 400), (500, 700)]:
            col_f, row_f = pixel_to_cell_float(cx, cy, 540, 960)
            col_i, row_i = pixel_to_cell(cx, cy, 540, 960)
            # Floor of float should match clamped int (for in-arena coords)
            col_floor = max(0, min(int(col_f), 17))
            row_floor = max(0, min(int(row_f), 31))
            assert col_floor == col_i, f"col mismatch at ({cx},{cy})"
            assert row_floor == row_i, f"row mismatch at ({cx},{cy})"

    def test_left_edge(self):
        """x=0 gives col_f=0.0."""
        col_f, _ = pixel_to_cell_float(0, 400, 540, 960)
        assert abs(col_f) < 0.01

    def test_right_edge(self):
        """x=540 gives col_f=18.0."""
        col_f, _ = pixel_to_cell_float(540, 400, 540, 960)
        assert abs(col_f - 18.0) < 0.01
```

### Step 2: Run tests to verify they fail

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. pytest docs/josh/state_encoder_module/tests/test_coord_utils.py -v
```

Expected: `ImportError: cannot import name 'pixel_to_cell_float'`

### Step 3: Add pixel_to_cell_float to coord_utils.py

Add this function after `pixel_to_cell()` (after line 50):

```python
def pixel_to_cell_float(
    cx: float, cy: float, frame_w: int, frame_h: int
) -> tuple[float, float]:
    """Convert pixel bbox center to continuous grid cell coordinates.

    Unlike pixel_to_cell() which returns clamped integers, this returns
    float coordinates for PositionFinder input. The fractional part
    captures sub-cell positioning, enabling better collision resolution.

    Args:
        cx: Center x in pixels.
        cy: Center y in pixels.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        (col_f, row_f) as continuous floats. NOT clamped to grid bounds.
        Caller (PositionFinder.find_position) handles clamping.
    """
    x_norm = cx / frame_w
    y_norm = cy / frame_h
    arena_y_frac = (y_norm - ARENA_Y_START_FRAC) / _ARENA_Y_SPAN
    col_f = x_norm * GRID_COLS
    row_f = arena_y_frac * GRID_ROWS
    return col_f, row_f
```

### Step 4: Run tests to verify they pass

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. pytest docs/josh/state_encoder_module/tests/test_coord_utils.py -v
```

Expected: All 5 tests PASS.

### Step 5: Commit

```bash
git add docs/josh/state_encoder_module/src/encoder/coord_utils.py docs/josh/state_encoder_module/tests/test_coord_utils.py
git commit -m "feat: add pixel_to_cell_float for PositionFinder continuous coords"
```

---

## Task 4: Rewrite state_encoder.py

**Files:**
- Modify: `src/encoder/state_encoder.py` (major rewrite)
- Create: `tests/test_state_encoder.py`

### Step 1: Write the failing tests

Create `tests/test_state_encoder.py`:

```python
"""Tests for the per-cell StateEncoder.

Tests cover:
- Arena shape and channel layout
- Per-cell unit placement via PositionFinder
- Spell bypass (additive channel, no PositionFinder)
- Tower HP encoding (unchanged)
- Vector encoding (unchanged)
- Action mask (unchanged)
- Observation space bounds
"""

import numpy as np
import pytest

from src.pipeline.game_state import GameState, Tower, Unit, Card
from src.encoder.state_encoder import StateEncoder
from src.encoder.encoder_constants import (
    CH_CLASS_ID, CH_BELONGING, CH_ARENA_MASK,
    CH_ALLY_TOWER_HP, CH_ENEMY_TOWER_HP, CH_SPELL,
    NUM_ARENA_CHANNELS, NUM_CLASSES, CLASS_NAME_TO_ID,
    NOOP_ACTION, CARD_ELIXIR_COST,
)


@pytest.fixture
def encoder():
    return StateEncoder()


@pytest.fixture
def empty_state():
    return GameState(frame_width=540, frame_height=960)


def _make_unit(class_name, belonging, cx, cy, conf=0.9):
    """Helper to create a Unit with a bbox centered at (cx, cy)."""
    return Unit(
        class_name=class_name,
        belonging=belonging,
        bbox=(cx - 10, cy - 10, cx + 10, cy + 10),
        confidence=conf,
    )


class TestArenaShape:
    """Arena tensor has correct shape and dtype."""

    def test_arena_shape(self, encoder, empty_state):
        obs = encoder.encode(empty_state)
        assert obs["arena"].shape == (32, 18, NUM_ARENA_CHANNELS)
        assert obs["arena"].shape == (32, 18, 6)

    def test_arena_dtype(self, encoder, empty_state):
        obs = encoder.encode(empty_state)
        assert obs["arena"].dtype == np.float32

    def test_empty_state_all_zeros(self, encoder, empty_state):
        obs = encoder.encode(empty_state)
        assert obs["arena"].sum() == 0.0


class TestPerCellEncoding:
    """Per-cell unit identity encoding."""

    def test_single_ally_unit(self, encoder):
        state = GameState(frame_width=540, frame_height=960)
        state.units = [_make_unit("knight", 0, 270, 480)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # Exactly one cell should be occupied
        assert arena[:, :, CH_ARENA_MASK].sum() == 1.0

        # Find the occupied cell
        mask = arena[:, :, CH_ARENA_MASK] > 0
        row, col = np.argwhere(mask)[0]

        # Verify class ID
        expected_class_id = CLASS_NAME_TO_ID["knight"] / NUM_CLASSES
        assert abs(arena[row, col, CH_CLASS_ID] - expected_class_id) < 1e-5

        # Verify belonging (ally = -1.0)
        assert arena[row, col, CH_BELONGING] == -1.0

        # Verify arena mask
        assert arena[row, col, CH_ARENA_MASK] == 1.0

    def test_single_enemy_unit(self, encoder):
        state = GameState(frame_width=540, frame_height=960)
        state.units = [_make_unit("skeleton", 1, 270, 200)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        mask = arena[:, :, CH_ARENA_MASK] > 0
        row, col = np.argwhere(mask)[0]

        # Verify belonging (enemy = +1.0)
        assert arena[row, col, CH_BELONGING] == 1.0

    def test_multiple_units_unique_cells(self, encoder):
        state = GameState(frame_width=540, frame_height=960)
        state.units = [
            _make_unit("knight", 0, 100, 600),
            _make_unit("archer", 1, 200, 300),
            _make_unit("musketeer", 0, 400, 500),
        ]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # All three should be in unique cells
        assert arena[:, :, CH_ARENA_MASK].sum() == 3.0

    def test_overlapping_units_displaced(self, encoder):
        """Two units at same pixel position get different cells."""
        state = GameState(frame_width=540, frame_height=960)
        state.units = [
            _make_unit("knight", 0, 270, 480),
            _make_unit("archer", 1, 270, 480),
        ]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # Both should be placed (PositionFinder displaces the second)
        assert arena[:, :, CH_ARENA_MASK].sum() == 2.0

    def test_unknown_class_skipped(self, encoder):
        """Units with class_name not in CLASS_NAME_TO_ID are skipped."""
        state = GameState(frame_width=540, frame_height=960)
        state.units = [_make_unit("totally_fake_class", 0, 270, 480)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        assert arena[:, :, CH_ARENA_MASK].sum() == 0.0


class TestSpellBypass:
    """Spells go to CH_SPELL, not per-cell channels."""

    def test_spell_in_spell_channel(self, encoder):
        state = GameState(frame_width=540, frame_height=960)
        state.units = [_make_unit("fireball", 0, 270, 400)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # No per-cell unit
        assert arena[:, :, CH_ARENA_MASK].sum() == 0.0
        # Spell channel has a count
        assert arena[:, :, CH_SPELL].sum() > 0.0

    def test_spell_additive(self, encoder):
        """Multiple spells in same cell accumulate."""
        state = GameState(frame_width=540, frame_height=960)
        state.units = [
            _make_unit("fireball", 0, 270, 400),
            _make_unit("zap", 1, 270, 400),
        ]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # Two spells, no per-cell units
        assert arena[:, :, CH_ARENA_MASK].sum() == 0.0
        assert arena[:, :, CH_SPELL].sum() == 2.0

    def test_spell_and_troop_coexist(self, encoder):
        """A spell and a troop can share a cell (different channels)."""
        state = GameState(frame_width=540, frame_height=960)
        state.units = [
            _make_unit("knight", 0, 270, 400),
            _make_unit("fireball", 0, 270, 400),
        ]
        obs = encoder.encode(state)
        arena = obs["arena"]

        assert arena[:, :, CH_ARENA_MASK].sum() == 1.0
        assert arena[:, :, CH_SPELL].sum() == 1.0


class TestTowerEncoding:
    """Tower HP fraction channels (unchanged behavior)."""

    def test_ally_tower_hp(self, encoder):
        state = GameState(frame_width=540, frame_height=960)
        state.player_king_tower = Tower(
            "king", "center", 0, hp=3204, max_hp=6408,
            bbox=(250, 800, 290, 850),
        )
        obs = encoder.encode(state)
        arena = obs["arena"]

        # Tower HP should be 3204/6408 = 0.5
        assert arena[:, :, CH_ALLY_TOWER_HP].max() == pytest.approx(0.5, abs=0.01)

    def test_enemy_tower_hp(self, encoder):
        state = GameState(frame_width=540, frame_height=960)
        state.enemy_king_tower = Tower(
            "king", "center", 1, hp=6408, max_hp=6408,
            bbox=(250, 100, 290, 150),
        )
        obs = encoder.encode(state)
        arena = obs["arena"]

        assert arena[:, :, CH_ENEMY_TOWER_HP].max() == pytest.approx(1.0, abs=0.01)


class TestOtherAndTowerUnitsSkipped:
    """UI elements ('other') and tower detections are not placed as units."""

    def test_bar_skipped(self, encoder):
        state = GameState(frame_width=540, frame_height=960)
        state.units = [_make_unit("bar", 0, 270, 400)]
        obs = encoder.encode(state)
        assert obs["arena"][:, :, CH_ARENA_MASK].sum() == 0.0

    def test_tower_detection_skipped(self, encoder):
        """Tower class_names in UNIT_TYPE_MAP are skipped (handled separately)."""
        state = GameState(frame_width=540, frame_height=960)
        state.units = [_make_unit("king-tower", 0, 270, 800)]
        obs = encoder.encode(state)
        assert obs["arena"][:, :, CH_ARENA_MASK].sum() == 0.0


class TestVectorEncoding:
    """Vector encoding is unchanged from before."""

    def test_vector_shape(self, encoder, empty_state):
        obs = encoder.encode(empty_state)
        assert obs["vector"].shape == (23,)

    def test_elixir_encoding(self, encoder):
        state = GameState(frame_width=540, frame_height=960, elixir=7)
        obs = encoder.encode(state)
        assert obs["vector"][0] == pytest.approx(0.7)

    def test_time_encoding(self, encoder):
        state = GameState(frame_width=540, frame_height=960, time_remaining=150)
        obs = encoder.encode(state)
        assert obs["vector"][1] == pytest.approx(0.5)


class TestActionMask:
    """Action mask is unchanged from before."""

    def test_noop_always_valid(self, encoder, empty_state):
        mask = encoder.action_mask(empty_state)
        assert mask[NOOP_ACTION] is np.True_

    def test_no_cards_only_noop(self, encoder, empty_state):
        mask = encoder.action_mask(empty_state)
        assert mask[:NOOP_ACTION].sum() == 0

    def test_playable_card_unmasked(self, encoder):
        state = GameState(frame_width=540, frame_height=960, elixir=5)
        state.cards = [Card(slot=0, class_name="zappies", elixir_cost=4, confidence=0.9)]
        mask = encoder.action_mask(state)
        # Card 0 should have 576 valid cells
        assert mask[0:576].sum() == 576
        # Card 1-3 should have 0 valid cells
        assert mask[576:2304].sum() == 0

    def test_insufficient_elixir_masked(self, encoder):
        state = GameState(frame_width=540, frame_height=960, elixir=2)
        state.cards = [Card(slot=0, class_name="royal-recruits", elixir_cost=7, confidence=0.9)]
        mask = encoder.action_mask(state)
        # 7 elixir card with only 2 elixir -- all masked
        assert mask[0:576].sum() == 0


class TestObservationSpace:
    """Observation space bounds match new encoding."""

    def test_arena_space_shape(self, encoder):
        arena_space = encoder.observation_space["arena"]
        assert arena_space.shape == (32, 18, 6)

    def test_arena_space_low(self, encoder):
        arena_space = encoder.observation_space["arena"]
        assert arena_space.low.min() == -1.0

    def test_arena_space_high(self, encoder):
        arena_space = encoder.observation_space["arena"]
        assert arena_space.high.max() == 10.0

    def test_action_space_size(self, encoder):
        assert encoder.action_space.n == 2305
```

### Step 2: Run tests to verify they fail

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. pytest docs/josh/state_encoder_module/tests/test_state_encoder.py -v
```

Expected: Import errors from old constant names (`CH_ALLY_GROUND`, etc.)

### Step 3: Rewrite state_encoder.py

Replace the entire file:

```python
"""StateEncoder -- converts GameState to SB3-compatible observation tensors.

This is the bridge between perception (StateBuilder) and learning (BC / PPO).
It produces a Dict observation and an action mask, both with fixed shapes that
don't change across frames.

Uses per-cell unit identity encoding (KataCR approach):
- Each arena grid cell holds at most one unit (enforced by PositionFinder)
- Per-cell features: normalized class ID, belonging flag, presence mask
- Spells bypass PositionFinder and use a separate additive channel
- Tower HP uses dedicated continuous channels

Usage:
    from src.encoder import StateEncoder

    encoder = StateEncoder()

    obs = encoder.encode(game_state)
    mask = encoder.action_mask(game_state)

    # obs is {"arena": np.float32(32,18,6), "vector": np.float32(23,)}
    # mask is np.bool_(2305,)

Observation space (gymnasium.spaces.Dict):
    "arena": Box(-1, 10, shape=(32, 18, 6), float32)
        Channel 0: normalized class ID (class_idx / 155, 0.0 = empty)
        Channel 1: belonging (-1.0 = ally, +1.0 = enemy, 0.0 = empty)
        Channel 2: arena mask (1.0 = unit present, 0.0 = empty)
        Channel 3: ally tower HP fraction (0-1)
        Channel 4: enemy tower HP fraction (0-1)
        Channel 5: spell effect count per cell (additive)
    "vector": Box(0, 1, shape=(23,), float32)
        [0]     elixir / 10
        [1]     time_remaining / 300
        [2]     is_overtime (0 or 1)
        [3-5]   player tower HP fracs (king, left princess, right princess)
        [6-8]   enemy tower HP fracs (king, left princess, right princess)
        [9]     player_tower_count / 3
        [10]    enemy_tower_count / 3
        [11-14] card present (binary, 4 slots)
        [15-18] card class index / num_deck_cards (normalized)
        [19-22] card elixir cost / 10 (normalized)

Action space: Discrete(2305)
    0..2303: card_id * 576 + row * 18 + col
    2304: no-op (wait)
"""

from typing import Optional

import gymnasium as gym
import numpy as np

from src.pipeline.game_state import GameState, Tower

from .coord_utils import pixel_to_cell, pixel_to_cell_float
from .encoder_constants import (
    ACTION_SPACE_SIZE,
    CARD_ELIXIR_COST,
    CH_ALLY_TOWER_HP,
    CH_ARENA_MASK,
    CH_BELONGING,
    CH_CLASS_ID,
    CH_ENEMY_TOWER_HP,
    CH_SPELL,
    CLASS_NAME_TO_ID,
    DECK_CARD_TO_IDX,
    DEFAULT_KING_MAX_HP,
    DEFAULT_PRINCESS_MAX_HP,
    GRID_CELLS,
    GRID_COLS,
    GRID_ROWS,
    MAX_ELIXIR,
    MAX_TIME_SECONDS,
    NOOP_ACTION,
    NUM_ARENA_CHANNELS,
    NUM_CARD_SLOTS,
    NUM_CLASSES,
    NUM_DECK_CARDS,
    NUM_VECTOR_FEATURES,
    UNIT_TYPE_MAP,
)
from .position_finder import PositionFinder


class StateEncoder:
    """Converts GameState to fixed-shape observation tensors for SB3.

    Uses per-cell unit identity encoding with PositionFinder collision
    resolution. Each arena grid cell holds at most one unit with its
    class ID and belonging flag.

    Thread-safe: encode() and action_mask() create a fresh PositionFinder
    per call and hold no mutable state between calls.
    """

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "arena": gym.spaces.Box(
                low=-1.0,
                high=10.0,
                shape=(GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
                dtype=np.float32,
            ),
            "vector": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(NUM_VECTOR_FEATURES,),
                dtype=np.float32,
            ),
        })
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def encode(self, state: GameState) -> dict[str, np.ndarray]:
        """Convert a GameState into the observation dict.

        Args:
            state: GameState from StateBuilder.build_state().

        Returns:
            Dict with "arena" (32, 18, 6) and "vector" (23,) arrays.
        """
        arena = self._encode_arena(state)
        vector = self._encode_vector(state)
        return {"arena": arena, "vector": vector}

    def action_mask(self, state: GameState) -> np.ndarray:
        """Build the action validity mask.

        A True value means the action is valid (can be taken).
        A False value means the action is masked out (cannot be taken).

        Masking rules:
        - If a card slot is empty (no card or "empty-slot"), all 576 cells
          for that card are masked False.
        - If elixir < card cost, all 576 cells for that card are masked False.
        - No-op (action 2304) is always valid.

        Args:
            state: GameState from StateBuilder.build_state().

        Returns:
            Boolean array of shape (2305,).
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)

        # No-op is always valid
        mask[NOOP_ACTION] = True

        current_elixir = state.elixir if state.elixir is not None else 0

        for slot_idx in range(NUM_CARD_SLOTS):
            card = self._get_card_at_slot(state, slot_idx)
            if card is None:
                continue

            card_name = card.class_name
            if card_name is None or card_name == "empty-slot":
                continue

            cost = CARD_ELIXIR_COST.get(card_name, 0)
            if current_elixir < cost:
                continue

            # This card is playable -- unmask all its grid cells
            start = slot_idx * GRID_CELLS
            end = start + GRID_CELLS
            mask[start:end] = True

        return mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_arena(self, state: GameState) -> np.ndarray:
        """Build the (32, 18, 6) spatial arena grid with per-cell features.

        Processing order:
        1. Separate units into placeable (ground/flying) and spells
        2. Sort placeable units: enemy first, then ally, bottom-to-top
        3. Run through PositionFinder for one-unit-per-cell placement
        4. Write per-cell features: class_id, belonging, arena_mask
        5. Spells go to CH_SPELL (additive, bypass PositionFinder)
        6. Towers go to CH_ALLY_TOWER_HP / CH_ENEMY_TOWER_HP
        """
        arena = np.zeros(
            (GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS), dtype=np.float32
        )
        fw = state.frame_width or 540
        fh = state.frame_height or 960

        # --- Separate units by type ---
        placeable_units = []
        for unit in state.units:
            unit_type = UNIT_TYPE_MAP.get(unit.class_name, "ground")
            if unit_type in ("tower", "other"):
                continue
            if unit_type == "spell":
                cx, cy = unit.center
                col, row = pixel_to_cell(cx, cy, fw, fh)
                arena[row, col, CH_SPELL] += 1.0
                continue
            placeable_units.append(unit)

        # --- Sort: enemy first (belonging=1), then ally (belonging=0) ---
        # Within each faction: bottom-to-top (higher center_y first)
        placeable_units.sort(key=lambda u: (-u.belonging, -u.center[1]))

        # --- Place units via PositionFinder ---
        pf = PositionFinder(rows=GRID_ROWS, cols=GRID_COLS)

        for unit in placeable_units:
            cx, cy = unit.center
            col_f, row_f = pixel_to_cell_float(cx, cy, fw, fh)
            col, row = pf.find_position(col_f, row_f)

            class_idx = CLASS_NAME_TO_ID.get(unit.class_name, 0)
            if class_idx == 0:
                continue  # Unknown class, skip

            arena[row, col, CH_CLASS_ID] = class_idx / NUM_CLASSES
            arena[row, col, CH_BELONGING] = (
                -1.0 if unit.belonging == 0 else 1.0
            )
            arena[row, col, CH_ARENA_MASK] = 1.0

        # --- Towers ---
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

    def _encode_vector(self, state: GameState) -> np.ndarray:
        """Build the (23,) scalar feature vector.

        All values are normalized to [0, 1]. Unknown/None values default
        to 0.0 which is a safe neutral value for neural networks.
        """
        vec = np.zeros(NUM_VECTOR_FEATURES, dtype=np.float32)

        # [0] Elixir
        if state.elixir is not None:
            vec[0] = state.elixir / MAX_ELIXIR

        # [1] Time remaining
        if state.time_remaining is not None:
            vec[1] = min(state.time_remaining / MAX_TIME_SECONDS, 1.0)

        # [2] Overtime flag
        vec[2] = 1.0 if state.is_overtime else 0.0

        # [3-5] Player tower HP fractions
        vec[3] = self._get_tower_hp_frac(state.player_king_tower)
        vec[4] = self._get_tower_hp_frac(state.player_left_princess)
        vec[5] = self._get_tower_hp_frac(state.player_right_princess)

        # [6-8] Enemy tower HP fractions
        vec[6] = self._get_tower_hp_frac(state.enemy_king_tower)
        vec[7] = self._get_tower_hp_frac(state.enemy_left_princess)
        vec[8] = self._get_tower_hp_frac(state.enemy_right_princess)

        # [9-10] Tower counts
        vec[9] = state.player_tower_count / 3.0
        vec[10] = state.enemy_tower_count / 3.0

        # [11-22] Card hand encoding (4 slots x 3 features each)
        for slot_idx in range(NUM_CARD_SLOTS):
            card = self._get_card_at_slot(state, slot_idx)
            base = 11 + slot_idx

            if card is not None and card.class_name and card.class_name != "empty-slot":
                # Card present
                vec[base] = 1.0
                # Class index (normalized)
                card_idx = DECK_CARD_TO_IDX.get(card.class_name, 0)
                vec[base + NUM_CARD_SLOTS] = card_idx / max(NUM_DECK_CARDS - 1, 1)
                # Elixir cost (normalized)
                cost = CARD_ELIXIR_COST.get(card.class_name, 0)
                vec[base + 2 * NUM_CARD_SLOTS] = cost / MAX_ELIXIR

        return vec

    def _get_card_at_slot(self, state: GameState, slot_idx: int):
        """Get the Card at a given slot index, or None."""
        for card in state.cards:
            if card.slot == slot_idx:
                return card
        return None

    def _get_tower_hp_frac(self, tower: Optional[Tower]) -> float:
        """Return tower HP as a 0-1 fraction.

        Returns 0.0 if tower is None, destroyed, or HP is unknown.
        If HP is known but max_hp is not, uses default max HP values.
        """
        if tower is None:
            return 0.0

        hp = tower.hp
        if hp is None or hp <= 0:
            return 0.0

        max_hp = tower.max_hp
        if max_hp is None or max_hp <= 0:
            if tower.tower_type == "king":
                max_hp = DEFAULT_KING_MAX_HP
            else:
                max_hp = DEFAULT_PRINCESS_MAX_HP

        return min(hp / max_hp, 1.0)
```

### Step 4: Run all tests

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. pytest docs/josh/state_encoder_module/tests/ -v
```

Expected: All tests PASS (position_finder + coord_utils + state_encoder).

### Step 5: Commit

```bash
git add docs/josh/state_encoder_module/src/encoder/state_encoder.py docs/josh/state_encoder_module/tests/test_state_encoder.py
git commit -m "feat: rewrite StateEncoder with per-cell unit identity encoding"
```

---

## Task 5: Update __init__.py

**Files:**
- Modify: `src/encoder/__init__.py`

### Step 1: Update exports

```python
"""StateEncoder module - converts GameState to RL observation tensors.

Uses per-cell unit identity encoding with PositionFinder collision resolution.
"""

from .state_encoder import StateEncoder
from .position_finder import PositionFinder
from .encoder_constants import (
    ACTION_SPACE_SIZE,
    NOOP_ACTION,
    NUM_CLASSES,
    CLASS_NAME_TO_ID,
)
from .coord_utils import (
    pixel_to_cell,
    pixel_to_cell_float,
    norm_to_cell,
    cell_to_norm,
    action_to_placement,
    placement_to_action,
)
```

### Step 2: Verify imports work

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. python -c "
from src.encoder import StateEncoder, PositionFinder, pixel_to_cell_float, NUM_CLASSES, CLASS_NAME_TO_ID
print('All imports successful')
print(f'NUM_CLASSES = {NUM_CLASSES}')
"
```

### Step 3: Commit

```bash
git add docs/josh/state_encoder_module/src/encoder/__init__.py
git commit -m "feat: update encoder __init__.py with new exports"
```

---

## Task 6: Rewrite CLAUDE.md (technical reference)

**Files:**
- Modify: `src/encoder/CLAUDE.md`

### Step 1: Rewrite with per-cell encoding documentation

The CLAUDE.md should document the new observation space, channel layout, PositionFinder role, and updated dependencies. Full content is provided in the implementation.

Key sections to include:
- File listing (now includes position_finder.py)
- Observation space (new 6-channel layout with per-cell features)
- Vector (unchanged, 23 features)
- Action space (unchanged, Discrete 2305)
- Action mask (unchanged)
- PositionFinder usage and sort order
- CLASS_NAME_TO_ID mapping description
- Coordinate system (unchanged)
- Dependencies (now includes scipy)
- Usage examples (updated for new encoding)
- Deck configuration (unchanged)

### Step 2: Commit

```bash
git add docs/josh/state_encoder_module/src/encoder/CLAUDE.md
git commit -m "docs: update encoder CLAUDE.md for per-cell encoding"
```

---

## Task 7: Rewrite state-encoder-docs.md (developer documentation)

**Files:**
- Modify: `docs/state-encoder-docs.md`

### Step 1: Rewrite with per-cell encoding context

This document should explain:
- What the StateEncoder is (updated pipeline diagram)
- Thought process: why per-cell encoding instead of additive counting
- PositionFinder explanation with examples
- New channel layout and what the agent sees
- Updated assumptions (including PositionFinder displacement caveats)
- How to use it (updated code examples with new shapes)
- Debugging/inspection examples (finding units by class ID)
- File reference (now includes position_finder.py)
- Common questions (updated for new encoding)
- BC network integration notes (CRFeatureExtractor with nn.Embedding)
- Next steps in the pipeline

### Step 2: Commit

```bash
git add docs/josh/state_encoder_module/docs/state-encoder-docs.md
git commit -m "docs: rewrite developer docs for per-cell unit encoding"
```

---

## Task 8: Run full test suite and validate

### Step 1: Run all tests

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. pytest docs/josh/state_encoder_module/tests/ -v --tb=short
```

Expected: All tests PASS.

### Step 2: Verify import chain from repo root

```bash
cd /Users/alanguo/Codin/CS175/Project/cr-object-detection
PYTHONPATH=docs/josh/state_encoder_module:. python -c "
from src.encoder import StateEncoder
from src.pipeline.game_state import GameState, Unit, Tower, Card

encoder = StateEncoder()
state = GameState(frame_width=540, frame_height=960, elixir=5)
state.units = [
    Unit('knight', 0, (260, 470, 280, 490), 0.9),
    Unit('skeleton', 1, (100, 200, 120, 220), 0.8),
    Unit('fireball', 0, (270, 400, 290, 420), 0.7),
]
state.player_king_tower = Tower('king', 'center', 0, hp=3204, max_hp=6408, bbox=(250, 800, 290, 850))

obs = encoder.encode(state)
mask = encoder.action_mask(state)

print(f'Arena shape: {obs[\"arena\"].shape}')
print(f'Vector shape: {obs[\"vector\"].shape}')
print(f'Mask shape: {mask.shape}')
print(f'Occupied cells: {(obs[\"arena\"][:,:,2] > 0).sum()}')
print(f'Spell cells: {(obs[\"arena\"][:,:,5] > 0).sum()}')
print(f'Elixir in vector: {obs[\"vector\"][0]:.2f}')
print('All checks passed')
"
```

Expected output:
```
Arena shape: (32, 18, 6)
Vector shape: (23,)
Mask shape: (2305,)
Occupied cells: 2
Spell cells: 1
Elixir in vector: 0.50
All checks passed
```

### Step 3: Final commit

```bash
git add docs/josh/state_encoder_module/
git commit -m "chore: per-cell unit encoding implementation complete with tests and docs"
```

---

## Summary

| Task | Component | Est. Time |
|------|-----------|-----------|
| 1 | PositionFinder + tests | Core collision resolution |
| 2 | encoder_constants.py | New channel constants, CLASS_NAME_TO_ID |
| 3 | coord_utils.py + tests | pixel_to_cell_float |
| 4 | state_encoder.py + tests | Main rewrite |
| 5 | __init__.py | Updated exports |
| 6 | CLAUDE.md | Technical reference |
| 7 | state-encoder-docs.md | Developer documentation |
| 8 | Full validation | Integration test |

**What changes:** `_encode_arena()`, observation space shape (32,18,7)->(32,18,6), channel semantics, new PositionFinder dependency.

**What does NOT change:** Action space (Discrete 2305), action mask, `_encode_vector()`, coord_utils action functions, GameState, StateBuilder, anything outside `src/encoder/`.
