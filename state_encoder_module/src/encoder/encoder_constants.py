"""
Constants for the StateEncoder and action space.

Single source of truth for grid dimensions, action encoding,
unit type classification, class identity encoding, and deck
configuration. Used by StateEncoder, coord_utils, PositionFinder,
and the future DatasetBuilder / Gym env.

Per-cell identity encoding (v2):
  Channel 0: class_id (1-indexed, 0 = empty) normalized by NUM_CLASSES
  Channel 1: belonging (-1 = ally, +1 = enemy)
  Channel 2: arena_mask (1 = unit present, 0 = empty)
  Channel 3: ally tower HP fraction
  Channel 4: enemy tower HP fraction
  Channel 5: spell effect count (additive, bypasses PositionFinder)
"""

from src.generation.label_list import (
    ground_unit_list,
    flying_unit_list,
    spell_unit_list,
    tower_unit_list,
    other_unit_list,
    unit_list,
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
# Arena grid channel indices (per-cell identity encoding)
# ---------------------------------------------------------------------------
CH_CLASS_ID = 0
CH_BELONGING = 1
CH_ARENA_MASK = 2
CH_ALLY_TOWER_HP = 3
CH_ENEMY_TOWER_HP = 4
CH_SPELL = 5
NUM_ARENA_CHANNELS = 6

# ---------------------------------------------------------------------------
# Class identity mapping
# All 155 detection class names mapped to 1-indexed integers.
# 0 is reserved for "empty cell" (no unit present).
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
# In the 32-row grid, rows 0-14 are enemy side, rows 17-31 are player side
# Rows 15-16 are the river area
PLAYER_HALF_ROW_START = 17

# Default tower HP (level 14 -- standard tournament level)
DEFAULT_KING_MAX_HP = 6408
DEFAULT_PRINCESS_MAX_HP = 4032
