"""Tests for StateEncoder with per-cell identity encoding."""

import numpy as np
import pytest

from src.encoder.state_encoder import StateEncoder
from src.encoder.encoder_constants import (
    ACTION_SPACE_SIZE,
    CARD_ELIXIR_COST,
    CH_ALLY_TOWER_HP,
    CH_ARENA_MASK,
    CH_BELONGING,
    CH_CLASS_ID,
    CH_ENEMY_TOWER_HP,
    CH_SPELL,
    CLASS_NAME_TO_ID,
    DEFAULT_KING_MAX_HP,
    DEFAULT_PRINCESS_MAX_HP,
    GRID_CELLS,
    GRID_COLS,
    GRID_ROWS,
    NOOP_ACTION,
    NUM_ARENA_CHANNELS,
    NUM_CLASSES,
    NUM_VECTOR_FEATURES,
)
from src.pipeline.game_state import GameState, Tower, Unit, Card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_state(fw: int = 540, fh: int = 960) -> GameState:
    """Create a minimal empty game state."""
    return GameState(frame_width=fw, frame_height=fh)


def _make_unit(
    class_name: str,
    belonging: int,
    cx: int = 270,
    cy: int = 400,
    w: int = 40,
    h: int = 40,
    confidence: float = 0.9,
) -> Unit:
    """Create a unit centered at (cx, cy)."""
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2
    return Unit(
        class_name=class_name,
        belonging=belonging,
        bbox=(x1, y1, x2, y2),
        confidence=confidence,
    )


def _make_tower(
    tower_type: str,
    position: str,
    belonging: int,
    hp: int = 4000,
    max_hp: int = None,
    cx: int = 270,
    cy: int = 400,
) -> Tower:
    """Create a tower with bbox centered at (cx, cy)."""
    return Tower(
        tower_type=tower_type,
        position=position,
        belonging=belonging,
        hp=hp,
        max_hp=max_hp,
        bbox=(cx - 30, cy - 30, cx + 30, cy + 30),
    )


# ---------------------------------------------------------------------------
# Arena encoding tests
# ---------------------------------------------------------------------------

class TestArenaShape:
    """Arena tensor shape, dtype, and default values."""

    def test_arena_shape(self):
        encoder = StateEncoder()
        obs = encoder.encode(_empty_state())
        assert obs["arena"].shape == (GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS)
        assert obs["arena"].shape == (32, 18, 6)

    def test_arena_dtype(self):
        encoder = StateEncoder()
        obs = encoder.encode(_empty_state())
        assert obs["arena"].dtype == np.float32

    def test_empty_state_all_zeros(self):
        encoder = StateEncoder()
        obs = encoder.encode(_empty_state())
        assert np.all(obs["arena"] == 0.0)


class TestSingleUnit:
    """Encoding of a single ground/flying unit."""

    def test_single_ally_unit(self):
        encoder = StateEncoder()
        state = _empty_state()
        state.units = [_make_unit("knight", belonging=0)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # Find the cell with arena_mask = 1
        mask_cells = np.argwhere(arena[:, :, CH_ARENA_MASK] == 1.0)
        assert len(mask_cells) == 1, "Exactly one cell should be marked"

        row, col = mask_cells[0]
        expected_class_id = CLASS_NAME_TO_ID["knight"] / NUM_CLASSES
        assert arena[row, col, CH_CLASS_ID] == pytest.approx(expected_class_id, abs=1e-6)
        assert arena[row, col, CH_BELONGING] == -1.0  # ally
        assert arena[row, col, CH_ARENA_MASK] == 1.0

    def test_single_enemy_unit(self):
        encoder = StateEncoder()
        state = _empty_state()
        state.units = [_make_unit("knight", belonging=1)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        mask_cells = np.argwhere(arena[:, :, CH_ARENA_MASK] == 1.0)
        assert len(mask_cells) == 1

        row, col = mask_cells[0]
        assert arena[row, col, CH_BELONGING] == 1.0  # enemy


class TestMultipleUnits:
    """Multiple units should each get their own cell."""

    def test_multiple_units_unique_cells(self):
        encoder = StateEncoder()
        state = _empty_state()
        # Place 3 units at different positions
        state.units = [
            _make_unit("knight", belonging=0, cx=100, cy=400),
            _make_unit("archer", belonging=0, cx=300, cy=600),
            _make_unit("goblin", belonging=1, cx=450, cy=200),
        ]
        obs = encoder.encode(state)
        arena = obs["arena"]

        mask_cells = np.argwhere(arena[:, :, CH_ARENA_MASK] == 1.0)
        assert len(mask_cells) == 3, "All 3 units should be placed"

        # All cells should be unique
        cell_set = set(tuple(c) for c in mask_cells)
        assert len(cell_set) == 3

    def test_overlapping_units_both_placed(self):
        """Two units at the same pixel position should both get cells."""
        encoder = StateEncoder()
        state = _empty_state()
        state.units = [
            _make_unit("knight", belonging=0, cx=270, cy=400),
            _make_unit("archer", belonging=0, cx=270, cy=400),
        ]
        obs = encoder.encode(state)
        arena = obs["arena"]

        mask_cells = np.argwhere(arena[:, :, CH_ARENA_MASK] == 1.0)
        assert len(mask_cells) == 2, "Both overlapping units should be placed"

        cell_set = set(tuple(c) for c in mask_cells)
        assert len(cell_set) == 2, "Cells must be unique"


class TestUnknownClass:
    """Units with class names not in CLASS_NAME_TO_ID should be skipped."""

    def test_unknown_class_skipped(self):
        encoder = StateEncoder()
        state = _empty_state()
        state.units = [_make_unit("totally-fake-unit-name", belonging=0)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # No cell should be marked
        assert np.all(arena[:, :, CH_ARENA_MASK] == 0.0)


class TestSpellEncoding:
    """Spell units go to CH_SPELL, not per-cell identity channels."""

    def test_spell_in_spell_channel(self):
        encoder = StateEncoder()
        state = _empty_state()
        # "zap" is in spell_unit_list
        state.units = [_make_unit("zap", belonging=1, cx=270, cy=400)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # Spell channel should have a nonzero value somewhere
        assert np.sum(arena[:, :, CH_SPELL]) > 0
        # Per-cell identity channels should be empty (spells bypass PositionFinder)
        assert np.all(arena[:, :, CH_ARENA_MASK] == 0.0)

    def test_spell_additive(self):
        """Two spells in the same cell should produce count 2."""
        encoder = StateEncoder()
        state = _empty_state()
        state.units = [
            _make_unit("zap", belonging=1, cx=270, cy=400),
            _make_unit("zap", belonging=0, cx=270, cy=400),
        ]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # Find cell with spell count
        spell_sum = np.sum(arena[:, :, CH_SPELL])
        assert spell_sum == pytest.approx(2.0)

    def test_spell_and_troop_coexist(self):
        """A spell and a troop in nearby cells use different channels."""
        encoder = StateEncoder()
        state = _empty_state()
        state.units = [
            _make_unit("zap", belonging=1, cx=270, cy=400),
            _make_unit("knight", belonging=0, cx=270, cy=400),
        ]
        obs = encoder.encode(state)
        arena = obs["arena"]

        # Spell channel has a value
        assert np.sum(arena[:, :, CH_SPELL]) > 0
        # Per-cell identity has a value (the knight)
        assert np.sum(arena[:, :, CH_ARENA_MASK]) == 1.0


class TestTowerHP:
    """Tower HP encoding (unchanged from old encoder)."""

    def test_ally_tower_hp(self):
        encoder = StateEncoder()
        state = _empty_state()
        state.player_king_tower = _make_tower(
            "king", "center", belonging=0, hp=3204, max_hp=6408
        )
        obs = encoder.encode(state)
        arena = obs["arena"]

        # Should have a non-zero value in the ally tower HP channel
        assert np.max(arena[:, :, CH_ALLY_TOWER_HP]) == pytest.approx(0.5, abs=0.01)

    def test_enemy_tower_hp(self):
        encoder = StateEncoder()
        state = _empty_state()
        state.enemy_left_princess = _make_tower(
            "princess", "left", belonging=1, hp=2016, max_hp=4032
        )
        obs = encoder.encode(state)
        arena = obs["arena"]

        assert np.max(arena[:, :, CH_ENEMY_TOWER_HP]) == pytest.approx(0.5, abs=0.01)


class TestOtherTowerSkipped:
    """Units with type 'other' or 'tower' should not appear in per-cell channels."""

    def test_other_unit_skipped(self):
        encoder = StateEncoder()
        state = _empty_state()
        # "bar" is in other_unit_list
        state.units = [_make_unit("bar", belonging=0)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        assert np.all(arena[:, :, CH_ARENA_MASK] == 0.0)

    def test_tower_unit_skipped(self):
        """A tower in the units list should be skipped (towers are handled separately)."""
        encoder = StateEncoder()
        state = _empty_state()
        # "king-tower" is in tower_unit_list
        state.units = [_make_unit("king-tower", belonging=0)]
        obs = encoder.encode(state)
        arena = obs["arena"]

        assert np.all(arena[:, :, CH_ARENA_MASK] == 0.0)


# ---------------------------------------------------------------------------
# Vector encoding tests
# ---------------------------------------------------------------------------

class TestVectorEncoding:
    """Vector observation shape and encoding."""

    def test_vector_shape(self):
        encoder = StateEncoder()
        obs = encoder.encode(_empty_state())
        assert obs["vector"].shape == (NUM_VECTOR_FEATURES,)
        assert obs["vector"].shape == (23,)

    def test_elixir_encoding(self):
        state = _empty_state()
        state.elixir = 5
        encoder = StateEncoder()
        obs = encoder.encode(state)
        assert obs["vector"][0] == pytest.approx(0.5)

    def test_time_encoding(self):
        state = _empty_state()
        state.time_remaining = 150
        encoder = StateEncoder()
        obs = encoder.encode(state)
        assert obs["vector"][1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Action mask tests
# ---------------------------------------------------------------------------

class TestActionMask:
    """Action mask encoding."""

    def test_noop_always_valid(self):
        encoder = StateEncoder()
        mask = encoder.action_mask(_empty_state())
        assert mask[NOOP_ACTION] is np.True_

    def test_no_cards_only_noop(self):
        encoder = StateEncoder()
        state = _empty_state()
        state.elixir = 10
        mask = encoder.action_mask(state)
        assert mask.sum() == 1  # only noop
        assert mask[NOOP_ACTION]

    def test_playable_card_unmasks_cells(self):
        encoder = StateEncoder()
        state = _empty_state()
        state.elixir = 10
        state.cards = [Card(slot=0, class_name="arrows", elixir_cost=3)]
        mask = encoder.action_mask(state)
        # 576 cells for card 0 + 1 noop
        assert mask.sum() == GRID_CELLS + 1

    def test_insufficient_elixir_masked(self):
        encoder = StateEncoder()
        state = _empty_state()
        state.elixir = 2  # arrows costs 3
        state.cards = [Card(slot=0, class_name="arrows", elixir_cost=3)]
        mask = encoder.action_mask(state)
        # Only noop should be valid
        assert mask.sum() == 1
        assert mask[NOOP_ACTION]


# ---------------------------------------------------------------------------
# Observation space tests
# ---------------------------------------------------------------------------

class TestObservationSpace:
    """Observation and action space definitions."""

    def test_arena_space_shape(self):
        encoder = StateEncoder()
        arena_space = encoder.observation_space["arena"]
        assert arena_space.shape == (32, 18, 6)

    def test_arena_space_low(self):
        encoder = StateEncoder()
        arena_space = encoder.observation_space["arena"]
        assert float(arena_space.low.min()) == -1.0

    def test_arena_space_high(self):
        encoder = StateEncoder()
        arena_space = encoder.observation_space["arena"]
        assert float(arena_space.high.max()) == 10.0

    def test_action_space_n(self):
        encoder = StateEncoder()
        assert encoder.action_space.n == ACTION_SPACE_SIZE
        assert encoder.action_space.n == 2305
