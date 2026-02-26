"""
Tests for ActionBuilder: click classification, click pairing, and timeline.
"""

import pytest

from src.action.action_builder import ActionBuilder, ActionEvent, ClickEvent
from src.action.action_constants import (
    ARENA_Y_MAX,
    ARENA_Y_MIN,
    CARD_BAR_Y_MAX,
    CARD_BAR_Y_MIN,
    CARD_SLOT_REGIONS,
)
from src.encoder.coord_utils import (
    action_to_placement,
    norm_to_cell,
    placement_to_action,
)
from src.encoder.encoder_constants import NOOP_ACTION


@pytest.fixture
def builder():
    return ActionBuilder()


# ---- Card slot region sanity ----

def test_card_slot_regions_count():
    """There should be exactly 4 card slot regions."""
    assert len(CARD_SLOT_REGIONS) == 4


def test_card_slot_regions_no_overlap():
    """Card slot X ranges should not overlap."""
    for i in range(3):
        x_max_i = CARD_SLOT_REGIONS[i][2]
        x_min_next = CARD_SLOT_REGIONS[i + 1][0]
        assert x_max_i <= x_min_next + 1e-9


def test_card_slot_regions_within_card_bar():
    """All card slot Y ranges should be within the card bar."""
    for x_min, y_min, x_max, y_max in CARD_SLOT_REGIONS:
        assert y_min >= CARD_BAR_Y_MIN - 1e-9
        assert y_max <= CARD_BAR_Y_MAX + 1e-9


# ---- classify_click ----

class TestClassifyClick:

    def test_card_0_center(self, builder):
        x_min, y_min, x_max, y_max = CARD_SLOT_REGIONS[0]
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        assert builder.classify_click(cx, cy) == "card_0"

    def test_card_1_center(self, builder):
        x_min, y_min, x_max, y_max = CARD_SLOT_REGIONS[1]
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        assert builder.classify_click(cx, cy) == "card_1"

    def test_card_2_center(self, builder):
        x_min, y_min, x_max, y_max = CARD_SLOT_REGIONS[2]
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        assert builder.classify_click(cx, cy) == "card_2"

    def test_card_3_center(self, builder):
        x_min, y_min, x_max, y_max = CARD_SLOT_REGIONS[3]
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        assert builder.classify_click(cx, cy) == "card_3"

    def test_card_0_top_left_edge(self, builder):
        """Click at exact top-left corner of card_0 is still card_0."""
        x_min, y_min, _, _ = CARD_SLOT_REGIONS[0]
        assert builder.classify_click(x_min, y_min) == "card_0"

    def test_card_3_bottom_right_edge(self, builder):
        """Click at exact bottom-right corner of card_3 is still card_3."""
        _, _, x_max, y_max = CARD_SLOT_REGIONS[3]
        assert builder.classify_click(x_max, y_max) == "card_3"

    def test_arena_center(self, builder):
        cy = (ARENA_Y_MIN + ARENA_Y_MAX) / 2
        assert builder.classify_click(0.5, cy) == "arena"

    def test_arena_top_edge(self, builder):
        assert builder.classify_click(0.5, ARENA_Y_MIN) == "arena"

    def test_arena_bottom_edge(self, builder):
        assert builder.classify_click(0.5, ARENA_Y_MAX) == "arena"

    def test_above_arena_is_other(self, builder):
        """Clicking the timer bar region is 'other'."""
        assert builder.classify_click(0.5, 0.01) == "other"

    def test_between_arena_and_cards_is_other(self, builder):
        """Gap between arena bottom and card bar top is 'other'."""
        gap_y = (ARENA_Y_MAX + CARD_BAR_Y_MIN) / 2
        assert builder.classify_click(0.5, gap_y) == "other"

    def test_below_cards_is_other(self, builder):
        """Below the card bar is 'other'."""
        assert builder.classify_click(0.5, 0.99) == "other"

    def test_left_of_cards_in_card_bar_y_range(self, builder):
        """In card bar Y range but left of card_0 is 'other'."""
        cy = (CARD_BAR_Y_MIN + CARD_BAR_Y_MAX) / 2
        assert builder.classify_click(0.05, cy) == "other"


# ---- clicks_to_actions ----

class TestClicksToActions:

    def test_normal_card_arena_pair(self, builder):
        """Card click followed by arena click produces one ActionEvent."""
        x_card, y_card = _card_center(0)
        clicks = [
            ClickEvent(1.0, x_card, y_card),
            ClickEvent(1.5, 0.5, 0.4),
        ]
        actions = builder.clicks_to_actions(clicks)
        assert len(actions) == 1
        assert actions[0].card_id == 0
        assert actions[0].timestamp == 1.5

    def test_action_encoding_roundtrip(self, builder):
        """ActionEvent.action_idx should decode back to card_id, col, row."""
        x_card, y_card = _card_center(2)
        arena_x, arena_y = 0.3, 0.5
        clicks = [
            ClickEvent(1.0, x_card, y_card),
            ClickEvent(1.5, arena_x, arena_y),
        ]
        actions = builder.clicks_to_actions(clicks)
        assert len(actions) == 1

        result = action_to_placement(actions[0].action_idx)
        assert result is not None
        card_id, col, row = result
        assert card_id == 2
        expected_col, expected_row = norm_to_cell(arena_x, arena_y)
        assert col == expected_col
        assert row == expected_row

    def test_changed_mind_uses_second_card(self, builder):
        """Two card clicks then arena uses the second card."""
        x_card0, y_card0 = _card_center(0)
        x_card2, y_card2 = _card_center(2)
        clicks = [
            ClickEvent(1.0, x_card0, y_card0),
            ClickEvent(1.2, x_card2, y_card2),
            ClickEvent(1.5, 0.5, 0.4),
        ]
        actions = builder.clicks_to_actions(clicks)
        assert len(actions) == 1
        assert actions[0].card_id == 2

    def test_arena_without_card_ignored(self, builder):
        """Arena click without prior card click produces no action."""
        clicks = [ClickEvent(1.0, 0.5, 0.4)]
        actions = builder.clicks_to_actions(clicks)
        assert len(actions) == 0

    def test_card_then_other_resets(self, builder):
        """Card click then 'other' click resets; subsequent arena ignored."""
        x_card, y_card = _card_center(1)
        clicks = [
            ClickEvent(1.0, x_card, y_card),
            ClickEvent(1.2, 0.5, 0.01),       # 'other' (timer bar)
            ClickEvent(1.5, 0.5, 0.4),         # arena, but state is idle
        ]
        actions = builder.clicks_to_actions(clicks)
        assert len(actions) == 0

    def test_two_complete_pairs(self, builder):
        """Two complete card+arena pairs produce two ActionEvents."""
        x_card0, y_card0 = _card_center(0)
        x_card3, y_card3 = _card_center(3)
        clicks = [
            ClickEvent(1.0, x_card0, y_card0),
            ClickEvent(1.5, 0.3, 0.3),
            ClickEvent(2.0, x_card3, y_card3),
            ClickEvent(2.5, 0.7, 0.5),
        ]
        actions = builder.clicks_to_actions(clicks)
        assert len(actions) == 2
        assert actions[0].card_id == 0
        assert actions[1].card_id == 3

    def test_unpaired_card_at_end(self, builder):
        """Card click at end of list without arena click is ignored."""
        x_card, y_card = _card_center(1)
        clicks = [ClickEvent(1.0, x_card, y_card)]
        actions = builder.clicks_to_actions(clicks)
        assert len(actions) == 0

    def test_empty_clicks(self, builder):
        """Empty click list produces empty action list."""
        assert builder.clicks_to_actions([]) == []

    def test_action_event_stores_arena_coords(self, builder):
        """ActionEvent should store the arena click's normalized coords."""
        x_card, y_card = _card_center(0)
        arena_x, arena_y = 0.25, 0.6
        clicks = [
            ClickEvent(1.0, x_card, y_card),
            ClickEvent(2.0, arena_x, arena_y),
        ]
        actions = builder.clicks_to_actions(clicks)
        assert len(actions) == 1
        assert actions[0].x_norm == arena_x
        assert actions[0].y_norm == arena_y


# ---- build_action_timeline ----

class TestBuildActionTimeline:

    def test_all_noop_no_clicks(self, builder):
        """No clicks means all frames are no-op."""
        frames = [0.0, 0.5, 1.0, 1.5]
        timeline = builder.build_action_timeline([], frames)
        assert timeline == [NOOP_ACTION] * 4

    def test_action_assigned_to_correct_frame(self, builder):
        """Action at t=1.3 should appear in the frame at t=1.5."""
        x_card, y_card = _card_center(0)
        clicks = [
            ClickEvent(1.2, x_card, y_card),
            ClickEvent(1.3, 0.5, 0.4),   # arena click (action time)
        ]
        frames = [0.5, 1.0, 1.5, 2.0]
        timeline = builder.build_action_timeline(clicks, frames)

        # Frames 0, 1, 3 should be no-op; frame 2 (t=1.5) should have the action
        assert timeline[0] == NOOP_ACTION
        assert timeline[1] == NOOP_ACTION
        assert timeline[2] != NOOP_ACTION
        assert timeline[3] == NOOP_ACTION

    def test_multiple_actions_in_timeline(self, builder):
        """Two actions at different times land in different frames."""
        x_card0, y_card0 = _card_center(0)
        x_card1, y_card1 = _card_center(1)
        clicks = [
            ClickEvent(0.8, x_card0, y_card0),
            ClickEvent(0.9, 0.3, 0.3),
            ClickEvent(2.1, x_card1, y_card1),
            ClickEvent(2.2, 0.7, 0.5),
        ]
        frames = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        timeline = builder.build_action_timeline(clicks, frames)

        # Action 1 at t=0.9 -> frame at t=1.0
        assert timeline[1] != NOOP_ACTION
        # Action 2 at t=2.2 -> frame at t=2.5
        assert timeline[4] != NOOP_ACTION
        # Other frames are no-op
        assert timeline[0] == NOOP_ACTION
        assert timeline[2] == NOOP_ACTION
        assert timeline[3] == NOOP_ACTION
        assert timeline[5] == NOOP_ACTION

    def test_empty_frame_timestamps(self, builder):
        """Empty frame list returns empty timeline."""
        x_card, y_card = _card_center(0)
        clicks = [
            ClickEvent(1.0, x_card, y_card),
            ClickEvent(1.5, 0.5, 0.4),
        ]
        assert builder.build_action_timeline(clicks, []) == []

    def test_action_exactly_at_frame_time(self, builder):
        """Action timestamp exactly matching a frame time is assigned to it."""
        x_card, y_card = _card_center(0)
        clicks = [
            ClickEvent(0.9, x_card, y_card),
            ClickEvent(1.0, 0.5, 0.4),   # exactly at frame time
        ]
        frames = [0.5, 1.0, 1.5]
        timeline = builder.build_action_timeline(clicks, frames)
        assert timeline[1] != NOOP_ACTION

    def test_action_before_first_frame(self, builder):
        """Action before first frame timestamp is assigned to first frame."""
        x_card, y_card = _card_center(0)
        clicks = [
            ClickEvent(0.1, x_card, y_card),
            ClickEvent(0.2, 0.5, 0.4),
        ]
        frames = [0.5, 1.0, 1.5]
        timeline = builder.build_action_timeline(clicks, frames)
        assert timeline[0] != NOOP_ACTION


# ---- Helpers ----

def _card_center(card_id: int) -> tuple[float, float]:
    """Return (x_norm, y_norm) for the center of a card slot."""
    x_min, y_min, x_max, y_max = CARD_SLOT_REGIONS[card_id]
    return (x_min + x_max) / 2, (y_min + y_max) / 2
