"""
Tests for ActionExecutor: coordinate conversion and PyAutoGUI execution.

All PyAutoGUI calls are mocked to avoid actual mouse movement.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from src.action.action_executor import ActionExecutor
from src.action.action_constants import CARD_SLOT_REGIONS, CLICK_DELAY_SECONDS
from src.encoder.coord_utils import cell_to_norm, placement_to_action
from src.encoder.encoder_constants import NOOP_ACTION, GRID_COLS, GRID_ROWS


@pytest.fixture
def executor():
    return ActionExecutor(frame_w=540, frame_h=960)


# ---- _norm_to_pixel ----

class TestNormToPixel:

    def test_origin(self, executor):
        assert executor._norm_to_pixel(0.0, 0.0) == (0, 0)

    def test_center(self, executor):
        x, y = executor._norm_to_pixel(0.5, 0.5)
        assert x == 270
        assert y == 480

    def test_bottom_right(self, executor):
        x, y = executor._norm_to_pixel(1.0, 1.0)
        assert x == 540
        assert y == 960

    def test_custom_resolution(self):
        ex = ActionExecutor(frame_w=1080, frame_h=1920)
        x, y = ex._norm_to_pixel(0.5, 0.5)
        assert x == 540
        assert y == 960


# ---- _card_slot_pixel ----

class TestCardSlotPixel:

    def test_card_0_center(self, executor):
        """Card slot 0 pixel should be center of its normalized region."""
        x_min, y_min, x_max, y_max = CARD_SLOT_REGIONS[0]
        expected_x = int(((x_min + x_max) / 2) * 540)
        expected_y = int(((y_min + y_max) / 2) * 960)
        px, py = executor._card_slot_pixel(0)
        assert px == expected_x
        assert py == expected_y

    def test_card_3_center(self, executor):
        x_min, y_min, x_max, y_max = CARD_SLOT_REGIONS[3]
        expected_x = int(((x_min + x_max) / 2) * 540)
        expected_y = int(((y_min + y_max) / 2) * 960)
        px, py = executor._card_slot_pixel(3)
        assert px == expected_x
        assert py == expected_y

    def test_all_slots_different_x(self, executor):
        """All 4 card slot centers should have different X values."""
        xs = [executor._card_slot_pixel(i)[0] for i in range(4)]
        assert len(set(xs)) == 4

    def test_all_slots_same_y(self, executor):
        """All 4 card slot centers should have the same Y value."""
        ys = [executor._card_slot_pixel(i)[1] for i in range(4)]
        assert len(set(ys)) == 1


# ---- execute ----

class TestExecute:

    @patch("src.action.action_executor.pyautogui")
    def test_noop_returns_false(self, mock_pyautogui, executor):
        """No-op action should return False and not click."""
        result = executor.execute(NOOP_ACTION)
        assert result is False
        mock_pyautogui.click.assert_not_called()

    @patch("src.action.action_executor.time")
    @patch("src.action.action_executor.pyautogui")
    def test_valid_action_returns_true(self, mock_pyautogui, mock_time, executor):
        """Valid card placement should return True and call click twice."""
        action_idx = placement_to_action(0, 9, 16)
        result = executor.execute(action_idx)
        assert result is True
        assert mock_pyautogui.click.call_count == 2

    @patch("src.action.action_executor.time")
    @patch("src.action.action_executor.pyautogui")
    def test_execute_calls_sleep_between_clicks(self, mock_pyautogui, mock_time, executor):
        """Should sleep CLICK_DELAY_SECONDS between card and arena clicks."""
        action_idx = placement_to_action(1, 5, 20)
        executor.execute(action_idx)
        mock_time.sleep.assert_called_once_with(CLICK_DELAY_SECONDS)


# ---- play_card ----

class TestPlayCard:

    @patch("src.action.action_executor.time")
    @patch("src.action.action_executor.pyautogui")
    def test_play_card_click_order(self, mock_pyautogui, mock_time, executor):
        """First click should be card slot, second should be arena position."""
        card_id = 2
        arena_x_norm, arena_y_norm = 0.5, 0.4

        executor.play_card(card_id, arena_x_norm, arena_y_norm)

        calls = mock_pyautogui.click.call_args_list
        assert len(calls) == 2

        # First call: card slot center
        card_px, card_py = executor._card_slot_pixel(card_id)
        assert calls[0] == call(card_px, card_py)

        # Second call: arena position
        arena_px, arena_py = executor._norm_to_pixel(arena_x_norm, arena_y_norm)
        assert calls[1] == call(arena_px, arena_py)

    @patch("src.action.action_executor.time")
    @patch("src.action.action_executor.pyautogui")
    def test_play_card_all_slots(self, mock_pyautogui, mock_time, executor):
        """Should work for all 4 card slots."""
        for card_id in range(4):
            mock_pyautogui.reset_mock()
            executor.play_card(card_id, 0.5, 0.5)
            assert mock_pyautogui.click.call_count == 2


# ---- Roundtrip: encode -> decode -> execute ----

class TestRoundtrip:

    @patch("src.action.action_executor.time")
    @patch("src.action.action_executor.pyautogui")
    def test_encode_decode_execute_roundtrip(self, mock_pyautogui, mock_time, executor):
        """placement_to_action -> execute -> clicks match expected coords."""
        card_id, col, row = 1, 10, 25
        action_idx = placement_to_action(card_id, col, row)
        executor.execute(action_idx)

        calls = mock_pyautogui.click.call_args_list
        assert len(calls) == 2

        # Verify arena click matches cell_to_norm output
        x_norm, y_norm = cell_to_norm(col, row)
        expected_px = executor._norm_to_pixel(x_norm, y_norm)
        assert calls[1] == call(*expected_px)

    @patch("src.action.action_executor.time")
    @patch("src.action.action_executor.pyautogui")
    def test_corner_cells(self, mock_pyautogui, mock_time, executor):
        """Actions at grid corners should execute without error."""
        corners = [
            (0, 0, 0),
            (0, GRID_COLS - 1, 0),
            (0, 0, GRID_ROWS - 1),
            (3, GRID_COLS - 1, GRID_ROWS - 1),
        ]
        for card_id, col, row in corners:
            mock_pyautogui.reset_mock()
            action_idx = placement_to_action(card_id, col, row)
            result = executor.execute(action_idx)
            assert result is True
            assert mock_pyautogui.click.call_count == 2
