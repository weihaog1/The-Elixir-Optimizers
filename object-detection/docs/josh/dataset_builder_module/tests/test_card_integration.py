"""Tests for EnhancedStateBuilder card integration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.dataset.card_integration import EnhancedStateBuilder
from src.pipeline.game_state import Card, GameState


def _make_mock_state_builder(frame_w=540, frame_h=960):
    """Create a mock StateBuilder that returns a blank GameState."""
    mock = MagicMock()
    mock.build_state.return_value = GameState(
        timestamp=0.0,
        frame_width=frame_w,
        frame_height=frame_h,
    )
    return mock


def _make_mock_card_predictor(predictions=None):
    """Create a mock CardPredictor with configurable predictions.

    Args:
        predictions: List of (class_name, confidence) tuples, one per slot.
            Defaults to 4 different card predictions.
    """
    if predictions is None:
        predictions = [
            ("royal-hogs", 0.95),
            ("arrows", 0.88),
            ("zappies", 0.91),
            ("goblin-cage", 0.87),
        ]
    mock = MagicMock()
    mock.predict.side_effect = predictions
    return mock


class TestEnhancedStateBuilder:
    """Tests for EnhancedStateBuilder."""

    def test_build_state_with_card_predictor(self):
        """Cards should be populated when card_predictor is provided."""
        mock_sb = _make_mock_state_builder()
        mock_cp = _make_mock_card_predictor()
        image = np.zeros((960, 540, 3), dtype=np.uint8)

        builder = EnhancedStateBuilder(mock_sb, mock_cp)
        state = builder.build_state(image, timestamp=1.0)

        assert len(state.cards) == 4
        assert state.cards[0].class_name == "royal-hogs"
        assert state.cards[0].slot == 0
        assert state.cards[0].confidence == 0.95
        assert state.cards[1].class_name == "arrows"
        assert state.cards[1].slot == 1
        assert state.cards[2].class_name == "zappies"
        assert state.cards[2].slot == 2
        assert state.cards[3].class_name == "goblin-cage"
        assert state.cards[3].slot == 3

    def test_build_state_without_card_predictor(self):
        """Cards should remain empty when card_predictor is None."""
        mock_sb = _make_mock_state_builder()
        image = np.zeros((960, 540, 3), dtype=np.uint8)

        builder = EnhancedStateBuilder(mock_sb, card_predictor=None)
        state = builder.build_state(image, timestamp=1.0)

        assert state.cards == []

    def test_build_state_delegates_to_base(self):
        """build_state should call the base state_builder.build_state."""
        mock_sb = _make_mock_state_builder()
        mock_cp = _make_mock_card_predictor()
        image = np.zeros((960, 540, 3), dtype=np.uint8)

        builder = EnhancedStateBuilder(mock_sb, mock_cp)
        builder.build_state(image, timestamp=5.5)

        mock_sb.build_state.assert_called_once_with(image, 5.5)

    def test_build_state_forwards_kwargs(self):
        """Extra kwargs should be forwarded to the base build_state."""
        mock_sb = _make_mock_state_builder()
        image = np.zeros((960, 540, 3), dtype=np.uint8)

        builder = EnhancedStateBuilder(mock_sb)
        builder.build_state(image, timestamp=0.0, run_detection=False)

        mock_sb.build_state.assert_called_once_with(
            image, 0.0, run_detection=False
        )

    def test_extract_cards_scales_regions(self):
        """Card slot crops should scale to the actual frame resolution."""
        mock_sb = _make_mock_state_builder(frame_w=1080, frame_h=1920)
        mock_cp = _make_mock_card_predictor()
        # 1080x1920 image
        image = np.zeros((1920, 1080, 3), dtype=np.uint8)

        builder = EnhancedStateBuilder(mock_sb, mock_cp)
        state = builder.build_state(image, timestamp=0.0)

        # CardPredictor.predict should have been called 4 times
        assert mock_cp.predict.call_count == 4
        assert len(state.cards) == 4

    def test_extract_cards_with_mock_image(self):
        """Card extraction should crop and classify from actual pixel data."""
        mock_sb = _make_mock_state_builder(frame_w=540, frame_h=960)
        predictions = [
            ("flying-machine", 0.92),
            ("eletro-spirit", 0.85),
            ("barbarian-barrel", 0.78),
            ("royal-recruits", 0.99),
        ]
        mock_cp = _make_mock_card_predictor(predictions)

        # Create image with non-zero data so crops are meaningful
        image = np.random.randint(0, 255, (960, 540, 3), dtype=np.uint8)

        builder = EnhancedStateBuilder(mock_sb, mock_cp)
        state = builder.build_state(image, timestamp=0.0)

        assert state.cards[0].class_name == "flying-machine"
        assert state.cards[3].class_name == "royal-recruits"
        assert state.cards[3].confidence == 0.99

        # Verify each predict call received a numpy array crop
        for call in mock_cp.predict.call_args_list:
            crop = call[0][0]
            assert isinstance(crop, np.ndarray)
            assert crop.ndim == 3
            assert crop.shape[2] == 3

    def test_card_elixir_cost_is_none(self):
        """Elixir cost should be None since CardPredictor does not provide it."""
        mock_sb = _make_mock_state_builder()
        mock_cp = _make_mock_card_predictor()
        image = np.zeros((960, 540, 3), dtype=np.uint8)

        builder = EnhancedStateBuilder(mock_sb, mock_cp)
        state = builder.build_state(image, timestamp=0.0)

        for card in state.cards:
            assert card.elixir_cost is None
