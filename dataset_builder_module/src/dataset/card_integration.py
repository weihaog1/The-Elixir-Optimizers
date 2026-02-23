"""EnhancedStateBuilder wraps StateBuilder + CardPredictor.

Produces complete GameState objects with card hand populated via
the MiniResNet card classifier. This bridges the gap where the base
StateBuilder leaves GameState.cards as an empty list.
"""

from typing import List, Optional

import numpy as np

from src.data.screen_regions import ScreenConfig
from src.pipeline.game_state import Card, GameState


class EnhancedStateBuilder:
    """StateBuilder wrapper that adds card classification to game state.

    Delegates detection and OCR to the underlying StateBuilder, then
    crops the 4 card slots from the raw image and classifies each
    using CardPredictor.

    Args:
        state_builder: Base StateBuilder instance (handles detection + OCR).
        card_predictor: Optional CardPredictor instance. If None, cards
            will remain empty (same behavior as base StateBuilder).
    """

    def __init__(self, state_builder, card_predictor=None):
        self.state_builder = state_builder
        self.card_predictor = card_predictor
        self._base_config = ScreenConfig()

    def build_state(
        self,
        image: np.ndarray,
        timestamp: float = 0.0,
        **kwargs,
    ) -> GameState:
        """Build game state with card hand classification.

        Calls the base state_builder.build_state(), then augments the
        result with card predictions if a card_predictor is available.

        Args:
            image: BGR numpy array of the game screenshot.
            timestamp: Frame timestamp (seconds since epoch or video time).
            **kwargs: Additional arguments forwarded to base build_state.

        Returns:
            GameState with cards populated when card_predictor is set.
        """
        state = self.state_builder.build_state(image, timestamp, **kwargs)
        if self.card_predictor is not None and image is not None:
            state.cards = self._extract_cards(
                image, state.frame_width, state.frame_height
            )
        return state

    def _extract_cards(
        self,
        image: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> List[Card]:
        """Crop 4 card slots and classify each.

        Uses ScreenConfig.scale_to_resolution() to get card slot regions
        at the actual frame resolution, then crops and classifies.

        Args:
            image: BGR numpy array of the full screenshot.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            List of 4 Card objects (one per slot).
        """
        config = self._base_config.scale_to_resolution(frame_w, frame_h)
        cards = []

        for i in range(4):
            slot_key = f"card_{i + 1}"
            region = config.card_slots.get(slot_key)
            if region is None:
                cards.append(Card(slot=i))
                continue

            crop = region.crop_image(image)
            class_name, confidence = self.card_predictor.predict(crop)
            cards.append(
                Card(
                    slot=i,
                    class_name=class_name,
                    elixir_cost=None,
                    confidence=confidence,
                )
            )

        return cards
