"""
ActionExecutor: converts Discrete(2305) action indices into PyAutoGUI commands.

Takes an action index from the policy network, decodes it into a card slot
and arena position, then executes the two-click sequence via PyAutoGUI.
"""

import time

try:
    import pyautogui
except ImportError:
    pyautogui = None  # type: ignore[assignment]

from src.encoder.coord_utils import action_to_placement, cell_to_norm

from .action_constants import (
    CARD_SLOT_REGIONS,
    CLICK_DELAY_SECONDS,
    PYAUTOGUI_PAUSE,
)


class ActionExecutor:
    """Executes Discrete(2305) actions via PyAutoGUI mouse clicks.

    Decodes an action index into (card_id, col, row), converts to pixel
    coordinates, and performs a two-click sequence: card slot then arena.

    Args:
        frame_w: Capture resolution width in pixels (default 540).
        frame_h: Capture resolution height in pixels (default 960).
    """

    def __init__(self, frame_w: int = 540, frame_h: int = 960) -> None:
        self.frame_w = frame_w
        self.frame_h = frame_h
        if pyautogui is not None:
            pyautogui.PAUSE = PYAUTOGUI_PAUSE

    def execute(self, action_idx: int) -> bool:
        """Execute an action via PyAutoGUI.

        Args:
            action_idx: Discrete(2305) action index.

        Returns:
            True if a card was played, False for no-op.
        """
        result = action_to_placement(action_idx)
        if result is None:
            return False

        card_id, col, row = result
        x_norm, y_norm = cell_to_norm(col, row)
        self.play_card(card_id, x_norm, y_norm)
        return True

    def play_card(self, card_id: int, x_norm: float, y_norm: float) -> None:
        """Click a card slot then click the arena position.

        Args:
            card_id: Card slot index (0-3).
            x_norm: Normalized arena x coordinate.
            y_norm: Normalized arena y coordinate.
        """
        # Click card slot
        card_x, card_y = self._card_slot_pixel(card_id)
        pyautogui.click(card_x, card_y)

        time.sleep(CLICK_DELAY_SECONDS)

        # Click arena position
        arena_x, arena_y = self._norm_to_pixel(x_norm, y_norm)
        pyautogui.click(arena_x, arena_y)

    def _card_slot_pixel(self, card_id: int) -> tuple[int, int]:
        """Get pixel coordinates for the center of a card slot.

        Args:
            card_id: Card slot index (0-3).

        Returns:
            (x_px, y_px) pixel coordinates.
        """
        x_min, y_min, x_max, y_max = CARD_SLOT_REGIONS[card_id]
        cx_norm = (x_min + x_max) / 2.0
        cy_norm = (y_min + y_max) / 2.0
        return self._norm_to_pixel(cx_norm, cy_norm)

    def _norm_to_pixel(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates.

        Args:
            x_norm: Normalized x (0-1).
            y_norm: Normalized y (0-1).

        Returns:
            (x_px, y_px) integer pixel coordinates.
        """
        return int(x_norm * self.frame_w), int(y_norm * self.frame_h)
