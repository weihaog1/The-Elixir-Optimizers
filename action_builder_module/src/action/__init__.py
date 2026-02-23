"""
Action module for behavior cloning data collection and game execution.

Converts click logger events into Discrete(2305) action indices,
and converts action indices back into PyAutoGUI commands.
"""

from .action_builder import ActionBuilder, ActionEvent, ClickEvent
from .action_executor import ActionExecutor
from .action_constants import (
    CARD_SLOT_REGIONS,
    CARD_BAR_Y_MIN,
    CARD_BAR_Y_MAX,
    ARENA_Y_MIN,
    ARENA_Y_MAX,
    CLICK_DELAY_SECONDS,
    PYAUTOGUI_PAUSE,
)

__all__ = [
    "ActionBuilder",
    "ActionEvent",
    "ClickEvent",
    "ActionExecutor",
    "CARD_SLOT_REGIONS",
    "CARD_BAR_Y_MIN",
    "CARD_BAR_Y_MAX",
    "ARENA_Y_MIN",
    "ARENA_Y_MAX",
    "CLICK_DELAY_SECONDS",
    "PYAUTOGUI_PAUSE",
]
