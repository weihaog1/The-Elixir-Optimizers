"""
Constants for the ActionBuilder and ActionExecutor.

Card slot regions and timing parameters derived from ScreenConfig
at 540x960 base resolution. All spatial values are normalized (0-1).
"""

from src.encoder.encoder_constants import ARENA_Y_START_FRAC, ARENA_Y_END_FRAC

# ---------------------------------------------------------------------------
# Screen dimensions (base resolution)
# ---------------------------------------------------------------------------
_BASE_WIDTH = 540
_BASE_HEIGHT = 960

# ---------------------------------------------------------------------------
# Card slot regions in normalized coordinates
# From ScreenConfig: card_start_x=110, card_width=100, y=[770, 920]
# card_0: x=[110, 210], card_1: x=[210, 310],
# card_2: x=[310, 410], card_3: x=[410, 510]
# ---------------------------------------------------------------------------
_CARD_START_X = 110
_CARD_WIDTH = 100
_CARD_Y_START = 770
_CARD_Y_END = 920

CARD_SLOT_REGIONS: list[tuple[float, float, float, float]] = []
for _i in range(4):
    _x_min = (_CARD_START_X + _i * _CARD_WIDTH) / _BASE_WIDTH
    _x_max = (_CARD_START_X + (_i + 1) * _CARD_WIDTH) / _BASE_WIDTH
    _y_min = _CARD_Y_START / _BASE_HEIGHT
    _y_max = _CARD_Y_END / _BASE_HEIGHT
    CARD_SLOT_REGIONS.append((_x_min, _y_min, _x_max, _y_max))

# ---------------------------------------------------------------------------
# Card bar Y range (normalized) - entire bottom card area
# ---------------------------------------------------------------------------
CARD_BAR_Y_MIN = _CARD_Y_START / _BASE_HEIGHT   # ~0.8021
CARD_BAR_Y_MAX = _CARD_Y_END / _BASE_HEIGHT      # ~0.9583

# ---------------------------------------------------------------------------
# Arena Y range (normalized) - reuse from encoder constants
# ---------------------------------------------------------------------------
ARENA_Y_MIN = ARENA_Y_START_FRAC   # ~0.0521
ARENA_Y_MAX = ARENA_Y_END_FRAC     # ~0.7813

# ---------------------------------------------------------------------------
# Timing constants for PyAutoGUI execution
# ---------------------------------------------------------------------------
CLICK_DELAY_SECONDS = 0.15   # Delay between card click and arena click
PYAUTOGUI_PAUSE = 0.05       # pyautogui.PAUSE setting
