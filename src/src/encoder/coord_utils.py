"""
Coordinate conversion utilities for the StateEncoder and action system.

Converts between three coordinate systems:
  1. Pixel coords -- raw bounding box centers from YOLO (resolution-dependent)
  2. Normalized coords -- 0-1 fractions of screen width/height (resolution-independent)
  3. Grid cells -- (col, row) in the 18x32 arena grid

Also provides action encoding/decoding for the Discrete(2305) action space:
  action = card_id * 576 + row * 18 + col   (0..2303 for card placements)
  action = 2304                               (no-op)
"""

from .encoder_constants import (
    GRID_COLS,
    GRID_ROWS,
    GRID_CELLS,
    NOOP_ACTION,
    ARENA_Y_START_FRAC,
    ARENA_Y_END_FRAC,
)

_ARENA_Y_SPAN = ARENA_Y_END_FRAC - ARENA_Y_START_FRAC


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


def pixel_to_cell(
    cx: float, cy: float, frame_w: int, frame_h: int
) -> tuple[int, int]:
    """Convert pixel bbox center to grid cell (col, row).

    Accounts for the arena not covering the full screen -- timer bar at
    top, card bar at bottom. Pixels outside the arena are clamped to the
    nearest edge cell.

    Args:
        cx: Center x in pixels.
        cy: Center y in pixels.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        (col, row) clamped to [0, GRID_COLS-1] x [0, GRID_ROWS-1].
    """
    x_norm = cx / frame_w
    y_norm = cy / frame_h
    return norm_to_cell(x_norm, y_norm)


def norm_to_cell(x_norm: float, y_norm: float) -> tuple[int, int]:
    """Convert normalized screen coords (0-1) to grid cell.

    Used by both StateEncoder (unit placement) and DatasetBuilder
    (encoding click logger actions to grid cells).

    Args:
        x_norm: Normalized x (0 = left edge, 1 = right edge).
        y_norm: Normalized y (0 = top edge, 1 = bottom edge).

    Returns:
        (col, row) clamped to valid grid range.
    """
    arena_y_frac = (y_norm - ARENA_Y_START_FRAC) / _ARENA_Y_SPAN

    col = _clamp(int(x_norm * GRID_COLS), 0, GRID_COLS - 1)
    row = _clamp(int(arena_y_frac * GRID_ROWS), 0, GRID_ROWS - 1)
    return col, row


def cell_to_norm(col: int, row: int) -> tuple[float, float]:
    """Convert grid cell to normalized screen coords (center of cell).

    Used by the action executor to convert the agent's chosen grid cell
    back into a click position on screen.

    Args:
        col: Column index (0..17).
        row: Row index (0..31).

    Returns:
        (x_norm, y_norm) in [0, 1] range.
    """
    x_norm = (col + 0.5) / GRID_COLS
    y_norm_arena = (row + 0.5) / GRID_ROWS
    y_norm = ARENA_Y_START_FRAC + y_norm_arena * _ARENA_Y_SPAN
    return x_norm, y_norm


def action_to_placement(action_idx: int) -> tuple[int, int, int] | None:
    """Decode a Discrete(2305) action index.

    Args:
        action_idx: Integer in [0, 2304].

    Returns:
        (card_id, col, row) for card placements, or None for no-op.
    """
    if action_idx == NOOP_ACTION:
        return None
    card_id = action_idx // GRID_CELLS
    cell = action_idx % GRID_CELLS
    row = cell // GRID_COLS
    col = cell % GRID_COLS
    return card_id, col, row


def placement_to_action(card_id: int, col: int, row: int) -> int:
    """Encode a card placement to a Discrete(2305) action index.

    Args:
        card_id: Card slot index (0..3).
        col: Grid column (0..17).
        row: Grid row (0..31).

    Returns:
        Action index in [0, 2303].
    """
    return card_id * GRID_CELLS + row * GRID_COLS + col
