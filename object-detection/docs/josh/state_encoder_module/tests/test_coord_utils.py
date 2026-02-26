"""Tests for pixel_to_cell_float() in coord_utils."""

from src.encoder.coord_utils import pixel_to_cell_float, pixel_to_cell
from src.encoder.encoder_constants import GRID_COLS, GRID_ROWS


# Standard test resolution: 540x960
FW = 540
FH = 960


class TestPixelToCellFloat:
    """Tests for the continuous coordinate conversion function."""

    def test_center_of_screen(self):
        """Center of the screen should produce mid-grid floats."""
        col_f, row_f = pixel_to_cell_float(FW / 2, FH / 2, FW, FH)
        # col_f should be around GRID_COLS / 2 = 9
        assert 8.5 < col_f < 9.5
        # row_f depends on arena fraction mapping, should be in valid range
        assert 0 < row_f < GRID_ROWS

    def test_returns_float_not_int(self):
        """Return values should be floats, not integers."""
        col_f, row_f = pixel_to_cell_float(270.0, 400.0, FW, FH)
        assert isinstance(col_f, float)
        assert isinstance(row_f, float)

    def test_consistent_with_pixel_to_cell(self):
        """Integer truncation of float result should match pixel_to_cell
        for coordinates well inside the grid."""
        # Use a point in the middle of the arena
        cx, cy = 270.0, 400.0
        col_f, row_f = pixel_to_cell_float(cx, cy, FW, FH)
        col_i, row_i = pixel_to_cell(cx, cy, FW, FH)

        # The float-to-int conversion should match (for in-bounds points)
        assert int(col_f) == col_i
        assert int(row_f) == row_i

    def test_left_edge(self):
        """Left edge of screen should produce col_f near 0."""
        col_f, _row_f = pixel_to_cell_float(0.0, 400.0, FW, FH)
        assert col_f == 0.0

    def test_right_edge(self):
        """Right edge of screen should produce col_f near GRID_COLS."""
        col_f, _row_f = pixel_to_cell_float(float(FW), 400.0, FW, FH)
        assert abs(col_f - GRID_COLS) < 0.01
