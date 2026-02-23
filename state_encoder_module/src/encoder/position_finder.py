"""
PositionFinder - resolves grid cell collisions for per-cell unit encoding.

When multiple units map to the same 18x32 arena grid cell, this class
displaces later units to the nearest unoccupied cell using Euclidean
distance (via scipy.spatial.distance.cdist).

Create a fresh instance per frame to reset the occupancy grid.
"""

import numpy as np
from scipy.spatial.distance import cdist


class PositionFinder:
    """Assigns unique grid cells to units, resolving collisions.

    Args:
        rows: Number of rows in the arena grid (default 32).
        cols: Number of columns in the arena grid (default 18).
    """

    def __init__(self, rows: int = 32, cols: int = 18):
        self._rows = rows
        self._cols = cols
        self._used = np.zeros((rows, cols), dtype=bool)
        self._used_count = 0

        # Pre-compute all cell center coordinates for distance lookups.
        # Shape: (rows * cols, 2) with each row as (col_center, row_center).
        coords = []
        for r in range(rows):
            for c in range(cols):
                coords.append((c + 0.5, r + 0.5))
        self._all_coords = np.array(coords, dtype=np.float64)

    @property
    def used_count(self) -> int:
        """Number of cells that have been marked as used."""
        return self._used_count

    def find_position(self, col_f: float, row_f: float) -> tuple[int, int]:
        """Find the best available grid cell for a unit.

        Clamps the continuous position to grid bounds, then either uses
        the natural cell (if free) or displaces to the nearest free cell
        by Euclidean distance.

        Args:
            col_f: Continuous column position (may be outside bounds).
            row_f: Continuous row position (may be outside bounds).

        Returns:
            (col, row) tuple of the assigned grid cell.

        Raises:
            RuntimeError: If all grid cells are occupied.
        """
        # Clamp to valid grid range
        col_clamped = max(0.0, min(col_f, self._cols - 1e-9))
        row_clamped = max(0.0, min(row_f, self._rows - 1e-9))

        nat_col = int(col_clamped)
        nat_row = int(row_clamped)

        # Clamp integer indices to valid bounds
        nat_col = max(0, min(nat_col, self._cols - 1))
        nat_row = max(0, min(nat_row, self._rows - 1))

        if not self._used[nat_row, nat_col]:
            self._used[nat_row, nat_col] = True
            self._used_count += 1
            return nat_col, nat_row

        # Natural cell is occupied -- find nearest free cell
        if self._used_count >= self._rows * self._cols:
            raise RuntimeError("All grid cells are occupied")

        # Build query point (continuous coordinates, same space as _all_coords)
        query = np.array([[col_clamped, row_clamped]])

        # Compute distances to all cells
        dists = cdist(query, self._all_coords, metric="euclidean")[0]

        # Mask out occupied cells with infinity
        dists[self._used.ravel()] = np.inf

        # Pick closest free cell
        best_idx = int(np.argmin(dists))
        best_row = best_idx // self._cols
        best_col = best_idx % self._cols

        self._used[best_row, best_col] = True
        self._used_count += 1
        return best_col, best_row
