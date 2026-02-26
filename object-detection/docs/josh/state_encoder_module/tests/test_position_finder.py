"""Tests for PositionFinder -- grid cell collision resolution."""

import pytest

from src.encoder.position_finder import PositionFinder


class TestNaturalPlacement:
    """Units placed in unoccupied cells get their natural position."""

    def test_first_unit_gets_natural_cell(self):
        pf = PositionFinder(rows=32, cols=18)
        col, row = pf.find_position(9.5, 16.5)
        assert (col, row) == (9, 16)

    def test_boundary_top_left(self):
        pf = PositionFinder(rows=32, cols=18)
        col, row = pf.find_position(0.0, 0.0)
        assert (col, row) == (0, 0)

    def test_boundary_bottom_right(self):
        pf = PositionFinder(rows=32, cols=18)
        col, row = pf.find_position(17.9, 31.9)
        assert (col, row) == (17, 31)

    def test_negative_coords_clamped(self):
        pf = PositionFinder(rows=32, cols=18)
        col, row = pf.find_position(-5.0, -10.0)
        assert (col, row) == (0, 0)

    def test_overflow_coords_clamped(self):
        pf = PositionFinder(rows=32, cols=18)
        col, row = pf.find_position(100.0, 200.0)
        assert (col, row) == (17, 31)


class TestCollisionDisplacement:
    """When the natural cell is occupied, the unit is displaced nearby."""

    def test_second_unit_same_cell_displaced(self):
        pf = PositionFinder(rows=32, cols=18)
        c1, r1 = pf.find_position(9.5, 16.5)
        c2, r2 = pf.find_position(9.5, 16.5)
        assert (c1, r1) != (c2, r2), "Second unit must not occupy same cell"

    def test_displaced_unit_is_adjacent(self):
        pf = PositionFinder(rows=32, cols=18)
        c1, r1 = pf.find_position(9.5, 16.5)
        c2, r2 = pf.find_position(9.5, 16.5)
        # Displaced unit should be within 1 cell of natural position
        assert abs(c2 - c1) <= 1 and abs(r2 - r1) <= 1, (
            f"Displaced cell ({c2},{r2}) too far from natural ({c1},{r1})"
        )

    def test_three_units_same_cell_all_unique(self):
        pf = PositionFinder(rows=32, cols=18)
        cells = set()
        for _ in range(3):
            c, r = pf.find_position(9.5, 16.5)
            cells.add((c, r))
        assert len(cells) == 3, "All three units must have unique cells"


class TestSwarm:
    """Stress test with many units targeting the same area."""

    def test_swarm_fifteen_units_all_unique(self):
        pf = PositionFinder(rows=32, cols=18)
        cells = set()
        for _ in range(15):
            c, r = pf.find_position(9.5, 16.5)
            cells.add((c, r))
        assert len(cells) == 15, "All 15 units must have unique cells"


class TestInstanceIndependence:
    """Each PositionFinder instance has its own occupancy grid."""

    def test_separate_instances_independent(self):
        pf1 = PositionFinder(rows=32, cols=18)
        pf2 = PositionFinder(rows=32, cols=18)

        c1, r1 = pf1.find_position(5.0, 10.0)
        c2, r2 = pf2.find_position(5.0, 10.0)
        # Both should get the same natural cell since they are independent
        assert (c1, r1) == (c2, r2)

    def test_used_count_tracks_placements(self):
        pf = PositionFinder(rows=32, cols=18)
        assert pf.used_count == 0
        pf.find_position(5.0, 10.0)
        assert pf.used_count == 1
        pf.find_position(5.0, 10.0)
        assert pf.used_count == 2
        pf.find_position(12.0, 20.0)
        assert pf.used_count == 3
