"""Tests for DatasetBuilder."""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.dataset.dataset_builder import DatasetBuilder, DatasetStats
from src.encoder.encoder_constants import (
    ACTION_SPACE_SIZE,
    GRID_COLS,
    GRID_ROWS,
    NOOP_ACTION,
    NUM_ARENA_CHANNELS,
    NUM_VECTOR_FEATURES,
)
from src.encoder.coord_utils import norm_to_cell, placement_to_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_session(tmp_path, num_frames=5, actions=None, name="match_20260222_120000"):
    """Create a minimal session directory for testing.

    Args:
        tmp_path: pytest tmp_path fixture.
        num_frames: Number of frames to generate.
        actions: List of action dicts. Defaults to empty.
        name: Session directory name.

    Returns:
        Path to the session directory.
    """
    session_dir = tmp_path / name
    screenshots_dir = session_dir / "screenshots"
    screenshots_dir.mkdir(parents=True)

    base_ts = 1740123456.0

    # Write frames.jsonl
    frames = []
    for i in range(num_frames):
        fname = f"frame_{i:06d}.jpg"
        frames.append({
            "frame_idx": i,
            "timestamp": base_ts + i * 0.5,
            "filename": fname,
            "width": 540,
            "height": 960,
        })
        # Create a tiny JPEG file (black 10x10 image)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(screenshots_dir / fname), img)

    with open(session_dir / "frames.jsonl", "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")

    # Write actions.jsonl
    if actions is None:
        actions = []
    with open(session_dir / "actions.jsonl", "w") as f:
        for action in actions:
            f.write(json.dumps(action) + "\n")

    # Write metadata.json
    metadata = {
        "window_title": "Test",
        "start_time": base_ts,
        "stop_time": base_ts + num_frames * 0.5,
        "frame_count": num_frames,
        "action_count": len(actions),
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return str(session_dir)


def _make_mock_state_builder():
    """Create a mock state builder that returns blank GameState."""
    from src.pipeline.game_state import GameState

    mock = MagicMock()
    mock.build_state.side_effect = lambda img, ts, **kw: GameState(
        timestamp=ts,
        frame_width=540,
        frame_height=960,
    )
    return mock


# ---------------------------------------------------------------------------
# Tests: _load_session
# ---------------------------------------------------------------------------

class TestLoadSession:
    """Tests for DatasetBuilder._load_session."""

    def test_load_session_basic(self, tmp_path):
        """Should parse frames.jsonl, actions.jsonl, and metadata.json."""
        session_dir = _create_session(tmp_path, num_frames=3, actions=[
            {"timestamp": 1740123457.0, "card_id": 1, "x_norm": 0.5, "y_norm": 0.5},
        ])
        builder = DatasetBuilder(state_encoder=MagicMock())
        frames, actions, metadata = builder._load_session(session_dir)

        assert len(frames) == 3
        assert len(actions) == 1
        assert actions[0]["card_id"] == 1
        assert metadata["frame_count"] == 3

    def test_load_session_empty_actions(self, tmp_path):
        """Should handle sessions with no actions."""
        session_dir = _create_session(tmp_path, num_frames=2)
        builder = DatasetBuilder(state_encoder=MagicMock())
        frames, actions, metadata = builder._load_session(session_dir)

        assert len(frames) == 2
        assert len(actions) == 0

    def test_load_session_no_metadata(self, tmp_path):
        """Should handle missing metadata.json gracefully."""
        session_dir = _create_session(tmp_path, num_frames=1)
        os.remove(os.path.join(session_dir, "metadata.json"))

        builder = DatasetBuilder(state_encoder=MagicMock())
        frames, actions, metadata = builder._load_session(session_dir)

        assert len(frames) == 1
        assert metadata == {}


# ---------------------------------------------------------------------------
# Tests: _convert_actions
# ---------------------------------------------------------------------------

class TestConvertActions:
    """Tests for DatasetBuilder._convert_actions."""

    def test_convert_known_coordinates(self):
        """Action with known normalized coords should produce correct index."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        actions = [
            {"timestamp": 100.0, "card_id": 0, "x_norm": 0.5, "y_norm": 0.5},
        ]
        events = builder._convert_actions(actions)

        assert len(events) == 1
        ts, action_idx = events[0]
        assert ts == 100.0
        assert 0 <= action_idx < NOOP_ACTION

        # Verify roundtrip
        col, row = norm_to_cell(0.5, 0.5)
        expected = placement_to_action(0, col, row)
        assert action_idx == expected

    def test_convert_multiple_actions(self):
        """Multiple actions should each produce correct indices."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        actions = [
            {"timestamp": 100.0, "card_id": 0, "x_norm": 0.1, "y_norm": 0.2},
            {"timestamp": 101.0, "card_id": 3, "x_norm": 0.9, "y_norm": 0.8},
        ]
        events = builder._convert_actions(actions)

        assert len(events) == 2
        # Card 0 and card 3 should produce different action ranges
        _, idx0 = events[0]
        _, idx3 = events[1]
        assert idx0 != idx3

    def test_convert_empty_actions(self):
        """Empty action list should produce empty events."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        events = builder._convert_actions([])
        assert events == []


# ---------------------------------------------------------------------------
# Tests: _assign_actions_to_frames
# ---------------------------------------------------------------------------

class TestAssignActionsToFrames:
    """Tests for DatasetBuilder._assign_actions_to_frames."""

    def test_action_assigned_to_nearest_frame(self):
        """Action should map to the frame with the closest timestamp."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        frame_timestamps = [10.0, 10.5, 11.0, 11.5, 12.0]
        action_events = [(10.6, 42)]

        labels = builder._assign_actions_to_frames(action_events, frame_timestamps)

        assert labels[1] == 42  # 10.5 is closest to 10.6
        assert labels[0] == NOOP_ACTION
        assert labels[2] == NOOP_ACTION

    def test_unmatched_frames_get_noop(self):
        """Frames without a matching action should get NOOP_ACTION."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        frame_timestamps = [1.0, 2.0, 3.0]
        action_events = []

        labels = builder._assign_actions_to_frames(action_events, frame_timestamps)

        assert all(l == NOOP_ACTION for l in labels)

    def test_two_actions_close_together(self):
        """Two actions near different frames should both be assigned."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        frame_timestamps = [10.0, 10.5, 11.0, 11.5]
        action_events = [(10.1, 100), (11.4, 200)]

        labels = builder._assign_actions_to_frames(action_events, frame_timestamps)

        assert labels[0] == 100  # 10.0 closest to 10.1
        assert labels[3] == 200  # 11.5 closest to 11.4
        assert labels[1] == NOOP_ACTION
        assert labels[2] == NOOP_ACTION

    def test_later_action_wins_same_frame(self):
        """When two actions map to the same frame, the later one should win."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        frame_timestamps = [10.0, 11.0]
        # Both actions are closest to frame at 10.0
        action_events = [(10.1, 100), (10.2, 200)]

        labels = builder._assign_actions_to_frames(action_events, frame_timestamps)

        # Later action (200) overwrites earlier (100)
        assert labels[0] == 200

    def test_empty_frames(self):
        """Empty frame list should produce empty labels."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        labels = builder._assign_actions_to_frames([(1.0, 42)], [])
        assert labels == []


# ---------------------------------------------------------------------------
# Tests: _downsample_noops
# ---------------------------------------------------------------------------

class TestDownsampleNoops:
    """Tests for DatasetBuilder._downsample_noops."""

    def test_all_action_frames_kept(self):
        """All frames with real actions should be kept regardless of ratio."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        action_labels = [42, NOOP_ACTION, NOOP_ACTION, 99, NOOP_ACTION]
        indices = list(range(5))

        kept = builder._downsample_noops(indices, action_labels, keep_ratio=0.0)

        # Even with 0 ratio, we keep at least 1 noop (min(1, ...))
        # but all action frames must be present
        assert 0 in kept  # action frame
        assert 3 in kept  # action frame

    def test_some_noops_kept(self):
        """A fraction of no-op frames should be kept."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        # 2 action frames, 8 noop frames
        action_labels = [42, NOOP_ACTION, NOOP_ACTION, NOOP_ACTION,
                         NOOP_ACTION, 99, NOOP_ACTION, NOOP_ACTION,
                         NOOP_ACTION, NOOP_ACTION]
        indices = list(range(10))

        np.random.seed(42)
        kept = builder._downsample_noops(indices, action_labels, keep_ratio=0.5)

        # Action frames (0, 5) must be present
        assert 0 in kept
        assert 5 in kept
        # Approximately 4 noops kept (50% of 8), plus the 2 action frames
        noop_kept = len(kept) - 2
        assert 1 <= noop_kept <= 8

    def test_result_is_sorted(self):
        """Kept indices should be in sorted order."""
        builder = DatasetBuilder(state_encoder=MagicMock())
        action_labels = [NOOP_ACTION] * 20 + [42]
        indices = list(range(21))

        kept = builder._downsample_noops(indices, action_labels, keep_ratio=0.3)

        assert kept == sorted(kept)


# ---------------------------------------------------------------------------
# Tests: build_dataset (end-to-end with mocks)
# ---------------------------------------------------------------------------

class TestBuildDataset:
    """End-to-end tests for DatasetBuilder.build_dataset."""

    def test_build_dataset_produces_npz(self, tmp_path):
        """build_dataset should produce a valid .npz file."""
        base_ts = 1740123456.0
        actions = [
            {"timestamp": base_ts + 1.1, "card_id": 0, "x_norm": 0.5, "y_norm": 0.5},
        ]
        session_dir = _create_session(tmp_path, num_frames=5, actions=actions)
        output_dir = str(tmp_path / "output")

        mock_sb = _make_mock_state_builder()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = {
            "arena": np.zeros((GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS), dtype=np.float32),
            "vector": np.zeros(NUM_VECTOR_FEATURES, dtype=np.float32),
        }
        mock_encoder.action_mask.return_value = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)

        builder = DatasetBuilder(
            enhanced_state_builder=mock_sb,
            state_encoder=mock_encoder,
        )
        stats = builder.build_dataset(session_dir, output_dir, noop_keep_ratio=0.5)

        assert isinstance(stats, DatasetStats)
        assert stats.total_frames == 5
        assert stats.total_actions == 1
        assert stats.action_frames == 1
        assert stats.noop_frames == 4
        assert os.path.exists(stats.output_path)

        # Verify .npz contents
        data = np.load(stats.output_path)
        assert "obs_arena" in data
        assert "obs_vector" in data
        assert "actions" in data
        assert "masks" in data
        assert "timestamps" in data
        assert data["actions"].dtype == np.int64
        assert data["timestamps"].dtype == np.float64

    def test_build_dataset_stats_consistency(self, tmp_path):
        """Stats fields should be internally consistent."""
        base_ts = 1740123456.0
        actions = [
            {"timestamp": base_ts + 0.3, "card_id": 1, "x_norm": 0.3, "y_norm": 0.4},
            {"timestamp": base_ts + 1.8, "card_id": 2, "x_norm": 0.7, "y_norm": 0.6},
        ]
        session_dir = _create_session(tmp_path, num_frames=4, actions=actions)
        output_dir = str(tmp_path / "output")

        mock_sb = _make_mock_state_builder()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = {
            "arena": np.zeros((GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS), dtype=np.float32),
            "vector": np.zeros(NUM_VECTOR_FEATURES, dtype=np.float32),
        }
        mock_encoder.action_mask.return_value = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)

        builder = DatasetBuilder(
            enhanced_state_builder=mock_sb,
            state_encoder=mock_encoder,
        )
        stats = builder.build_dataset(session_dir, output_dir, noop_keep_ratio=1.0)

        assert stats.noop_frames + stats.action_frames == stats.total_frames
        assert stats.total_actions == 2
        # With keep_ratio=1.0, all frames should be kept
        assert stats.kept_after_downsample == stats.total_frames


# ---------------------------------------------------------------------------
# Tests: build_from_multiple
# ---------------------------------------------------------------------------

class TestBuildFromMultiple:
    """Tests for DatasetBuilder.build_from_multiple."""

    def test_build_from_two_sessions(self, tmp_path):
        """Should process multiple sessions and return stats for each."""
        base_ts = 1740123456.0

        # Session 1
        session1 = _create_session(
            tmp_path, num_frames=3,
            actions=[
                {"timestamp": base_ts + 0.3, "card_id": 0, "x_norm": 0.5, "y_norm": 0.5},
            ],
            name="match_session_1",
        )
        # Session 2
        session2 = _create_session(
            tmp_path, num_frames=4,
            actions=[
                {"timestamp": base_ts + 0.8, "card_id": 1, "x_norm": 0.3, "y_norm": 0.4},
                {"timestamp": base_ts + 1.5, "card_id": 2, "x_norm": 0.7, "y_norm": 0.6},
            ],
            name="match_session_2",
        )
        output_dir = str(tmp_path / "output")

        mock_sb = _make_mock_state_builder()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = {
            "arena": np.zeros((GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS), dtype=np.float32),
            "vector": np.zeros(NUM_VECTOR_FEATURES, dtype=np.float32),
        }
        mock_encoder.action_mask.return_value = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)

        builder = DatasetBuilder(
            enhanced_state_builder=mock_sb,
            state_encoder=mock_encoder,
        )
        all_stats = builder.build_from_multiple(
            [session1, session2], output_dir, noop_keep_ratio=1.0
        )

        assert len(all_stats) == 2
        assert all_stats[0].total_frames == 3
        assert all_stats[0].total_actions == 1
        assert all_stats[1].total_frames == 4
        assert all_stats[1].total_actions == 2

        # Verify two separate .npz files were created
        npz_files = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
        assert len(npz_files) == 2
