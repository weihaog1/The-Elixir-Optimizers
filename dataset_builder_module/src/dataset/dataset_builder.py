"""DatasetBuilder - converts click_logger sessions into training-ready .npz files.

Processes session directories produced by MatchRecorder (screenshots + actions)
through the perception and encoding pipeline to produce (obs, action, mask)
tuples suitable for behavior cloning with SB3 MaskableMultiInputPolicy.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.encoder.coord_utils import norm_to_cell, placement_to_action
from src.encoder.encoder_constants import NOOP_ACTION


@dataclass
class DatasetStats:
    """Summary statistics for a processed session."""

    total_frames: int
    total_actions: int
    noop_frames: int
    action_frames: int
    kept_after_downsample: int
    session_dir: str
    output_path: str


class DatasetBuilder:
    """Converts click_logger session directories into .npz training files.

    Pipeline per session:
      1. Load frames.jsonl, actions.jsonl, metadata.json
      2. Convert pre-paired actions to Discrete(2305) action indices
      3. Assign actions to frames by nearest timestamp
      4. Run each frame through perception (EnhancedStateBuilder) + encoding
      5. Downsample no-op frames to balance the dataset
      6. Save as .npz

    Args:
        enhanced_state_builder: EnhancedStateBuilder (or any object with
            build_state(image, timestamp) -> GameState). If None, frames
            will produce zero-filled observations.
        state_encoder: StateEncoder instance. Created with defaults if None.
    """

    def __init__(self, enhanced_state_builder=None, state_encoder=None):
        self.state_builder = enhanced_state_builder
        if state_encoder is None:
            from src.encoder.state_encoder import StateEncoder
            state_encoder = StateEncoder()
        self.state_encoder = state_encoder

    def build_dataset(
        self,
        session_dir: str,
        output_dir: str,
        noop_keep_ratio: float = 0.15,
    ) -> DatasetStats:
        """Process one session directory into a .npz file.

        Args:
            session_dir: Path to a click_logger session (contains
                screenshots/, frames.jsonl, actions.jsonl).
            output_dir: Directory where the .npz file will be saved.
            noop_keep_ratio: Fraction of no-op frames to keep (0-1).

        Returns:
            DatasetStats summarizing the processing result.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load session data
        frames, actions, metadata = self._load_session(session_dir)

        # 2. Convert pre-paired actions to action indices
        action_events = self._convert_actions(actions)

        # 3. Assign actions to frames by nearest timestamp
        frame_timestamps = [f["timestamp"] for f in frames]
        action_labels = self._assign_actions_to_frames(
            action_events, frame_timestamps
        )

        # 4. Process each frame through perception + encoding
        obs_arenas = []
        obs_vectors = []
        masks = []
        for frame_info in frames:
            img_path = os.path.join(
                session_dir, "screenshots", frame_info["filename"]
            )
            obs, mask = self._process_frame(img_path, frame_info["timestamp"])
            obs_arenas.append(obs["arena"])
            obs_vectors.append(obs["vector"])
            masks.append(mask)

        # 5. Downsample no-ops
        all_indices = list(range(len(frames)))
        noop_count = sum(1 for a in action_labels if a == NOOP_ACTION)
        action_count = len(action_labels) - noop_count
        kept_indices = self._downsample_noops(
            all_indices, action_labels, noop_keep_ratio
        )

        # 6. Save as .npz
        session_name = os.path.basename(session_dir.rstrip("/"))
        output_path = os.path.join(output_dir, f"{session_name}.npz")

        np.savez(
            output_path,
            obs_arena=np.array([obs_arenas[i] for i in kept_indices]),
            obs_vector=np.array([obs_vectors[i] for i in kept_indices]),
            actions=np.array(
                [action_labels[i] for i in kept_indices], dtype=np.int64
            ),
            masks=np.array([masks[i] for i in kept_indices]),
            timestamps=np.array(
                [frame_timestamps[i] for i in kept_indices], dtype=np.float64
            ),
        )

        return DatasetStats(
            total_frames=len(frames),
            total_actions=len(actions),
            noop_frames=noop_count,
            action_frames=action_count,
            kept_after_downsample=len(kept_indices),
            session_dir=session_dir,
            output_path=output_path,
        )

    def build_from_multiple(
        self,
        session_dirs: List[str],
        output_dir: str,
        noop_keep_ratio: float = 0.15,
    ) -> List[DatasetStats]:
        """Process multiple session directories.

        Args:
            session_dirs: List of session directory paths.
            output_dir: Directory where .npz files will be saved.
            noop_keep_ratio: Fraction of no-op frames to keep.

        Returns:
            List of DatasetStats, one per session.
        """
        stats = []
        for session_dir in session_dirs:
            s = self.build_dataset(session_dir, output_dir, noop_keep_ratio)
            stats.append(s)
        return stats

    def _load_session(
        self, session_dir: str
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """Parse frames.jsonl, actions.jsonl, and metadata.json.

        Args:
            session_dir: Path to a click_logger session directory.

        Returns:
            Tuple of (frames_list, actions_list, metadata_dict).
        """
        frames_path = os.path.join(session_dir, "frames.jsonl")
        actions_path = os.path.join(session_dir, "actions.jsonl")
        metadata_path = os.path.join(session_dir, "metadata.json")

        frames = []
        with open(frames_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))

        actions = []
        if os.path.exists(actions_path):
            with open(actions_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        actions.append(json.loads(line))

        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        return frames, actions, metadata

    def _convert_actions(
        self, actions: List[Dict]
    ) -> List[Tuple[float, int]]:
        """Convert pre-paired click_logger actions to action indices.

        Each action has {timestamp, card_id, x_norm, y_norm}. We convert
        the normalized coordinates to a grid cell and encode as a
        Discrete(2305) action index.

        Args:
            actions: List of action dicts from actions.jsonl.

        Returns:
            List of (timestamp, action_index) tuples.
        """
        events = []
        for action in actions:
            col, row = norm_to_cell(action["x_norm"], action["y_norm"])
            action_idx = placement_to_action(action["card_id"], col, row)
            events.append((action["timestamp"], action_idx))
        return events

    def _assign_actions_to_frames(
        self,
        action_events: List[Tuple[float, int]],
        frame_timestamps: List[float],
    ) -> List[int]:
        """Assign each action to the nearest frame by timestamp.

        If multiple actions map to the same frame, the later action wins.
        Frames without a matching action get NOOP_ACTION.

        Args:
            action_events: List of (timestamp, action_index) tuples.
            frame_timestamps: List of frame timestamps.

        Returns:
            List of action indices, one per frame.
        """
        labels = [NOOP_ACTION] * len(frame_timestamps)

        if not frame_timestamps or not action_events:
            return labels

        for action_ts, action_idx in action_events:
            best_frame = 0
            best_diff = abs(frame_timestamps[0] - action_ts)
            for i, ft in enumerate(frame_timestamps):
                diff = abs(ft - action_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_frame = i
            labels[best_frame] = action_idx

        return labels

    def _process_frame(
        self,
        image_path: str,
        timestamp: float,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load image, run state builder and encoder.

        Args:
            image_path: Path to the screenshot JPEG.
            timestamp: Frame timestamp.

        Returns:
            Tuple of (obs_dict, action_mask).
        """
        from src.pipeline.game_state import GameState

        if self.state_builder is not None and os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                state = self.state_builder.build_state(image, timestamp)
            else:
                state = GameState(timestamp=timestamp)
        else:
            state = GameState(timestamp=timestamp)

        obs = self.state_encoder.encode(state)
        mask = self.state_encoder.action_mask(state)
        return obs, mask

    def _downsample_noops(
        self,
        indices: List[int],
        action_labels: List[int],
        keep_ratio: float,
    ) -> List[int]:
        """Downsample no-op frames while keeping all action frames.

        Args:
            indices: List of frame indices to consider.
            action_labels: Action label per frame.
            keep_ratio: Fraction of no-op frames to keep (0-1).

        Returns:
            Filtered list of frame indices.
        """
        action_indices = [
            i for i in indices if action_labels[i] != NOOP_ACTION
        ]
        noop_indices = [
            i for i in indices if action_labels[i] == NOOP_ACTION
        ]

        num_keep = max(1, int(len(noop_indices) * keep_ratio))
        num_keep = min(num_keep, len(noop_indices))
        kept_noops = sorted(random.sample(noop_indices, num_keep))

        result = sorted(action_indices + kept_noops)
        return result
