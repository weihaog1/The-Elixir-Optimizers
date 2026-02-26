"""
MatchRecorder - orchestrates ClickLogger + ScreenCapture for BC data.

Creates a session output directory, wires up both recording components,
manages their lifecycle, and writes session metadata on stop.

Output structure:
    recordings/match_YYYYMMDD_HHMMSS/
        screenshots/         JPEG frames from mss
        actions.jsonl        Click logger output (paired card+arena events)
        frames.jsonl         Timestamp-to-filename manifest
        metadata.json        Session info (written on stop)
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Tuple

import pygetwindow as gw

from click_logger import ClickLogger
from screen_capture import ScreenCapture


class MatchRecorder:
    """Records a single Clash Royale match for behavior cloning.

    Args:
        window_title: Exact window title for pygetwindow lookup.
        card_positions: {card_id: (x_norm, y_norm)} for ClickLogger.
        arena_bounds: (x_min, y_min, x_max, y_max) normalized.
        output_root: Parent directory for session folders.
        fps: Screen capture FPS.
        jpeg_quality: JPEG quality for screenshots.
        slot_threshold: ClickLogger card slot radius.
    """

    def __init__(
        self,
        window_title: str,
        card_positions: Dict[int, Tuple[float, float]],
        arena_bounds: Tuple[float, float, float, float],
        output_root: str = "recordings",
        fps: float = 2.0,
        jpeg_quality: int = 85,
        slot_threshold: float = 0.035,
    ):
        self._window_title = window_title
        self._card_positions = card_positions
        self._arena_bounds = arena_bounds
        self._fps = fps
        self._jpeg_quality = jpeg_quality

        # Resolve window (fail-fast if not found)
        self._window = self._find_window()
        print(
            f"[MatchRecorder] Window: {self._window.width}x"
            f"{self._window.height} at "
            f"({self._window.left}, {self._window.top})"
        )

        # Create session directory
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(
            output_root, f"match_{timestamp_str}"
        )
        os.makedirs(self._session_dir, exist_ok=True)

        self._metadata_path = os.path.join(
            self._session_dir, "metadata.json"
        )
        self._start_time = None

        # Construct sub-components
        actions_path = os.path.join(self._session_dir, "actions.jsonl")

        self._click_logger = ClickLogger(
            window_title=window_title,
            card_positions=card_positions,
            arena_bounds=arena_bounds,
            output_path=actions_path,
            slot_threshold=slot_threshold,
        )
        self._screen_capture = ScreenCapture(
            window=self._window,
            output_dir=self._session_dir,
            fps=fps,
            jpeg_quality=jpeg_quality,
        )

    @property
    def session_dir(self) -> str:
        return self._session_dir

    def start(self):
        """Start both ClickLogger and ScreenCapture."""
        self._start_time = time.time()
        self._click_logger.start()
        self._screen_capture.start()
        print(f"[MatchRecorder] Session: {self._session_dir}")

    def stop(self):
        """Stop both components and write metadata."""
        stop_time = time.time()
        self._screen_capture.stop()
        self._click_logger.stop()

        # Count action lines
        actions_path = os.path.join(self._session_dir, "actions.jsonl")
        action_count = 0
        if os.path.exists(actions_path):
            with open(actions_path, "r") as f:
                action_count = sum(1 for _ in f)

        duration = round(stop_time - self._start_time, 2)

        # Write metadata
        metadata = {
            "window_title": self._window_title,
            "window_geometry": {
                "left": self._window.left,
                "top": self._window.top,
                "width": self._window.width,
                "height": self._window.height,
            },
            "fps": self._fps,
            "jpeg_quality": self._jpeg_quality,
            "start_time": self._start_time,
            "stop_time": stop_time,
            "duration_seconds": duration,
            "frame_count": self._screen_capture.frame_count,
            "action_count": action_count,
            "card_positions": {
                str(k): list(v)
                for k, v in self._card_positions.items()
            },
            "arena_bounds": list(self._arena_bounds),
        }
        with open(self._metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"[MatchRecorder] Stopped. "
            f"{self._screen_capture.frame_count} frames, "
            f"{action_count} actions, "
            f"{duration}s duration."
        )

    def _find_window(self):
        """Resolve the game window or raise."""
        windows = gw.getWindowsWithTitle(self._window_title)
        if not windows:
            raise RuntimeError(
                f"Window not found: '{self._window_title}'"
            )
        return windows[0]
