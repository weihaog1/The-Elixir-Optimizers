"""
ScreenCapture - threaded mss screen grabber for BC data recording.

Captures the game window at a configurable FPS and saves timestamped
JPEG screenshots. Runs in a daemon thread to avoid blocking the main
thread or the click logger.
"""

import json
import os
import threading
import time
from typing import Optional

import mss
from PIL import Image


class ScreenCapture:
    """Captures the game window region at a fixed FPS using mss.

    Each frame is saved as a JPEG file in the output directory,
    and a manifest entry (timestamp, filename) is appended to
    a JSONL file for later merging with click logs.

    Args:
        window: pygetwindow window object. Geometry is read each
            frame to handle window movement.
        output_dir: Path to the session directory.
        fps: Target capture rate in frames per second.
        jpeg_quality: JPEG compression quality (1-100).
    """

    def __init__(
        self,
        window,
        output_dir: str,
        fps: float = 2.0,
        jpeg_quality: int = 85,
    ):
        self._window = window
        self._output_dir = output_dir
        self._screenshot_dir = os.path.join(output_dir, "screenshots")
        self._manifest_path = os.path.join(output_dir, "frames.jsonl")
        self._fps = fps
        self._interval = 1.0 / fps
        self._jpeg_quality = jpeg_quality

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0

        os.makedirs(self._screenshot_dir, exist_ok=True)
        self._manifest_file = open(self._manifest_path, "a")

    def start(self):
        """Start the capture thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()
        print(f"[ScreenCapture] Started at {self._fps} FPS")

    def stop(self):
        """Stop the capture thread and flush the manifest."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._manifest_file.flush()
        self._manifest_file.close()
        print(
            f"[ScreenCapture] Stopped. "
            f"{self._frame_count} frames captured."
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _get_region(self) -> dict:
        """Read current window geometry as an mss monitor dict."""
        return {
            "left": self._window.left,
            "top": self._window.top,
            "width": self._window.width,
            "height": self._window.height,
        }

    def _capture_loop(self):
        """Main capture loop running in a daemon thread.

        Creates its own mss context (mss is not thread-safe across
        contexts). Re-reads window geometry each frame to track
        window movement. Converts BGRA to RGB via Pillow and saves
        as JPEG.
        """
        with mss.mss() as sct:
            while self._running:
                t_start = time.time()

                region = self._get_region()
                screenshot = sct.grab(region)

                filename = f"frame_{self._frame_count:06d}.jpg"
                filepath = os.path.join(self._screenshot_dir, filename)

                # mss returns BGRA pixels. Convert to RGB via Pillow.
                img = Image.frombytes(
                    "RGB",
                    screenshot.size,
                    screenshot.bgra,
                    "raw",
                    "BGRX",
                )
                img.save(filepath, "JPEG", quality=self._jpeg_quality)

                # Write manifest entry
                entry = {
                    "frame_idx": self._frame_count,
                    "timestamp": t_start,
                    "filename": filename,
                    "width": region["width"],
                    "height": region["height"],
                }
                self._manifest_file.write(json.dumps(entry) + "\n")
                self._manifest_file.flush()

                self._frame_count += 1

                # Sleep to maintain target FPS
                elapsed = time.time() - t_start
                sleep_time = self._interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
