import json
import time
import threading
from typing import Dict, Tuple, Optional

import pygetwindow as gw
from pynput import mouse

class ClickLogger:
    def __init__(
            self,
            window_title: str,
            card_positions: Dict[int, Tuple[float, float]],
            arena_bounds: Tuple[float, float, float, float],
            output_path: str = "click_log.jsonl",
            slot_threshold: float = 0.035
    ):
        """
        window_title: exact window title string
        card_positions: {card_id: (x_norm, y_norm)}
        arena_bounds: (x_min, y_min, x_max, y_max) normalized
        """
        self.window_title = window_title
        self.card_positions = card_positions
        self.arena_bounds = arena_bounds
        self.output_path = output_path
        self.slot_threshold = slot_threshold

        self._selected_slot: Optional[int] = None
        self._listener:Optional[mouse.Listener] = None
        self._window = None
        self._running = False

        self._lock = threading.Lock()

        self._file = open(self.output_path, "a")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def start(self):
        self._window = self._get_window()
        self._running = True

        self._listener = mouse.Listener(on_click=self._on_click)
        self._listener.start()

        print(f"Started [ClickLogger] for window '{self.window_title}'")
    
    def stop(self):
        self._running = False

        if self._listener:
            self._listener.stop()
        
        self._file.flush()
        self._file.close()

        print("[ClickLogger] Stopped.")

    # --------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------

    def _get_window(self):
        windows = gw.getWindowsWithTitle(self.window_title)
        if not windows:
            raise Exception(f"Window '{self.window_title}' not found")
        return windows[0]
    
    def _pixel_to_norm(self, x_pixel, y_pixel):
        wx, wy = self._window.left, self._window.top
        w, h = self._window.width, self._window.height

        x_norm = (x_pixel - wx) / w
        y_norm = (y_pixel - wy) / h

        return x_norm, y_norm
    
    def _in_window(self, x_norm, y_norm):
        return 0 <= x_norm <= 1 and 0 <= y_norm <= 1
    
    def _identify_slot(self, x_norm, y_norm):
        for card_id, (sx, sy) in self.card_positions.items():
            if abs(x_norm - sx) < self.slot_threshold and \
               abs(y_norm - sy) < self.slot_threshold:
                return card_id
        return None
    
    def _in_arena(self, x_norm, y_norm):
        x_min, y_min, x_max, y_max = self.arena_bounds
        return x_min <= x_norm <= x_max and \
               y_min <= y_norm <= y_max
    
    # --------------------------------------------------
    # Mouse Callback
    # --------------------------------------------------
    def _on_click(self, x, y, button, pressed):
        if not self._running:
            return

        if button != mouse.Button.left:
            return

        x_norm, y_norm = self._pixel_to_norm(x, y)

        if not self._in_window(x_norm, y_norm):
            return

        # -------------------------
        # MOUSE DOWN
        # -------------------------
        if pressed:
            slot = self._identify_slot(x_norm, y_norm)
            if slot is not None:
                self._selected_slot = slot
            return

        # -------------------------
        # MOUSE RELEASE
        # -------------------------
        if not pressed:
            if self._selected_slot is not None and \
            self._in_arena(x_norm, y_norm):

                action = {
                    "timestamp": time.time(),
                    "card_id": self._selected_slot,
                    "x_norm": float(x_norm),
                    "y_norm": float(y_norm),
                }

                self._file.write(json.dumps(action) + "\n")
                self._file.flush()

                print(f"[ClickLogger] Logged: {action}")

                self._selected_slot = None


