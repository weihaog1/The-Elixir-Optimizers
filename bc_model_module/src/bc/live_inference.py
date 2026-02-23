"""Live game inference engine for the BC model.

Captures the Clash Royale game screen, runs the BC policy to predict
card placement actions, and executes them via PyAutoGUI.

Architecture:
    GameCapture (mss) -> PerceptionAdapter (YOLO/fallback) ->
    BCPolicy (predict) -> ActionDispatcher (PyAutoGUI with window offset)

Perception Tiers:
    Tier 2: CRDetector + CardPredictor (real class IDs, card classification)
            - Available when ultralytics + model files present
    Tier 3: Zero-filled observations (empty arena, mid-game defaults)
            - Always works, model relies on learned biases only

Usage:
    from src.bc.live_inference import LiveConfig, LiveInferenceEngine

    config = LiveConfig(
        model_path="models/bc/best_bc.pt",
        capture_region=(0, 0, 540, 960),
        dry_run=True,
    )
    engine = LiveInferenceEngine(config)
    engine.run()
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

try:
    import mss
except ImportError:
    mss = None  # type: ignore[assignment]

import torch

# Enable DPI awareness on Windows so window coordinates match physical pixels
# (must be called before any window/coordinate APIs)
if sys.platform == "win32":
    try:
        import ctypes as _ctypes
        _ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LiveConfig:
    """Configuration for the live inference engine."""

    # --- Model ---
    model_path: str = ""
    device: str = "cpu"

    # --- Capture ---
    window_title: str = ""
    capture_region: Optional[tuple[int, int, int, int]] = None
    capture_fps: float = 2.0
    frame_w: int = 540
    frame_h: int = 960

    # --- Window offset (for PyAutoGUI absolute coords) ---
    window_left: int = 0
    window_top: int = 0

    # --- Perception ---
    use_perception: bool = True
    detector_model_paths: list[str] = field(default_factory=lambda: [
        os.path.join("models", "best_yolov8s_50epochs_fixed_pregen_set.pt"),
    ])
    card_classifier_path: str = os.path.join("models", "card_classifier.pt")

    # --- Policy ---
    confidence_threshold: float = 0.0
    action_cooldown: float = 0.5
    max_actions_per_minute: int = 20

    # --- Safety ---
    dry_run: bool = False

    # --- Logging ---
    log_dir: str = os.path.join("logs", "live")
    verbose: bool = True


# ---------------------------------------------------------------------------
# Constants (inlined from encoder_constants / action_constants to avoid
# fragile import chains through src/__init__.py)
# ---------------------------------------------------------------------------

_GRID_ROWS = 32
_GRID_COLS = 18
_GRID_CELLS = _GRID_ROWS * _GRID_COLS  # 576
_NUM_ARENA_CHANNELS = 6
_NUM_VECTOR_FEATURES = 23
_ACTION_SPACE_SIZE = 4 * _GRID_CELLS + 1  # 2305
_NOOP_ACTION = _ACTION_SPACE_SIZE - 1  # 2304

_ARENA_Y_START_FRAC = 50.0 / 960.0  # ~0.0521
_ARENA_Y_END_FRAC = 750.0 / 960.0  # ~0.7813
_ARENA_Y_SPAN = _ARENA_Y_END_FRAC - _ARENA_Y_START_FRAC

# Card slot regions at 540x960 base resolution (from action_constants.py)
_CARD_START_X = 110
_CARD_WIDTH = 100
_CARD_Y_START = 770
_CARD_Y_END = 920
_BASE_W = 540
_BASE_H = 960


def _action_to_placement(action_idx: int):
    """Decode Discrete(2305) action. Returns (card_id, col, row) or None."""
    if action_idx == _NOOP_ACTION:
        return None
    card_id = action_idx // _GRID_CELLS
    cell = action_idx % _GRID_CELLS
    row = cell // _GRID_COLS
    col = cell % _GRID_COLS
    return card_id, col, row


def _cell_to_norm(col: int, row: int) -> tuple[float, float]:
    """Grid cell center -> normalized screen coords."""
    x_norm = (col + 0.5) / _GRID_COLS
    y_norm_arena = (row + 0.5) / _GRID_ROWS
    y_norm = _ARENA_Y_START_FRAC + y_norm_arena * _ARENA_Y_SPAN
    return x_norm, y_norm


def _card_slot_center_norm(card_id: int) -> tuple[float, float]:
    """Get normalized center of a card slot."""
    cx = (_CARD_START_X + card_id * _CARD_WIDTH + _CARD_WIDTH / 2) / _BASE_W
    cy = (_CARD_Y_START + _CARD_Y_END) / 2 / _BASE_H
    return cx, cy


# ---------------------------------------------------------------------------
# GameCapture
# ---------------------------------------------------------------------------

class GameCapture:
    """Screen capture via mss with rate limiting."""

    def __init__(self, config: LiveConfig) -> None:
        if mss is None:
            raise ImportError("mss is required: pip install mss")
        self._config = config
        self._sct = mss.mss()
        self._monitor = self._resolve_monitor()
        self._frame_interval = 1.0 / max(config.capture_fps, 0.1)
        self._last_capture_time = 0.0

    def _resolve_monitor(self) -> dict:
        """Build mss monitor dict from config."""
        if self._config.capture_region:
            left, top, w, h = self._config.capture_region
            return {"left": left, "top": top, "width": w, "height": h}

        if self._config.window_title:
            return self._find_window_region()

        # Fallback: primary monitor
        return self._sct.monitors[1]

    def _find_window_region(self) -> dict:
        """Try to find the window via pygetwindow.

        On Windows, uses ctypes to get the client area (game content
        without the title bar) so that capture and click coordinates
        align with the actual game viewport.
        """
        try:
            import pygetwindow as gw

            windows = gw.getWindowsWithTitle(self._config.window_title)
            if windows:
                w = windows[0]
                # Prefer client area (excludes title bar / borders)
                client = self._get_client_region(w)
                if client:
                    return client
                return {
                    "left": w.left,
                    "top": w.top,
                    "width": w.width,
                    "height": w.height,
                }
        except ImportError:
            pass

        raise RuntimeError(
            f"Cannot find window '{self._config.window_title}' and no "
            f"--capture-region specified. Install pygetwindow or set "
            f"--capture-region left,top,width,height manually."
        )

    @staticmethod
    def _get_client_region(window) -> Optional[dict]:
        """Get the client area (game content without title bar) on Windows.

        Returns None on non-Windows platforms or on failure.
        """
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            hwnd = window._hWnd
            rect = wintypes.RECT()
            ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
            point = wintypes.POINT(0, 0)
            ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
            client = {
                "left": point.x,
                "top": point.y,
                "width": rect.right - rect.left,
                "height": rect.bottom - rect.top,
            }
            if client["width"] > 0 and client["height"] > 0:
                return client
        except Exception:
            pass
        return None

    def capture(self) -> np.ndarray:
        """Capture a single frame. Returns BGR numpy array."""
        now = time.time()
        elapsed = now - self._last_capture_time
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)

        screenshot = self._sct.grab(self._monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self._last_capture_time = time.time()
        return frame

    def get_window_offset(self) -> tuple[int, int]:
        """Return (left, top) of the capture region on screen."""
        return self._monitor["left"], self._monitor["top"]

    def get_frame_size(self) -> tuple[int, int]:
        """Return (width, height) of the capture region."""
        return self._monitor["width"], self._monitor["height"]

    def release(self) -> None:
        """Release mss resources."""
        self._sct.close()


# ---------------------------------------------------------------------------
# PerceptionAdapter
# ---------------------------------------------------------------------------

class PerceptionAdapter:
    """Converts captured frames into observation tensors for the BC policy.

    Tier 2 (CRDetector + CardPredictor): Uses CRDetector for YOLO
    inference with real CLASS_NAME_TO_ID mapping, and CardPredictor for
    card hand classification. Arena uses proper 6-channel encoding.

    Tier 3 (zero-fill): Returns empty arena, mid-game default vector,
    and all-valid action mask. Always works as fallback.
    """

    def __init__(self, config: LiveConfig, project_root: str = "") -> None:
        self._config = config
        self._project_root = project_root
        self._detector = None
        self._card_predictor = None
        self.perception_active = False

        # Class mapping attributes (populated by _try_load_mappings)
        self._class_name_to_id: Optional[dict] = None
        self._unit_type_map: Optional[dict] = None
        self._num_classes: int = 155
        self._deck_card_to_idx: Optional[dict] = None
        self._card_elixir_cost: Optional[dict] = None
        self._num_deck_cards: int = 8
        self._max_elixir: int = 10

        if config.use_perception:
            self._try_init_detector(config)
            self._try_load_mappings()
            self._try_init_card_predictor(config)

    def _try_init_detector(self, config: LiveConfig) -> None:
        """Attempt to load CRDetector (YOLOv8 wrapper)."""
        try:
            from src.detection.model import CRDetector
        except ImportError as e:
            print(f"[Perception] CRDetector not available: {e}")
            print("[Perception] Using zero-filled observations.")
            return

        for path in config.detector_model_paths:
            full_path = path
            if self._project_root and not os.path.isabs(path):
                full_path = os.path.join(self._project_root, path)

            if os.path.exists(full_path):
                try:
                    self._detector = CRDetector(
                        model_path=full_path,
                        confidence_threshold=0.5,
                        iou_threshold=0.45,
                    )
                    self.perception_active = True
                    print(f"[Perception] Loaded CRDetector: {full_path}")
                    return
                except Exception as e:
                    print(f"[Perception] Failed to load {full_path}: {e}")
            else:
                print(f"[Perception] Model not found: {full_path}")

        print("[Perception] No detector models loaded. "
              "Using zero-filled observations.")

    def _try_load_mappings(self) -> None:
        """Load class name -> ID and unit type mappings from encoder_constants."""
        try:
            from src.encoder.encoder_constants import (
                CLASS_NAME_TO_ID,
                NUM_CLASSES,
                UNIT_TYPE_MAP,
                DECK_CARD_TO_IDX,
                CARD_ELIXIR_COST,
                NUM_DECK_CARDS,
                MAX_ELIXIR,
            )
            self._class_name_to_id = CLASS_NAME_TO_ID
            self._num_classes = NUM_CLASSES
            self._unit_type_map = UNIT_TYPE_MAP
            self._deck_card_to_idx = DECK_CARD_TO_IDX
            self._card_elixir_cost = CARD_ELIXIR_COST
            self._num_deck_cards = NUM_DECK_CARDS
            self._max_elixir = MAX_ELIXIR
            print(f"[Perception] Loaded class mappings: {NUM_CLASSES} classes")
        except Exception as e:
            print(f"[Perception] Could not load encoder_constants: {e}")
            print("[Perception] Class IDs will use placeholder values.")

    def _try_init_card_predictor(self, config: LiveConfig) -> None:
        """Attempt to load CardPredictor for card hand classification."""
        if not config.card_classifier_path:
            return

        try:
            from src.classification.card_classifier import CardPredictor
        except ImportError as e:
            print(f"[Perception] CardPredictor not available: {e}")
            return

        full_path = config.card_classifier_path
        if self._project_root and not os.path.isabs(full_path):
            full_path = os.path.join(self._project_root, full_path)

        if not os.path.exists(full_path):
            print(f"[Perception] Card classifier not found: {full_path}")
            return

        try:
            self._card_predictor = CardPredictor(full_path)
            print(f"[Perception] Loaded CardPredictor: {full_path}")
        except Exception as e:
            print(f"[Perception] Failed to load CardPredictor: {e}")

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a frame into obs tensors, action mask, and metadata.

        Returns:
            dict with keys:
              "obs": {"arena": (1,32,18,6) tensor, "vector": (1,23) tensor}
              "mask": (1,2305) bool tensor
              "detections": list of detection dicts (for logging)
              "perception_active": bool
        """
        if self.perception_active:
            return self._process_with_detection(frame)
        return self._process_fallback()

    def _process_fallback(self) -> dict:
        """Tier 3: Zero-filled obs with mid-game default vector."""
        arena = np.zeros(
            (1, _GRID_ROWS, _GRID_COLS, _NUM_ARENA_CHANNELS),
            dtype=np.float32,
        )
        vector = np.zeros((1, _NUM_VECTOR_FEATURES), dtype=np.float32)

        # Mid-game defaults so the model sees a plausible state
        vector[0, 0] = 0.5   # elixir / 10 (assume 5 elixir)
        vector[0, 1] = 0.4   # time_remaining / 300 (assume 120s left)
        vector[0, 3:9] = 1.0  # all 6 towers alive at full HP
        vector[0, 9] = 1.0   # player tower count / 3
        vector[0, 10] = 1.0  # enemy tower count / 3
        vector[0, 11:15] = 1.0  # all 4 card slots present

        # All actions valid (no elixir check possible)
        mask = np.ones(_ACTION_SPACE_SIZE, dtype=np.bool_)

        return {
            "obs": {
                "arena": torch.from_numpy(arena).float(),
                "vector": torch.from_numpy(vector).float(),
            },
            "mask": torch.from_numpy(mask).bool().unsqueeze(0),
            "detections": [],
            "perception_active": False,
        }

    def _process_with_detection(self, frame: np.ndarray) -> dict:
        """Tier 2: CRDetector with proper 6-channel arena encoding.

        Uses real CLASS_NAME_TO_ID for unit identity and UNIT_TYPE_MAP for
        channel assignment. CardPredictor populates vector card features.

        Arena channels:
          0: CH_CLASS_ID   - class_idx / NUM_CLASSES (0 = empty)
          1: CH_BELONGING  - -1.0 = ally, +1.0 = enemy
          2: CH_ARENA_MASK - 1.0 = unit present
          3: CH_ALLY_TOWER_HP   - ally tower HP fraction
          4: CH_ENEMY_TOWER_HP  - enemy tower HP fraction
          5: CH_SPELL      - spell effect count (additive)
        """
        # Run CRDetector
        detections = self._detector.detect(frame)

        # Build arena tensor
        arena = np.zeros(
            (1, _GRID_ROWS, _GRID_COLS, _NUM_ARENA_CHANNELS),
            dtype=np.float32,
        )
        fh, fw = frame.shape[:2]

        detection_dicts = []

        for det in detections:
            cx, cy = det.center
            x_norm = cx / fw
            y_norm = cy / fh
            arena_y_frac = (y_norm - _ARENA_Y_START_FRAC) / _ARENA_Y_SPAN

            col = max(0, min(_GRID_COLS - 1, int(x_norm * _GRID_COLS)))
            row = max(0, min(_GRID_ROWS - 1, int(arena_y_frac * _GRID_ROWS)))

            # Belonging heuristic: top half = enemy, bottom half = ally
            is_ally = row >= 16
            belonging_val = -1.0 if is_ally else 1.0

            # Look up unit type
            unit_type = "ground"
            if self._unit_type_map:
                unit_type = self._unit_type_map.get(det.class_name, "ground")

            # Look up class ID (1-indexed, 0 = empty)
            class_idx = 0
            if self._class_name_to_id:
                class_idx = self._class_name_to_id.get(det.class_name, 0)

            if unit_type == "tower":
                # Assume alive at full HP (no OCR in Tier 2)
                ch = 3 if is_ally else 4  # CH_ALLY_TOWER_HP / CH_ENEMY_TOWER_HP
                arena[0, row, col, ch] = 1.0
            elif unit_type == "spell":
                # Additive spell count
                arena[0, row, col, 5] += 1.0  # CH_SPELL
            elif unit_type == "other":
                pass  # UI elements - skip
            else:
                # Ground or flying unit - per-cell identity encoding
                if class_idx > 0 and arena[0, row, col, 2] == 0.0:
                    arena[0, row, col, 0] = class_idx / self._num_classes
                    arena[0, row, col, 1] = belonging_val
                    arena[0, row, col, 2] = 1.0  # CH_ARENA_MASK

            detection_dicts.append({
                "bbox": list(det.bbox),
                "confidence": det.confidence,
                "class_name": det.class_name,
                "unit_type": unit_type,
            })

        # Vector features with mid-game defaults
        vector = np.zeros((1, _NUM_VECTOR_FEATURES), dtype=np.float32)
        vector[0, 0] = 0.5    # elixir / 10
        vector[0, 1] = 0.4    # time_remaining / 300
        vector[0, 3:9] = 1.0  # all 6 towers alive at full HP
        vector[0, 9] = 1.0    # player tower count / 3
        vector[0, 10] = 1.0   # enemy tower count / 3

        # Card classification (if CardPredictor available)
        if self._card_predictor is not None:
            self._populate_card_vector(frame, vector, fw, fh)
        else:
            vector[0, 11:15] = 1.0  # assume all 4 cards present

        mask = np.ones(_ACTION_SPACE_SIZE, dtype=np.bool_)

        return {
            "obs": {
                "arena": torch.from_numpy(arena).float(),
                "vector": torch.from_numpy(vector).float(),
            },
            "mask": torch.from_numpy(mask).bool().unsqueeze(0),
            "detections": detection_dicts,
            "perception_active": True,
        }

    def _populate_card_vector(
        self, frame: np.ndarray, vector: np.ndarray, fw: int, fh: int
    ) -> None:
        """Crop 4 card slots from frame and classify each.

        Populates vector indices:
          [11-14] card present (1.0 if classified)
          [15-18] card class index / (NUM_DECK_CARDS - 1)
          [19-22] card elixir cost / MAX_ELIXIR
        """
        x_scale = fw / _BASE_W
        y_scale = fh / _BASE_H

        for i in range(4):
            x_start = int((_CARD_START_X + i * _CARD_WIDTH) * x_scale)
            x_end = int((_CARD_START_X + (i + 1) * _CARD_WIDTH) * x_scale)
            y_start = int(_CARD_Y_START * y_scale)
            y_end = int(_CARD_Y_END * y_scale)

            # Bounds check
            y_start = max(0, min(y_start, fh - 1))
            y_end = max(y_start + 1, min(y_end, fh))
            x_start = max(0, min(x_start, fw - 1))
            x_end = max(x_start + 1, min(x_end, fw))

            crop = frame[y_start:y_end, x_start:x_end]
            if crop.size == 0:
                continue

            try:
                class_name, confidence = self._card_predictor.predict(crop)
            except Exception:
                continue

            # Card present
            vector[0, 11 + i] = 1.0

            # Card class index (normalized)
            if self._deck_card_to_idx is not None:
                card_idx = self._deck_card_to_idx.get(class_name, 0)
                vector[0, 15 + i] = card_idx / max(self._num_deck_cards - 1, 1)

            # Card elixir cost (normalized)
            if self._card_elixir_cost is not None:
                cost = self._card_elixir_cost.get(class_name, 0)
                vector[0, 19 + i] = cost / self._max_elixir


# ---------------------------------------------------------------------------
# ActionDispatcher
# ---------------------------------------------------------------------------

class ActionDispatcher:
    """Executes actions via PyAutoGUI with window offset correction.

    Unlike ActionExecutor (which converts norm coords to frame-relative
    pixels), this class adds the window's absolute screen position so
    PyAutoGUI clicks land in the correct spot.
    """

    def __init__(self, config: LiveConfig) -> None:
        self._config = config
        self._frame_w = config.frame_w
        self._frame_h = config.frame_h
        self._window_left = config.window_left
        self._window_top = config.window_top
        self._last_action_time = 0.0
        self._actions_this_minute = 0
        self._minute_start = time.time()

        self._pyautogui = None
        if not config.dry_run:
            try:
                import pyautogui

                pyautogui.PAUSE = 0.05
                self._pyautogui = pyautogui
            except ImportError:
                print("[ActionDispatcher] pyautogui not installed. "
                      "Forcing dry-run mode.")
                self._config.dry_run = True

    def update_window_offset(self, left: int, top: int) -> None:
        """Update window position (call if window moves)."""
        self._window_left = left
        self._window_top = top

    def execute(self, action_idx: int, logit_score: float) -> dict:
        """Execute an action with safety checks.

        Returns a dict describing the outcome (for logging).
        """
        result: dict = {
            "action_idx": action_idx,
            "logit_score": round(logit_score, 4),
            "executed": False,
            "reason": "",
        }

        # Decode action
        placement = _action_to_placement(action_idx)
        if placement is None:
            result["reason"] = "noop"
            return result

        card_id, col, row = placement
        x_norm, y_norm = _cell_to_norm(col, row)

        result["card_id"] = card_id
        result["col"] = col
        result["row"] = row
        result["x_norm"] = round(x_norm, 4)
        result["y_norm"] = round(y_norm, 4)

        # Confidence check
        if logit_score < self._config.confidence_threshold:
            result["reason"] = "below_confidence"
            return result

        # Cooldown check
        now = time.time()
        if now - self._last_action_time < self._config.action_cooldown:
            result["reason"] = "cooldown"
            return result

        # Rate limit check
        if now - self._minute_start >= 60:
            self._actions_this_minute = 0
            self._minute_start = now
        if self._actions_this_minute >= self._config.max_actions_per_minute:
            result["reason"] = "rate_limited"
            return result

        # Dry run
        if self._config.dry_run:
            result["reason"] = "dry_run"
            return result

        # Execute the two-click sequence
        self._click_card_then_arena(card_id, x_norm, y_norm)
        self._last_action_time = time.time()
        self._actions_this_minute += 1
        result["executed"] = True
        result["reason"] = "played"
        return result

    def _click_card_then_arena(
        self, card_id: int, x_norm: float, y_norm: float
    ) -> None:
        """Two-click sequence: card slot then arena position.

        Applies window offset for absolute screen coordinates.
        """
        # Card slot center (normalized)
        cx_norm, cy_norm = _card_slot_center_norm(card_id)

        # Convert to absolute screen pixels
        card_px = int(cx_norm * self._frame_w) + self._window_left
        card_py = int(cy_norm * self._frame_h) + self._window_top

        arena_px = int(x_norm * self._frame_w) + self._window_left
        arena_py = int(y_norm * self._frame_h) + self._window_top

        # Click card slot, wait, click arena
        self._pyautogui.click(card_px, card_py)
        time.sleep(0.15)  # CLICK_DELAY_SECONDS
        self._pyautogui.click(arena_px, arena_py)


# ---------------------------------------------------------------------------
# LiveInferenceEngine
# ---------------------------------------------------------------------------

class LiveInferenceEngine:
    """Main inference loop: capture -> perceive -> predict -> act.

    Ties together GameCapture, PerceptionAdapter, BCPolicy, and
    ActionDispatcher into a real-time loop running at ~2 FPS.

    Stop with Ctrl+C. Produces a JSONL log file per session.
    """

    def __init__(self, config: LiveConfig, project_root: str = "") -> None:
        self._config = config
        self._project_root = project_root

        # Set up logging first
        self._setup_logging(config)

        # Load BC policy
        print(f"[Engine] Loading model from {config.model_path}...")
        self._policy = self._load_policy(config)

        # Initialize pipeline components
        print("[Engine] Initializing screen capture...")
        self._capture = GameCapture(config)

        print("[Engine] Initializing perception...")
        self._perception = PerceptionAdapter(config, project_root)

        print("[Engine] Initializing action dispatcher...")
        self._dispatcher = ActionDispatcher(config)

        # Sync window offset from capture to dispatcher
        wl, wt = self._capture.get_window_offset()
        fw, fh = self._capture.get_frame_size()
        self._capture_w = fw
        self._capture_h = fh

        # Detect game content bounds (exclude black pillarbox bars)
        print("[Engine] Detecting game content bounds...")
        probe_frame = self._capture.capture()
        gx, gy, gw, gh = self._detect_game_bounds(probe_frame)
        self._game_x_offset = gx
        self._game_y_offset = gy
        self._game_w = gw
        self._game_h = gh

        if gx > 0 or gy > 0 or gw != fw or gh != fh:
            print(f"[Engine] Game content: ({gx},{gy}) {gw}x{gh} "
                  f"within {fw}x{fh} capture (black bars detected)")
        else:
            print(f"[Engine] Game fills entire capture: {fw}x{fh}")

        # Offset dispatcher by window + game content position
        self._dispatcher.update_window_offset(wl + gx, wt + gy)
        config.window_left = wl + gx
        config.window_top = wt + gy

        # Frame dimensions = game content (not full capture)
        config.frame_w = gw
        config.frame_h = gh
        self._dispatcher._frame_w = gw
        self._dispatcher._frame_h = gh

        # Session stats
        self._frame_count = 0
        self._action_count = 0
        self._noop_count = 0
        self._start_time: Optional[float] = None

    @staticmethod
    def _detect_game_bounds(
        frame: np.ndarray, threshold: int = 15
    ) -> tuple[int, int, int, int]:
        """Detect game content within a frame by scanning for non-black regions.

        Google Play Games renders the 9:16 Clash Royale game centered
        within a potentially wider window, padding with black bars
        (pillarboxing).  This method finds the actual game rectangle.

        Returns:
            (x_offset, y_offset, game_w, game_h) of the game content
            within the frame.  Falls back to (0, 0, frame_w, frame_h)
            when no black bars are detected.
        """
        fh, fw = frame.shape[:2]

        # Max pixel intensity per column / row (across height and channels)
        col_max = frame.max(axis=(0, 2))  # shape (fw,)
        row_max = frame.max(axis=(1, 2))  # shape (fh,)

        non_black_cols = np.where(col_max > threshold)[0]
        non_black_rows = np.where(row_max > threshold)[0]

        if len(non_black_cols) == 0 or len(non_black_rows) == 0:
            # No content found – fall back to full frame
            return 0, 0, fw, fh

        x_start = int(non_black_cols[0])
        x_end = int(non_black_cols[-1]) + 1
        y_start = int(non_black_rows[0])
        y_end = int(non_black_rows[-1]) + 1

        return x_start, y_start, x_end - x_start, y_end - y_start

    def _load_policy(self, config: LiveConfig):
        """Load BCPolicy from checkpoint."""
        from src.bc.bc_policy import BCPolicy

        device = torch.device(config.device)
        model_path = config.model_path
        if self._project_root and not os.path.isabs(model_path):
            model_path = os.path.join(self._project_root, model_path)

        policy = BCPolicy.load(model_path)
        policy = policy.to(device)
        policy.eval()
        print(f"[Engine] Model loaded on {device}")
        return policy

    def run(self) -> None:
        """Main inference loop. Ctrl+C to stop."""
        self._start_time = time.time()
        print(f"\n{'=' * 60}")
        print("[Engine] Starting live inference loop")
        print(f"  Dry run:      {self._config.dry_run}")
        print(f"  Perception:   {self._perception.perception_active}")
        print(f"  Confidence:   {self._config.confidence_threshold}")
        print(f"  Cooldown:     {self._config.action_cooldown}s")
        print(f"  Capture FPS:  {self._config.capture_fps}")
        print(f"  Game size:    {self._game_w}x{self._game_h}")
        print(f"  Capture size: {self._capture_w}x{self._capture_h}")
        print(f"  Game offset:  ({self._game_x_offset}, {self._game_y_offset})")
        print(f"  Window offset: ({self._config.window_left}, "
              f"{self._config.window_top})")
        print(f"  Log file:     {self._log_path}")
        print(f"{'=' * 60}")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                self._step()
        except KeyboardInterrupt:
            print("\n[Engine] Stopped by user (Ctrl+C)")
        finally:
            self._shutdown()

    def _step(self) -> None:
        """Single inference step: capture -> perceive -> predict -> act."""
        self._frame_count += 1
        step_start = time.time()

        # 1. Capture and crop to game content (exclude black bars)
        frame = self._capture.capture()
        frame = frame[
            self._game_y_offset:self._game_y_offset + self._game_h,
            self._game_x_offset:self._game_x_offset + self._game_w,
        ]

        # 2. Perception -> obs tensors
        perception_result = self._perception.process_frame(frame)
        obs = perception_result["obs"]
        mask = perception_result["mask"]

        # 3. Move to device
        device = next(self._policy.parameters()).device
        obs_device = {
            "arena": obs["arena"].to(device),
            "vector": obs["vector"].to(device),
        }
        mask_device = mask.to(device)

        # 4. Predict
        with torch.no_grad():
            logits = self._policy.forward(obs_device)  # (1, 2305)
            # Apply mask
            masked_logits = logits.clone()
            if mask_device.dim() == 2:
                masked_logits[~mask_device] = float("-inf")
            else:
                masked_logits[0, ~mask_device] = float("-inf")

            action_idx = masked_logits.argmax(dim=1).item()
            logit_score = masked_logits[0, action_idx].item()

        # 5. Execute
        exec_result = self._dispatcher.execute(action_idx, logit_score)

        if exec_result.get("executed", False):
            self._action_count += 1
        elif exec_result["reason"] == "noop":
            self._noop_count += 1

        # 6. Log
        step_time = time.time() - step_start
        self._log_step(step_time, exec_result, perception_result)

    def _log_step(
        self,
        step_time: float,
        exec_result: dict,
        perception_result: dict,
    ) -> None:
        """Log this inference step to console and JSONL file."""
        det_count = len(perception_result["detections"])
        action_idx = exec_result["action_idx"]
        logit_score = exec_result["logit_score"]
        reason = exec_result["reason"]

        if self._config.verbose:
            status = "ACT" if exec_result.get("executed") else reason.upper()
            card_info = ""
            if "card_id" in exec_result:
                card_info = (
                    f" card={exec_result['card_id']} "
                    f"cell=({exec_result['col']},{exec_result['row']})"
                )
            print(
                f"[{self._frame_count:04d}] {step_time * 1000:.0f}ms | "
                f"action={action_idx:4d} logit={logit_score:+.2f} | "
                f"{status}{card_info} | "
                f"det={det_count}"
            )

        log_entry = {
            "frame": self._frame_count,
            "timestamp": time.time(),
            "step_time_ms": round(step_time * 1000, 1),
            **exec_result,
            "detection_count": det_count,
            "perception_active": perception_result["perception_active"],
        }
        self._log_file.write(json.dumps(log_entry) + "\n")
        self._log_file.flush()

    def _setup_logging(self, config: LiveConfig) -> None:
        """Create log directory and open JSONL log file."""
        log_dir = config.log_dir
        if self._project_root and not os.path.isabs(log_dir):
            log_dir = os.path.join(self._project_root, log_dir)

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(log_dir, f"session_{timestamp}.jsonl")
        self._log_file = open(self._log_path, "w")
        print(f"[Engine] Logging to {self._log_path}")

    def _shutdown(self) -> None:
        """Cleanup and print session summary."""
        duration = time.time() - self._start_time if self._start_time else 0
        summary = {
            "duration_seconds": round(duration, 1),
            "total_frames": self._frame_count,
            "actions_executed": self._action_count,
            "noops": self._noop_count,
            "avg_fps": round(
                self._frame_count / max(duration, 0.1), 2
            ),
        }

        print(f"\n{'=' * 60}")
        print("Session Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"  Log: {self._log_path}")
        print(f"{'=' * 60}")

        # Write summary as last line
        self._log_file.write(json.dumps({"summary": summary}) + "\n")
        self._log_file.close()
        self._capture.release()
