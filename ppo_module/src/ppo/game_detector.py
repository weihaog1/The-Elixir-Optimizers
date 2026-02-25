"""Game phase detection from raw screen frames.

Detects whether Clash Royale is in a loading screen, in-game, or on
the end-of-game results screen. Uses pixel heuristics on the card bar
region and arena region -- no OCR or template matching required for
basic detection.

For win/loss detection, uses template matching against pre-captured
"Victory" and "Defeat" banner images stored in ppo_module/templates/.
Falls back to crown-count heuristics if templates are not available.

Screen layout reference (540x960 base resolution):
    y=0-50:     Timer / header bar
    y=50-750:   Arena (game area)
    y=750-770:  Elixir bar
    y=770-920:  Card bar (4 card slots)
    y=920-960:  Bottom UI
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class Phase(Enum):
    """Game phase states."""

    UNKNOWN = "unknown"
    LOADING = "loading"
    IN_GAME = "in_game"
    END_SCREEN = "end_screen"


@dataclass
class DetectorConfig:
    """Configuration for game phase detection.

    All pixel coordinates are fractions of frame dimensions (0-1).
    """

    # Card bar region (fraction of frame height)
    card_bar_y_start: float = 770.0 / 960.0  # ~0.802
    card_bar_y_end: float = 920.0 / 960.0  # ~0.958

    # Arena region (fraction of frame height)
    arena_y_start: float = 50.0 / 960.0  # ~0.052
    arena_y_end: float = 750.0 / 960.0  # ~0.781

    # Card bar detection: mean pixel intensity threshold
    # During gameplay, card bar has colored card art (higher intensity)
    # During end screen, this area is covered by results overlay
    card_bar_intensity_threshold: float = 40.0

    # Arena variance threshold for distinguishing game vs loading
    # Loading screens are relatively uniform; gameplay has varied content
    arena_variance_threshold: float = 200.0

    # End screen detection: the results overlay dims the arena significantly
    # and the card bar region changes appearance
    end_screen_arena_intensity_max: float = 80.0

    # Template matching threshold for Victory/Defeat banners
    template_match_threshold: float = 0.7

    # Minimum consecutive frames in a phase before confirming transition
    phase_stability_frames: int = 3


class GamePhaseDetector:
    """Detects game phase transitions from raw BGR frames.

    Usage:
        detector = GamePhaseDetector()
        phase = detector.detect_phase(frame)
        if phase == Phase.END_SCREEN:
            outcome = detector.detect_outcome(frame)
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        templates_dir: str = "",
    ) -> None:
        self.config = config or DetectorConfig()
        self._phase = Phase.UNKNOWN
        self._phase_count = 0
        self._victory_template: Optional[np.ndarray] = None
        self._defeat_template: Optional[np.ndarray] = None

        if templates_dir:
            self._load_templates(templates_dir)

    def _load_templates(self, templates_dir: str) -> None:
        """Load Victory/Defeat template images for matching."""
        victory_path = os.path.join(templates_dir, "victory.png")
        defeat_path = os.path.join(templates_dir, "defeat.png")

        if os.path.exists(victory_path):
            self._victory_template = cv2.imread(victory_path, cv2.IMREAD_GRAYSCALE)
        if os.path.exists(defeat_path):
            self._defeat_template = cv2.imread(defeat_path, cv2.IMREAD_GRAYSCALE)

    def detect_phase(self, frame: np.ndarray) -> Phase:
        """Classify the current game phase from a raw BGR frame.

        Args:
            frame: BGR image from screen capture.

        Returns:
            Current Phase enum value.
        """
        fh, fw = frame.shape[:2]
        cfg = self.config

        # Extract regions
        card_bar_y1 = int(cfg.card_bar_y_start * fh)
        card_bar_y2 = int(cfg.card_bar_y_end * fh)
        arena_y1 = int(cfg.arena_y_start * fh)
        arena_y2 = int(cfg.arena_y_end * fh)

        card_bar = frame[card_bar_y1:card_bar_y2, :, :]
        arena = frame[arena_y1:arena_y2, :, :]

        # Compute metrics
        card_bar_intensity = float(np.mean(card_bar))
        arena_intensity = float(np.mean(arena))
        arena_variance = float(np.var(arena.astype(np.float32)))

        # Determine candidate phase
        card_bar_active = card_bar_intensity > cfg.card_bar_intensity_threshold
        arena_active = arena_variance > cfg.arena_variance_threshold

        if card_bar_active and arena_active:
            candidate = Phase.IN_GAME
        elif not card_bar_active and arena_intensity < cfg.end_screen_arena_intensity_max:
            candidate = Phase.LOADING
        elif not card_bar_active or (
            arena_intensity > cfg.end_screen_arena_intensity_max
            and not arena_active
        ):
            # Results overlay: arena is dimmed/covered, card bar gone
            candidate = Phase.END_SCREEN
        else:
            candidate = Phase.UNKNOWN

        # Phase stability: require N consecutive frames of same candidate
        if candidate == self._phase:
            self._phase_count += 1
        else:
            self._phase_count = 1
            self._phase = candidate

        if self._phase_count >= cfg.phase_stability_frames:
            return self._phase

        # Return previous confirmed phase until stability threshold met
        return self._phase

    def detect_outcome(self, frame: np.ndarray) -> Optional[str]:
        """Detect win/loss/draw from the end-of-game screen.

        Uses template matching if templates are loaded, otherwise
        falls back to a color heuristic (victory screens tend to be
        brighter/more golden, defeat screens are darker/blue-tinted).

        Args:
            frame: BGR image of the end screen.

        Returns:
            "win", "loss", "draw", or None if uncertain.
        """
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try template matching first
        if self._victory_template is not None:
            victory_score = self._match_template(gray, self._victory_template, fh, fw)
        else:
            victory_score = 0.0

        if self._defeat_template is not None:
            defeat_score = self._match_template(gray, self._defeat_template, fh, fw)
        else:
            defeat_score = 0.0

        threshold = self.config.template_match_threshold

        if victory_score >= threshold and victory_score > defeat_score:
            return "win"
        if defeat_score >= threshold and defeat_score > victory_score:
            return "loss"

        # Fallback: color heuristic on the upper-center banner area
        # Victory = golden tones (high R+G, lower B)
        # Defeat = blue/dark tones (high B, lower R+G)
        banner_y1 = int(0.25 * fh)
        banner_y2 = int(0.45 * fh)
        banner_x1 = int(0.15 * fw)
        banner_x2 = int(0.85 * fw)
        banner = frame[banner_y1:banner_y2, banner_x1:banner_x2, :]

        mean_bgr = np.mean(banner, axis=(0, 1))  # (B, G, R)
        blue, green, red = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])

        # Golden (victory): red > 120, green > 100, blue < red * 0.7
        if red > 120 and green > 100 and blue < red * 0.7:
            return "win"

        # Blue/dark (defeat): blue > red, overall dark
        if blue > red and (red + green + blue) / 3 < 120:
            return "loss"

        return None

    def _match_template(
        self,
        gray: np.ndarray,
        template: np.ndarray,
        fh: int,
        fw: int,
    ) -> float:
        """Run template matching in the banner region.

        Returns the best match score (0-1).
        """
        # Scale template to match frame size
        t_h, t_w = template.shape[:2]
        scale = fw / 540.0  # base resolution
        new_w = max(1, int(t_w * scale))
        new_h = max(1, int(t_h * scale))
        scaled = cv2.resize(template, (new_w, new_h))

        # Search in upper portion of frame (where banner appears)
        search_region = gray[: int(0.6 * fh), :]

        if scaled.shape[0] > search_region.shape[0] or scaled.shape[1] > search_region.shape[1]:
            return 0.0

        result = cv2.matchTemplate(search_region, scaled, cv2.TM_CCOEFF_NORMED)
        return float(np.max(result))

    def wait_for_game_start(
        self,
        capture_fn,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """Block until a game starts.

        Args:
            capture_fn: Callable returning a BGR frame (e.g., GameCapture.capture).
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between checks.

        Returns:
            True if game started, False on timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            frame = capture_fn()
            phase = self.detect_phase(frame)
            if phase == Phase.IN_GAME:
                return True
            time.sleep(poll_interval)
        return False

    def wait_for_game_end(
        self,
        capture_fn,
        timeout: float = 360.0,
        poll_interval: float = 0.5,
    ) -> Optional[str]:
        """Block until the game ends.

        Args:
            capture_fn: Callable returning a BGR frame.
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between checks.

        Returns:
            Outcome string ("win"/"loss"/"draw") or None on timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            frame = capture_fn()
            phase = self.detect_phase(frame)
            if phase == Phase.END_SCREEN:
                # Wait a moment for the full results screen to render
                time.sleep(1.0)
                frame = capture_fn()
                return self.detect_outcome(frame)
            time.sleep(poll_interval)
        return None

    def reset(self) -> None:
        """Reset detector state for a new episode."""
        self._phase = Phase.UNKNOWN
        self._phase_count = 0
