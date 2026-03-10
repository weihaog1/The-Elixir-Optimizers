"""Gymnasium environment wrapper for live Clash Royale gameplay.

Wraps screen capture, perception, action execution, reward computation,
and game phase detection into a standard Gymnasium interface compatible
with sb3_contrib.MaskablePPO.

Supports N-frame stacking: the policy sees concatenated observations from
the last *n_frames* steps (arena channels and vector features), while the
reward computer always operates on single-frame observations.

Architecture:
    GameCapture (mss) -> PerceptionAdapter (YOLO/fallback) ->
    RewardComputer (obs deltas) -> ActionDispatcher (PyAutoGUI)

The environment operates in real-time at ~2 FPS. Each step() call:
1. Executes the chosen action (or noop)
2. Captures the next frame
3. Runs perception to produce observations
4. Computes reward from observation deltas
5. Checks for game end

Usage:
    from src.ppo.clash_royale_env import ClashRoyaleEnv, EnvConfig

    config = EnvConfig(capture_region=(0, 0, 540, 960), dry_run=True)
    env = ClashRoyaleEnv(config)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import msvcrt
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from src.bc.live_inference import (
    ActionDispatcher,
    GameCapture,
    LiveConfig,
    PerceptionAdapter,
    _NOOP_ACTION,
)
from src.encoder.encoder_constants import CARD_ELIXIR_COST
from src.ppo.game_detector import DetectorConfig, GamePhaseDetector, Phase
from src.ppo.reward import RewardComputer, RewardConfig


# Constants (match encoder_constants.py)
_GRID_ROWS = 32
_GRID_COLS = 18
_NUM_ARENA_CHANNELS = 6
_NUM_VECTOR_FEATURES = 23
_ACTION_SPACE_SIZE = 4 * _GRID_ROWS * _GRID_COLS + 1  # 2305


@dataclass
class EnvConfig:
    """Configuration for ClashRoyaleEnv."""

    # Capture settings
    window_title: str = ""
    capture_region: Optional[tuple[int, int, int, int]] = None
    capture_fps: float = 2.0
    frame_w: int = 540
    frame_h: int = 960

    # Game region within the captured window (left, top, width, height).
    # Use when the game window is landscape (e.g. Google Play Games 1920x1080)
    # and the Clash Royale portrait game area is a subset of the window.
    # If None, auto-detected via _detect_game_bounds().
    game_region: Optional[tuple[int, int, int, int]] = None

    # Perception
    use_perception: bool = True
    detector_model_paths: list[str] = field(default_factory=lambda: [
        "models/dual_d1_best.pt",
        "models/dual_d2_best.pt",
    ])
    split_config_path: str = os.path.join("configs", "split_config.json")
    detector_conf: float = 0.25
    detector_imgsz: int = 960
    ocr_interval: int = 5
    card_classifier_path: str = "models/card_classifier.pt"
    card_confidence_threshold: float = 0.6  # min softmax confidence to trust card classification

    # Safety
    dry_run: bool = False

    # Reward
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Game detection
    detector_config: DetectorConfig = field(default_factory=DetectorConfig)
    templates_dir: str = ""

    # Episode limits
    max_episode_steps: int = 700  # 300s game * 2 FPS = 600 + overtime margin
    min_episode_steps: int = 60  # Ignore END_SCREEN before this (30s at 2 FPS)

    # Timeouts
    step_timeout: float = 5.0  # Max seconds to wait for a frame
    identical_frame_limit: int = 5  # Truncate after N identical frames
    game_start_timeout: float = 120.0  # Max seconds to wait for game start

    # Device
    device: str = "cpu"

    # Frame stacking
    n_frames: int = 3  # Number of frames to stack (1 = no stacking)

    # Operator control
    pause_between_episodes: bool = True  # Wait for Enter between episodes

    # Visualization
    visualize: bool = False  # Show live observation heatmaps
    vis_save_dir: str = ""  # Save visualization frames to this directory
    vis_backend: str = "cv"  # "cv" for OpenCV side-by-side, "mpl" for matplotlib heatmaps
    vis_record: bool = False  # Record video (cv backend only)
    vis_output_path: str = ""  # Video output path (cv backend only)
    vis_show_window: bool = True  # Show cv2.imshow window (cv backend only)

    # Logging
    verbose: bool = True


class ClashRoyaleEnv(gymnasium.Env):
    """Gymnasium environment wrapping live Clash Royale gameplay.

    Observation space (with n_frames=3):
        Dict(arena=Box(32, 18, 18), vector=Box(69,))
    Action space: Discrete(2305)
    Supports action masking via action_masks() for MaskablePPO.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        project_root: str = "",
    ) -> None:
        super().__init__()
        self._config = config or EnvConfig()
        self._project_root = project_root
        n = self._config.n_frames

        # Define spaces (frame-stacked dimensions)
        self.observation_space = spaces.Dict({
            "arena": spaces.Box(
                low=-1.0, high=10.0,
                shape=(_GRID_ROWS, _GRID_COLS, _NUM_ARENA_CHANNELS * n),
                dtype=np.float32,
            ),
            "vector": spaces.Box(
                low=0.0, high=1.0,
                shape=(_NUM_VECTOR_FEATURES * n,),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(_ACTION_SPACE_SIZE)

        # Build LiveConfig for reusable components
        self._live_config = LiveConfig(
            window_title=self._config.window_title,
            capture_region=self._config.capture_region,
            capture_fps=self._config.capture_fps,
            frame_w=self._config.frame_w,
            frame_h=self._config.frame_h,
            use_perception=self._config.use_perception,
            detector_model_paths=self._config.detector_model_paths,
            split_config_path=self._config.split_config_path,
            detector_conf=self._config.detector_conf,
            detector_imgsz=self._config.detector_imgsz,
            ocr_interval=self._config.ocr_interval,
            card_classifier_path=self._config.card_classifier_path,
            dry_run=self._config.dry_run,
            card_confidence_threshold=self._config.card_confidence_threshold,
            action_cooldown=0.0,  # PPO controls timing, not cooldown
            max_actions_per_minute=999,  # No rate limiting for PPO
            confidence_threshold=0.0,
            noop_frames_after_play=0,
        )

        # Initialize components
        self._capture = GameCapture(self._live_config)

        # Detect game bounds within the captured window.
        if self._config.game_region is not None:
            # Manual override: use exact game region coordinates
            gx, gy, gw, gh = self._config.game_region
        else:
            # Auto-detect: wait for focus, then probe
            hwnd = self._capture.get_window_hwnd()
            if hwnd is not None:
                import ctypes
                for _ in range(30):  # wait up to 15s for focus
                    try:
                        if ctypes.windll.user32.GetForegroundWindow() == hwnd:
                            break
                    except Exception:
                        break
                    time.sleep(0.5)
                else:
                    print("[Env] WARNING: Game window not focused for probe. "
                          "Bounds may be incorrect.")

            probe = self._capture.capture()
            gx, gy, gw, gh = self._detect_game_bounds(probe)

        self._game_x_offset = gx
        self._game_y_offset = gy
        self._game_w = gw
        self._game_h = gh

        # Sanity check: game region should be portrait (aspect < 0.7)
        cap_w, cap_h = self._capture.get_frame_size()
        aspect = gw / max(gh, 1)
        src = "manual" if self._config.game_region else "auto-detected"
        print(f"[Env] Capture: {cap_w}x{cap_h}. "
              f"Game bounds ({src}): ({gx},{gy}) {gw}x{gh} "
              f"(aspect={aspect:.2f})")
        if aspect > 0.7:
            print(f"[Env] WARNING: Game region is landscape ({gw}x{gh}). "
                  f"Expected portrait (~609x1077). Check --game-region values.")

        # Update window offset for dispatcher
        wl, wt = self._capture.get_window_offset()
        self._live_config.window_left = wl + gx
        self._live_config.window_top = wt + gy
        self._live_config.frame_w = gw
        self._live_config.frame_h = gh

        self._perception = PerceptionAdapter(self._live_config, project_root)
        self._dispatcher = ActionDispatcher(
            self._live_config,
            game_hwnd=self._capture.get_window_hwnd(),
        )
        self._dispatcher.update_window_offset(wl + gx, wt + gy)
        self._dispatcher._frame_w = gw
        self._dispatcher._frame_h = gh

        self._reward_computer = RewardComputer(self._config.reward_config)
        self._game_detector = GamePhaseDetector(
            config=self._config.detector_config,
            templates_dir=self._config.templates_dir,
        )

        # Episode state
        self._raw_prev_obs: Optional[dict[str, np.ndarray]] = None
        self._current_mask: np.ndarray = np.ones(
            _ACTION_SPACE_SIZE, dtype=np.bool_
        )
        self._step_count = 0
        self._episode_reward = 0.0
        self._cards_played = 0
        self._card_costs_played: list[float] = []  # elixir costs for metrics
        self._noop_count = 0
        self._last_card_names: list[str] = ["", "", "", ""]  # from previous perception
        self._last_elixir: int = 5  # from previous perception
        self._identical_frame_count = 0
        self._prev_frame_hash: Optional[int] = None

        # Frame stacking history (stores single-frame raw obs dicts)
        self._obs_history: deque = deque(maxlen=n)

        # Visualization
        self._visualizer = None
        if self._config.visualize:
            if self._config.vis_backend == "cv":
                from src.ppo.cv_visualizer import CVVisualizer, CVVisConfig
                vis_cfg = CVVisConfig(
                    record=self._config.vis_record,
                    output_path=self._config.vis_output_path or "vis_output.mp4",
                    show_window=self._config.vis_show_window,
                    game_frame_width=self._game_w,
                    game_frame_height=self._game_h,
                )
                self._visualizer = CVVisualizer(vis_cfg)
            else:
                from src.ppo.obs_visualizer import ObsVisualizer
                self._visualizer = ObsVisualizer(
                    save_dir=self._config.vis_save_dir,
                )

        if self._config.verbose:
            aspect = gw / max(gh, 1)
            orient = "portrait" if aspect < 0.7 else "LANDSCAPE — use --game-region"
            src = "manual" if self._config.game_region else "auto-detected"
            print(f"[Env] Initialized. Game bounds: ({gx},{gy}) {gw}x{gh} "
                  f"(aspect={aspect:.2f}, {orient}, {src})")
            print(f"[Env] Perception: {self._perception.perception_active}")
            print(f"[Env] Dry run: {self._config.dry_run}")
            print(f"[Env] Frame stacking: {n} frames")
            if self._visualizer:
                print(f"[Env] Visualization ON (save_dir={self._config.vis_save_dir or 'none'})")

    # ------------------------------------------------------------------
    # Frame stacking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_obs() -> dict[str, np.ndarray]:
        """Return a zero-filled single-frame observation for padding."""
        return {
            "arena": np.zeros((_GRID_ROWS, _GRID_COLS, _NUM_ARENA_CHANNELS), dtype=np.float32),
            "vector": np.zeros((_NUM_VECTOR_FEATURES,), dtype=np.float32),
        }

    def _stack_obs(self) -> dict[str, np.ndarray]:
        """Concatenate the obs history deque into a frame-stacked observation."""
        arenas = [o["arena"] for o in self._obs_history]
        vectors = [o["vector"] for o in self._obs_history]
        return {
            "arena": np.concatenate(arenas, axis=-1),   # (32, 18, 6*n)
            "vector": np.concatenate(vectors, axis=-1),  # (23*n,)
        }

    # ------------------------------------------------------------------
    # Window focus
    # ------------------------------------------------------------------

    def _is_game_focused(self) -> bool:
        """Check if the game window is the foreground window.

        mss captures screen pixels (not window content), so if another
        window covers the game, captured frames will be garbage. This
        method detects that situation.

        Returns True if focused or if focus cannot be checked.
        """
        hwnd = self._capture.get_window_hwnd()
        if hwnd is None:
            return True  # Can't check, assume focused
        try:
            import ctypes
            return ctypes.windll.user32.GetForegroundWindow() == hwnd
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_game_bounds(
        frame: np.ndarray, threshold: int = 15,
    ) -> tuple[int, int, int, int]:
        """Detect game content within frame (exclude pillarbox bars).

        Two-pass approach:
        1. Strip pure-black bars (pixels < threshold)
        2. If result is landscape (wider than portrait), find the portrait
           game region by detecting the column band with highest variance.
           This handles Google Play Games windows where the sidebar UI is
           dark gray (not black) so pass 1 doesn't catch it.

        Returns:
            (x_offset, y_offset, width, height) of the game region.
        """
        fh, fw = frame.shape[:2]

        # Pass 1: strip black bars
        col_max = frame.max(axis=(0, 2))
        row_max = frame.max(axis=(1, 2))
        non_black_cols = np.where(col_max > threshold)[0]
        non_black_rows = np.where(row_max > threshold)[0]
        if len(non_black_cols) == 0 or len(non_black_rows) == 0:
            return 0, 0, fw, fh
        gx = int(non_black_cols[0])
        gw = int(non_black_cols[-1]) + 1 - gx
        gy = int(non_black_rows[0])
        gh = int(non_black_rows[-1]) + 1 - gy

        # Pass 2: if result is landscape, find portrait game region
        # Portrait games are ~9:16 ratio (0.5625). If aspect > 0.7, the
        # detected region includes non-game UI (e.g. GPG sidebar).
        aspect = gw / max(gh, 1)
        if aspect > 0.7:
            # Per-column variance: game columns have varied content
            # (troops, cards, effects), sidebar columns are uniform
            region = frame[gy:gy + gh, gx:gx + gw]
            gray = region.mean(axis=2)  # (H, W) grayscale
            col_var = np.var(gray, axis=0)  # (W,) variance per column

            # Find contiguous band of high-variance columns
            var_thresh = np.percentile(col_var, 60)
            high_var = col_var > var_thresh

            # Longest contiguous run = game region
            best_start, best_len = 0, 0
            run_start, run_len = 0, 0
            for i, v in enumerate(high_var):
                if v:
                    if run_len == 0:
                        run_start = i
                    run_len += 1
                else:
                    if run_len > best_len:
                        best_start, best_len = run_start, run_len
                    run_len = 0
            if run_len > best_len:
                best_start, best_len = run_start, run_len

            if best_len > 100:  # minimum viable game width
                gx = gx + best_start
                gw = best_len

        return gx, gy, gw, gh

    def _crop_game_region(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to game content region."""
        return frame[
            self._game_y_offset:self._game_y_offset + self._game_h,
            self._game_x_offset:self._game_x_offset + self._game_w,
        ]

    def _obs_to_numpy(self, obs_dict: dict) -> dict[str, np.ndarray]:
        """Convert torch tensor obs to numpy for reward computation."""
        arena = obs_dict["arena"]
        vector = obs_dict["vector"]
        if hasattr(arena, "numpy"):
            arena = arena.numpy()
        if hasattr(vector, "numpy"):
            vector = vector.numpy()
        # Squeeze batch dim if present
        if arena.ndim == 4:
            arena = arena[0]
        if vector.ndim == 2:
            vector = vector[0]
        return {"arena": arena, "vector": vector}

    def _frame_hash(self, frame: np.ndarray) -> int:
        """Quick hash for identical-frame detection."""
        return hash(frame[::10, ::10, 0].tobytes())

    def _check_manual_stop(self) -> tuple[bool, Optional[str], int, int]:
        """Non-blocking check if operator pressed Enter to end the episode.

        Uses msvcrt.kbhit() (Windows) to detect keypress without blocking
        the step loop. When Enter is pressed, prompts for game results.

        Returns:
            (should_stop, outcome, enemy_crowns, ally_crowns_lost)
        """
        if not msvcrt.kbhit():
            return False, None, 0, 0
        key = msvcrt.getwch()
        if key != "\r":  # Not Enter
            return False, None, 0, 0

        # Flush any remaining buffered input
        while msvcrt.kbhit():
            msvcrt.getwch()

        print("\n[Env] Manual stop. Enter game results:")
        outcome_key = input("  Outcome (w=win, l=loss, d=draw): ").strip().lower()
        outcome_map = {"w": "win", "l": "loss", "d": "draw"}
        outcome = outcome_map.get(outcome_key)
        if outcome is None:
            print(f"  Unknown outcome '{outcome_key}', skipping terminal reward.")

        try:
            enemy_crowns = int(input("  Crowns scored (0-3): ").strip())
            enemy_crowns = max(0, min(3, enemy_crowns))
        except ValueError:
            enemy_crowns = 0

        try:
            ally_crowns_lost = int(input("  Towers lost (0-3): ").strip())
            ally_crowns_lost = max(0, min(3, ally_crowns_lost))
        except ValueError:
            ally_crowns_lost = 0

        if self._config.verbose:
            print(f"  -> outcome={outcome}, crowns_scored={enemy_crowns}, "
                  f"towers_lost={ally_crowns_lost}")

        return True, outcome, enemy_crowns, ally_crowns_lost

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Wait for a game to start and return the initial observation.

        Blocks until GamePhaseDetector detects IN_GAME phase.
        The operator should click "Battle" in Clash Royale.
        """
        super().reset(seed=seed, options=options)

        # Pause between episodes: wait for operator to press Enter
        # Skip on the very first episode (step_count == 0 means no game played yet)
        if self._config.pause_between_episodes and self._step_count > 0:
            print("\n[Env] Episode ended. Queue the next match, then press Enter to continue...")
            print("[Env] Type 'stop' + Enter to save and end training.")
            try:
                user_input = input()
                if user_input.strip().lower() == "stop":
                    raise KeyboardInterrupt("User requested stop")
            except EOFError:
                pass  # SB3 auto-reset — no interactive stdin available

        # Reset episode state
        self._reward_computer.reset()
        self._game_detector.reset()
        self._step_count = 0
        self._episode_reward = 0.0
        self._cards_played = 0
        self._card_costs_played = []
        self._noop_count = 0
        self._identical_frame_count = 0
        self._prev_frame_hash = None

        # Fill frame history with zero-padded observations
        self._obs_history.clear()
        for _ in range(self._config.n_frames - 1):
            self._obs_history.append(self._zero_obs())

        if self._config.verbose:
            print("[Env] Waiting for game start... Press Battle in Clash Royale.")
            print("[Env] Keep game window visible (don't alt-tab over it).")

        # Focus-aware capture: only feed frames to detector when game is focused.
        # When unfocused, return a black frame so the detector stays in UNKNOWN/LOADING
        # instead of falsely triggering IN_GAME or END_SCREEN from non-game pixels.
        _black_frame = None

        def _focused_capture():
            nonlocal _black_frame
            frame = self._crop_game_region(self._capture.capture())
            if not self._is_game_focused():
                if _black_frame is None or _black_frame.shape != frame.shape:
                    _black_frame = np.zeros_like(frame)
                return _black_frame
            return frame

        started = self._game_detector.wait_for_game_start(
            capture_fn=_focused_capture,
            timeout=self._config.game_start_timeout,
        )
        if not started:
            if self._config.verbose:
                print("[Env] WARNING: Game start timeout. Proceeding anyway.")

        # Wait for the game state to stabilize after detection.
        # This prevents capturing loading/transition frames that would cause
        # tower count anomalies on the first few steps.
        time.sleep(2.0)

        # Capture initial frame and run perception (wait for focus if needed)
        frame = self._capture.capture()
        frame = self._crop_game_region(frame)
        perception_result = self._perception.process_frame(frame)

        obs_np = self._obs_to_numpy(perception_result["obs"])
        self._raw_prev_obs = obs_np  # single-frame obs for reward computation
        self._obs_history.append(obs_np)

        # Update action mask
        mask = perception_result["mask"]
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        if mask.ndim == 2:
            mask = mask[0]
        self._current_mask = mask.astype(np.bool_)

        stacked_obs = self._stack_obs()

        # Store card names and elixir for action logging
        card_names = perception_result.get("card_names", [])
        self._last_card_names = card_names if len(card_names) == 4 else ["", "", "", ""]
        self._last_elixir = perception_result.get("elixir", 5)
        valid_card_actions = int(self._current_mask[:_ACTION_SPACE_SIZE - 1].sum())
        info = {
            "perception_active": perception_result.get("perception_active", False),
            "step": 0,
        }

        if self._config.verbose:
            print(f"[Env] Game started. Initial cards: {card_names}, elixir={self._last_elixir}")
            print(f"[Env] Initial mask: {int(self._current_mask.sum())} valid "
                  f"actions ({valid_card_actions} card placements)")

        return stacked_obs, info

    def step(
        self, action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Execute one step: capture -> detect phase -> action -> perceive -> reward.

        Frame capture and phase detection happen BEFORE action execution so that
        the end screen is detected before the model can click on it (which would
        dismiss the results overlay and start a new game).

        Args:
            action: Integer action index (0-2304).

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        terminated = False
        truncated = False
        truncation_reason = ""
        outcome = None
        exec_result = {"executed": False, "reason": "skipped"}

        # Manual stop: operator presses Enter to end the episode
        manual_crown_reward = 0.0
        should_stop, manual_outcome, enemy_crowns, ally_crowns_lost = (
            self._check_manual_stop()
        )
        if should_stop:
            terminated = True
            outcome = manual_outcome
            manual_crown_reward = self._reward_computer.compute_manual_crowns(
                enemy_crowns, ally_crowns_lost,
            )

        # Layer 1: Max episode length
        if self._step_count >= self._config.max_episode_steps:
            truncated = True
            truncation_reason = "max_steps"

        # 1. Capture frame FIRST (before any action)
        frame = self._capture.capture()
        frame = self._crop_game_region(frame)

        # Focus guard: if the game window is not the foreground window,
        # mss captured non-game pixels (e.g. VS Code). Skip processing
        # and reuse the previous observation to avoid corrupting the
        # episode with garbage frames.
        game_focused = self._is_game_focused()
        if not game_focused and not terminated:
            self._unfocused_count = getattr(self, "_unfocused_count", 0) + 1
            if self._unfocused_count == 1 and self._config.verbose:
                print(f"[Env] WARNING step {self._step_count}: Game window lost "
                      f"focus — skipping frame capture until refocused.")

            # Reuse previous stacked obs, give survival-only reward
            stacked_obs = self._stack_obs()
            cfg = self._config.reward_config
            reward = cfg.survival_bonus * cfg.reward_scale
            self._episode_reward += reward

            info = {
                "step": self._step_count,
                "action": action,
                "action_executed": False,
                "action_reason": "window_not_focused",
                "phase": "in_game",
                "perception_active": False,
                "episode_reward": self._episode_reward,
                "cards_played": self._cards_played,
                "anomaly_detected": False,
                "truncation_reason": truncation_reason,
            }
            return stacked_obs, reward, terminated, truncated, info
        else:
            self._unfocused_count = 0

        # 2. Check for identical frames (freeze detection)
        fhash = self._frame_hash(frame)
        if fhash == self._prev_frame_hash:
            self._identical_frame_count += 1
        else:
            self._identical_frame_count = 0
        self._prev_frame_hash = fhash

        if self._identical_frame_count >= self._config.identical_frame_limit:
            truncated = True
            truncation_reason = truncation_reason or "identical_frames"
            if self._config.verbose:
                print(f"[Env] Truncated: {self._identical_frame_count} identical frames.")

        # 3. Check game phase BEFORE executing action
        # This prevents the model from clicking on the end screen and dismissing it
        phase = self._game_detector.detect_phase(frame)
        if phase == Phase.END_SCREEN and self._step_count >= self._config.min_episode_steps:
            terminated = True
            # Wait briefly for full results screen
            time.sleep(1.0)
            end_frame = self._crop_game_region(self._capture.capture())
            outcome = self._game_detector.detect_outcome(end_frame)
            if self._config.verbose:
                print(f"[Env] Game ended. Outcome: {outcome}")
        elif phase == Phase.END_SCREEN and self._config.verbose:
            print(f"[Env] Ignoring END_SCREEN at step {self._step_count} "
                  f"(min_episode_steps={self._config.min_episode_steps})")

        # 4. Execute action ONLY if game is still in progress
        #    Also suppress if END_SCREEN is the raw candidate (debounce not yet
        #    confirmed) — prevents clicking "Play Again" during the 3-frame window
        candidate_end = self._game_detector.candidate_phase == Phase.END_SCREEN
        action_card_name = ""
        action_card_id = -1
        action_row = -1
        action_col = -1
        if not terminated and not candidate_end:
            exec_result = self._dispatcher.execute(action, logit_score=0.0)
            if action != _NOOP_ACTION and exec_result.get("executed", False):
                self._cards_played += 1
                # Decode action for logging
                action_card_id = action // 576
                cell = action % 576
                action_row = cell // 18
                action_col = cell % 18
                if 0 <= action_card_id < len(self._last_card_names):
                    action_card_name = self._last_card_names[action_card_id]
                cost = 0
                if action_card_name:
                    cost = CARD_ELIXIR_COST.get(action_card_name, 0)
                if self._config.verbose:
                    print(f"[Env] Step {self._step_count}: Played "
                          f"{action_card_name or '?'} (slot {action_card_id}, "
                          f"cost {cost}) at row={action_row} col={action_col} "
                          f"| elixir={self._last_elixir}")
            elif action == _NOOP_ACTION:
                self._noop_count += 1
                if self._config.verbose and self._step_count % 50 == 0:
                    print(f"[Env] Step {self._step_count}: NOOP "
                          f"| elixir={self._last_elixir}")
            elif action != _NOOP_ACTION:
                # Card action attempted but not executed
                action_card_id = action // 576
                if 0 <= action_card_id < len(self._last_card_names):
                    action_card_name = self._last_card_names[action_card_id]
                if self._config.verbose:
                    print(f"[Env] Step {self._step_count}: {action_card_name or '?'} "
                          f"NOT executed ({exec_result.get('reason', '?')}) "
                          f"| elixir={self._last_elixir}")

        # 5. Run perception (produces single-frame obs)
        perception_result = self._perception.process_frame(frame)
        curr_raw_obs = self._obs_to_numpy(perception_result["obs"])

        # Update card names and elixir for next step's action logging
        new_card_names = perception_result.get("card_names", [])
        self._last_card_names = new_card_names if len(new_card_names) == 4 else self._last_card_names
        self._last_elixir = perception_result.get("elixir", self._last_elixir)

        # Update action mask
        mask = perception_result["mask"]
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        if mask.ndim == 2:
            mask = mask[0]
        self._current_mask = mask.astype(np.bool_)

        # Mask validation: warn if all card actions are masked
        valid_card_actions = int(self._current_mask[:_ACTION_SPACE_SIZE - 1].sum())
        if valid_card_actions == 0 and self._config.verbose:
            card_names = perception_result.get("card_names", [])
            print(f"[Env] WARNING step {self._step_count}: All card actions "
                  f"masked! Only NOOP available. cards={card_names}")

        # 5b. Update visualization (uses single-frame obs)
        if self._visualizer is not None:
            if self._config.vis_backend == "cv":
                self._visualizer.update(
                    game_frame=frame,
                    obs=curr_raw_obs,
                    step=self._step_count,
                    info={
                        "phase": phase.value,
                        "episode_reward": self._episode_reward,
                        "cards_played": self._cards_played,
                    },
                )
            else:
                self._visualizer.update(curr_raw_obs, self._step_count)

        # 6. Compute reward from SINGLE-FRAME obs (not stacked)
        reward = 0.0
        if self._raw_prev_obs is not None:
            reward = self._reward_computer.compute(
                self._raw_prev_obs, curr_raw_obs,
                terminal_outcome=outcome, action=action,
            )
        # Add manual crown reward (non-zero only when operator manually stopped)
        reward += manual_crown_reward
        self._episode_reward += reward
        self._raw_prev_obs = curr_raw_obs

        # Update frame stacking history and build stacked observation
        self._obs_history.append(curr_raw_obs)
        stacked_obs = self._stack_obs()

        # Layer 2: Observation anomaly detection (new game started mid-episode)
        anomaly_detected = self._reward_computer.new_game_detected
        if anomaly_detected and not terminated:
            terminated = True
            if self._config.verbose:
                print("[Env] Anomaly: tower counts jumped up — new game detected. Ending episode.")

        # Track card cost from observation vector (card costs at indices 19-22)
        if action != _NOOP_ACTION and exec_result.get("executed", False):
            card_id = action // 576  # GRID_CELLS
            if 0 <= card_id < 4:
                raw_vec = curr_raw_obs["vector"]
                if raw_vec.ndim == 2:
                    raw_vec = raw_vec[0]
                card_cost = float(raw_vec[19 + card_id]) * 10.0  # denormalize
                self._card_costs_played.append(card_cost)

        # Compute episode-level metrics
        noop_ratio = self._noop_count / max(self._step_count, 1)
        card_cost_avg = (
            sum(self._card_costs_played) / len(self._card_costs_played)
            if self._card_costs_played else 0.0
        )

        # 7. Build info dict
        info = {
            "step": self._step_count,
            "action": action,
            "action_executed": exec_result.get("executed", False),
            "action_reason": exec_result.get("reason", ""),
            "action_card_name": action_card_name,
            "action_card_id": action_card_id,
            "action_row": action_row,
            "action_col": action_col,
            "elixir": self._last_elixir,
            "phase": phase.value,
            "perception_active": perception_result.get("perception_active", False),
            "episode_reward": self._episode_reward,
            "cards_played": self._cards_played,
            "card_cost_avg": round(card_cost_avg, 2),
            "noop_ratio": round(noop_ratio, 3),
            "anomaly_detected": anomaly_detected,
            "truncation_reason": truncation_reason,
        }
        if outcome is not None:
            info["outcome"] = outcome
        if terminated:
            info["episode_length"] = self._step_count

        return stacked_obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return valid action mask for MaskablePPO.

        Returns:
            Boolean array of shape (2305,). True = action is valid.
        """
        return self._current_mask

    def close(self) -> None:
        """Release resources."""
        if self._visualizer is not None:
            self._visualizer.close()
        self._capture.release()

    def render(self) -> None:
        """No-op render (game is already visible on screen)."""
        pass
