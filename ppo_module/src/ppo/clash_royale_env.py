"""Gymnasium environment wrapper for live Clash Royale gameplay.

Wraps screen capture, perception, action execution, reward computation,
and game phase detection into a standard Gymnasium interface compatible
with sb3_contrib.MaskablePPO.

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

import time
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

    # Perception
    use_perception: bool = True
    detector_model_paths: list[str] = field(default_factory=lambda: [
        "models/best_yolov8s_50epochs_fixed_pregen_set.pt",
    ])
    card_classifier_path: str = "models/card_classifier.pt"

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

    # Logging
    verbose: bool = True


class ClashRoyaleEnv(gymnasium.Env):
    """Gymnasium environment wrapping live Clash Royale gameplay.

    Observation space: Dict(arena=Box(32,18,6), vector=Box(23,))
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

        # Define spaces
        self.observation_space = spaces.Dict({
            "arena": spaces.Box(
                low=-1.0, high=10.0,
                shape=(_GRID_ROWS, _GRID_COLS, _NUM_ARENA_CHANNELS),
                dtype=np.float32,
            ),
            "vector": spaces.Box(
                low=0.0, high=1.0,
                shape=(_NUM_VECTOR_FEATURES,),
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
            card_classifier_path=self._config.card_classifier_path,
            dry_run=self._config.dry_run,
            action_cooldown=0.0,  # PPO controls timing, not cooldown
            max_actions_per_minute=999,  # No rate limiting for PPO
            confidence_threshold=0.0,
            noop_frames_after_play=0,
        )

        # Initialize components
        self._capture = GameCapture(self._live_config)

        # Detect game bounds from probe frame
        probe = self._capture.capture()
        gx, gy, gw, gh = self._detect_game_bounds(probe)
        self._game_x_offset = gx
        self._game_y_offset = gy
        self._game_w = gw
        self._game_h = gh

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
        self._prev_obs: Optional[dict[str, np.ndarray]] = None
        self._current_mask: np.ndarray = np.ones(
            _ACTION_SPACE_SIZE, dtype=np.bool_
        )
        self._step_count = 0
        self._episode_reward = 0.0
        self._cards_played = 0
        self._identical_frame_count = 0
        self._prev_frame_hash: Optional[int] = None

        if self._config.verbose:
            print(f"[Env] Initialized. Game bounds: ({gx},{gy}) {gw}x{gh}")
            print(f"[Env] Perception: {self._perception.perception_active}")
            print(f"[Env] Dry run: {self._config.dry_run}")

    @staticmethod
    def _detect_game_bounds(
        frame: np.ndarray, threshold: int = 15,
    ) -> tuple[int, int, int, int]:
        """Detect game content within frame (exclude black pillarbox bars)."""
        fh, fw = frame.shape[:2]
        col_max = frame.max(axis=(0, 2))
        row_max = frame.max(axis=(1, 2))
        non_black_cols = np.where(col_max > threshold)[0]
        non_black_rows = np.where(row_max > threshold)[0]
        if len(non_black_cols) == 0 or len(non_black_rows) == 0:
            return 0, 0, fw, fh
        x_start = int(non_black_cols[0])
        x_end = int(non_black_cols[-1]) + 1
        y_start = int(non_black_rows[0])
        y_end = int(non_black_rows[-1]) + 1
        return x_start, y_start, x_end - x_start, y_end - y_start

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

        # Reset episode state
        self._reward_computer.reset()
        self._game_detector.reset()
        self._step_count = 0
        self._episode_reward = 0.0
        self._cards_played = 0
        self._identical_frame_count = 0
        self._prev_frame_hash = None

        if self._config.verbose:
            print("[Env] Waiting for game start... Press Battle in Clash Royale.")

        # Wait for game to start (use cropped frames so detector regions align)
        started = self._game_detector.wait_for_game_start(
            capture_fn=lambda: self._crop_game_region(self._capture.capture()),
            timeout=self._config.game_start_timeout,
        )
        if not started:
            if self._config.verbose:
                print("[Env] WARNING: Game start timeout. Proceeding anyway.")

        # Capture initial frame and run perception
        frame = self._capture.capture()
        frame = self._crop_game_region(frame)
        perception_result = self._perception.process_frame(frame)

        obs_np = self._obs_to_numpy(perception_result["obs"])
        self._prev_obs = obs_np

        # Update action mask
        mask = perception_result["mask"]
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        if mask.ndim == 2:
            mask = mask[0]
        self._current_mask = mask.astype(np.bool_)

        info = {
            "perception_active": perception_result.get("perception_active", False),
            "step": 0,
        }

        if self._config.verbose:
            print("[Env] Game started. First observation captured.")

        return obs_np, info

    def step(
        self, action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Execute one step: action -> capture -> perceive -> reward.

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

        # Layer 1: Max episode length
        if self._step_count >= self._config.max_episode_steps:
            truncated = True
            truncation_reason = "max_steps"

        # 1. Execute action
        exec_result = self._dispatcher.execute(action, logit_score=0.0)
        if action != _NOOP_ACTION and exec_result.get("executed", False):
            self._cards_played += 1

        # 2. Capture next frame and crop to game region
        frame = self._capture.capture()
        frame = self._crop_game_region(frame)

        # 3. Check for identical frames (freeze detection)
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

        # 4. Check game phase (use cropped frame so detector regions align)
        # Only trust END_SCREEN after min_episode_steps to avoid false positives
        # from spell effects, overlays, or animations that dim the card bar
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

        # 5. Run perception
        perception_result = self._perception.process_frame(frame)
        curr_obs = self._obs_to_numpy(perception_result["obs"])

        # Update action mask
        mask = perception_result["mask"]
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        if mask.ndim == 2:
            mask = mask[0]
        self._current_mask = mask.astype(np.bool_)

        # 6. Compute reward
        reward = 0.0
        if self._prev_obs is not None:
            reward = self._reward_computer.compute(
                self._prev_obs, curr_obs, terminal_outcome=outcome,
            )
        self._episode_reward += reward
        self._prev_obs = curr_obs

        # Layer 2: Observation anomaly detection (new game started mid-episode)
        anomaly_detected = self._reward_computer.new_game_detected
        if anomaly_detected and not terminated:
            terminated = True
            if self._config.verbose:
                print("[Env] Anomaly: tower counts jumped up — new game detected. Ending episode.")

        # 7. Build info dict
        info = {
            "step": self._step_count,
            "action": action,
            "action_executed": exec_result.get("executed", False),
            "action_reason": exec_result.get("reason", ""),
            "phase": phase.value,
            "perception_active": perception_result.get("perception_active", False),
            "episode_reward": self._episode_reward,
            "cards_played": self._cards_played,
            "anomaly_detected": anomaly_detected,
            "truncation_reason": truncation_reason,
        }
        if outcome is not None:
            info["outcome"] = outcome
        if terminated:
            info["episode_length"] = self._step_count

        return curr_obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return valid action mask for MaskablePPO.

        Returns:
            Boolean array of shape (2305,). True = action is valid.
        """
        return self._current_mask

    def close(self) -> None:
        """Release resources."""
        self._capture.release()

    def render(self) -> None:
        """No-op render (game is already visible on screen)."""
        pass
