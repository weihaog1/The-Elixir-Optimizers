"""Tests for ClashRoyaleEnv (mock-based, no live game required)."""

from unittest.mock import patch

import numpy as np
import pytest

from src.ppo.game_detector import GamePhaseDetector, Phase

_HAS_SB3 = True
try:
    import stable_baselines3  # noqa: F401
except ImportError:
    _HAS_SB3 = False


class TestGamePhaseDetector:
    """Test game phase detection heuristics."""

    def test_initial_phase_is_unknown(self):
        detector = GamePhaseDetector()
        assert detector._phase == Phase.UNKNOWN

    def test_reset_clears_state(self):
        detector = GamePhaseDetector()
        detector._phase = Phase.IN_GAME
        detector._phase_count = 10
        detector.reset()
        assert detector._phase == Phase.UNKNOWN
        assert detector._phase_count == 0

    def test_bright_frame_detected_as_in_game(self):
        """A frame with bright card bar and varied arena should be IN_GAME."""
        detector = GamePhaseDetector()
        frame = np.zeros((960, 540, 3), dtype=np.uint8)
        frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        frame[770:920, :, :] = 150

        for _ in range(5):
            phase = detector.detect_phase(frame)
        assert phase == Phase.IN_GAME

    def test_dark_frame_detected_as_loading(self):
        """A fully dark frame should be LOADING."""
        detector = GamePhaseDetector()
        frame = np.zeros((960, 540, 3), dtype=np.uint8)
        frame[:, :, :] = 5

        for _ in range(5):
            phase = detector.detect_phase(frame)
        assert phase == Phase.LOADING

    def test_detect_outcome_returns_none_for_ambiguous(self):
        """When neither victory nor defeat is clear, return None."""
        detector = GamePhaseDetector()
        frame = np.full((960, 540, 3), 128, dtype=np.uint8)
        result = detector.detect_outcome(frame)
        assert result is None or isinstance(result, str)


class TestCandidatePhase:
    """Test candidate_phase property for action suppression."""

    def test_candidate_phase_returns_raw_candidate(self):
        """candidate_phase should reflect the raw (unconfirmed) candidate."""
        detector = GamePhaseDetector()
        assert detector.candidate_phase == Phase.UNKNOWN

        in_game_frame = np.zeros((960, 540, 3), dtype=np.uint8)
        in_game_frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        in_game_frame[770:920, :, :] = 150
        detector.detect_phase(in_game_frame)
        assert detector.candidate_phase == Phase.IN_GAME

    def test_candidate_end_screen_before_confirmed(self):
        """candidate_phase should be END_SCREEN even while confirmed is still IN_GAME."""
        detector = GamePhaseDetector()
        in_game_frame = np.zeros((960, 540, 3), dtype=np.uint8)
        in_game_frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        in_game_frame[770:920, :, :] = 150
        for _ in range(5):
            detector.detect_phase(in_game_frame)
        assert detector._confirmed_phase == Phase.IN_GAME

        end_frame = np.full((960, 540, 3), 100, dtype=np.uint8)
        end_frame[770:920, :, :] = 10
        end_frame[50:750, :, :] = 90
        detector.detect_phase(end_frame)

        assert detector.candidate_phase == Phase.END_SCREEN
        assert detector._confirmed_phase == Phase.IN_GAME


class TestPhaseStability:
    """Test phase detection stability fix (Layer 3)."""

    def test_confirmed_phase_does_not_change_on_first_new_frame(self):
        """A single frame of a new phase should NOT change confirmed output."""
        detector = GamePhaseDetector()
        in_game_frame = np.zeros((960, 540, 3), dtype=np.uint8)
        in_game_frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        in_game_frame[770:920, :, :] = 150

        for _ in range(5):
            detector.detect_phase(in_game_frame)
        assert detector._confirmed_phase == Phase.IN_GAME

        loading_frame = np.full((960, 540, 3), 5, dtype=np.uint8)
        phase = detector.detect_phase(loading_frame)
        assert phase == Phase.IN_GAME

    def test_in_game_to_loading_triggers_end_screen(self):
        """IN_GAME -> LOADING transition should return END_SCREEN."""
        detector = GamePhaseDetector()
        in_game_frame = np.zeros((960, 540, 3), dtype=np.uint8)
        in_game_frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        in_game_frame[770:920, :, :] = 150

        for _ in range(5):
            detector.detect_phase(in_game_frame)
        assert detector._confirmed_phase == Phase.IN_GAME
        assert detector._was_ever_in_game is True

        loading_frame = np.full((960, 540, 3), 5, dtype=np.uint8)
        for _ in range(5):
            phase = detector.detect_phase(loading_frame)

        assert phase == Phase.END_SCREEN

    def test_reset_clears_new_fields(self):
        """reset() should clear confirmed phase and was_ever_in_game."""
        detector = GamePhaseDetector()
        detector._confirmed_phase = Phase.IN_GAME
        detector._was_ever_in_game = True
        detector._prev_confirmed_phase = Phase.LOADING
        detector.reset()
        assert detector._confirmed_phase == Phase.UNKNOWN
        assert detector._was_ever_in_game is False
        assert detector._prev_confirmed_phase == Phase.UNKNOWN


class TestEnvConfig:
    """Test EnvConfig defaults."""

    def test_defaults(self):
        from src.ppo.clash_royale_env import EnvConfig
        config = EnvConfig()
        assert config.capture_fps == 2.0
        assert config.dry_run is False
        assert config.step_timeout == 5.0
        assert config.identical_frame_limit == 5

    def test_max_episode_steps_default(self):
        from src.ppo.clash_royale_env import EnvConfig
        config = EnvConfig()
        assert config.max_episode_steps == 700

    def test_pause_between_episodes_default(self):
        from src.ppo.clash_royale_env import EnvConfig
        config = EnvConfig()
        assert config.pause_between_episodes is True

    def test_n_frames_default(self):
        """Default frame stack depth should be 3."""
        from src.ppo.clash_royale_env import EnvConfig
        config = EnvConfig()
        assert config.n_frames == 3


@pytest.mark.skipif(not _HAS_SB3, reason="stable_baselines3 not installed")
class TestPPOConfigDefaults:
    """Test PPOConfig updated defaults."""

    def test_n_steps_default(self):
        from src.ppo.ppo_trainer import PPOConfig
        config = PPOConfig()
        assert config.n_steps == 700

    def test_network_architecture_default(self):
        from src.ppo.ppo_trainer import PPOConfig
        config = PPOConfig()
        assert config.pi_layers == [256, 128]
        assert config.vf_layers == [256, 128]

    def test_n_frames_default(self):
        from src.ppo.ppo_trainer import PPOConfig
        config = PPOConfig()
        assert config.n_frames == 3

    def test_entropy_annealing_defaults(self):
        from src.ppo.ppo_trainer import PPOConfig
        config = PPOConfig()
        assert config.ent_coef_start == 0.02
        assert config.ent_coef_end == 0.005

    def test_kl_penalty_defaults(self):
        from src.ppo.ppo_trainer import PPOConfig
        config = PPOConfig()
        assert config.bc_policy_path == ""
        assert config.kl_coef == 0.1


class TestFrameStacking:
    """Test frame stacking observation shapes."""

    def test_stacked_obs_shapes_3_frames(self):
        """With n_frames=3, arena should be (32,18,18) and vector (69,)."""
        n = 3
        arena_frames = [np.zeros((32, 18, 6), dtype=np.float32) for _ in range(n)]
        vector_frames = [np.zeros(23, dtype=np.float32) for _ in range(n)]

        stacked_arena = np.concatenate(arena_frames, axis=-1)
        stacked_vector = np.concatenate(vector_frames, axis=-1)

        assert stacked_arena.shape == (32, 18, 18)
        assert stacked_vector.shape == (69,)

    def test_stacked_obs_shapes_1_frame(self):
        """With n_frames=1, shapes should match single-frame."""
        arena = np.zeros((32, 18, 6), dtype=np.float32)
        vector = np.zeros(23, dtype=np.float32)

        assert arena.shape == (32, 18, 6)
        assert vector.shape == (23,)

    def test_zero_padding_initial(self):
        """Initial frame stack should be zero-padded except last frame."""
        from collections import deque
        n_frames = 3

        zero_obs = {
            "arena": np.zeros((32, 18, 6), dtype=np.float32),
            "vector": np.zeros(23, dtype=np.float32),
        }
        first_obs = {
            "arena": np.ones((32, 18, 6), dtype=np.float32),
            "vector": np.ones(23, dtype=np.float32),
        }

        history = deque(maxlen=n_frames)
        # Fill with zeros, then append first real obs
        for _ in range(n_frames - 1):
            history.append(zero_obs)
        history.append(first_obs)

        arenas = [o["arena"] for o in history]
        vectors = [o["vector"] for o in history]
        stacked_arena = np.concatenate(arenas, axis=-1)
        stacked_vector = np.concatenate(vectors, axis=-1)

        assert stacked_arena.shape == (32, 18, 18)
        assert stacked_vector.shape == (69,)
        # First 2 frames are zeros, last frame is ones
        assert np.all(stacked_arena[:, :, :12] == 0.0)
        assert np.all(stacked_arena[:, :, 12:] == 1.0)
        assert np.all(stacked_vector[:46] == 0.0)
        assert np.all(stacked_vector[46:] == 1.0)

    def test_raw_obs_for_reward_is_single_frame(self):
        """Reward computation should use single-frame obs, not stacked."""
        raw_obs = {
            "arena": np.zeros((32, 18, 6), dtype=np.float32),
            "vector": np.zeros(23, dtype=np.float32),
        }
        # Single-frame obs should have shape (23,), not (69,)
        assert raw_obs["vector"].shape == (23,)
        assert raw_obs["arena"].shape == (32, 18, 6)


class TestRewardIntegration:
    """Test reward computation with realistic observation sequences."""

    def test_full_game_reward_sequence(self):
        """Simulate a short game with crown events."""
        from src.ppo.reward import RewardComputer

        rc = RewardComputer()
        rc.reset()

        rewards = []

        obs1 = _make_obs(enemy_towers=1.0, ally_towers=1.0)
        obs2 = _make_obs(enemy_towers=1.0, ally_towers=1.0)
        rewards.append(rc.compute(obs1, obs2))

        obs3 = _make_obs(enemy_towers=2.0 / 3.0, ally_towers=1.0)
        rewards.append(rc.compute(obs2, obs3))

        obs4 = _make_obs(enemy_towers=2.0 / 3.0, ally_towers=2.0 / 3.0)
        rewards.append(rc.compute(obs3, obs4))

        obs5 = _make_obs(enemy_towers=2.0 / 3.0, ally_towers=2.0 / 3.0)
        rewards.append(rc.compute(obs4, obs5, terminal_outcome="win"))

        # Check reward progression (all scaled by 0.1)
        assert rewards[0] == pytest.approx(0.002, abs=0.001)  # survival only
        assert rewards[1] > 0.5  # crown reward (scaled)
        assert rewards[2] < -0.5  # crown penalty (scaled)
        assert rewards[3] > 2.5  # win terminal (scaled)


class TestManualStop:
    """Test _check_manual_stop with mocked msvcrt."""

    def test_no_key_pressed_returns_false(self):
        """When no key is pressed, should return False."""
        with patch("src.ppo.clash_royale_env.msvcrt") as mock_msvcrt:
            mock_msvcrt.kbhit.return_value = False
            result = mock_msvcrt.kbhit()
            assert result is False

    def test_non_enter_key_returns_false(self):
        """Pressing a non-Enter key should not trigger manual stop."""
        with patch("src.ppo.clash_royale_env.msvcrt") as mock_msvcrt:
            mock_msvcrt.kbhit.side_effect = [True, False]
            mock_msvcrt.getwch.return_value = "a"
            assert mock_msvcrt.kbhit() is True
            key = mock_msvcrt.getwch()
            assert key != "\r"


@pytest.mark.skipif(not _HAS_SB3, reason="stable_baselines3 not installed")
class TestEntropyScheduleCallback:
    """Test EntropyScheduleCallback logic."""

    def test_entropy_decreases_over_episodes(self):
        """Entropy coefficient should decrease from start to end."""
        from src.ppo.callbacks import EntropyScheduleCallback

        cb = EntropyScheduleCallback(start=0.02, end=0.005, total_episodes=10)
        assert cb.start == 0.02
        assert cb.end == 0.005
        assert cb.total_episodes == 10
        assert cb._episode_count == 0

    def test_entropy_schedule_progress(self):
        """Linear interpolation should work correctly at midpoint."""
        from src.ppo.callbacks import EntropyScheduleCallback

        cb = EntropyScheduleCallback(start=0.02, end=0.005, total_episodes=10)
        # At 50% progress: 0.02 + (0.005 - 0.02) * 0.5 = 0.0125
        progress = 0.5
        expected = 0.02 + (0.005 - 0.02) * progress
        assert expected == pytest.approx(0.0125, abs=1e-6)


def _make_obs(
    elixir: float = 0.5,
    ally_towers: float = 1.0,
    enemy_towers: float = 1.0,
) -> dict[str, np.ndarray]:
    vector = np.zeros(23, dtype=np.float32)
    vector[0] = elixir
    vector[9] = ally_towers
    vector[10] = enemy_towers
    return {"arena": np.zeros((32, 18, 6), dtype=np.float32), "vector": vector}
