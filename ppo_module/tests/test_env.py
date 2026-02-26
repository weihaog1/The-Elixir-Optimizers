"""Tests for ClashRoyaleEnv (mock-based, no live game required)."""

import numpy as np
import pytest

from src.ppo.game_detector import GamePhaseDetector, Phase


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
        # Create frame with bright card bar and varied arena
        frame = np.zeros((960, 540, 3), dtype=np.uint8)
        # Arena region (y=50-750): varied colors
        frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        # Card bar region (y=770-920): bright
        frame[770:920, :, :] = 150

        # Need multiple frames for phase stability
        for _ in range(5):
            phase = detector.detect_phase(frame)
        assert phase == Phase.IN_GAME

    def test_dark_frame_detected_as_loading(self):
        """A fully dark frame should be LOADING."""
        detector = GamePhaseDetector()
        frame = np.zeros((960, 540, 3), dtype=np.uint8)
        # Very dark everywhere
        frame[:, :, :] = 5

        for _ in range(5):
            phase = detector.detect_phase(frame)
        assert phase == Phase.LOADING

    def test_detect_outcome_returns_none_for_ambiguous(self):
        """When neither victory nor defeat is clear, return None."""
        detector = GamePhaseDetector()
        # Uniform gray frame (ambiguous)
        frame = np.full((960, 540, 3), 128, dtype=np.uint8)
        result = detector.detect_outcome(frame)
        # Should be None or a string, not crash
        assert result is None or isinstance(result, str)


class TestCandidatePhase:
    """Test candidate_phase property for action suppression."""

    def test_candidate_phase_returns_raw_candidate(self):
        """candidate_phase should reflect the raw (unconfirmed) candidate."""
        detector = GamePhaseDetector()
        # Initially UNKNOWN
        assert detector.candidate_phase == Phase.UNKNOWN

        # Feed 1 IN_GAME frame — candidate updates immediately
        in_game_frame = np.zeros((960, 540, 3), dtype=np.uint8)
        in_game_frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        in_game_frame[770:920, :, :] = 150
        detector.detect_phase(in_game_frame)
        assert detector.candidate_phase == Phase.IN_GAME

    def test_candidate_end_screen_before_confirmed(self):
        """candidate_phase should be END_SCREEN even while confirmed is still IN_GAME."""
        detector = GamePhaseDetector()
        # Establish IN_GAME (confirmed)
        in_game_frame = np.zeros((960, 540, 3), dtype=np.uint8)
        in_game_frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        in_game_frame[770:920, :, :] = 150
        for _ in range(5):
            detector.detect_phase(in_game_frame)
        assert detector._confirmed_phase == Phase.IN_GAME

        # Feed 1 END_SCREEN frame — card bar dim + arena uniform
        end_frame = np.full((960, 540, 3), 100, dtype=np.uint8)
        end_frame[770:920, :, :] = 10  # dim card bar
        end_frame[50:750, :, :] = 90   # uniform arena (low variance)
        detector.detect_phase(end_frame)

        # candidate should be END_SCREEN, but confirmed still IN_GAME
        assert detector.candidate_phase == Phase.END_SCREEN
        assert detector._confirmed_phase == Phase.IN_GAME


class TestPhaseStability:
    """Test phase detection stability fix (Layer 3)."""

    def test_confirmed_phase_does_not_change_on_first_new_frame(self):
        """A single frame of a new phase should NOT change confirmed output."""
        detector = GamePhaseDetector()
        # Feed 5 IN_GAME frames to establish confirmed phase
        in_game_frame = np.zeros((960, 540, 3), dtype=np.uint8)
        in_game_frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        in_game_frame[770:920, :, :] = 150

        for _ in range(5):
            detector.detect_phase(in_game_frame)
        assert detector._confirmed_phase == Phase.IN_GAME

        # Feed 1 dark/loading frame — confirmed should still be IN_GAME
        loading_frame = np.full((960, 540, 3), 5, dtype=np.uint8)
        phase = detector.detect_phase(loading_frame)
        assert phase == Phase.IN_GAME  # Debounced — not changed yet

    def test_in_game_to_loading_triggers_end_screen(self):
        """IN_GAME -> LOADING transition should return END_SCREEN."""
        detector = GamePhaseDetector()
        # Establish IN_GAME
        in_game_frame = np.zeros((960, 540, 3), dtype=np.uint8)
        in_game_frame[50:750, :, :] = np.random.randint(30, 200, (700, 540, 3), dtype=np.uint8)
        in_game_frame[770:920, :, :] = 150

        for _ in range(5):
            detector.detect_phase(in_game_frame)
        assert detector._confirmed_phase == Phase.IN_GAME
        assert detector._was_ever_in_game is True

        # Feed enough LOADING frames to trigger transition
        loading_frame = np.full((960, 540, 3), 5, dtype=np.uint8)
        for _ in range(5):
            phase = detector.detect_phase(loading_frame)

        # Should detect as END_SCREEN (missed end screen transition)
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


class TestRewardIntegration:
    """Test reward computation with realistic observation sequences."""

    def test_full_game_reward_sequence(self):
        """Simulate a short game with crown events."""
        from src.ppo.reward import RewardComputer

        rc = RewardComputer()
        rc.reset()

        rewards = []

        # Step 1: Normal play (no events)
        obs1 = _make_obs(enemy_towers=1.0, ally_towers=1.0)
        obs2 = _make_obs(enemy_towers=1.0, ally_towers=1.0)
        rewards.append(rc.compute(obs1, obs2))

        # Step 2: We score a crown
        obs3 = _make_obs(enemy_towers=2.0 / 3.0, ally_towers=1.0)
        rewards.append(rc.compute(obs2, obs3))

        # Step 3: They score a crown
        obs4 = _make_obs(enemy_towers=2.0 / 3.0, ally_towers=2.0 / 3.0)
        rewards.append(rc.compute(obs3, obs4))

        # Step 4: We win
        obs5 = _make_obs(enemy_towers=2.0 / 3.0, ally_towers=2.0 / 3.0)
        rewards.append(rc.compute(obs4, obs5, terminal_outcome="win"))

        # Check reward progression
        assert rewards[0] == pytest.approx(0.02, abs=0.01)  # survival only
        assert rewards[1] > 5  # crown reward
        assert rewards[2] < -5  # crown penalty
        assert rewards[3] > 25  # win terminal


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
