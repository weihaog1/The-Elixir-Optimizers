"""Tests for RewardComputer."""

import numpy as np
import pytest

from src.ppo.reward import RewardComputer, RewardConfig


def _make_obs(
    elixir: float = 0.5,
    ally_towers: float = 1.0,
    enemy_towers: float = 1.0,
) -> dict[str, np.ndarray]:
    """Create a minimal observation dict with specified vector values."""
    vector = np.zeros(23, dtype=np.float32)
    vector[0] = elixir  # elixir / 10
    vector[9] = ally_towers  # ally tower count / 3
    vector[10] = enemy_towers  # enemy tower count / 3
    return {"arena": np.zeros((32, 18, 6), dtype=np.float32), "vector": vector}


class TestRewardComputer:
    """Test the reward computation logic."""

    def test_survival_bonus(self):
        """Every step should give a small survival bonus."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02, abs=1e-6)

    def test_enemy_crown_reward(self):
        """Destroying an enemy tower should give +10."""
        rc = RewardComputer()
        obs1 = _make_obs(enemy_towers=1.0)  # 3/3 towers
        obs2 = _make_obs(enemy_towers=2.0 / 3.0)  # 2/3 towers
        reward = rc.compute(obs1, obs2)
        # enemy_crown_reward (10) + survival_bonus (0.02)
        assert reward == pytest.approx(10.02, abs=0.1)

    def test_ally_crown_penalty(self):
        """Losing an ally tower should give -10."""
        rc = RewardComputer()
        obs1 = _make_obs(ally_towers=1.0)  # 3/3 towers
        obs2 = _make_obs(ally_towers=2.0 / 3.0)  # 2/3 towers
        reward = rc.compute(obs1, obs2)
        # ally_crown_penalty (-10) + survival_bonus (0.02)
        assert reward == pytest.approx(-9.98, abs=0.1)

    def test_win_terminal(self):
        """Winning should give a large positive reward."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2, terminal_outcome="win")
        # win_reward (30) + survival_bonus (0.02)
        assert reward == pytest.approx(30.02, abs=0.1)

    def test_loss_terminal(self):
        """Losing should give a large negative reward."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2, terminal_outcome="loss")
        # loss_penalty (-30) + survival_bonus (0.02)
        assert reward == pytest.approx(-29.98, abs=0.1)

    def test_draw_terminal(self):
        """Drawing should give a small negative reward."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2, terminal_outcome="draw")
        # draw_penalty (-5) + survival_bonus (0.02)
        assert reward == pytest.approx(-4.98, abs=0.1)

    def test_elixir_waste_penalty(self):
        """Sitting at max elixir should penalize."""
        rc = RewardComputer()
        obs1 = _make_obs(elixir=0.5)
        obs2 = _make_obs(elixir=0.98)  # 9.8/10 = above threshold
        reward = rc.compute(obs1, obs2)
        # survival_bonus (0.02) + elixir_waste (-0.05) = -0.03
        assert reward == pytest.approx(-0.03, abs=1e-6)

    def test_no_elixir_penalty_below_threshold(self):
        """Elixir below threshold should not penalize."""
        rc = RewardComputer()
        obs1 = _make_obs(elixir=0.5)
        obs2 = _make_obs(elixir=0.8)  # 8/10 = below threshold
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02, abs=1e-6)

    def test_batched_vectors(self):
        """Should handle (1, 23) shaped vectors."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs1["vector"] = obs1["vector"].reshape(1, 23)
        obs2 = _make_obs()
        obs2["vector"] = obs2["vector"].reshape(1, 23)
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02, abs=1e-6)

    def test_custom_config(self):
        """Should respect custom reward weights."""
        cfg = RewardConfig(
            enemy_crown_reward=20.0,
            survival_bonus=0.0,
            reward_clamp=25.0,  # High enough to not clip 20.0
        )
        rc = RewardComputer(cfg)
        obs1 = _make_obs(enemy_towers=1.0)
        obs2 = _make_obs(enemy_towers=2.0 / 3.0)
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(20.0, abs=0.1)

    def test_reset_clears_state(self):
        """reset() should not cause errors on next compute()."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        rc.compute(obs1, obs2)
        rc.reset()
        # Should work fine after reset
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02, abs=1e-6)

    def test_combined_crown_and_win(self):
        """Scoring a crown and winning on same step."""
        rc = RewardComputer()
        obs1 = _make_obs(enemy_towers=1.0 / 3.0)  # 1 tower left
        obs2 = _make_obs(enemy_towers=0.0)  # 0 towers (3-crown)
        reward = rc.compute(obs1, obs2, terminal_outcome="win")
        # crown(10) clamped to 15 max, then + win(30) + survival(0.02)
        # non-terminal = 10 + 0.02 = 10.02, clamped to 10.02, then + 30 = 40.02
        assert reward == pytest.approx(40.02, abs=0.1)


class TestAnomalyDetection:
    """Test new-game anomaly detection (Layer 2)."""

    def test_tower_jump_anomaly_detected(self):
        """Both towers jumping from damaged to full = new game."""
        rc = RewardComputer()
        # First call: establish tower tracking state
        obs1 = _make_obs(ally_towers=2.0 / 3.0, enemy_towers=2.0 / 3.0)
        obs2 = _make_obs(ally_towers=2.0 / 3.0, enemy_towers=2.0 / 3.0)
        rc.compute(obs1, obs2)

        # Second call: towers jump back to 1.0 (new game)
        obs3 = _make_obs(ally_towers=1.0, enemy_towers=1.0)
        reward = rc.compute(obs2, obs3)

        assert rc.new_game_detected is True
        # Should return survival-only reward
        assert reward == pytest.approx(0.02, abs=1e-6)

    def test_single_tower_slight_increase_not_anomaly(self):
        """Small noise in one tower should not trigger anomaly."""
        rc = RewardComputer()
        obs1 = _make_obs(ally_towers=0.65, enemy_towers=0.65)
        obs2 = _make_obs(ally_towers=0.65, enemy_towers=0.65)
        rc.compute(obs1, obs2)

        # Only one tower increases slightly (noise)
        obs3 = _make_obs(ally_towers=0.70, enemy_towers=0.65)
        rc.compute(obs2, obs3)
        assert rc.new_game_detected is False

    def test_anomaly_not_triggered_on_first_call(self):
        """First compute() has no prior state — no anomaly possible."""
        rc = RewardComputer()
        obs1 = _make_obs(ally_towers=2.0 / 3.0, enemy_towers=2.0 / 3.0)
        obs2 = _make_obs(ally_towers=1.0, enemy_towers=1.0)
        rc.compute(obs1, obs2)
        assert rc.new_game_detected is False

    def test_anomaly_resets_internal_state(self):
        """After anomaly, next compute() should work cleanly."""
        rc = RewardComputer()
        # Build up state
        obs1 = _make_obs(ally_towers=2.0 / 3.0, enemy_towers=2.0 / 3.0)
        obs2 = _make_obs(ally_towers=2.0 / 3.0, enemy_towers=2.0 / 3.0)
        rc.compute(obs1, obs2)

        # Trigger anomaly
        obs3 = _make_obs(ally_towers=1.0, enemy_towers=1.0)
        rc.compute(obs2, obs3)
        assert rc.new_game_detected is True

        # Next call should be clean (no anomaly)
        obs4 = _make_obs(ally_towers=1.0, enemy_towers=1.0)
        reward = rc.compute(obs3, obs4)
        assert rc.new_game_detected is False
        assert reward == pytest.approx(0.02, abs=1e-6)


class TestRewardClamping:
    """Test reward clamping (Layer 4)."""

    def test_reward_clamping_positive(self):
        """Non-terminal reward should be clamped to reward_clamp."""
        cfg = RewardConfig(enemy_crown_reward=20.0, reward_clamp=15.0)
        rc = RewardComputer(cfg)
        obs1 = _make_obs(enemy_towers=1.0)
        obs2 = _make_obs(enemy_towers=2.0 / 3.0)
        reward = rc.compute(obs1, obs2)
        # 20 + 0.02 would be clamped to 15
        assert reward == pytest.approx(15.0, abs=0.1)

    def test_reward_clamping_negative(self):
        """Negative non-terminal reward should be clamped."""
        cfg = RewardConfig(ally_crown_penalty=-20.0, reward_clamp=15.0)
        rc = RewardComputer(cfg)
        obs1 = _make_obs(ally_towers=1.0)
        obs2 = _make_obs(ally_towers=2.0 / 3.0)
        reward = rc.compute(obs1, obs2)
        # -20 + 0.02 clamped to -15
        assert reward == pytest.approx(-15.0, abs=0.1)

    def test_terminal_reward_bypasses_clamp(self):
        """Terminal rewards (win/loss) are added AFTER clamping."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2, terminal_outcome="win")
        # survival (0.02) clamped to 0.02, then + 30 = 30.02
        assert reward == pytest.approx(30.02, abs=0.1)

    def test_crown_plus_terminal_with_clamp(self):
        """Crown reward gets clamped, terminal added on top."""
        cfg = RewardConfig(enemy_crown_reward=20.0, reward_clamp=15.0)
        rc = RewardComputer(cfg)
        obs1 = _make_obs(enemy_towers=1.0)
        obs2 = _make_obs(enemy_towers=2.0 / 3.0)
        reward = rc.compute(obs1, obs2, terminal_outcome="win")
        # non-terminal: 20 + 0.02 = 20.02, clamped to 15, then + 30 = 45
        assert reward == pytest.approx(45.0, abs=0.1)


class TestManualCrowns:
    """Test manual crown reward computation."""

    def test_enemy_crowns_only(self):
        """Scoring 2 crowns with no losses."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=2, ally_crowns_lost=0)
        assert reward == pytest.approx(20.0, abs=0.1)

    def test_ally_crowns_lost_only(self):
        """Losing 1 tower with no crowns scored."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=0, ally_crowns_lost=1)
        assert reward == pytest.approx(-10.0, abs=0.1)

    def test_mixed_crowns(self):
        """2 crowns scored, 1 tower lost."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=2, ally_crowns_lost=1)
        # 2*10 + 1*(-10) = 10
        assert reward == pytest.approx(10.0, abs=0.1)

    def test_zero_crowns(self):
        """No crowns either side."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=0, ally_crowns_lost=0)
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_three_crown_victory(self):
        """3-crown with no losses."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=3, ally_crowns_lost=0)
        assert reward == pytest.approx(30.0, abs=0.1)
