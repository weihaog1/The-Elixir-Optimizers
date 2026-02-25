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
        # crown(10) + win(30) + survival(0.02) = 40.02
        assert reward == pytest.approx(40.02, abs=0.1)
