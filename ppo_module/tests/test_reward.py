"""Tests for RewardComputer."""

import numpy as np
import pytest

from src.ppo.reward import RewardComputer, RewardConfig

# Default reward_scale = 0.1
_SCALE = 0.1


def _make_obs(
    elixir: float = 0.5,
    ally_towers: float = 1.0,
    enemy_towers: float = 1.0,
    arena: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Create a minimal observation dict with specified vector values."""
    vector = np.zeros(23, dtype=np.float32)
    vector[0] = elixir  # elixir / 10
    vector[9] = ally_towers  # ally tower count / 3
    vector[10] = enemy_towers  # enemy tower count / 3
    if arena is None:
        arena = np.zeros((32, 18, 6), dtype=np.float32)
    return {"arena": arena, "vector": vector}


class TestRewardComputer:
    """Test the reward computation logic."""

    def test_survival_bonus(self):
        """Every step should give a small survival bonus."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2)
        # survival_bonus (0.02) * scale (0.1) = 0.002
        assert reward == pytest.approx(0.02 * _SCALE, abs=1e-6)

    def test_enemy_crown_reward(self):
        """Destroying an enemy tower should give +10 (scaled)."""
        rc = RewardComputer()
        obs1 = _make_obs(enemy_towers=1.0)  # 3/3 towers
        obs2 = _make_obs(enemy_towers=2.0 / 3.0)  # 2/3 towers
        reward = rc.compute(obs1, obs2)
        # (enemy_crown_reward (10) + survival_bonus (0.02)) * 0.1 = 1.002
        assert reward == pytest.approx(10.02 * _SCALE, abs=0.01)

    def test_ally_crown_penalty(self):
        """Losing an ally tower should give -10 (scaled)."""
        rc = RewardComputer()
        obs1 = _make_obs(ally_towers=1.0)  # 3/3 towers
        obs2 = _make_obs(ally_towers=2.0 / 3.0)  # 2/3 towers
        reward = rc.compute(obs1, obs2)
        # (ally_crown_penalty (-10) + survival_bonus (0.02)) * 0.1 = -0.998
        assert reward == pytest.approx(-9.98 * _SCALE, abs=0.01)

    def test_win_terminal(self):
        """Winning should give a large positive reward."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2, terminal_outcome="win")
        # (survival 0.02 + win 30) * 0.1 = 3.002
        assert reward == pytest.approx(30.02 * _SCALE, abs=0.01)

    def test_loss_terminal(self):
        """Losing should give a large negative reward."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2, terminal_outcome="loss")
        # (survival 0.02 + loss -30) * 0.1 = -2.998
        assert reward == pytest.approx(-29.98 * _SCALE, abs=0.01)

    def test_draw_terminal(self):
        """Drawing should give a small negative reward."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2, terminal_outcome="draw")
        # (survival 0.02 + draw -5) * 0.1 = -0.498
        assert reward == pytest.approx(-4.98 * _SCALE, abs=0.01)

    def test_elixir_waste_penalty_full(self):
        """Elixir at 9.5+ should get full waste penalty."""
        rc = RewardComputer()
        obs1 = _make_obs(elixir=0.5)
        obs2 = _make_obs(elixir=0.98)  # 9.8/10 = above waste threshold
        reward = rc.compute(obs1, obs2)
        # (survival 0.02 + waste -0.1) * 0.1 = -0.008
        assert reward == pytest.approx(-0.08 * _SCALE, abs=1e-5)

    def test_elixir_high_penalty_graduated(self):
        """Elixir at 8.0-9.5 should get mild penalty."""
        rc = RewardComputer()
        obs1 = _make_obs(elixir=0.5)
        obs2 = _make_obs(elixir=0.85)  # 8.5/10 = above high threshold
        reward = rc.compute(obs1, obs2)
        # (survival 0.02 + high_penalty -0.02) * 0.1 = 0.0
        assert reward == pytest.approx(0.0 * _SCALE, abs=1e-5)

    def test_no_elixir_penalty_below_threshold(self):
        """Elixir below 8.0 should not penalize."""
        rc = RewardComputer()
        obs1 = _make_obs(elixir=0.5)
        obs2 = _make_obs(elixir=0.7)  # 7.0/10 = below high threshold
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02 * _SCALE, abs=1e-5)

    def test_batched_vectors(self):
        """Should handle (1, 23) shaped vectors."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs1["vector"] = obs1["vector"].reshape(1, 23)
        obs2 = _make_obs()
        obs2["vector"] = obs2["vector"].reshape(1, 23)
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02 * _SCALE, abs=1e-5)

    def test_custom_config(self):
        """Should respect custom reward weights."""
        cfg = RewardConfig(
            enemy_crown_reward=20.0,
            survival_bonus=0.0,
            reward_clamp=25.0,
            reward_scale=1.0,  # No scaling for this test
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
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02 * _SCALE, abs=1e-5)

    def test_combined_crown_and_win(self):
        """Scoring a crown and winning on same step."""
        rc = RewardComputer()
        obs1 = _make_obs(enemy_towers=1.0 / 3.0)
        obs2 = _make_obs(enemy_towers=0.0)
        reward = rc.compute(obs1, obs2, terminal_outcome="win")
        # non-terminal: 10 + 0.02 = 10.02, clamped to 10.02, + win 30 = 40.02
        assert reward == pytest.approx(40.02 * _SCALE, abs=0.01)


class TestRewardScaling:
    """Test that reward_scale applies correctly."""

    def test_scale_factor_applied(self):
        """All rewards should be multiplied by reward_scale."""
        cfg = RewardConfig(reward_scale=0.5)
        rc = RewardComputer(cfg)
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02 * 0.5, abs=1e-5)

    def test_scale_one_gives_raw_values(self):
        """reward_scale=1.0 should give unscaled rewards."""
        cfg = RewardConfig(reward_scale=1.0)
        rc = RewardComputer(cfg)
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2, terminal_outcome="win")
        assert reward == pytest.approx(30.02, abs=0.1)

    def test_manual_crowns_scaled(self):
        """compute_manual_crowns should also apply reward_scale."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=1, ally_crowns_lost=0)
        assert reward == pytest.approx(10.0 * _SCALE, abs=0.01)


class TestUnitAdvantage:
    """Test unit count advantage reward."""

    def _arena_with_units(self, ally: int, enemy: int) -> np.ndarray:
        """Create arena with specified ally/enemy unit counts."""
        arena = np.zeros((32, 18, 6), dtype=np.float32)
        # Place ally units: belonging=-1, mask=1
        for i in range(ally):
            arena[20 + i, 0, 1] = -1.0  # belonging = ally
            arena[20 + i, 0, 2] = 1.0   # arena mask = occupied
        # Place enemy units: belonging=+1, mask=1
        for i in range(enemy):
            arena[5 + i, 0, 1] = 1.0   # belonging = enemy
            arena[5 + i, 0, 2] = 1.0   # arena mask = occupied
        return arena

    def test_ally_advantage_positive(self):
        """More ally units than enemy should give positive reward."""
        rc = RewardComputer()
        obs1 = _make_obs()
        arena = self._arena_with_units(ally=5, enemy=2)
        obs2 = _make_obs(arena=arena)
        reward = rc.compute(obs1, obs2)
        # survival 0.02 + advantage 0.01 * (5-2) = 0.02 + 0.03 = 0.05
        assert reward == pytest.approx(0.05 * _SCALE, abs=1e-4)

    def test_enemy_advantage_negative(self):
        """More enemy units than ally should give negative reward component."""
        rc = RewardComputer()
        obs1 = _make_obs()
        arena = self._arena_with_units(ally=1, enemy=4)
        obs2 = _make_obs(arena=arena)
        reward = rc.compute(obs1, obs2)
        # survival 0.02 + advantage 0.01 * (1-4) = 0.02 - 0.03 = -0.01
        assert reward == pytest.approx(-0.01 * _SCALE, abs=1e-4)

    def test_equal_units_no_advantage(self):
        """Equal units should give no advantage reward."""
        rc = RewardComputer()
        obs1 = _make_obs()
        arena = self._arena_with_units(ally=3, enemy=3)
        obs2 = _make_obs(arena=arena)
        reward = rc.compute(obs1, obs2)
        # survival 0.02 + advantage 0 = 0.02
        assert reward == pytest.approx(0.02 * _SCALE, abs=1e-4)

    def test_no_units_no_advantage(self):
        """Empty arena should give no advantage reward."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02 * _SCALE, abs=1e-5)


class TestGraduatedElixir:
    """Test graduated elixir penalty thresholds."""

    def test_below_high_threshold_no_penalty(self):
        """Elixir at 7.0 (below 8.0) should have no elixir penalty."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs(elixir=0.7)
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(0.02 * _SCALE, abs=1e-5)

    def test_at_high_threshold_mild_penalty(self):
        """Elixir at exactly 8.0 should get mild penalty."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs(elixir=0.8)  # exactly at threshold
        reward = rc.compute(obs1, obs2)
        # survival 0.02 + high_penalty -0.02 = 0.0
        assert reward == pytest.approx(0.0, abs=1e-5)

    def test_between_thresholds_mild_penalty(self):
        """Elixir at 9.0 (between 8.0 and 9.5) should get mild penalty."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs(elixir=0.9)
        reward = rc.compute(obs1, obs2)
        # survival 0.02 + high_penalty -0.02 = 0.0
        assert reward == pytest.approx(0.0, abs=1e-5)

    def test_above_waste_threshold_full_penalty(self):
        """Elixir at 9.6 (above 9.5) should get full waste penalty."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs(elixir=0.96)
        reward = rc.compute(obs1, obs2)
        # survival 0.02 + waste -0.1 = -0.08
        assert reward == pytest.approx(-0.08 * _SCALE, abs=1e-5)

    def test_max_elixir_full_penalty(self):
        """Elixir at 10/10 should get full waste penalty."""
        rc = RewardComputer()
        obs1 = _make_obs()
        obs2 = _make_obs(elixir=1.0)
        reward = rc.compute(obs1, obs2)
        assert reward == pytest.approx(-0.08 * _SCALE, abs=1e-5)


class TestAnomalyDetection:
    """Test new-game anomaly detection (Layer 2)."""

    def test_tower_jump_anomaly_detected(self):
        """Both towers jumping from damaged to full = new game."""
        rc = RewardComputer()
        obs1 = _make_obs(ally_towers=2.0 / 3.0, enemy_towers=2.0 / 3.0)
        obs2 = _make_obs(ally_towers=2.0 / 3.0, enemy_towers=2.0 / 3.0)
        rc.compute(obs1, obs2)

        obs3 = _make_obs(ally_towers=1.0, enemy_towers=1.0)
        reward = rc.compute(obs2, obs3)

        assert rc.new_game_detected is True
        # Anomaly returns survival_bonus * scale
        assert reward == pytest.approx(0.02 * _SCALE, abs=1e-5)

    def test_single_tower_slight_increase_not_anomaly(self):
        """Small noise in one tower should not trigger anomaly."""
        rc = RewardComputer()
        obs1 = _make_obs(ally_towers=0.65, enemy_towers=0.65)
        obs2 = _make_obs(ally_towers=0.65, enemy_towers=0.65)
        rc.compute(obs1, obs2)

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
        obs1 = _make_obs(ally_towers=2.0 / 3.0, enemy_towers=2.0 / 3.0)
        obs2 = _make_obs(ally_towers=2.0 / 3.0, enemy_towers=2.0 / 3.0)
        rc.compute(obs1, obs2)

        obs3 = _make_obs(ally_towers=1.0, enemy_towers=1.0)
        rc.compute(obs2, obs3)
        assert rc.new_game_detected is True

        obs4 = _make_obs(ally_towers=1.0, enemy_towers=1.0)
        reward = rc.compute(obs3, obs4)
        assert rc.new_game_detected is False
        assert reward == pytest.approx(0.02 * _SCALE, abs=1e-5)


class TestRewardClamping:
    """Test reward clamping (Layer 4)."""

    def test_reward_clamping_positive(self):
        """Non-terminal reward should be clamped to reward_clamp."""
        cfg = RewardConfig(enemy_crown_reward=20.0, reward_clamp=15.0, reward_scale=1.0)
        rc = RewardComputer(cfg)
        obs1 = _make_obs(enemy_towers=1.0)
        obs2 = _make_obs(enemy_towers=2.0 / 3.0)
        reward = rc.compute(obs1, obs2)
        # 20 + 0.02 clamped to 15
        assert reward == pytest.approx(15.0, abs=0.1)

    def test_reward_clamping_negative(self):
        """Negative non-terminal reward should be clamped."""
        cfg = RewardConfig(ally_crown_penalty=-20.0, reward_clamp=15.0, reward_scale=1.0)
        rc = RewardComputer(cfg)
        obs1 = _make_obs(ally_towers=1.0)
        obs2 = _make_obs(ally_towers=2.0 / 3.0)
        reward = rc.compute(obs1, obs2)
        # -20 + 0.02 clamped to -15
        assert reward == pytest.approx(-15.0, abs=0.1)

    def test_terminal_reward_bypasses_clamp(self):
        """Terminal rewards (win/loss) are added AFTER clamping."""
        cfg = RewardConfig(reward_scale=1.0)
        rc = RewardComputer(cfg)
        obs1 = _make_obs()
        obs2 = _make_obs()
        reward = rc.compute(obs1, obs2, terminal_outcome="win")
        # survival (0.02) clamped to 0.02, then + 30 = 30.02
        assert reward == pytest.approx(30.02, abs=0.1)

    def test_crown_plus_terminal_with_clamp(self):
        """Crown reward gets clamped, terminal added on top."""
        cfg = RewardConfig(enemy_crown_reward=20.0, reward_clamp=15.0, reward_scale=1.0)
        rc = RewardComputer(cfg)
        obs1 = _make_obs(enemy_towers=1.0)
        obs2 = _make_obs(enemy_towers=2.0 / 3.0)
        reward = rc.compute(obs1, obs2, terminal_outcome="win")
        # non-terminal: 20 + 0.02 clamped to 15, + 30 = 45
        assert reward == pytest.approx(45.0, abs=0.1)


class TestManualCrowns:
    """Test manual crown reward computation."""

    def test_enemy_crowns_only(self):
        """Scoring 2 crowns with no losses."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=2, ally_crowns_lost=0)
        assert reward == pytest.approx(20.0 * _SCALE, abs=0.01)

    def test_ally_crowns_lost_only(self):
        """Losing 1 tower with no crowns scored."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=0, ally_crowns_lost=1)
        assert reward == pytest.approx(-10.0 * _SCALE, abs=0.01)

    def test_mixed_crowns(self):
        """2 crowns scored, 1 tower lost."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=2, ally_crowns_lost=1)
        assert reward == pytest.approx(10.0 * _SCALE, abs=0.01)

    def test_zero_crowns(self):
        """No crowns either side."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=0, ally_crowns_lost=0)
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_three_crown_victory(self):
        """3-crown with no losses."""
        rc = RewardComputer()
        reward = rc.compute_manual_crowns(enemy_crowns=3, ally_crowns_lost=0)
        assert reward == pytest.approx(30.0 * _SCALE, abs=0.01)

    def test_manual_crowns_unscaled(self):
        """With reward_scale=1.0, returns raw values."""
        cfg = RewardConfig(reward_scale=1.0)
        rc = RewardComputer(cfg)
        reward = rc.compute_manual_crowns(enemy_crowns=2, ally_crowns_lost=0)
        assert reward == pytest.approx(20.0, abs=0.1)
