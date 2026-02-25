"""Reward function for PPO training on Clash Royale.

Computes per-step rewards from observation vector deltas. Signals used:
- Tower count changes (crown scored / crown lost)
- Win/loss/draw terminal outcome
- Survival bonus (small positive reward per step)
- Elixir waste penalty (sitting at max elixir)

No OCR required -- all signals come from the observation vector.

Vector indices used:
    0: elixir / 10
    9: player tower count / 3
   10: enemy tower count / 3
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RewardConfig:
    """Reward function weights."""

    enemy_crown_reward: float = 10.0
    ally_crown_penalty: float = -10.0
    win_reward: float = 30.0
    loss_penalty: float = -30.0
    draw_penalty: float = -5.0
    survival_bonus: float = 0.02
    elixir_waste_penalty: float = -0.05
    elixir_waste_threshold: float = 0.95  # 9.5/10 elixir


class RewardComputer:
    """Stateful reward computer tracking tower counts between steps.

    Call reset() at the start of each episode.
    Call compute() at each step with prev and current observations.

    Args:
        config: Reward weights and thresholds.
    """

    # Vector feature indices
    _ELIXIR_IDX = 0
    _ALLY_TOWER_COUNT_IDX = 9
    _ENEMY_TOWER_COUNT_IDX = 10

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()
        self._prev_ally_towers: Optional[float] = None
        self._prev_enemy_towers: Optional[float] = None

    def reset(self) -> None:
        """Reset state for a new episode."""
        self._prev_ally_towers = None
        self._prev_enemy_towers = None

    def compute(
        self,
        prev_obs: dict[str, np.ndarray],
        curr_obs: dict[str, np.ndarray],
        terminal_outcome: Optional[str] = None,
    ) -> float:
        """Compute reward for a single step.

        Args:
            prev_obs: Previous observation dict with "vector" key.
            curr_obs: Current observation dict with "vector" key.
            terminal_outcome: "win", "loss", or "draw" on last step, else None.

        Returns:
            Scalar reward for this step.
        """
        prev_vec = prev_obs["vector"]
        curr_vec = curr_obs["vector"]

        # Handle batched (1, 23) or unbatched (23,) vectors
        if prev_vec.ndim == 2:
            prev_vec = prev_vec[0]
        if curr_vec.ndim == 2:
            curr_vec = curr_vec[0]

        reward = 0.0
        cfg = self.config

        # --- Crown rewards (tower count changes) ---
        curr_enemy_towers = float(curr_vec[self._ENEMY_TOWER_COUNT_IDX])
        curr_ally_towers = float(curr_vec[self._ALLY_TOWER_COUNT_IDX])
        prev_enemy_towers = float(prev_vec[self._ENEMY_TOWER_COUNT_IDX])
        prev_ally_towers = float(prev_vec[self._ALLY_TOWER_COUNT_IDX])

        # Enemy tower count decreased = we scored a crown
        enemy_delta = prev_enemy_towers - curr_enemy_towers
        if enemy_delta > 0.01:  # threshold for float comparison
            reward += cfg.enemy_crown_reward

        # Ally tower count decreased = we lost a crown
        ally_delta = prev_ally_towers - curr_ally_towers
        if ally_delta > 0.01:
            reward += cfg.ally_crown_penalty

        # --- Survival bonus ---
        reward += cfg.survival_bonus

        # --- Elixir waste penalty ---
        curr_elixir = float(curr_vec[self._ELIXIR_IDX])
        if curr_elixir >= cfg.elixir_waste_threshold:
            reward += cfg.elixir_waste_penalty

        # --- Terminal reward ---
        if terminal_outcome == "win":
            reward += cfg.win_reward
        elif terminal_outcome == "loss":
            reward += cfg.loss_penalty
        elif terminal_outcome == "draw":
            reward += cfg.draw_penalty

        return reward
