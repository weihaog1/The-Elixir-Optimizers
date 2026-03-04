"""Reward function for PPO training on Clash Royale.

Computes per-step rewards from observation vector deltas. Signals used:
- Tower count changes (crown scored / crown lost)
- Win/loss/draw terminal outcome
- Survival bonus (small positive reward per step)
- Graduated elixir waste penalty (mild at 8+, full at 9.5+)
- Unit count advantage (ally vs enemy troop presence)

All signals come from the observation tensors (vector + arena).

Vector indices used:
    0: elixir / 10
    9: player tower count / 3
   10: enemy tower count / 3

Arena channels used for unit advantage:
    1: CH_BELONGING  (-1=ally, +1=enemy)
    2: CH_ARENA_MASK (1.0=unit present)
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
    elixir_waste_penalty: float = -0.1  # full penalty at max elixir
    elixir_waste_threshold: float = 0.95  # 9.5/10 elixir
    elixir_high_penalty: float = -0.02  # mild penalty at 8+ elixir
    elixir_high_threshold: float = 0.8  # 8/10 elixir
    unit_advantage_weight: float = 0.01  # per-unit advantage reward
    reward_clamp: float = 15.0  # Per-step non-terminal reward ceiling
    reward_scale: float = 0.1  # Scale all rewards for value fn stability
    tower_jump_threshold: float = 0.15  # Tower increase > this = new game anomaly


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
        self.new_game_detected: bool = False

    def reset(self) -> None:
        """Reset state for a new episode."""
        self._prev_ally_towers = None
        self._prev_enemy_towers = None
        self.new_game_detected = False

    def _detect_new_game_anomaly(
        self,
        prev_ally: float,
        curr_ally: float,
        prev_enemy: float,
        curr_enemy: float,
    ) -> bool:
        """Detect if a new game started mid-episode.

        Tower counts only decrease in-game (towers get destroyed, never rebuilt).
        If both jump UP significantly, a new game has started.
        """
        threshold = self.config.tower_jump_threshold
        ally_jumped = (curr_ally - prev_ally) > threshold
        enemy_jumped = (curr_enemy - prev_enemy) > threshold
        return ally_jumped and enemy_jumped

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
        self.new_game_detected = False

        # --- Crown rewards (tower count changes) ---
        curr_enemy_towers = float(curr_vec[self._ENEMY_TOWER_COUNT_IDX])
        curr_ally_towers = float(curr_vec[self._ALLY_TOWER_COUNT_IDX])
        prev_enemy_towers = float(prev_vec[self._ENEMY_TOWER_COUNT_IDX])
        prev_ally_towers = float(prev_vec[self._ALLY_TOWER_COUNT_IDX])

        # --- Anomaly detection: new game started mid-episode ---
        if self._prev_ally_towers is not None and self._prev_enemy_towers is not None:
            if self._detect_new_game_anomaly(
                prev_ally_towers, curr_ally_towers,
                prev_enemy_towers, curr_enemy_towers,
            ):
                self.new_game_detected = True
                # Reset internal state — skip crown deltas, return survival only
                self._prev_ally_towers = curr_ally_towers
                self._prev_enemy_towers = curr_enemy_towers
                return cfg.survival_bonus * cfg.reward_scale

        # Track tower counts for anomaly detection on subsequent calls
        self._prev_ally_towers = curr_ally_towers
        self._prev_enemy_towers = curr_enemy_towers

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

        # --- Graduated elixir waste penalty ---
        curr_elixir = float(curr_vec[self._ELIXIR_IDX])
        if curr_elixir >= cfg.elixir_waste_threshold:
            reward += cfg.elixir_waste_penalty  # full penalty at 9.5+
        elif curr_elixir >= cfg.elixir_high_threshold:
            reward += cfg.elixir_high_penalty  # mild penalty at 8.0+

        # --- Unit count advantage (from arena grid) ---
        if "arena" in curr_obs:
            arena = curr_obs["arena"]
            if arena.ndim == 4:
                arena = arena[0]
            arena_mask = arena[:, :, 2]    # CH_ARENA_MASK
            belonging = arena[:, :, 1]     # CH_BELONGING: -1=ally, +1=enemy
            occupied = arena_mask > 0.5
            ally_units = int(np.sum(occupied & (belonging < 0)))
            enemy_units = int(np.sum(occupied & (belonging > 0)))
            advantage = ally_units - enemy_units
            reward += cfg.unit_advantage_weight * advantage

        # --- Clamp non-terminal reward ---
        reward = max(-cfg.reward_clamp, min(cfg.reward_clamp, reward))

        # --- Terminal reward (added AFTER clamping) ---
        if terminal_outcome == "win":
            reward += cfg.win_reward
        elif terminal_outcome == "loss":
            reward += cfg.loss_penalty
        elif terminal_outcome == "draw":
            reward += cfg.draw_penalty

        return reward * cfg.reward_scale

    def compute_manual_crowns(
        self, enemy_crowns: int, ally_crowns_lost: int,
    ) -> float:
        """Compute crown reward from manually-provided crown counts.

        Used when the operator manually ends an episode and reports
        crowns scored/lost. Serves as a fallback/override — automatic
        crown tracking from YOLO tower detection now works for normal
        gameplay, but manual input is still useful when the operator
        stops the episode early.

        Args:
            enemy_crowns: Number of enemy towers destroyed (0-3).
            ally_crowns_lost: Number of ally towers lost (0-3).

        Returns:
            Crown reward (can be positive or negative).
        """
        reward = 0.0
        reward += enemy_crowns * self.config.enemy_crown_reward
        reward += ally_crowns_lost * self.config.ally_crown_penalty
        return reward * self.config.reward_scale
