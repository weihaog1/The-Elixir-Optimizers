"""Custom SB3 callbacks for Clash Royale PPO training.

Tracks per-episode metrics (win rate, crowns, reward, cards played)
and logs them to TensorBoard and JSONL.
"""

import json
import os
from collections import deque
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class CRMetricsCallback(BaseCallback):
    """Tracks and logs per-episode Clash Royale metrics.

    Monitors the info dict returned by ClashRoyaleEnv for episode
    completion signals and game outcomes.

    Logged metrics (rolling window of last N episodes):
        - win_rate: fraction of wins
        - avg_crowns_taken: enemy crowns scored
        - avg_episode_reward: total reward per episode
        - avg_cards_played: card actions per episode
        - avg_episode_length: steps per episode

    Args:
        window_size: Number of recent episodes for rolling averages.
        log_path: Path for JSONL episode log file.
        verbose: Print episode summaries to console.
    """

    def __init__(
        self,
        window_size: int = 10,
        log_path: str = "",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self._window_size = window_size
        self._log_path = log_path
        self._log_file = None

        # Rolling metrics
        self._outcomes: deque[str] = deque(maxlen=window_size)
        self._rewards: deque[float] = deque(maxlen=window_size)
        self._cards_played: deque[int] = deque(maxlen=window_size)
        self._episode_lengths: deque[int] = deque(maxlen=window_size)

        self._total_episodes = 0
        self._anomaly_count = 0
        self._truncation_count = 0

    def _on_training_start(self) -> None:
        if self._log_path:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            self._log_file = open(self._log_path, "a")

    def _on_step(self) -> bool:
        # Check if any environment reported episode completion
        infos = self.locals.get("infos", [])
        for info in infos:
            # Track anomalies per step (even if episode doesn't end)
            if info.get("anomaly_detected", False):
                self._anomaly_count += 1
            if info.get("truncation_reason"):
                self._truncation_count += 1

            if "outcome" in info or "episode_length" in info:
                self._record_episode(info)
        return True

    def _record_episode(self, info: dict) -> None:
        """Record metrics for a completed episode."""
        self._total_episodes += 1

        outcome = info.get("outcome", "unknown")
        ep_reward = info.get("episode_reward", 0.0)
        cards = info.get("cards_played", 0)
        ep_length = info.get("episode_length", 0)

        self._outcomes.append(outcome)
        self._rewards.append(ep_reward)
        self._cards_played.append(cards)
        self._episode_lengths.append(ep_length)

        # Compute rolling metrics
        n = len(self._outcomes)
        win_count = sum(1 for o in self._outcomes if o == "win")
        win_rate = win_count / n if n > 0 else 0.0
        avg_reward = sum(self._rewards) / n if n > 0 else 0.0
        avg_cards = sum(self._cards_played) / n if n > 0 else 0.0
        avg_length = sum(self._episode_lengths) / n if n > 0 else 0.0

        # Log to TensorBoard
        if self.logger is not None:
            self.logger.record("cr/win_rate", win_rate)
            self.logger.record("cr/episode_reward", ep_reward)
            self.logger.record("cr/avg_reward", avg_reward)
            self.logger.record("cr/cards_played", cards)
            self.logger.record("cr/avg_cards_played", avg_cards)
            self.logger.record("cr/episode_length", ep_length)
            self.logger.record("cr/avg_episode_length", avg_length)
            self.logger.record("cr/total_episodes", self._total_episodes)
            self.logger.record("cr/anomaly_count", self._anomaly_count)
            self.logger.record("cr/truncation_count", self._truncation_count)

        # Log to JSONL
        anomaly = info.get("anomaly_detected", False)
        truncation_reason = info.get("truncation_reason", "")

        entry = {
            "episode": self._total_episodes,
            "outcome": outcome,
            "reward": round(ep_reward, 3),
            "cards_played": cards,
            "episode_length": ep_length,
            "win_rate": round(win_rate, 3),
            "avg_reward": round(avg_reward, 3),
            "timestep": self.num_timesteps,
            "anomaly_detected": anomaly,
            "truncation_reason": truncation_reason,
        }
        if self._log_file is not None:
            self._log_file.write(json.dumps(entry) + "\n")
            self._log_file.flush()

        # Console output
        if self.verbose > 0:
            result_str = outcome.upper() if outcome else "?"
            print(
                f"[Episode {self._total_episodes}] "
                f"{result_str} | "
                f"reward={ep_reward:+.1f} | "
                f"cards={cards} | "
                f"steps={ep_length} | "
                f"win_rate={win_rate:.0%} (last {n})"
            )

    def _on_training_end(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        if self.verbose > 0 and self._total_episodes > 0:
            n = len(self._outcomes)
            wins = sum(1 for o in self._outcomes if o == "win")
            print(f"\n{'=' * 50}")
            print(f"Training Summary: {self._total_episodes} episodes")
            print(f"  Win rate (last {n}): {wins}/{n} = {wins / n:.0%}")
            print(f"  Avg reward: {sum(self._rewards) / n:.1f}")
            print(f"  Avg cards/episode: {sum(self._cards_played) / n:.1f}")
            print(f"{'=' * 50}")
