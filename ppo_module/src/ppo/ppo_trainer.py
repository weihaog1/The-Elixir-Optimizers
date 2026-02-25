"""PPO training orchestrator for Clash Royale.

Manages MaskablePPO model creation, BC weight loading, freeze/unfreeze
schedule, and the semi-automated episode training loop.

Usage:
    from src.ppo.ppo_trainer import PPOTrainer, PPOConfig

    trainer = PPOTrainer(PPOConfig(
        bc_weights_path="models/bc/bc_feature_extractor.pt",
        output_dir="models/ppo/",
    ))
    trainer.train(num_episodes=15)
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.ppo.callbacks import CRMetricsCallback
from src.ppo.clash_royale_env import ClashRoyaleEnv, EnvConfig
from src.ppo.reward import RewardConfig
from src.ppo.sb3_feature_extractor import SB3CRFeatureExtractor


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # BC weight loading
    bc_weights_path: str = ""

    # Environment
    env_config: EnvConfig = field(default_factory=EnvConfig)
    project_root: str = ""

    # PPO hyperparameters
    learning_rate: float = 1e-4
    clip_range: float = 0.1
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Feature extractor
    features_dim: int = 192
    freeze_extractor: bool = False

    # Network architecture
    pi_layers: list[int] = field(default_factory=lambda: [128, 64])
    vf_layers: list[int] = field(default_factory=lambda: [128, 64])

    # Output
    output_dir: str = "models/ppo/"
    log_dir: str = "logs/ppo/"

    # Training
    device: str = "cpu"
    verbose: int = 1

    # Resume
    resume_path: str = ""


class PPOTrainer:
    """Orchestrates MaskablePPO training on live Clash Royale.

    Handles:
    - Model creation with BC-pretrained feature extractor
    - Freeze/unfreeze schedule
    - Semi-automated episode loop (operator queues matches)
    - Per-episode checkpointing and logging
    """

    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self._model = None
        self._env = None
        self._best_win_rate = 0.0

    def _create_env(self) -> ClashRoyaleEnv:
        """Create and return the Gymnasium environment."""
        return ClashRoyaleEnv(
            config=self.config.env_config,
            project_root=self.config.project_root,
        )

    def _create_model(self, env: ClashRoyaleEnv):
        """Create MaskablePPO model with BC feature extractor."""
        from sb3_contrib import MaskablePPO

        cfg = self.config

        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": SB3CRFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": cfg.features_dim},
                "net_arch": dict(pi=cfg.pi_layers, vf=cfg.vf_layers),
                "activation_fn": torch.nn.ReLU,
            },
            learning_rate=cfg.learning_rate,
            clip_range=cfg.clip_range,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            verbose=cfg.verbose,
            tensorboard_log=cfg.log_dir,
            device=cfg.device,
        )

        # Load BC pretrained weights
        if cfg.bc_weights_path and os.path.exists(cfg.bc_weights_path):
            bc_weights = torch.load(
                cfg.bc_weights_path, map_location=cfg.device,
            )
            model.policy.features_extractor.load_bc_weights(bc_weights)
            print(f"[PPOTrainer] Loaded BC weights from {cfg.bc_weights_path}")

        # Freeze feature extractor if requested
        if cfg.freeze_extractor:
            model.policy.features_extractor.freeze()
            print("[PPOTrainer] Feature extractor FROZEN.")

        return model

    def _load_model(self, env: ClashRoyaleEnv):
        """Load a saved MaskablePPO model for resuming training."""
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(
            self.config.resume_path,
            env=env,
            device=self.config.device,
        )
        print(f"[PPOTrainer] Resumed from {self.config.resume_path}")
        return model

    def train(self, num_episodes: int) -> dict:
        """Run the semi-automated PPO training loop.

        Each episode:
        1. env.reset() blocks until operator starts a match
        2. Agent plays the match autonomously
        3. After game ends, model is saved
        4. Operator clicks "Battle" for the next match

        Args:
            num_episodes: Number of games to train on.

        Returns:
            Training summary dict.
        """
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        # Create environment and model
        print("[PPOTrainer] Initializing environment...")
        self._env = self._create_env()

        if cfg.resume_path:
            print(f"[PPOTrainer] Loading model from {cfg.resume_path}...")
            self._model = self._load_model(self._env)
        else:
            print("[PPOTrainer] Creating new MaskablePPO model...")
            self._model = self._create_model(self._env)

        # Metrics callback
        metrics_cb = CRMetricsCallback(
            window_size=10,
            log_path=os.path.join(cfg.log_dir, "training_log.jsonl"),
            verbose=cfg.verbose,
        )

        print(f"\n{'=' * 60}")
        print(f"[PPOTrainer] Starting PPO training: {num_episodes} episodes")
        print(f"  BC weights: {cfg.bc_weights_path or '(none)'}")
        print(f"  Frozen extractor: {cfg.freeze_extractor}")
        print(f"  LR: {cfg.learning_rate}")
        print(f"  Clip range: {cfg.clip_range}")
        print(f"  n_steps: {cfg.n_steps}")
        print(f"  n_epochs: {cfg.n_epochs}")
        print(f"  Device: {cfg.device}")
        print(f"  Output: {cfg.output_dir}")
        print(f"{'=' * 60}\n")

        # Semi-automated episode loop
        # We use model.learn() with n_steps matching roughly one episode.
        # After each learn() call, we save and log.
        episode_results = []
        training_start = time.time()

        for ep in range(num_episodes):
            print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

            try:
                # learn() for one rollout (n_steps)
                # This calls env.reset() internally on first call or after done
                self._model.learn(
                    total_timesteps=cfg.n_steps,
                    callback=metrics_cb,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
            except KeyboardInterrupt:
                print("\n[PPOTrainer] Training interrupted by user.")
                break

            # Save checkpoint
            latest_path = os.path.join(cfg.output_dir, "latest_ppo")
            self._model.save(latest_path)

            ep_path = os.path.join(cfg.output_dir, f"ppo_ep{ep + 1}")
            self._model.save(ep_path)

            print(f"[PPOTrainer] Saved checkpoint: {ep_path}.zip")
            print(
                f"[PPOTrainer] Queue next match in Clash Royale "
                f"(or Ctrl+C to stop)."
            )

        # Final save
        final_path = os.path.join(cfg.output_dir, "final_ppo")
        self._model.save(final_path)

        duration = time.time() - training_start
        summary = {
            "total_episodes": min(num_episodes, len(episode_results)),
            "duration_seconds": round(duration, 1),
            "final_model": f"{final_path}.zip",
        }

        print(f"\n{'=' * 60}")
        print(f"[PPOTrainer] Training complete.")
        print(f"  Duration: {duration / 60:.1f} minutes")
        print(f"  Final model: {final_path}.zip")
        print(f"{'=' * 60}")

        return summary

    def unfreeze_extractor(self, new_lr: float = 3e-5) -> None:
        """Unfreeze the feature extractor and reduce learning rate.

        Call this between Phase 1 (frozen) and Phase 2 (fine-tuning).
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call train() first.")

        self._model.policy.features_extractor.unfreeze()
        self._model.learning_rate = new_lr
        print(f"[PPOTrainer] Feature extractor UNFROZEN. LR set to {new_lr}")

    def close(self) -> None:
        """Clean up resources."""
        if self._env is not None:
            self._env.close()
