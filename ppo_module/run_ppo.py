#!/usr/bin/env python3
"""PPO training entry point for Clash Royale.

Semi-automated training: the agent plays matches autonomously while
the operator queues new games between episodes.

Usage:
    # Phase 1: frozen feature extractor
    python ppo_module/run_ppo.py \
        --bc-weights models/bc/bc_feature_extractor.pt \
        --window-title "Clash Royale" \
        --num-episodes 15 \
        --freeze-extractor

    # Phase 2: full fine-tuning (resume from phase 1)
    python ppo_module/run_ppo.py \
        --resume models/ppo/latest_ppo.zip \
        --window-title "Clash Royale" \
        --num-episodes 25 \
        --lr 3e-5

    # Dry run (no clicks)
    python ppo_module/run_ppo.py \
        --bc-weights models/bc/bc_feature_extractor.pt \
        --capture-region 0,0,540,960 \
        --num-episodes 1 \
        --dry-run
"""

import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Namespace package setup (same pattern as bc_model_module/run_live.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

import types as _types

_ppo_src = os.path.join(_SCRIPT_DIR, "src")
_bc_src = os.path.join(PROJECT_ROOT, "bc_model_module", "src")
_encoder_src = os.path.join(PROJECT_ROOT, "state_encoder_module", "src")
_action_src = os.path.join(PROJECT_ROOT, "action_builder_module", "src")

_src_paths = [_ppo_src, _bc_src, _encoder_src, _action_src]

# Pre-register src as namespace package
_src_pkg = _types.ModuleType("src")
_src_pkg.__path__ = _src_paths
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

# Pre-register subpackages to bypass __init__.py imports
_sub_pkgs = {
    "src.ppo": os.path.join(_ppo_src, "ppo"),
    "src.bc": os.path.join(_bc_src, "bc"),
    "src.encoder": os.path.join(_encoder_src, "encoder"),
}
for pkg_name, pkg_path in _sub_pkgs.items():
    mod = _types.ModuleType(pkg_name)
    mod.__path__ = [pkg_path]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod

# Main codebase subpackages
_main_src = os.path.join(PROJECT_ROOT, "src", "src")
for sub in ("generation", "detection", "classification", "data", "pipeline"):
    pkg_name = f"src.{sub}"
    mod = _types.ModuleType(pkg_name)
    mod.__path__ = [os.path.join(_main_src, sub)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO training for Clash Royale agent"
    )

    # Model
    parser.add_argument(
        "--bc-weights", type=str, default="",
        help="Path to BC feature extractor weights (bc_feature_extractor.pt)",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to a saved PPO model to resume training",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for training (default: cpu)",
    )

    # Capture
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--window-title", type=str, default="",
        help="Game window title for auto-detection",
    )
    capture_group.add_argument(
        "--capture-region", type=str, default="",
        help="Manual capture region: left,top,width,height",
    )

    # Game region within captured window (for landscape windows like GPG)
    parser.add_argument(
        "--game-region", type=str, default="",
        help="Game area within window: left,top,width,height (e.g. 655,1,609,1077)",
    )

    # Card classification
    parser.add_argument(
        "--card-confidence", type=float, default=0.6,
        help="Minimum card classifier confidence to unmask a card (default: 0.6)",
    )

    # Training
    parser.add_argument(
        "--num-episodes", type=int, default=15,
        help="Number of games to train on (default: 15)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--clip-range", type=float, default=0.1,
        help="PPO clip range (default: 0.1)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=700,
        help="Steps per rollout (default: 700)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10,
        help="Epochs per batch (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.02,
        help="Starting entropy coefficient (default: 0.02)",
    )
    parser.add_argument(
        "--ent-start", type=float, default=0.02,
        help="Starting entropy coefficient for annealing (default: 0.02)",
    )
    parser.add_argument(
        "--ent-end", type=float, default=0.005,
        help="Ending entropy coefficient for annealing (default: 0.005)",
    )
    parser.add_argument(
        "--n-frames", type=int, default=3,
        help="Frame stack depth (default: 3)",
    )
    parser.add_argument(
        "--reward-scale", type=float, default=0.1,
        help="Reward scaling factor (default: 0.1)",
    )
    parser.add_argument(
        "--bc-policy", type=str, default="",
        help="Path to full BC policy for KL penalty reference",
    )
    parser.add_argument(
        "--kl-coef", type=float, default=0.0,
        help="KL penalty coefficient from BC reference (0 = disabled)",
    )
    parser.add_argument(
        "--freeze-extractor", action="store_true",
        help="Freeze the BC feature extractor (Phase 1 training)",
    )

    # Safety
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log actions but do not execute clicks",
    )
    parser.add_argument(
        "--no-perception", action="store_true",
        help="Disable YOLO detection (use zero-filled observations)",
    )
    parser.add_argument(
        "--no-pause", action="store_true",
        help="Skip Enter prompt between episodes (auto-continue)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="models/ppo/",
        help="Directory for model checkpoints (default: models/ppo/)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/ppo/",
        help="Directory for training logs (default: logs/ppo/)",
    )

    # Templates
    parser.add_argument(
        "--templates-dir", type=str, default="",
        help="Directory with victory.png/defeat.png templates for game end detection",
    )

    # Visualization
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show live observation tensor heatmaps during training",
    )
    parser.add_argument(
        "--vis-save-dir", type=str, default="",
        help="Save visualization frames to this directory (for video)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    from src.ppo.clash_royale_env import EnvConfig
    from src.ppo.ppo_trainer import PPOConfig, PPOTrainer
    from src.ppo.reward import RewardConfig

    # Parse capture region
    capture_region = None
    if args.capture_region:
        parts = args.capture_region.split(",")
        if len(parts) != 4:
            print("Error: --capture-region must be left,top,width,height")
            return 1
        capture_region = tuple(int(p.strip()) for p in parts)

    # Parse game region within window
    game_region = None
    if args.game_region:
        parts = args.game_region.split(",")
        if len(parts) != 4:
            print("Error: --game-region must be left,top,width,height")
            return 1
        game_region = tuple(int(p.strip()) for p in parts)

    # Build configs
    reward_config = RewardConfig(reward_scale=args.reward_scale)

    env_config = EnvConfig(
        window_title=args.window_title,
        capture_region=capture_region,
        use_perception=not args.no_perception,
        dry_run=args.dry_run,
        reward_config=reward_config,
        templates_dir=args.templates_dir,
        device=args.device,
        pause_between_episodes=not args.no_pause,
        visualize=args.visualize,
        vis_save_dir=args.vis_save_dir,
        n_frames=args.n_frames,
        game_region=game_region,
        card_confidence_threshold=args.card_confidence,
    )

    ppo_config = PPOConfig(
        bc_weights_path=args.bc_weights,
        env_config=env_config,
        project_root=PROJECT_ROOT,
        learning_rate=args.lr,
        clip_range=args.clip_range,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        ent_coef_start=args.ent_start,
        ent_coef_end=args.ent_end,
        n_frames=args.n_frames,
        freeze_extractor=args.freeze_extractor,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device,
        resume_path=args.resume,
        bc_policy_path=args.bc_policy,
        kl_coef=args.kl_coef,
    )

    trainer = PPOTrainer(ppo_config)
    try:
        trainer.train(num_episodes=args.num_episodes)
    except KeyboardInterrupt:
        print("\n[run_ppo] Interrupted. Saving final model...")
    finally:
        trainer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
