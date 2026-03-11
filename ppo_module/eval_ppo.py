#!/usr/bin/env python3
"""Evaluate a trained PPO agent on live Clash Royale games.

Plays N games using inference only (no training updates) and reports
win rate, average reward, cards played, and other metrics.

Usage:
    python ppo_module/eval_ppo.py --model models/ppo/latest_ppo.zip --game-region 655,1,609,1077 --num-games 5
    python ppo_module/eval_ppo.py --model models/ppo/latest_ppo.zip --game-region 655,1,609,1077 --num-games 3 --visualize
"""

import argparse
import json
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Namespace package setup (same as run_ppo.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

import types as _types

_ppo_src = os.path.join(_SCRIPT_DIR, "src")
_bc_src = os.path.join(PROJECT_ROOT, "bc_model_module", "src")
_encoder_src = os.path.join(PROJECT_ROOT, "state_encoder_module", "src")
_action_src = os.path.join(PROJECT_ROOT, "action_builder_module", "src")

_src_paths = [_ppo_src, _bc_src, _encoder_src, _action_src]

_src_pkg = _types.ModuleType("src")
_src_pkg.__path__ = _src_paths
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

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

_main_src = os.path.join(PROJECT_ROOT, "src", "src")
for sub in ("generation", "detection", "classification", "data", "pipeline", "yolov8_custom", "ocr"):
    pkg_name = f"src.{sub}"
    mod = _types.ModuleType(pkg_name)
    mod.__path__ = [os.path.join(_main_src, sub)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent")

    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model (.zip)")
    parser.add_argument("--num-games", type=int, default=5, help="Number of games to evaluate (default: 5)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Capture
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument("--window-title", type=str, default="", help="Game window title")
    capture_group.add_argument("--capture-region", type=str, default="", help="Manual capture: left,top,width,height")
    parser.add_argument("--game-region", type=str, default="", help="Game area within window: left,top,width,height")

    # Options
    parser.add_argument("--visualize", action="store_true", help="Show live observation visualizer")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling")
    parser.add_argument("--n-frames", type=int, default=3, help="Frame stack depth (default: 3)")
    parser.add_argument("--card-confidence", type=float, default=0.6, help="Card classifier confidence threshold")
    parser.add_argument("--templates-dir", type=str, default="", help="Game end detection templates directory")
    parser.add_argument("--output", type=str, default="logs/ppo/eval_results.jsonl", help="Output JSONL path")

    return parser.parse_args()


def main():
    args = parse_args()

    from sb3_contrib import MaskablePPO
    from src.ppo.clash_royale_env import ClashRoyaleEnv, EnvConfig
    from src.ppo.reward import RewardConfig

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return 1

    # Parse regions
    capture_region = None
    if args.capture_region:
        parts = args.capture_region.split(",")
        if len(parts) != 4:
            print("Error: --capture-region must be left,top,width,height")
            return 1
        capture_region = tuple(int(p.strip()) for p in parts)

    game_region = None
    if args.game_region:
        parts = args.game_region.split(",")
        if len(parts) != 4:
            print("Error: --game-region must be left,top,width,height")
            return 1
        game_region = tuple(int(p.strip()) for p in parts)

    # Create environment (no training, no pauses)
    env_config = EnvConfig(
        window_title=args.window_title,
        capture_region=capture_region,
        game_region=game_region,
        use_perception=True,
        dry_run=False,  # Actually play the game
        reward_config=RewardConfig(),
        templates_dir=args.templates_dir,
        device=args.device,
        pause_between_episodes=True,  # Wait for operator to queue next game
        visualize=args.visualize,
        n_frames=args.n_frames,
        card_confidence_threshold=args.card_confidence,
    )

    env = ClashRoyaleEnv(config=env_config, project_root=PROJECT_ROOT)

    # Load trained model
    print(f"[Eval] Loading model: {args.model}")
    model = MaskablePPO.load(args.model, env=env, device=args.device)
    print(f"[Eval] Model loaded. Evaluating {args.num_games} games.")
    print(f"[Eval] Deterministic: {args.deterministic}")
    print(f"={'=' * 60}")

    # Metrics tracking
    results = []
    start_time = time.time()

    try:
        for game_num in range(1, args.num_games + 1):
            print(f"\n--- Game {game_num}/{args.num_games} ---")
            print("[Eval] Waiting for game to start...")

            obs, info = env.reset()
            done = False
            step_count = 0

            while not done:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1

            # Extract episode metrics
            outcome = info.get("outcome", "unknown")
            ep_data = {
                "game": game_num,
                "outcome": outcome,
                "reward": round(info.get("episode_reward", 0.0), 3),
                "cards_played": info.get("cards_played", 0),
                "card_cost_avg": round(info.get("card_cost_avg", 0.0), 2),
                "noop_ratio": round(info.get("noop_ratio", 0.0), 3),
                "episode_length": info.get("episode_length", step_count),
                "truncation_reason": info.get("truncation_reason", ""),
            }
            results.append(ep_data)

            # Print per-game result
            icon = {"win": "W", "loss": "L", "draw": "D"}.get(outcome, "?")
            print(f"[Eval] Game {game_num}: {icon} | reward={ep_data['reward']:+.1f} | "
                  f"cards={ep_data['cards_played']} | steps={ep_data['episode_length']} | "
                  f"noop={ep_data['noop_ratio']:.0%}")

    except KeyboardInterrupt:
        print("\n[Eval] Interrupted by user.")
    finally:
        env.close()

    # Print summary
    if not results:
        print("\n[Eval] No games completed.")
        return 0

    duration = time.time() - start_time
    n = len(results)
    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    draws = sum(1 for r in results if r["outcome"] == "draw")
    rewards = [r["reward"] for r in results]
    cards = [r["cards_played"] for r in results]
    lengths = [r["episode_length"] for r in results]
    noops = [r["noop_ratio"] for r in results]
    costs = [r["card_cost_avg"] for r in results if r["card_cost_avg"] > 0]

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION SUMMARY ({n} games, {duration / 60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"  Model:       {args.model}")
    print(f"  Win rate:    {wins}/{n} = {wins / n:.0%}")
    print(f"  Outcomes:    {wins}W / {losses}L / {draws}D")
    print(f"  Avg reward:  {sum(rewards) / n:+.1f}  (min={min(rewards):+.1f}, max={max(rewards):+.1f})")
    print(f"  Avg cards:   {sum(cards) / n:.1f}  (min={min(cards)}, max={max(cards)})")
    print(f"  Avg length:  {sum(lengths) / n:.0f} steps")
    print(f"  Avg noop:    {sum(noops) / n:.0%}")
    if costs:
        print(f"  Avg card cost: {sum(costs) / len(costs):.1f} elixir")
    print(f"{'=' * 60}")

    # Per-game table
    print(f"\n  {'Game':>4}  {'Result':>6}  {'Reward':>7}  {'Cards':>5}  {'Steps':>5}  {'Noop%':>5}")
    print(f"  {'----':>4}  {'------':>6}  {'-------':>7}  {'-----':>5}  {'-----':>5}  {'-----':>5}")
    for r in results:
        icon = {"win": "WIN", "loss": "LOSS", "draw": "DRAW"}.get(r["outcome"], "?")
        print(f"  {r['game']:>4}  {icon:>6}  {r['reward']:>+7.1f}  {r['cards_played']:>5}  "
              f"{r['episode_length']:>5}  {r['noop_ratio']:>4.0%}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[Eval] Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
