#!/usr/bin/env python3
"""Standalone observation tensor visualizer for Clash Royale.

Captures frames from the game, runs perception, and displays the
observation tensors as live heatmaps — no training, no clicks.

Usage:
    # Live visualization (watch what the model sees)
    python ppo_module/visualize_obs.py \
        --window-title "Clash Royale - thegoodpersonplayer2"

    # Record frames for video
    python ppo_module/visualize_obs.py \
        --window-title "Clash Royale - thegoodpersonplayer2" \
        --save-dir vis_frames/ \
        --num-steps 100

    # Assemble video from saved frames
    ffmpeg -framerate 2 -i vis_frames/step_%04d.png -c:v libx264 -pix_fmt yuv420p obs_viz.mp4
"""

import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Namespace package setup (same pattern as run_ppo.py)
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
for sub in ("generation", "detection", "classification", "data", "pipeline"):
    pkg_name = f"src.{sub}"
    mod = _types.ModuleType(pkg_name)
    mod.__path__ = [os.path.join(_main_src, sub)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Clash Royale observation tensors (no training)"
    )
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--window-title", type=str, default="",
        help="Game window title for auto-detection",
    )
    capture_group.add_argument(
        "--capture-region", type=str, default="",
        help="Manual capture region: left,top,width,height",
    )
    parser.add_argument(
        "--save-dir", type=str, default="",
        help="Save visualization frames to this directory",
    )
    parser.add_argument(
        "--num-steps", type=int, default=0,
        help="Number of steps to capture (0 = run until Ctrl+C)",
    )
    parser.add_argument(
        "--no-perception", action="store_true",
        help="Disable YOLO detection (use zero-filled observations)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from src.ppo.clash_royale_env import ClashRoyaleEnv, EnvConfig

    capture_region = None
    if args.capture_region:
        parts = args.capture_region.split(",")
        if len(parts) != 4:
            print("Error: --capture-region must be left,top,width,height")
            return 1
        capture_region = tuple(int(p.strip()) for p in parts)

    config = EnvConfig(
        window_title=args.window_title,
        capture_region=capture_region,
        use_perception=not args.no_perception,
        dry_run=True,  # Never click
        visualize=True,
        vis_save_dir=args.save_dir,
        pause_between_episodes=False,
    )

    print("[Visualizer] Initializing environment (dry-run mode)...")
    env = ClashRoyaleEnv(config, project_root=PROJECT_ROOT)

    print("[Visualizer] Waiting for game to start...")
    obs, info = env.reset()
    print("[Visualizer] Game detected. Visualizing observations...")
    print("[Visualizer] Press Ctrl+C to stop.\n")

    # NOOP action index
    noop = 2304
    step = 0

    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(noop)
            step += 1

            if step % 10 == 0:
                print(f"  Step {step} | reward={reward:.3f} | "
                      f"phase={info.get('phase', '?')}")

            if args.num_steps > 0 and step >= args.num_steps:
                print(f"\n[Visualizer] Reached {args.num_steps} steps. Done.")
                break

            if terminated or truncated:
                print(f"\n[Visualizer] Episode ended at step {step}. Resetting...")
                obs, info = env.reset()
                step = 0

    except KeyboardInterrupt:
        print(f"\n[Visualizer] Stopped at step {step}.")
    finally:
        env.close()

    if args.save_dir:
        print(f"\n[Visualizer] Frames saved to {args.save_dir}/")
        print("  To make a video:")
        print(f"  ffmpeg -framerate 2 -i {args.save_dir}/step_%04d.png "
              "-c:v libx264 -pix_fmt yuv420p obs_viz.mp4")

    return 0


if __name__ == "__main__":
    sys.exit(main())
