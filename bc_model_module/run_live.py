#!/usr/bin/env python3
"""
Live game inference entry point.

Captures the Clash Royale game screen, runs the BC policy, and executes
card placements via PyAutoGUI. Ctrl+C to stop.

Usage:
    python bc_model_module/run_live.py --model-path models/bc/best_bc.pt --window-title "Clash Royale - thegoodpersonplayer2" --dry-run
    python bc_model_module/run_live.py --model-path models/bc/best_bc.pt --window-title "Clash Royale - thegoodpersonplayer2"
    python bc_model_module/run_live.py --help
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Import path setup (same pattern as process_recordings.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

# ---------------------------------------------------------------------------
# Namespace package setup for `src.*` imports
#
# The root src/__init__.py eagerly imports detector.py which does a bare
# `from config import ...` that fails. To avoid this, we pre-register `src`
# as a namespace package in sys.modules whose __path__ only includes the
# module-local src/ directories. This lets `from src.bc ...`,
# `from src.encoder ...`, etc. resolve to module-local copies without ever
# touching the root src/__init__.py.
# ---------------------------------------------------------------------------
import types as _types

_bc_src = os.path.join(_SCRIPT_DIR, "src")
_encoder_src = os.path.join(PROJECT_ROOT, "state_encoder_module", "src")
_action_src = os.path.join(PROJECT_ROOT, "action_builder_module", "src")

_src_paths = [_bc_src, _encoder_src, _action_src]

# Pre-register src as namespace package
_src_pkg = _types.ModuleType("src")
_src_pkg.__path__ = _src_paths  # type: ignore[attr-defined]
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

# Pre-register src.bc to bypass its __init__.py (which eagerly imports
# CRFeatureExtractor -> encoder_constants -> gymnasium)
_bc_pkg = _types.ModuleType("src.bc")
_bc_pkg.__path__ = [os.path.join(_bc_src, "bc")]  # type: ignore[attr-defined]
_bc_pkg.__package__ = "src.bc"
sys.modules["src.bc"] = _bc_pkg

# Pre-register src.encoder to bypass its __init__.py (which eagerly
# imports StateEncoder -> gymnasium)
_enc_pkg = _types.ModuleType("src.encoder")
_enc_pkg.__path__ = [os.path.join(_encoder_src, "encoder")]  # type: ignore[attr-defined]
_enc_pkg.__package__ = "src.encoder"
sys.modules["src.encoder"] = _enc_pkg

# ---------------------------------------------------------------------------
# Pre-register subpackages from the main src/src/ codebase.
# Each is registered as a namespace package (bypassing __init__.py) so that
# only the specific .py files we import are loaded.
# ---------------------------------------------------------------------------
_main_src = os.path.join(PROJECT_ROOT, "src", "src")

# src.generation -- REAL label_list.py with 155 class names
# (needed by encoder_constants.py for CLASS_NAME_TO_ID and UNIT_TYPE_MAP)
_gen_pkg = _types.ModuleType("src.generation")
_gen_pkg.__path__ = [os.path.join(_main_src, "generation")]  # type: ignore[attr-defined]
_gen_pkg.__package__ = "src.generation"
sys.modules["src.generation"] = _gen_pkg

# src.detection -- CRDetector YOLO wrapper
_det_pkg = _types.ModuleType("src.detection")
_det_pkg.__path__ = [os.path.join(_main_src, "detection")]  # type: ignore[attr-defined]
_det_pkg.__package__ = "src.detection"
sys.modules["src.detection"] = _det_pkg

# src.classification -- CardPredictor (MiniResNet card classifier)
_cls_pkg = _types.ModuleType("src.classification")
_cls_pkg.__path__ = [os.path.join(_main_src, "classification")]  # type: ignore[attr-defined]
_cls_pkg.__package__ = "src.classification"
sys.modules["src.classification"] = _cls_pkg

# src.data -- ScreenConfig for card slot regions
_data_pkg = _types.ModuleType("src.data")
_data_pkg.__path__ = [os.path.join(_main_src, "data")]  # type: ignore[attr-defined]
_data_pkg.__package__ = "src.data"
sys.modules["src.data"] = _data_pkg

# src.pipeline -- GameState dataclasses
_pipe_pkg = _types.ModuleType("src.pipeline")
_pipe_pkg.__path__ = [os.path.join(_main_src, "pipeline")]  # type: ignore[attr-defined]
_pipe_pkg.__package__ = "src.pipeline"
sys.modules["src.pipeline"] = _pipe_pkg


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Live BC model inference for Clash Royale"
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the BC model checkpoint (best_bc.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)",
    )

    # Capture
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--window-title",
        type=str,
        default="",
        help="Game window title for auto-detection (requires pygetwindow)",
    )
    capture_group.add_argument(
        "--capture-region",
        type=str,
        default="",
        help="Manual capture region: left,top,width,height (e.g. 0,0,540,960)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Capture frames per second (default: 2.0)",
    )

    # Perception
    parser.add_argument(
        "--no-perception",
        action="store_true",
        help="Disable YOLO detection (use zero-filled observations)",
    )
    parser.add_argument(
        "--card-classifier",
        type=str,
        default=os.path.join("models", "card_classifier.pt"),
        help="Path to card classifier weights (default: models/card_classifier.pt)",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="",
        help="Path to YOLO detector weights (default: models/best_yolov8s_50epochs_fixed_pregen_set.pt)",
    )

    # Policy
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Minimum logit score to execute an action (default: 0.0 = always)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.5,
        help="Minimum seconds between card plays (default: 0.5)",
    )
    parser.add_argument(
        "--max-apm",
        type=int,
        default=20,
        help="Maximum actions per minute (default: 20)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (>1 = more diverse, <1 = more greedy, default: 1.0)",
    )
    parser.add_argument(
        "--noop-frames",
        type=int,
        default=0,
        help="Force noop for N frames after each card play (0=auto from rate limit, default: 0)",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=0.5,
        help="Logit penalty for recently-used actions (default: 0.5)",
    )
    parser.add_argument(
        "--repeat-memory",
        type=int,
        default=5,
        help="Number of recent actions to penalize (default: 5)",
    )

    # Safety
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions but do not execute clicks",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join("logs", "live"),
        help="Directory for session logs (default: logs/live)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-frame console output",
    )

    return parser.parse_args()


def main():
    """Build config from CLI args and run the inference engine."""
    args = parse_args()

    from src.bc.live_inference import LiveConfig, LiveInferenceEngine

    # Parse capture region if provided
    capture_region = None
    if args.capture_region:
        parts = args.capture_region.split(",")
        if len(parts) != 4:
            print("Error: --capture-region must be left,top,width,height")
            return 1
        capture_region = tuple(int(p.strip()) for p in parts)

    config = LiveConfig(
        model_path=args.model_path,
        device=args.device,
        window_title=args.window_title,
        capture_region=capture_region,
        capture_fps=args.fps,
        use_perception=not args.no_perception,
        card_classifier_path=args.card_classifier,
        confidence_threshold=args.confidence,
        action_cooldown=args.cooldown,
        max_actions_per_minute=args.max_apm,
        temperature=args.temperature,
        noop_frames_after_play=args.noop_frames,
        repeat_penalty=args.repeat_penalty,
        repeat_memory=args.repeat_memory,
        dry_run=args.dry_run,
        log_dir=args.log_dir,
        verbose=not args.quiet,
    )

    # Override detector model path if provided
    if args.detector_model:
        config.detector_model_paths = [args.detector_model]

    engine = LiveInferenceEngine(config, project_root=PROJECT_ROOT)
    engine.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
