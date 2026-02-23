#!/usr/bin/env python3
"""
Process recorded matches into BC training datasets.

Reads all session directories from the recordings folder and converts them
into .npz training files using DatasetBuilder. Each session produces one
.npz file containing (obs_arena, obs_vector, actions, masks, timestamps).

Usage:
    python dataset_builder_module/process_recordings.py
    python dataset_builder_module/process_recordings.py --recordings-dir click_logger/recordings
    python dataset_builder_module/process_recordings.py --output-dir data/bc_training --noop-ratio 0.2
"""

import argparse
import glob
import os
import sys
import types as _types

# ---------------------------------------------------------------------------
# Namespace package setup for src.* imports
#
# Pre-register `src` as a namespace package whose __path__ includes the
# module-local src/ directories. Then pre-register subpackages from the
# main codebase (src/src/) as namespace packages so their __init__.py
# files are bypassed (they eagerly import heavy dependencies).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

_dataset_src = os.path.join(_SCRIPT_DIR, "src")
_encoder_src = os.path.join(PROJECT_ROOT, "state_encoder_module", "src")
_action_src = os.path.join(PROJECT_ROOT, "action_builder_module", "src")

# Pre-register src as namespace package
_src_pkg = _types.ModuleType("src")
_src_pkg.__path__ = [_dataset_src, _encoder_src, _action_src]
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

# Bypass encoder/__init__.py (eagerly imports StateEncoder -> gymnasium)
_enc_pkg = _types.ModuleType("src.encoder")
_enc_pkg.__path__ = [os.path.join(_encoder_src, "encoder")]
_enc_pkg.__package__ = "src.encoder"
sys.modules["src.encoder"] = _enc_pkg

# Pre-register subpackages from main codebase (src/src/)
# Each bypasses __init__.py to avoid cascading import chains
_main_src = os.path.join(PROJECT_ROOT, "src", "src")

for _name, _subdir in [
    ("src.generation", os.path.join(_main_src, "generation")),
    ("src.detection", os.path.join(_main_src, "detection")),
    ("src.classification", os.path.join(_main_src, "classification")),
    ("src.data", os.path.join(_main_src, "data")),
    ("src.pipeline", os.path.join(_main_src, "pipeline")),
    ("src.ocr", os.path.join(_main_src, "ocr")),
]:
    _pkg = _types.ModuleType(_name)
    _pkg.__path__ = [_subdir]
    _pkg.__package__ = _name
    sys.modules[_name] = _pkg


def find_session_dirs(recordings_dir: str) -> list[str]:
    """Find all valid session directories in the recordings folder.

    A valid session has at least frames.jsonl and a screenshots folder.

    Args:
        recordings_dir: Path to the recordings root folder.

    Returns:
        List of absolute paths to session directories.
    """
    sessions = []
    if not os.path.isdir(recordings_dir):
        print(f"Warning: Recordings directory not found: {recordings_dir}")
        return sessions

    pattern = os.path.join(recordings_dir, "match_*")
    for session_path in sorted(glob.glob(pattern)):
        if not os.path.isdir(session_path):
            continue

        frames_path = os.path.join(session_path, "frames.jsonl")
        screenshots_dir = os.path.join(session_path, "screenshots")

        if os.path.exists(frames_path) and os.path.isdir(screenshots_dir):
            sessions.append(session_path)
        else:
            print(f"Skipping incomplete session: {session_path}")

    return sessions


def create_state_builder(detector_model=None, card_classifier_model=None):
    """Create EnhancedStateBuilder with perception pipeline if available.

    Args:
        detector_model: Path to YOLO detection model, or None for default.
        card_classifier_model: Path to card classifier model, or None for default.

    Returns None if the perception modules are not installed.
    """
    try:
        from src.pipeline.state_builder import StateBuilder
        from src.classification.card_classifier import CardPredictor
        from src.dataset.card_integration import EnhancedStateBuilder

        # Resolve model paths relative to project root
        if detector_model is None:
            detector_model = os.path.join(
                PROJECT_ROOT, "models", "best_yolov8s_50epochs_fixed_pregen_set.pt"
            )
        elif not os.path.isabs(detector_model):
            detector_model = os.path.join(PROJECT_ROOT, detector_model)

        if card_classifier_model is None:
            card_classifier_model = os.path.join(
                PROJECT_ROOT, "models", "card_classifier.pt"
            )
        elif not os.path.isabs(card_classifier_model):
            card_classifier_model = os.path.join(PROJECT_ROOT, card_classifier_model)

        if not os.path.exists(detector_model):
            print(f"Detection model not found: {detector_model}")
            print("Running without perception pipeline (zero-filled observations)")
            return None

        state_builder = StateBuilder(
            detection_model_path=detector_model,
            enable_ocr=False,
        )

        card_predictor = None
        if os.path.exists(card_classifier_model):
            card_predictor = CardPredictor(weights_path=card_classifier_model)
            print(f"[Perception] Loaded CardPredictor: {card_classifier_model}")
        else:
            print(f"Card classifier not found: {card_classifier_model}")
            print("Running without card classification")

        print(f"[Perception] Loaded detector: {detector_model}")
        return EnhancedStateBuilder(state_builder, card_predictor)

    except ImportError as e:
        print(f"Perception pipeline not available: {e}")
        print("Running without perception pipeline (zero-filled observations)")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Process recorded matches into BC training datasets"
    )
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "click_logger", "recordings"),
        help="Path to the recordings folder containing match_* sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "bc_training"),
        help="Output directory for .npz training files",
    )
    parser.add_argument(
        "--noop-ratio",
        type=float,
        default=0.15,
        help="Fraction of no-op frames to keep (0-1, default: 0.15)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sessions that already have an output .npz file",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default=None,
        help="Path to YOLO detection model (default: models/best_yolov8s_50epochs_fixed_pregen_set.pt)",
    )
    parser.add_argument(
        "--card-classifier",
        type=str,
        default=None,
        help="Path to card classifier model (default: models/card_classifier.pt)",
    )
    args = parser.parse_args()

    # Find all session directories
    print(f"Scanning for sessions in: {args.recordings_dir}")
    sessions = find_session_dirs(args.recordings_dir)

    if not sessions:
        print("No valid session directories found.")
        print("\nExpected structure:")
        print("  recordings/")
        print("    match_YYYYMMDD_HHMMSS/")
        print("      screenshots/")
        print("        frame_000000.jpg")
        print("        ...")
        print("      frames.jsonl")
        print("      actions.jsonl")
        print("      metadata.json")
        return 1

    print(f"Found {len(sessions)} session(s)")

    # Filter out existing if requested
    if args.skip_existing:
        os.makedirs(args.output_dir, exist_ok=True)
        filtered = []
        for session in sessions:
            session_name = os.path.basename(session)
            output_path = os.path.join(args.output_dir, f"{session_name}.npz")
            if os.path.exists(output_path):
                print(f"  Skipping (exists): {session_name}")
            else:
                filtered.append(session)
        sessions = filtered

        if not sessions:
            print("All sessions already processed.")
            return 0

        print(f"Processing {len(sessions)} new session(s)")

    # Create state builder (may be None if perception pipeline unavailable)
    state_builder = create_state_builder(
        detector_model=args.detector_model,
        card_classifier_model=args.card_classifier,
    )

    # Create state encoder
    from src.encoder.state_encoder import StateEncoder
    state_encoder = StateEncoder()

    # Create dataset builder
    from src.dataset.dataset_builder import DatasetBuilder
    builder = DatasetBuilder(
        enhanced_state_builder=state_builder,
        state_encoder=state_encoder,
    )

    # Process all sessions
    print(f"\nProcessing sessions (noop_keep_ratio={args.noop_ratio})...")
    print("-" * 60)

    all_stats = builder.build_from_multiple(
        session_dirs=sessions,
        output_dir=args.output_dir,
        noop_keep_ratio=args.noop_ratio,
    )

    # Print summary
    print("-" * 60)
    print("\nProcessing complete!")
    print(f"Output directory: {args.output_dir}\n")

    total_frames = 0
    total_actions = 0
    total_kept = 0

    for stats in all_stats:
        session_name = os.path.basename(stats.session_dir)
        print(f"{session_name}:")
        print(f"  Frames: {stats.total_frames} total, {stats.kept_after_downsample} kept")
        print(f"  Actions: {stats.action_frames} placements, {stats.noop_frames} no-ops")
        print(f"  Output: {stats.output_path}")
        print()

        total_frames += stats.total_frames
        total_actions += stats.action_frames
        total_kept += stats.kept_after_downsample

    print("=" * 60)
    print(f"Total: {len(all_stats)} sessions, {total_frames} frames, "
          f"{total_actions} actions, {total_kept} kept after downsampling")

    return 0


if __name__ == "__main__":
    sys.exit(main())
