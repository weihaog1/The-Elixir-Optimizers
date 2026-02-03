#!/usr/bin/env python3
"""
Extract battle frames from Clash Royale gameplay video.

This script extracts frames specifically from battle sequences,
skipping menus, loading screens, and end screens.

Part of Phase 5: Fine-tuning for Google Play Games.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def is_battle_frame(frame: np.ndarray) -> Tuple[bool, str]:
    """
    Detect if a frame is from an active battle.

    Uses color analysis and region detection to identify battle frames.
    Battle frames typically have:
    - Grass/arena colors in the middle section
    - Card bar at the bottom with purple/orange colors
    - Timer at the top

    Returns:
        Tuple of (is_battle, reason)
    """
    height, width = frame.shape[:2]

    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Region definitions (normalized to handle different resolutions)
    # Arena region (middle 60% of screen)
    arena_start = int(height * 0.1)
    arena_end = int(height * 0.78)
    arena_region = hsv[arena_start:arena_end, :, :]

    # Card bar region (bottom 20%)
    card_start = int(height * 0.78)
    card_region = hsv[card_start:, :, :]

    # Check for grass green in arena (common in all arenas)
    # Green: H=35-85, S>50, V>50
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(arena_region, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    # Check for elixir bar purple/pink at bottom
    # Purple: H=130-160, S>50
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(card_region, lower_purple, upper_purple)
    purple_ratio = np.sum(purple_mask > 0) / purple_mask.size

    # Check for overall brightness (loading screens are often darker)
    avg_brightness = np.mean(frame)

    # Battle frame criteria
    if green_ratio > 0.05 and purple_ratio > 0.01 and avg_brightness > 60:
        return True, "battle_detected"

    # Additional check: look for card icons at bottom
    # Cards have distinct colors
    bottom_strip = frame[int(height * 0.85):, int(width * 0.1):int(width * 0.9)]
    color_variance = np.std(bottom_strip)

    if color_variance > 40 and avg_brightness > 50:
        return True, "card_bar_detected"

    reasons = []
    if green_ratio < 0.05:
        reasons.append(f"low_green={green_ratio:.3f}")
    if purple_ratio < 0.01:
        reasons.append(f"low_purple={purple_ratio:.3f}")
    if avg_brightness < 60:
        reasons.append(f"dark={avg_brightness:.1f}")

    return False, ";".join(reasons) if reasons else "unknown"


def detect_scene_changes(
    video_path: str,
    threshold: float = 30.0,
    min_scene_frames: int = 30,
) -> List[Tuple[int, int]]:
    """
    Detect scene changes in video.

    Returns list of (start_frame, end_frame) tuples for each scene.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    scenes = []
    prev_frame = None
    scene_start = 0
    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Detecting scenes") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))  # Downscale for speed

            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = np.mean(diff)

                if mean_diff > threshold:
                    # Scene change detected
                    if frame_idx - scene_start >= min_scene_frames:
                        scenes.append((scene_start, frame_idx))
                    scene_start = frame_idx

            prev_frame = gray
            frame_idx += 1
            pbar.update(1)

    # Add final scene
    if frame_idx - scene_start >= min_scene_frames:
        scenes.append((scene_start, frame_idx))

    cap.release()
    return scenes


def extract_battle_frames(
    video_path: str,
    output_dir: str,
    target_fps: float = 1.0,
    min_confidence: int = 2,
    resize: Optional[Tuple[int, int]] = (540, 960),
    max_frames: Optional[int] = None,
) -> dict:
    """
    Extract frames from battle sequences in the video.

    Args:
        video_path: Path to input video
        output_dir: Directory for output frames
        target_fps: Frames per second to extract
        min_confidence: Minimum consecutive battle detections
        resize: Target resolution (width, height)
        max_frames: Maximum frames to extract

    Returns:
        Dictionary with extraction statistics
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / target_fps)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "video_path": video_path,
        "total_video_frames": total_frames,
        "video_fps": video_fps,
        "target_fps": target_fps,
        "frames_examined": 0,
        "battle_frames_found": 0,
        "frames_extracted": 0,
        "non_battle_reasons": {},
    }

    consecutive_battle = 0
    extracted_count = 0
    frame_idx = 0

    with tqdm(total=total_frames, desc="Extracting battle frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                stats["frames_examined"] += 1

                is_battle, reason = is_battle_frame(frame)

                if is_battle:
                    consecutive_battle += 1
                    stats["battle_frames_found"] += 1

                    # Only extract after confidence threshold
                    if consecutive_battle >= min_confidence:
                        # Resize if specified
                        if resize:
                            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

                        # Save frame
                        timestamp = frame_idx / video_fps
                        filename = f"battle_{extracted_count:05d}_{timestamp:.2f}s.png"
                        cv2.imwrite(str(output_path / filename), frame)

                        extracted_count += 1
                        stats["frames_extracted"] += 1

                        if max_frames and extracted_count >= max_frames:
                            break
                else:
                    consecutive_battle = 0
                    # Track non-battle reasons
                    if reason not in stats["non_battle_reasons"]:
                        stats["non_battle_reasons"][reason] = 0
                    stats["non_battle_reasons"][reason] += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()

    # Calculate percentages
    if stats["frames_examined"] > 0:
        stats["battle_frame_percentage"] = round(
            stats["battle_frames_found"] / stats["frames_examined"] * 100, 1
        )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract battle frames from Clash Royale gameplay video"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to gameplay video"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/battle_frames",
        help="Output directory for frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="540x960",
        help="Resize frames to WIDTHxHEIGHT (default: 540x960)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract"
    )
    parser.add_argument(
        "--confidence",
        type=int,
        default=2,
        help="Consecutive battle detections required (default: 2)"
    )
    parser.add_argument(
        "--scenes",
        action="store_true",
        help="Detect and print scene boundaries"
    )

    args = parser.parse_args()

    # Parse resize
    resize = None
    if args.resize:
        parts = args.resize.lower().split("x")
        if len(parts) == 2:
            resize = (int(parts[0]), int(parts[1]))

    if args.scenes:
        # Just detect scenes
        print("Detecting scene changes...")
        scenes = detect_scene_changes(args.video)
        print(f"\nFound {len(scenes)} scenes:")
        for i, (start, end) in enumerate(scenes):
            print(f"  Scene {i+1}: frames {start}-{end} ({end-start} frames)")
        return

    # Extract battle frames
    print(f"Extracting battle frames from: {args.video}")
    print(f"Output directory: {args.output}")
    print(f"Target FPS: {args.fps}")
    print(f"Resize: {resize}")

    stats = extract_battle_frames(
        video_path=args.video,
        output_dir=args.output,
        target_fps=args.fps,
        min_confidence=args.confidence,
        resize=resize,
        max_frames=args.max_frames,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Video frames: {stats['total_video_frames']}")
    print(f"Frames examined: {stats['frames_examined']}")
    print(f"Battle frames found: {stats['battle_frames_found']}")
    print(f"Frames extracted: {stats['frames_extracted']}")
    print(f"Battle frame %: {stats.get('battle_frame_percentage', 0):.1f}%")

    if stats["non_battle_reasons"]:
        print("\nNon-battle frame reasons:")
        for reason, count in sorted(stats["non_battle_reasons"].items(), key=lambda x: -x[1])[:5]:
            print(f"  {reason}: {count}")

    print(f"\nFrames saved to: {args.output}")


if __name__ == "__main__":
    main()
