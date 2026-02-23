"""
Frame extraction from Clash Royale gameplay videos.

This module extracts frames at a specified FPS for training data creation.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def get_video_info(video_path: str) -> dict:
    """Get video metadata.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary containing video metadata (fps, frame_count, width, height, duration).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: float = 2.0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None,
    image_format: str = "png",
    prefix: str = "frame",
) -> int:
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        target_fps: Target frames per second for extraction (default: 2.0).
        start_time: Start time in seconds (optional).
        end_time: End time in seconds (optional).
        resize: Optional (width, height) tuple to resize frames.
        image_format: Output image format (png, jpg).
        prefix: Prefix for output filenames.

    Returns:
        Number of frames extracted.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame interval
    frame_interval = int(video_fps / target_fps)
    if frame_interval < 1:
        frame_interval = 1

    # Set start and end frames
    start_frame = int(start_time * video_fps) if start_time else 0
    end_frame = int(end_time * video_fps) if end_time else total_frames
    end_frame = min(end_frame, total_frames)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    extracted_count = 0

    pbar = tqdm(
        total=(end_frame - start_frame) // frame_interval,
        desc="Extracting frames",
        unit="frames"
    )

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_frame - start_frame) % frame_interval == 0:
            # Resize if specified
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

            # Calculate timestamp
            timestamp = current_frame / video_fps

            # Save frame with timestamp in filename
            filename = f"{prefix}_{extracted_count:05d}_{timestamp:.2f}s.{image_format}"
            output_file = output_path / filename
            cv2.imwrite(str(output_file), frame)

            extracted_count += 1
            pbar.update(1)

        current_frame += 1

    pbar.close()
    cap.release()

    return extracted_count


def extract_specific_times(
    video_path: str,
    output_dir: str,
    timestamps: list[float],
    resize: Optional[Tuple[int, int]] = None,
    image_format: str = "png",
    prefix: str = "frame",
) -> int:
    """Extract frames at specific timestamps.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        timestamps: List of timestamps in seconds to extract.
        resize: Optional (width, height) tuple to resize frames.
        image_format: Output image format (png, jpg).
        prefix: Prefix for output filenames.

    Returns:
        Number of frames extracted.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extracted_count = 0

    for ts in tqdm(timestamps, desc="Extracting specific frames"):
        frame_num = int(ts * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame at {ts}s")
            continue

        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

        filename = f"{prefix}_{extracted_count:05d}_{ts:.2f}s.{image_format}"
        output_file = output_path / filename
        cv2.imwrite(str(output_file), frame)
        extracted_count += 1

    cap.release()
    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from Clash Royale gameplay video"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target FPS for extraction (default: 2.0)"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds"
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds"
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Resize frames to WIDTHxHEIGHT (e.g., 540x960)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Output image format (default: png)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print video info and exit"
    )

    args = parser.parse_args()

    # Print video info
    if args.info:
        info = get_video_info(args.video)
        print(f"Video: {args.video}")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']:.2f}")
        print(f"  Duration: {info['duration']:.2f}s")
        print(f"  Total frames: {info['frame_count']}")
        return

    # Parse resize argument
    resize = None
    if args.resize:
        parts = args.resize.lower().split("x")
        if len(parts) == 2:
            resize = (int(parts[0]), int(parts[1]))

    # Extract frames
    count = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        target_fps=args.fps,
        start_time=args.start,
        end_time=args.end,
        resize=resize,
        image_format=args.format,
    )

    print(f"\nExtracted {count} frames to {args.output}")


if __name__ == "__main__":
    main()
