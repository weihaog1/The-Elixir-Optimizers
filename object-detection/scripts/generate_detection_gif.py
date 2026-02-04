"""
Generate an animated GIF showing YOLOv8 object detection on Clash Royale gameplay.

Loads the trained model, runs detection on sampled video frames, draws bounding
boxes with class labels and confidence scores, and saves an animated GIF.

Usage:
    python scripts/generate_detection_gif.py \
        --model models/best_yolov8s_50epochs_fixed_pregen_set.pt \
        --video "../../gameplay-videos/pigs_lose_0_1_crowns(1).mp4" \
        --output ../../The-Elixir-Optimizers/docs/images/status/detection_demo.gif \
        --start-sec 30 --duration 8 --fps 4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# Color palette (BGR) for drawing - visually distinct colors
COLORS_BGR = [
    (46, 204, 113),   # Emerald green
    (52, 152, 219),   # Blue
    (231, 76, 60),    # Red
    (241, 196, 15),   # Yellow
    (155, 89, 182),   # Purple
    (230, 126, 34),   # Orange
    (26, 188, 156),   # Teal
    (243, 156, 18),   # Gold
    (192, 57, 43),    # Dark red
    (41, 128, 185),   # Dark blue
    (39, 174, 96),    # Dark green
    (142, 68, 173),   # Dark purple
]


def get_color(class_id: int) -> tuple:
    """Get a consistent color for a class ID."""
    return COLORS_BGR[class_id % len(COLORS_BGR)]


def draw_detections(frame: np.ndarray, results, model_names: dict) -> np.ndarray:
    """Draw bounding boxes with labels and confidence on a frame.

    Args:
        frame: BGR image (numpy array).
        results: Ultralytics detection results for this frame.
        model_names: Class ID -> name mapping from the model.

    Returns:
        Annotated frame (BGR).
    """
    img = frame.copy()

    if results[0].boxes is None:
        return img

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    det_count = len(boxes)

    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        color = get_color(cls_id)
        class_name = model_names.get(cls_id, f"cls_{cls_id}")

        # Shorten long class names for readability
        short_name = class_name.replace("-", " ").title()
        if len(short_name) > 16:
            short_name = short_name[:14] + ".."

        label = f"{short_name} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # Draw detection count overlay in top-left
    count_text = f"Detections: {det_count}"
    cv2.putText(img, count_text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, count_text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Model name tag in top-left below count
    model_text = "YOLOv8s | 155 classes"
    cv2.putText(img, model_text, (12, 66), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, model_text, (12, 66), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return img


def generate_gif(
    model_path: str,
    video_path: str,
    output_path: str,
    start_sec: float = 30.0,
    duration: float = 8.0,
    gif_fps: float = 4.0,
    conf: float = 0.35,
    iou: float = 0.45,
    max_width: int = 480,
    quality: int = 60,
):
    """Generate detection demo GIF from gameplay video.

    Args:
        model_path: Path to YOLOv8 weights.
        video_path: Path to gameplay video.
        output_path: Path to save the output GIF.
        start_sec: Start time in seconds (skip menu/loading).
        duration: Duration of the GIF in seconds.
        gif_fps: Frames per second in the output GIF.
        conf: Confidence threshold for detections.
        iou: IoU threshold for NMS.
        max_width: Maximum width of the GIF (height scales proportionally).
        quality: Color quantization quality (lower = smaller file, fewer colors).
    """
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    model_names = model.names

    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / video_fps

    print(f"Video: {video_w}x{video_h} @ {video_fps:.1f} FPS, {video_duration:.1f}s")

    # Calculate frame sampling
    start_frame = int(start_sec * video_fps)
    end_frame = min(int((start_sec + duration) * video_fps), total_frames)
    frame_interval = max(1, int(video_fps / gif_fps))
    target_frames = list(range(start_frame, end_frame, frame_interval))

    print(f"Sampling {len(target_frames)} frames from {start_sec:.1f}s to "
          f"{start_sec + duration:.1f}s (every {frame_interval} frames)")

    # Calculate resize dimensions
    scale = max_width / video_w
    out_w = max_width
    out_h = int(video_h * scale)
    print(f"Output size: {out_w}x{out_h}")

    # Process frames
    gif_frames = []
    for i, target_frame_idx in enumerate(target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: could not read frame {target_frame_idx}")
            continue

        # Run detection at full resolution for best results
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            device="cpu",
            verbose=False,
            imgsz=960,
        )

        # Draw detections on the frame
        annotated = draw_detections(frame, results, model_names)

        # Resize for GIF
        resized = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB for Pillow
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # Quantize to reduce file size
        pil_frame = pil_frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT)

        gif_frames.append(pil_frame)
        det_count = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"  Frame {i+1}/{len(target_frames)} "
              f"(video frame {target_frame_idx}): {det_count} detections")

    cap.release()

    if not gif_frames:
        print("Error: No frames were processed.")
        return

    # Save as animated GIF
    print(f"\nSaving GIF to: {output_path}")
    frame_duration_ms = int(1000 / gif_fps)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    # Report file size
    size_bytes = Path(output_path).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"GIF saved: {size_mb:.2f} MB ({len(gif_frames)} frames, "
          f"{gif_fps} FPS, {len(gif_frames)/gif_fps:.1f}s)")

    if size_mb > 10:
        print("Warning: GIF is larger than 10 MB. Consider reducing max_width, "
              "fps, duration, or quality.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate detection demo GIF from gameplay video"
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Path to YOLOv8 model weights"
    )
    parser.add_argument(
        "--video", "-v", type=str, required=True,
        help="Path to gameplay video"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Path to save output GIF"
    )
    parser.add_argument(
        "--start-sec", type=float, default=30.0,
        help="Start time in seconds (default: 30)"
    )
    parser.add_argument(
        "--duration", type=float, default=8.0,
        help="Duration in seconds (default: 8)"
    )
    parser.add_argument(
        "--fps", type=float, default=4.0,
        help="GIF frames per second (default: 4)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--max-width", type=int, default=480,
        help="Max GIF width in pixels (default: 480)"
    )

    args = parser.parse_args()

    generate_gif(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        start_sec=args.start_sec,
        duration=args.duration,
        gif_fps=args.fps,
        conf=args.conf,
        max_width=args.max_width,
    )


if __name__ == "__main__":
    main()
