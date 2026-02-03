"""Evaluate model on gameplay video - extract frames, run inference, compare."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def evaluate_on_video(
    video_path: str,
    model_path: str,
    output_dir: str = "results/video_eval",
    device: str = "mps",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 960,
    frame_interval: float = 1.0,
    comparison_model: str = None,
):
    """Evaluate model on gameplay video frames.

    Args:
        video_path: Path to gameplay video.
        model_path: Path to model weights.
        output_dir: Output directory for annotated frames.
        device: Inference device.
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        imgsz: Inference image size.
        frame_interval: Extract a frame every N seconds.
        comparison_model: Optional second model for side-by-side comparison.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    comp_model = YOLO(comparison_model) if comparison_model else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_step = max(1, int(fps * frame_interval))

    print(f"Video: {video_path}")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {fps:.1f}, Duration: {duration:.1f}s, Total frames: {total_frames}")
    print(f"  Extracting every {frame_interval}s ({total_frames // frame_step} frames)")
    print(f"Model: {model_path}, device={device}, imgsz={imgsz}, conf={conf}")

    all_det_counts = []
    all_fps = []
    false_positives = {"the-log": 0}
    frame_num = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_step == 0:
            timestamp = frame_num / fps

            # Run inference
            t0 = time.time()
            results = model.predict(
                frame, conf=conf, iou=iou, device=device,
                verbose=False, imgsz=imgsz,
            )
            infer_time = time.time() - t0
            infer_fps = 1.0 / infer_time if infer_time > 0 else 0

            n_dets = len(results[0].boxes)
            all_det_counts.append(n_dets)
            all_fps.append(infer_fps)

            # Count specific class detections for analysis
            if results[0].boxes is not None:
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                class_names = [results[0].names[c] for c in class_ids]
                for name in class_names:
                    if name in false_positives:
                        false_positives[name] += 1

            # Save annotated frame
            annotated = results[0].plot()

            # Add HUD
            cv2.putText(
                annotated,
                f"t={timestamp:.1f}s | {n_dets} dets | {infer_fps:.1f} FPS",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            frame_path = output_path / f"frame_{extracted:04d}_t{timestamp:.0f}s.jpg"
            cv2.imwrite(str(frame_path), annotated)

            # Side-by-side comparison if comparison model provided
            if comp_model:
                comp_results = comp_model.predict(
                    frame, conf=conf, iou=iou, device=device,
                    verbose=False, imgsz=imgsz,
                )
                comp_annotated = comp_results[0].plot()
                n_comp = len(comp_results[0].boxes)

                cv2.putText(
                    comp_annotated,
                    f"OLD: {n_comp} dets",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

                # Stack side by side
                h1, w1 = annotated.shape[:2]
                h2, w2 = comp_annotated.shape[:2]
                max_h = max(h1, h2)
                scale1 = max_h / h1
                scale2 = max_h / h2
                a1 = cv2.resize(annotated, (int(w1 * scale1), max_h))
                a2 = cv2.resize(comp_annotated, (int(w2 * scale2), max_h))
                combined = np.hstack([a1, a2])

                comp_path = output_path / f"compare_{extracted:04d}_t{timestamp:.0f}s.jpg"
                cv2.imwrite(str(comp_path), combined)

            extracted += 1

        frame_num += 1

    cap.release()

    # Print summary
    if all_det_counts:
        avg_dets = np.mean(all_det_counts)
        avg_fps = np.mean(all_fps)
        print(f"\n--- Evaluation Summary ---")
        print(f"Frames analyzed: {extracted}")
        print(f"Avg detections/frame: {avg_dets:.1f}")
        print(f"Min/Max detections: {min(all_det_counts)}/{max(all_det_counts)}")
        print(f"Avg inference FPS: {avg_fps:.1f}")
        for name, count in false_positives.items():
            print(f"  '{name}' detections: {count}")
        print(f"Annotated frames saved to: {output_path}")


def compare_timestamps(
    video_path: str,
    model_path: str,
    comparison_model: str,
    timestamps: list = None,
    output_dir: str = "results/comparison",
    device: str = "mps",
    conf: float = 0.25,
    imgsz: int = 960,
):
    """Run side-by-side comparison at specific timestamps."""
    if timestamps is None:
        timestamps = [15, 30, 50, 70, 90, 110, 125, 140, 155, 165]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_new = YOLO(model_path)
    model_old = YOLO(comparison_model)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Comparing at timestamps: {timestamps}")

    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Could not read frame at t={ts}s")
            continue

        # New model
        r_new = model_new.predict(frame, conf=conf, device=device, verbose=False, imgsz=imgsz)
        ann_new = r_new[0].plot()
        n_new = len(r_new[0].boxes)

        # Old model
        r_old = model_old.predict(frame, conf=conf, device=device, verbose=False, imgsz=imgsz)
        ann_old = r_old[0].plot()
        n_old = len(r_old[0].boxes)

        # Labels
        cv2.putText(ann_new, f"NEW: {n_new} dets", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(ann_old, f"OLD: {n_old} dets", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Stack
        combined = np.hstack([ann_new, ann_old])
        out_path = output_path / f"compare_t{ts:03d}s.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"  t={ts:3d}s: NEW={n_new} dets, OLD={n_old} dets")

    cap.release()
    print(f"Comparisons saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on gameplay video")
    sub = parser.add_subparsers(dest="command")

    # Full evaluation
    eval_p = sub.add_parser("eval", help="Full video evaluation")
    eval_p.add_argument("--video", required=True)
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--output", default="results/video_eval")
    eval_p.add_argument("--device", default="mps")
    eval_p.add_argument("--conf", type=float, default=0.25)
    eval_p.add_argument("--iou", type=float, default=0.45)
    eval_p.add_argument("--imgsz", type=int, default=960)
    eval_p.add_argument("--interval", type=float, default=1.0, help="Seconds between frames")
    eval_p.add_argument("--compare", default=None, help="Comparison model path")

    # Timestamp comparison
    comp_p = sub.add_parser("compare", help="Side-by-side at specific timestamps")
    comp_p.add_argument("--video", required=True)
    comp_p.add_argument("--new-model", required=True)
    comp_p.add_argument("--old-model", required=True)
    comp_p.add_argument("--output", default="results/comparison")
    comp_p.add_argument("--device", default="mps")
    comp_p.add_argument("--conf", type=float, default=0.25)
    comp_p.add_argument("--imgsz", type=int, default=960)

    args = parser.parse_args()

    if args.command == "eval":
        evaluate_on_video(
            video_path=args.video,
            model_path=args.model,
            output_dir=args.output,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            frame_interval=args.interval,
            comparison_model=args.compare,
        )
    elif args.command == "compare":
        compare_timestamps(
            video_path=args.video,
            model_path=args.new_model,
            comparison_model=args.old_model,
            output_dir=args.output,
            device=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
        )
    else:
        parser.print_help()
