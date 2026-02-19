"""Frame-by-frame video viewer with YOLO detection overlay.

Uses custom NMS for belonging (ally/enemy) prediction from the model.
No Y-position heuristic - belonging comes directly from the model output.

Controls:
  Right arrow  - advance by `skip` frames
  Left arrow   - go back by `skip` frames
  1-9          - set frame skip amount
  q            - quit
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from src.yolov8_custom.custom_utils import non_max_suppression

ARENA_CUTOFF_Y = 1550


def run_belonging_inference(model, frame, conf, iou, device, imgsz):
    """Run model and apply custom NMS to get 7-column output with belonging."""
    results = model.predict(
        frame, conf=conf, iou=iou, device=device,
        verbose=False, imgsz=imgsz,
    )
    # Get raw prediction tensor before standard NMS
    # Re-run forward pass to get raw output
    im = results[0].orig_img
    # Use the preprocessed tensor from results
    preds = model.model(results[0].orig_img if False else
                        torch.from_numpy(
                            cv2.resize(frame, (imgsz, imgsz))
                            .transpose(2, 0, 1)[np.newaxis]
                            .astype(np.float32) / 255.0
                        ).to(next(model.model.parameters()).device))

    # Apply custom NMS with belonging
    dets = non_max_suppression(
        preds, conf_thres=conf, iou_thres=iou,
        nc=model.model.model[-1].nc,
    )
    return dets[0], results[0].names


def draw_detections(display, dets, names, scale):
    """Draw detections with model-predicted belonging. No Y heuristic."""
    if len(dets) == 0:
        return 0

    for det in dets:
        x1, y1, x2, y2 = [int(v * scale) for v in det[:4]]
        conf = float(det[4])
        cls_id = int(det[5])
        belonging = int(det[6])  # 0=ally, 1=enemy

        color = (255, 150, 0) if belonging == 0 else (0, 0, 255)

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        class_name = names.get(cls_id, f"cls_{cls_id}")
        side_label = "A" if belonging == 0 else "E"
        label = f"{class_name} {conf:.0%} [{side_label}]"

        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(
            display, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1
        )
        cv2.putText(
            display, label, (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
        )

    return len(dets)


def run(video_path, model_path, device="mps", conf=0.25, iou=0.45, imgsz=960):
    model = YOLO(model_path)
    nc = model.model.model[-1].nc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Video: {frame_width}x{frame_height} @ {vid_fps:.0f}fps, {total_frames} frames")
    print(f"Model: {model_path} (nc={nc}) on {device}, imgsz={imgsz}, conf={conf}")
    print(f"Belonging from model output (0=ally/orange, 1=enemy/red)")
    print(f"Controls: [left/right] navigate, [1-9] set skip, [q] quit")

    # Warmup
    ret, frame = cap.read()
    if not ret:
        return
    print("Warming up model...")
    model.predict(frame, conf=conf, iou=iou, device=device, verbose=False, imgsz=imgsz)
    print("Ready.")

    frame_num = 0
    skip = 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()

    while True:
        arena_frame = frame[:ARENA_CUTOFF_Y, :]

        # Use ultralytics preprocessing + custom NMS for belonging
        results = model.predict(
            arena_frame, conf=conf, iou=iou, device=device,
            verbose=False, imgsz=imgsz,
        )
        # Get raw predictions by running forward pass again through the model
        # Use the preprocessed image from ultralytics
        preprocessed = results[0].speed  # just to trigger processing
        # Actually, we need raw preds. Use model() directly with proper preprocessing.
        from ultralytics.data.augment import LetterBox
        letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
        img = letterbox(image=arena_frame)
        img = img.transpose(2, 0, 1)[::-1].copy()  # HWC->CHW, BGR->RGB
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        img = img.to(next(model.model.parameters()).device)

        with torch.no_grad():
            preds = model.model(img)

        dets = non_max_suppression(
            preds, conf_thres=conf, iou_thres=iou, nc=nc,
        )[0]

        # Scale detections back from letterboxed coords to original arena frame
        if len(dets) > 0:
            from ultralytics.utils.ops import scale_boxes
            dets_scaled = dets.clone()
            dets_scaled[:, :4] = scale_boxes(
                img.shape[2:], dets[:, :4], arena_frame.shape[:2]
            )
            dets = dets_scaled.cpu().numpy()
        else:
            dets = np.zeros((0, 7))

        names = results[0].names

        # Scale for display
        display_h = 900
        scale = display_h / frame.shape[0]
        display_w = int(frame.shape[1] * scale)
        display = cv2.resize(frame, (display_w, display_h))

        # Draw arena cutoff line
        cutoff_scaled = int(ARENA_CUTOFF_Y * scale)
        cv2.line(
            display, (0, cutoff_scaled), (display_w, cutoff_scaled),
            (0, 255, 0), 2, cv2.LINE_AA,
        )

        # Draw detections with model belonging
        n_dets = draw_detections(display, dets, names, scale)

        # HUD
        hud = f"Frame {frame_num}/{total_frames} | Skip: {skip} | {n_dets} dets"
        cv2.putText(
            display, hud, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
        )

        cv2.imshow("Frame Viewer", display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == 83 or key == 3:  # right arrow
            frame_num = min(frame_num + skip, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
        elif key == 81 or key == 2:  # left arrow
            frame_num = max(frame_num - skip, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
        elif ord('1') <= key <= ord('9'):
            skip = key - ord('0')
            print(f"Skip set to {skip}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame-by-frame video viewer with YOLO detections")
    parser.add_argument("--video", required=True, help="Path to gameplay video")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    run(args.video, args.model, args.device, args.conf, args.iou, args.imgsz)
