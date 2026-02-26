"""Real-time YOLOv8 inference on Clash Royale gameplay video."""

import argparse
import time
import cv2
from ultralytics import YOLO
from src.classification.card_classifier import CardPredictor
from src.ocr.text_extractor import GameTextExtractor

# Card slot coordinates for 1080x1920
CARD_SLOTS = [
    (242, 1595, 430, 1830),
    (445, 1595, 633, 1830),
    (648, 1595, 836, 1830),
    (851, 1595, 1039, 1830),
]

# YOLO arena cutoff - exclude card/UI area below this y coordinate
ARENA_CUTOFF_Y = 1550


def run(video_path, model_path, device="mps", conf=0.25, iou=0.45, imgsz=960,
        card_model_path=None, enable_ocr=False, ocr_interval=5):
    model = YOLO(model_path)

    card_predictor = None
    if card_model_path:
        card_predictor = CardPredictor(card_model_path)
        print(f"Card classifier: {card_model_path} ({len(card_predictor.classes)} classes)")

    ocr = None
    if enable_ocr:
        ocr = GameTextExtractor(use_gpu=(device != "cpu"))
        print(f"OCR enabled (every {ocr_interval} frames)")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    arena_mid_y = frame_height * 0.47  # Arena vertical midpoint
    paused = False
    frame_num = 0

    print(f"Video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{frame_height} @ {vid_fps:.0f}fps, {total_frames} frames")
    print(f"Model: {model_path} on {device}, imgsz={imgsz}, conf={conf}")
    print(f"Arena midpoint Y: {arena_mid_y:.0f}px (ally below, enemy above)")
    print(f"Controls: [space] pause/resume, [q] quit, [s] save frame, [</>] step when paused")

    # Warmup the model
    ret, frame = cap.read()
    if not ret:
        return
    frame_num = 1
    print("Warming up model...")
    model.predict(frame, conf=conf, iou=iou, device=device, verbose=False, imgsz=imgsz)
    print("Ready. Playing at native speed.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_num = 0

    # OCR state (cached between updates)
    ocr_elixir = None
    ocr_timer = None
    ocr_regions = None

    start_time = time.time()

    while cap.isOpened():
        if not paused:
            # Calculate which frame we SHOULD be on based on elapsed time
            elapsed = time.time() - start_time
            target_frame = int(elapsed * vid_fps)

            # Skip ahead if we're behind
            if target_frame > frame_num + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_num = target_frame

            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

        t0 = time.time()
        arena_frame = frame[:ARENA_CUTOFF_Y, :]
        results = model.predict(
            arena_frame,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            imgsz=imgsz,
        )
        infer_ms = (time.time() - t0) * 1000

        n_dets = len(results[0].boxes) if results[0].boxes is not None else 0

        # OCR extraction (run periodically, cache results)
        if ocr is not None and frame_num % ocr_interval == 0:
            h, w = frame.shape[:2]
            if ocr_regions is None:
                ocr_regions = {k: v for k, v in ocr.scale_regions(w, h).items()
                               if k in ("timer", "elixir")}
            ocr_results = ocr.extract_game_text(frame, regions=ocr_regions)
            if ocr_results.elixir is not None:
                ocr_elixir = ocr_results.elixir
            if ocr_results.timer is not None:
                ocr_timer = ocr_results.timer

        # Resize first, then draw annotations (so text is legible)
        display_h = 900
        scale = display_h / frame.shape[0]
        display_w = int(frame.shape[1] * scale)
        display = cv2.resize(frame, (display_w, display_h))
        mid_y_scaled = int(arena_mid_y * scale)

        # Draw arena midline
        cv2.line(display, (0, mid_y_scaled), (display_w, mid_y_scaled),
                (0, 255, 255), 1, cv2.LINE_AA)

        # Draw YOLO arena cutoff line (neon green)
        cutoff_scaled = int(ARENA_CUTOFF_Y * scale)
        cv2.line(display, (0, cutoff_scaled), (display_w, cutoff_scaled),
                (0, 255, 0), 2, cv2.LINE_AA)

        # Draw OCR region bounding boxes (neon purple)
        if ocr_regions is not None:
            for region_name, (rx1, ry1, rx2, ry2) in ocr_regions.items():
                sx1, sy1 = int(rx1 * scale), int(ry1 * scale)
                sx2, sy2 = int(rx2 * scale), int(ry2 * scale)
                cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (255, 0, 255), 2)
                cv2.putText(display, region_name, (sx1, sy1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        if results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, c, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = [int(v * scale) for v in box]
                center_y = (y1 + y2) / 2

                side = 0 if center_y > mid_y_scaled else 1
                color = (255, 150, 0) if side == 0 else (0, 0, 255)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                class_name = results[0].names.get(cls_id, f"cls_{cls_id}")
                side_label = "A" if side == 0 else "E"
                label = f"{class_name} {c:.0%} [{side_label}]"

                font_scale = 0.45
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                label_y = max(y1 - 4, th + 4)
                cv2.rectangle(display, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
                cv2.putText(display, label, (x1 + 2, label_y - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Card classification - append bar below the display
        if card_predictor is not None:
            card_results = []
            for x1, y1, x2, y2 in CARD_SLOTS:
                crop = frame[y1:y2, x1:x2]
                name, card_conf = card_predictor.predict(crop)
                card_results.append((name, card_conf))

            import numpy as np
            bar_h = 50
            bar = np.zeros((bar_h, display_w, 3), dtype=np.uint8)
            cv2.putText(bar, "HAND:", (5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            for i, (name, card_conf) in enumerate(card_results):
                x_pos = 60 + i * (display_w // 4)
                color = (0, 255, 0) if card_conf > 0.8 else (0, 200, 255)
                cv2.putText(bar, f"{name}", (x_pos, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.putText(bar, f"{card_conf:.0%}", (x_pos, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            # Elixir on right side of bar
            if ocr_elixir is not None:
                elx_text = f"Elixir: {ocr_elixir}"
                cv2.putText(bar, elx_text, (display_w - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            display = np.vstack([display, bar])

        # OCR overlay (top-right)
        if ocr is not None:
            elixir_text = f"Elixir: {ocr_elixir}" if ocr_elixir is not None else "Elixir: ?"
            timer_text = ""
            if ocr_timer is not None:
                ot = " OT" if ocr_timer.is_overtime else ""
                timer_text = f"  Timer: {ocr_timer.minutes}:{ocr_timer.seconds:02d}{ot}"
            ocr_label = elixir_text + timer_text
            (tw, _), _ = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ocr_x = display_w - tw - 10
            cv2.putText(display, ocr_label, (ocr_x, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        # HUD overlay
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(display, f"{status} | Frame {frame_num}/{total_frames} | {infer_ms:.0f}ms | {n_dets} dets",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("Clash Royale Detection", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if not paused:
                # Reset clock so playback resumes from current position
                start_time = time.time() - (frame_num / vid_fps)
        elif key == ord('s'):
            save_path = f"frame_{frame_num}.jpg"
            cv2.imwrite(save_path, annotated)
            print(f"Saved: {save_path}")
        elif key == ord('.') and paused:
            ret, frame = cap.read()
            if ret:
                frame_num += 1
        elif key == ord(',') and paused:
            frame_num = max(0, frame_num - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to gameplay video")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960, help="Inference resolution (960 recommended)")
    parser.add_argument("--class-names", default=None, help="Path to class names YAML (e.g. 155-class config)")
    parser.add_argument("--card-model", default=None, help="Path to card classifier .pt weights")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for elixir and timer")
    parser.add_argument("--ocr-interval", type=int, default=5, help="Run OCR every N frames")
    args = parser.parse_args()

    run(args.video, args.model, args.device, args.conf, args.iou, args.imgsz,
        card_model_path=args.card_model, enable_ocr=args.ocr,
        ocr_interval=args.ocr_interval)
