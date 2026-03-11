"""Dual-detector real-time demo for Clash Royale.

Runs both YOLOv8m detectors (small-sprite + large-sprite) via ComboDetector
and displays merged detections with belonging prediction, card hand, and OCR.

Controls:
  Space     - toggle pause/play
  Right/Left arrow or . / , - step forward/back (works in both modes)
  +/-       - adjust frame skip (1-30)
  [ / ]     - decrease/increase playback speed (0.25x - 4x)
  c         - toggle confidence display
  d         - toggle detector source overlay (D1 vs D2)
  s         - save current frame
  q         - quit

Usage:
  python dual_demo.py --video gameplay.mp4
  python dual_demo.py --video gameplay.mp4 --autoplay --speed 0.5
  python dual_demo.py --video gameplay.mp4 --card-model models/card_classifier.pt --ocr
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from src.detection.combo_detector import ComboDetector

ARENA_CUTOFF_Y = 1550

# Card slot coordinates for 1080x1920
CARD_SLOTS = [
    (242, 1595, 430, 1830),
    (445, 1595, 633, 1830),
    (648, 1595, 836, 1830),
    (851, 1595, 1039, 1830),
]

ALLY_COLOR = (255, 150, 0)   # Orange
ENEMY_COLOR = (0, 0, 255)    # Red
BASE_COLOR = (200, 200, 200) # Gray for base classes (towers/UI)

# Classes that are scene elements, not combatants
NON_COMBAT_CLASSES = {
    "bar", "bar-level", "clock", "emote", "elixir",
    "tower-bar", "king-tower-bar", "skeleton-king-bar",
    "dagger-duchess-tower-bar", "evolution-symbol",
    "ice-spirit-evolution-symbol",
}

SPEED_STEPS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def draw_detections(display, dets, names, scale, show_conf=True, show_source=False,
                    base_indices=None, d1_globals=None, d2_globals=None):
    """Draw detections with belonging coloring."""
    if len(dets) == 0:
        return 0

    for d in dets:
        x1, y1, x2, y2 = [int(v * scale) for v in d[:4]]
        conf = float(d[4])
        cls_id = int(d[5])
        belonging = int(d[6])

        class_name = names.get(cls_id, f"cls_{cls_id}")

        if class_name in NON_COMBAT_CLASSES:
            color = BASE_COLOR
        elif belonging == 0:
            color = ALLY_COLOR
        else:
            color = ENEMY_COLOR

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        parts = [class_name]
        if show_conf:
            parts.append(f"{conf:.0%}")
        side_char = "A" if belonging == 0 else "E"
        parts.append(f"[{side_char}]")

        if show_source and d1_globals and d2_globals:
            if cls_id in d1_globals and cls_id not in d2_globals:
                parts.append("D1")
            elif cls_id in d2_globals and cls_id not in d1_globals:
                parts.append("D2")
            else:
                parts.append("B")

        label = " ".join(parts)

        font_scale = 0.4
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(display, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
        cv2.putText(display, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return len(dets)


def draw_info_panel(display_w, card_results, ocr_elixir, ocr_timer):
    """Draw a bottom info panel with cards, elixir, and timer."""
    panel_h = 70
    panel = np.zeros((panel_h, display_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark background

    # Divider line at top
    cv2.line(panel, (0, 0), (display_w, 0), (80, 80, 80), 1)

    # -- Left section: Cards --
    if card_results:
        cv2.putText(panel, "HAND", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        slot_w = (display_w - 200) // 4  # Reserve right side for OCR
        for i, (name, card_conf) in enumerate(card_results):
            x = 8 + i * slot_w

            # Card name
            if card_conf > 0.8:
                name_color = (100, 255, 100)  # Green for high conf
            elif card_conf > 0.5:
                name_color = (100, 220, 255)  # Cyan for medium
            else:
                name_color = (100, 100, 100)  # Dim for low

            cv2.putText(panel, name, (x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, name_color, 1)

            # Confidence bar
            bar_x = x
            bar_y = 50
            bar_w = max(int(slot_w * 0.7 * card_conf), 1)
            bar_max_w = int(slot_w * 0.7)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 6),
                         (60, 60, 60), -1)
            bar_color = (100, 255, 100) if card_conf > 0.8 else (100, 220, 255)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6),
                         bar_color, -1)

            # Percentage right-aligned to bar
            cv2.putText(panel, f"{card_conf:.0%}", (bar_x + bar_max_w + 4, bar_y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    # -- Right section: Elixir + Timer --
    right_x = display_w - 160

    if ocr_elixir is not None:
        # Elixir with colored fill bar
        cv2.putText(panel, "ELIXIR", (right_x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        elixir_val = min(ocr_elixir, 10)
        bar_full_w = 100
        bar_fill_w = int(bar_full_w * elixir_val / 10)
        cv2.rectangle(panel, (right_x, 24), (right_x + bar_full_w, 36),
                     (60, 60, 60), -1)
        # Purple elixir color
        cv2.rectangle(panel, (right_x, 24), (right_x + bar_fill_w, 36),
                     (220, 80, 220), -1)
        cv2.putText(panel, str(ocr_elixir), (right_x + bar_full_w + 6, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 80, 220), 1)

    if ocr_timer is not None:
        ot = " OT" if ocr_timer.is_overtime else ""
        timer_str = f"{ocr_timer.minutes}:{ocr_timer.seconds:02d}{ot}"
        timer_color = (80, 80, 255) if ocr_timer.is_overtime else (255, 255, 255)
        cv2.putText(panel, timer_str, (right_x, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, timer_color, 1)
    elif ocr_elixir is not None:
        # Show placeholder if OCR enabled but no timer yet
        cv2.putText(panel, "--:--", (right_x, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    return panel


def run(video_path, d1_model, d2_model, split_config, device="mps",
        conf=0.25, iou=0.45, imgsz=960, autoplay=False, speed=1.0,
        card_model_path=None, enable_ocr=False, ocr_interval=5):
    combo = ComboDetector(
        model_paths=[d1_model, d2_model],
        split_config_path=split_config,
        conf=conf, iou=iou, device=device, imgsz=imgsz,
    )

    # Precompute detector-exclusive class sets for source overlay
    d1_globals = set(combo.split_config["detector1"]["global_indices"])
    d2_globals = set(combo.split_config["detector2"]["global_indices"])
    base_indices = set(combo.split_config["base_indices"])

    # Card classifier
    card_predictor = None
    if card_model_path:
        from src.classification.card_classifier import CardPredictor
        card_predictor = CardPredictor(card_model_path)
        print(f"Card classifier: {card_model_path} ({len(card_predictor.classes)} classes)")

    # OCR
    ocr = None
    if enable_ocr:
        from src.ocr.text_extractor import GameTextExtractor
        ocr = GameTextExtractor(use_gpu=(device != "cpu"))
        print(f"OCR enabled (every {ocr_interval} frames)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Video: {frame_w}x{frame_h} @ {vid_fps:.0f}fps, {total_frames} frames")
    print(f"D1: {d1_model}")
    print(f"D2: {d2_model}")
    print(f"Device: {device}, imgsz={imgsz}, conf={conf}, iou={iou}")
    print("Warming up...")

    combo.warmup()

    ret, frame = cap.read()
    if not ret:
        return
    frame_num = 0

    print("Ready.")
    print("Controls: [space] pause, [</>] step, [+/-] skip, [[ / ]] speed, [c] conf, [d] source, [s] save, [q] quit")

    paused = not autoplay
    show_conf = True
    show_source = False
    skip = 1
    speed_idx = min(range(len(SPEED_STEPS)), key=lambda i: abs(SPEED_STEPS[i] - speed))
    playback_speed = SPEED_STEPS[speed_idx]
    last_frame_time = time.time()

    # Cache inference results
    cached_dets = None
    cached_infer_ms = 0
    frame_changed = True

    # OCR state (cached between updates)
    ocr_elixir = None
    ocr_timer = None
    ocr_regions = None

    # Card state (cached between updates)
    card_results = None

    while True:
        if frame_changed:
            t0 = time.time()
            cached_dets = combo.infer(frame, arena_cutoff=ARENA_CUTOFF_Y)
            cached_infer_ms = (time.time() - t0) * 1000

            # Card classification on frame change
            if card_predictor is not None:
                card_results = []
                for x1, y1, x2, y2 in CARD_SLOTS:
                    crop = frame[y1:y2, x1:x2]
                    name, card_conf = card_predictor.predict(crop)
                    card_results.append((name, card_conf))

            # OCR on frame change (respecting interval)
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

            frame_changed = False

        dets = cached_dets
        infer_ms = cached_infer_ms

        # Scale for display
        display_h = 900
        scale = display_h / frame.shape[0]
        display_w = int(frame.shape[1] * scale)
        display = cv2.resize(frame, (display_w, display_h))

        # Arena cutoff line
        cutoff_scaled = int(ARENA_CUTOFF_Y * scale)
        cv2.line(display, (0, cutoff_scaled), (display_w, cutoff_scaled),
                 (0, 255, 0), 1, cv2.LINE_AA)

        # Draw detections
        n_dets = draw_detections(
            display, dets, combo.names, scale,
            show_conf=show_conf, show_source=show_source,
            base_indices=base_indices, d1_globals=d1_globals, d2_globals=d2_globals,
        )

        # HUD (top bar)
        status = "PAUSED" if paused else f"PLAYING {playback_speed}x"
        hud = f"{status} | Frame {frame_num}/{total_frames} | {infer_ms:.0f}ms | {n_dets} dets | skip={skip}"
        cv2.putText(display, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Detector stats
        n_d1 = sum(1 for d in dets if int(d[5]) in d1_globals and int(d[5]) not in d2_globals)
        n_d2 = sum(1 for d in dets if int(d[5]) in d2_globals and int(d[5]) not in d1_globals)
        n_base = sum(1 for d in dets if int(d[5]) in base_indices)
        stats = f"D1:{n_d1} D2:{n_d2} Base:{n_base}"
        cv2.putText(display, stats, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Bottom info panel (cards + OCR)
        has_panel_data = card_results or ocr_elixir is not None or ocr_timer is not None
        if has_panel_data:
            panel = draw_info_panel(display_w, card_results, ocr_elixir, ocr_timer)
            display = np.vstack([display, panel])

        cv2.imshow("Dual Detector Demo", display)

        # Input handling
        wait_ms = 1 if not paused else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if not paused:
                last_frame_time = time.time()
        elif key == ord('c'):
            show_conf = not show_conf
        elif key == ord('d'):
            show_source = not show_source
        elif key == ord('s'):
            save_path = f"dual_frame_{frame_num}.jpg"
            cv2.imwrite(save_path, display)
            print(f"Saved: {save_path}")
        elif key in (ord('+'), ord('=')):
            skip = min(skip + 1, 30)
        elif key in (ord('-'), ord('_')):
            skip = max(skip - 1, 1)
        elif key == ord(']'):
            speed_idx = min(speed_idx + 1, len(SPEED_STEPS) - 1)
            playback_speed = SPEED_STEPS[speed_idx]
            last_frame_time = time.time()
        elif key == ord('['):
            speed_idx = max(speed_idx - 1, 0)
            playback_speed = SPEED_STEPS[speed_idx]
            last_frame_time = time.time()

        # Frame advancement
        advance = False
        direction = 1
        if key in (ord('.'), 83, 3):  # . or right arrow
            advance = True
            direction = 1
        elif key in (ord(','), 81, 2):  # , or left arrow
            advance = True
            direction = -1
        elif not paused:
            frame_interval = 1.0 / (vid_fps * playback_speed)
            if time.time() - last_frame_time >= frame_interval:
                advance = True
                direction = 1

        if advance:
            if direction == 1:
                new_frame = min(frame_num + skip, total_frames - 1)
            else:
                new_frame = max(frame_num - skip, 0)

            if new_frame != frame_num:
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num = new_frame
                frame_changed = True
                last_frame_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-detector real-time Clash Royale demo")
    parser.add_argument("--video", required=True, help="Path to gameplay video")
    parser.add_argument("--d1", default="models/dual_d1_best.pt", help="Detector 1 (small sprites) weights")
    parser.add_argument("--d2", default="models/dual_d2_best.pt", help="Detector 2 (large sprites) weights")
    parser.add_argument("--split-config", default="configs/split_config.json", help="Split config JSON")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--autoplay", action="store_true", help="Start playing immediately")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.25-4.0)")
    parser.add_argument("--card-model", default=None, help="Path to card classifier .pt weights")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for elixir and timer")
    parser.add_argument("--ocr-interval", type=int, default=5, help="Run OCR every N frames")
    args = parser.parse_args()

    run(args.video, args.d1, args.d2, args.split_config,
        args.device, args.conf, args.iou, args.imgsz,
        autoplay=args.autoplay, speed=args.speed,
        card_model_path=args.card_model, enable_ocr=args.ocr,
        ocr_interval=args.ocr_interval)
