"""Resize gameplay frames to training resolution and compare detection results."""

import cv2
import json
import os
from ultralytics import YOLO

VIDEO = "/Users/alanguo/Codin/CS175/gameplay-videos/pigs_lose_0_1_crowns(1).mp4"
MODEL = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/models/best.pt"
OUT_ORIG = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/analysis/detections"
OUT_RESIZED_FRAMES = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/analysis/resized_frames"
OUT_RESIZED_DETS = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/analysis/resized_detections"

# Training data resolution (from remote instance)
TRAIN_W, TRAIN_H = 568, 896

TIMESTAMPS = [15, 30, 50, 70, 90, 110, 125, 140, 155, 165]

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video FPS: {fps}")
print(f"Original video res: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Resizing to: {TRAIN_W}x{TRAIN_H} (training resolution)")
print(f"Model classes: {len(model.names)}")
print()

# Also test with different imgsz values
IMGSZ_OPTIONS = [640, 960]
CONF_THRESHOLDS = [0.25, 0.5]

comparison_summary = []

for i, ts in enumerate(TIMESTAMPS):
    frame_idx = int(ts * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame at {ts}s")
        continue

    # Resize to training resolution
    resized = cv2.resize(frame, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
    resized_path = os.path.join(OUT_RESIZED_FRAMES, f"frame_{i:02d}_t{ts}s_resized.jpg")
    cv2.imwrite(resized_path, resized)

    frame_results = {"timestamp": ts, "frame_index": frame_idx, "configs": {}}

    for imgsz in IMGSZ_OPTIONS:
        for conf_thresh in CONF_THRESHOLDS:
            # Run on RESIZED frame
            results = model.predict(resized, conf=conf_thresh, iou=0.45,
                                    device="mps", verbose=False, imgsz=imgsz)

            annotated = results[0].plot()
            tag = f"resized_imgsz{imgsz}_conf{conf_thresh}"
            ann_path = os.path.join(OUT_RESIZED_DETS, f"frame_{i:02d}_t{ts}s_{tag}.jpg")
            cv2.imwrite(ann_path, annotated)

            dets = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                dets.append({
                    "class_id": cls_id,
                    "class_name": model.names[cls_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                })

            json_path = os.path.join(OUT_RESIZED_DETS, f"frame_{i:02d}_t{ts}s_{tag}.json")
            with open(json_path, "w") as f:
                json.dump({
                    "timestamp": ts,
                    "input": "resized",
                    "input_resolution": f"{TRAIN_W}x{TRAIN_H}",
                    "imgsz": imgsz,
                    "conf_threshold": conf_thresh,
                    "num_detections": len(dets),
                    "detections": dets,
                }, f, indent=2)

            config_key = f"resized_imgsz{imgsz}_conf{conf_thresh}"
            frame_results["configs"][config_key] = len(dets)

            # Also run on ORIGINAL frame with same imgsz for comparison
            results_orig = model.predict(frame, conf=conf_thresh, iou=0.45,
                                         device="mps", verbose=False, imgsz=imgsz)

            orig_dets = []
            for box in results_orig[0].boxes:
                cls_id = int(box.cls[0])
                orig_dets.append({
                    "class_id": cls_id,
                    "class_name": model.names[cls_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                })

            orig_tag = f"original_imgsz{imgsz}_conf{conf_thresh}"
            orig_json = os.path.join(OUT_RESIZED_DETS, f"frame_{i:02d}_t{ts}s_{orig_tag}.json")
            with open(orig_json, "w") as f:
                json.dump({
                    "timestamp": ts,
                    "input": "original",
                    "input_resolution": "1080x1920",
                    "imgsz": imgsz,
                    "conf_threshold": conf_thresh,
                    "num_detections": len(orig_dets),
                    "detections": orig_dets,
                }, f, indent=2)

            # Also save annotated original with this imgsz
            ann_orig = results_orig[0].plot()
            ann_orig_path = os.path.join(OUT_RESIZED_DETS, f"frame_{i:02d}_t{ts}s_{orig_tag}.jpg")
            cv2.imwrite(ann_orig_path, ann_orig)

            frame_results["configs"][orig_tag] = len(orig_dets)

    comparison_summary.append(frame_results)
    print(f"Frame {i} (t={ts}s):")
    for k, v in sorted(frame_results["configs"].items()):
        print(f"  {k}: {v} detections")
    print()

cap.release()

# Save summary
with open(os.path.join(OUT_RESIZED_DETS, "comparison_summary.json"), "w") as f:
    json.dump(comparison_summary, f, indent=2)

print("Done. All comparisons saved.")
