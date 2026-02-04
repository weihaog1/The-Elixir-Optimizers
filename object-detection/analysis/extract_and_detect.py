"""Extract frames from gameplay video, run detection, and save results."""

import cv2
import json
import os
from ultralytics import YOLO

VIDEO = "/Users/alanguo/Codin/CS175/gameplay-videos/pigs_lose_0_1_crowns(1).mp4"
MODEL = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/models/best.pt"
OUT_FRAMES = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/analysis/frames"
OUT_DETS = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/analysis/detections"

# Extract 10 frames at evenly spaced timestamps
# Video is 177s, skip first 10s (lobby) and last 5s (end screen)
TIMESTAMPS = [15, 30, 50, 70, 90, 110, 125, 140, 155, 165]

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video FPS: {fps}")
print(f"Model classes: {len(model.names)}")

for i, ts in enumerate(TIMESTAMPS):
    frame_idx = int(ts * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame at {ts}s")
        continue

    # Save raw frame
    raw_path = os.path.join(OUT_FRAMES, f"frame_{i:02d}_t{ts}s.jpg")
    cv2.imwrite(raw_path, frame)

    # Run detection at multiple confidence thresholds
    for conf_thresh in [0.25, 0.5]:
        results = model.predict(frame, conf=conf_thresh, iou=0.45, device="mps",
                                verbose=False, imgsz=640)

        # Save annotated image
        annotated = results[0].plot()
        ann_path = os.path.join(OUT_DETS, f"frame_{i:02d}_t{ts}s_conf{conf_thresh}.jpg")
        cv2.imwrite(ann_path, annotated)

        # Save detection data as JSON
        dets = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            dets.append({
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
            })

        json_path = os.path.join(OUT_DETS, f"frame_{i:02d}_t{ts}s_conf{conf_thresh}.json")
        with open(json_path, "w") as f:
            json.dump({
                "timestamp": ts,
                "frame_index": frame_idx,
                "conf_threshold": conf_thresh,
                "num_detections": len(dets),
                "detections": dets,
            }, f, indent=2)

        print(f"Frame {i} (t={ts}s) conf={conf_thresh}: {len(dets)} detections")

cap.release()
print("Done. Frames and detections saved.")
