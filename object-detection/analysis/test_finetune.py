"""Test fine-tuned model vs original on the same 10 frames."""

import cv2
import json
import os
from ultralytics import YOLO

VIDEO = "/Users/alanguo/Codin/CS175/gameplay-videos/pigs_lose_0_1_crowns(1).mp4"
OLD_MODEL = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/models/best.pt"
NEW_MODEL = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/models/best_finetune960.pt"
OUT_DIR = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/analysis/finetune_detections"

TIMESTAMPS = [15, 30, 50, 70, 90, 110, 125, 140, 155, 165]
CONF = 0.25
IMGSZ = 960
DEVICE = "mps"

TOWER_CLASSES = {"king-tower", "queen-tower", "king-tower-bar", "queen-tower-bar",
                 "king-tower-ruin", "queen-tower-ruin", "cannoneer"}
BAR_CLASSES = {"bar", "tower-bar", "king-tower-bar", "queen-tower-bar"}
UI_CLASSES = {"text", "emote", "elixir-icon", "clock", "bar", "background",
              "tower-bar", "king-tower-bar", "queen-tower-bar"}

def classify(name):
    if name in TOWER_CLASSES:
        return "tower"
    if name in UI_CLASSES:
        return "ui"
    return "troop"

def run_model(model, frame, conf, imgsz, device):
    results = model.predict(frame, conf=conf, iou=0.45, device=device,
                            verbose=False, imgsz=imgsz)
    dets = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        dets.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
            "category": classify(model.names[cls_id]),
        })
    return results, dets

print("Loading models...")
old_model = YOLO(OLD_MODEL)
new_model = YOLO(NEW_MODEL)
print(f"Old model classes: {len(old_model.names)}")
print(f"New model classes: {len(new_model.names)}")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

# Warmup both models
ret, warmup_frame = cap.read()
old_model.predict(warmup_frame, conf=CONF, iou=0.45, device=DEVICE, verbose=False, imgsz=IMGSZ)
new_model.predict(warmup_frame, conf=CONF, iou=0.45, device=DEVICE, verbose=False, imgsz=IMGSZ)
print("Warmup done.\n")

old_totals = {"troops": [], "towers": [], "all": []}
new_totals = {"troops": [], "towers": [], "all": []}
old_troop_confs = []
new_troop_confs = []
old_troop_classes = {}
new_troop_classes = {}

for i, ts in enumerate(TIMESTAMPS):
    frame_idx = int(ts * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed at {ts}s")
        continue

    old_results, old_dets = run_model(old_model, frame, CONF, IMGSZ, DEVICE)
    new_results, new_dets = run_model(new_model, frame, CONF, IMGSZ, DEVICE)

    # Save annotated images
    old_ann = old_results[0].plot()
    new_ann = new_results[0].plot()
    cv2.imwrite(os.path.join(OUT_DIR, f"frame_{i:02d}_t{ts}s_OLD.jpg"), old_ann)
    cv2.imwrite(os.path.join(OUT_DIR, f"frame_{i:02d}_t{ts}s_NEW.jpg"), new_ann)

    # Save side-by-side comparison
    h = max(old_ann.shape[0], new_ann.shape[0])
    scale_old = h / old_ann.shape[0]
    scale_new = h / new_ann.shape[0]
    old_resized = cv2.resize(old_ann, (int(old_ann.shape[1] * scale_old), h))
    new_resized = cv2.resize(new_ann, (int(new_ann.shape[1] * scale_new), h))
    # Add labels
    cv2.putText(old_resized, "OLD MODEL", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(new_resized, "NEW MODEL (finetune960)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    sidebyside = cv2.hconcat([old_resized, new_resized])
    cv2.imwrite(os.path.join(OUT_DIR, f"frame_{i:02d}_t{ts}s_COMPARE.jpg"), sidebyside)

    # Save JSON
    for tag, dets in [("OLD", old_dets), ("NEW", new_dets)]:
        json_path = os.path.join(OUT_DIR, f"frame_{i:02d}_t{ts}s_{tag}.json")
        with open(json_path, "w") as f:
            json.dump({"timestamp": ts, "model": tag, "detections": dets}, f, indent=2)

    # Stats
    old_troops = [d for d in old_dets if d["category"] == "troop"]
    new_troops = [d for d in new_dets if d["category"] == "troop"]
    old_towers = [d for d in old_dets if d["category"] == "tower"]
    new_towers = [d for d in new_dets if d["category"] == "tower"]

    old_totals["troops"].append(len(old_troops))
    old_totals["towers"].append(len(old_towers))
    old_totals["all"].append(len(old_dets))
    new_totals["troops"].append(len(new_troops))
    new_totals["towers"].append(len(new_towers))
    new_totals["all"].append(len(new_dets))

    for t in old_troops:
        old_troop_confs.append(t["confidence"])
        old_troop_classes[t["class_name"]] = old_troop_classes.get(t["class_name"], 0) + 1
    for t in new_troops:
        new_troop_confs.append(t["confidence"])
        new_troop_classes[t["class_name"]] = new_troop_classes.get(t["class_name"], 0) + 1

    old_troop_str = ", ".join(f"{d['class_name']}({d['confidence']:.2f})" for d in old_troops)
    new_troop_str = ", ".join(f"{d['class_name']}({d['confidence']:.2f})" for d in new_troops)

    print(f"Frame {i} (t={ts}s):")
    print(f"  OLD: {len(old_dets)} total, {len(old_troops)} troops, {len(old_towers)} towers")
    if old_troops:
        print(f"       Troops: {old_troop_str}")
    print(f"  NEW: {len(new_dets)} total, {len(new_troops)} troops, {len(new_towers)} towers")
    if new_troops:
        print(f"       Troops: {new_troop_str}")

    diff = len(new_troops) - len(old_troops)
    sign = "+" if diff >= 0 else ""
    print(f"  DIFF: {sign}{diff} troops")
    print()

cap.release()

# Summary
print("=" * 80)
print("AGGREGATE COMPARISON")
print("=" * 80)
avg = lambda lst: sum(lst) / len(lst) if lst else 0
print(f"\n{'Metric':<25} {'Old Model':>12} {'New Model':>12} {'Change':>12}")
print("-" * 65)
print(f"{'Avg troops/frame':<25} {avg(old_totals['troops']):>12.1f} {avg(new_totals['troops']):>12.1f} {avg(new_totals['troops'])-avg(old_totals['troops']):>+12.1f}")
print(f"{'Avg towers/frame':<25} {avg(old_totals['towers']):>12.1f} {avg(new_totals['towers']):>12.1f} {avg(new_totals['towers'])-avg(old_totals['towers']):>+12.1f}")
print(f"{'Avg total/frame':<25} {avg(old_totals['all']):>12.1f} {avg(new_totals['all']):>12.1f} {avg(new_totals['all'])-avg(old_totals['all']):>+12.1f}")
print(f"{'Avg troop confidence':<25} {avg(old_troop_confs):>12.3f} {avg(new_troop_confs):>12.3f} {avg(new_troop_confs)-avg(old_troop_confs):>+12.3f}")
print(f"{'Total troop detections':<25} {sum(old_totals['troops']):>12} {sum(new_totals['troops']):>12} {sum(new_totals['troops'])-sum(old_totals['troops']):>+12}")
print(f"{'Unique troop classes':<25} {len(old_troop_classes):>12} {len(new_troop_classes):>12} {len(new_troop_classes)-len(old_troop_classes):>+12}")

print(f"\nOLD model troop classes:")
for name, count in sorted(old_troop_classes.items(), key=lambda x: -x[1]):
    print(f"  {name}: {count}")

print(f"\nNEW model troop classes:")
for name, count in sorted(new_troop_classes.items(), key=lambda x: -x[1]):
    print(f"  {name}: {count}")

# New classes found
new_only = set(new_troop_classes.keys()) - set(old_troop_classes.keys())
old_only = set(old_troop_classes.keys()) - set(new_troop_classes.keys())
if new_only:
    print(f"\nNEW classes detected by fine-tuned model: {new_only}")
if old_only:
    print(f"LOST classes (old detected, new doesn't): {old_only}")
