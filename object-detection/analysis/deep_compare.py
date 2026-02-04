"""Deep comparison of detection results across all configurations."""

import json
import os
from collections import defaultdict

DET_DIR = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/analysis/resized_detections"
ORIG_DIR = "/Users/alanguo/Codin/CS175/Project/cr-object-detection/analysis/detections"

TIMESTAMPS = [15, 30, 50, 70, 90, 110, 125, 140, 155, 165]

# Categorize classes
TOWER_CLASSES = {"king-tower", "queen-tower", "king-tower-bar", "queen-tower-bar",
                 "king-tower-ruin", "queen-tower-ruin", "cannoneer"}
BAR_CLASSES = {"bar", "tower-bar", "king-tower-bar", "queen-tower-bar"}
UI_CLASSES = {"text", "emote", "elixir-icon", "clock", "bar", "background"}

def load_json(path):
    with open(path) as f:
        return json.load(f)

def classify_detection(name):
    if name in TOWER_CLASSES:
        return "tower"
    if name in BAR_CLASSES:
        return "bar/ui"
    if name in UI_CLASSES:
        return "ui"
    return "troop"

# Compare all configs per frame
configs = [
    ("original_imgsz640_conf0.25", "Original 640 c0.25"),
    ("original_imgsz640_conf0.5", "Original 640 c0.5"),
    ("original_imgsz960_conf0.25", "Original 960 c0.25"),
    ("original_imgsz960_conf0.5", "Original 960 c0.5"),
    ("resized_imgsz640_conf0.25", "Resized 640 c0.25"),
    ("resized_imgsz640_conf0.5", "Resized 640 c0.5"),
    ("resized_imgsz960_conf0.25", "Resized 960 c0.25"),
    ("resized_imgsz960_conf0.5", "Resized 960 c0.5"),
]

print("=" * 100)
print("DEEP DETECTION ANALYSIS: Original vs Resized Input")
print("=" * 100)

# Aggregate stats
config_troop_counts = defaultdict(list)
config_tower_counts = defaultdict(list)
config_total_counts = defaultdict(list)
all_troop_names = defaultdict(lambda: defaultdict(int))
troop_confidence_by_config = defaultdict(list)

for i, ts in enumerate(TIMESTAMPS):
    print(f"\n{'='*80}")
    print(f"FRAME {i} (t={ts}s)")
    print(f"{'='*80}")

    for config_tag, config_label in configs:
        json_path = os.path.join(DET_DIR, f"frame_{i:02d}_t{ts}s_{config_tag}.json")
        if not os.path.exists(json_path):
            continue

        data = load_json(json_path)
        dets = data["detections"]

        troops = [d for d in dets if classify_detection(d["class_name"]) == "troop"]
        towers = [d for d in dets if classify_detection(d["class_name"]) == "tower"]
        other = [d for d in dets if classify_detection(d["class_name"]) not in ("troop", "tower")]

        config_troop_counts[config_tag].append(len(troops))
        config_tower_counts[config_tag].append(len(towers))
        config_total_counts[config_tag].append(len(dets))

        for t in troops:
            all_troop_names[config_tag][t["class_name"]] += 1
            troop_confidence_by_config[config_tag].append(t["confidence"])

        troop_names = [f"{d['class_name']}({d['confidence']:.2f})" for d in troops]
        tower_names = [f"{d['class_name']}({d['confidence']:.2f})" for d in towers]

        print(f"\n  {config_label}:")
        print(f"    Total: {len(dets)} | Troops: {len(troops)} | Towers: {len(towers)} | Other: {len(other)}")
        if troops:
            print(f"    Troops: {', '.join(troop_names)}")

print("\n\n" + "=" * 100)
print("AGGREGATE SUMMARY")
print("=" * 100)

print(f"\n{'Config':<35} {'Avg Total':>10} {'Avg Troops':>12} {'Avg Towers':>12} {'Avg Troop Conf':>15}")
print("-" * 85)
for config_tag, config_label in configs:
    avg_total = sum(config_total_counts[config_tag]) / len(config_total_counts[config_tag])
    avg_troops = sum(config_troop_counts[config_tag]) / len(config_troop_counts[config_tag])
    avg_towers = sum(config_tower_counts[config_tag]) / len(config_tower_counts[config_tag])
    confs = troop_confidence_by_config[config_tag]
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"{config_label:<35} {avg_total:>10.1f} {avg_troops:>12.1f} {avg_towers:>12.1f} {avg_conf:>15.3f}")

print(f"\n\nUNIQUE TROOP CLASSES DETECTED PER CONFIG:")
print("-" * 85)
for config_tag, config_label in configs:
    troop_dict = all_troop_names[config_tag]
    sorted_troops = sorted(troop_dict.items(), key=lambda x: -x[1])
    print(f"\n  {config_label}:")
    print(f"    Unique classes: {len(troop_dict)}")
    for name, count in sorted_troops:
        print(f"      {name}: {count}")

# Compare: which troops appear ONLY in resized or ONLY in original?
print(f"\n\n{'='*100}")
print("DIFFERENTIAL ANALYSIS: Resized vs Original (conf 0.25, imgsz 640)")
print("=" * 100)

for i, ts in enumerate(TIMESTAMPS):
    orig_path = os.path.join(DET_DIR, f"frame_{i:02d}_t{ts}s_original_imgsz640_conf0.25.json")
    resized_path = os.path.join(DET_DIR, f"frame_{i:02d}_t{ts}s_resized_imgsz640_conf0.25.json")

    if not os.path.exists(orig_path) or not os.path.exists(resized_path):
        continue

    orig_dets = load_json(orig_path)["detections"]
    resized_dets = load_json(resized_path)["detections"]

    orig_troops = {d["class_name"] for d in orig_dets if classify_detection(d["class_name"]) == "troop"}
    resized_troops = {d["class_name"] for d in resized_dets if classify_detection(d["class_name"]) == "troop"}

    only_orig = orig_troops - resized_troops
    only_resized = resized_troops - orig_troops
    both = orig_troops & resized_troops

    if only_orig or only_resized:
        print(f"\n  Frame {i} (t={ts}s):")
        if only_orig:
            print(f"    ONLY in original: {only_orig}")
        if only_resized:
            print(f"    ONLY in resized:  {only_resized}")
        print(f"    In both:          {both}")
