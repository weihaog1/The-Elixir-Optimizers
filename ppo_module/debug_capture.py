"""Full pipeline diagnostic: capture -> crop -> detect -> perceive -> mask -> action.

Run while IN A MATCH (cards visible, arena active).
Tests every stage and reports where the pipeline breaks.
"""

import sys
import os
import time

# Namespace setup (same as run_ppo.py)
_here = os.path.dirname(os.path.abspath(__file__))
_project = os.path.abspath(os.path.join(_here, ".."))
sys.path.insert(0, _project)

import types
_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [
    os.path.join(_here, "src"),
    os.path.join(_project, "bc_model_module", "src"),
    os.path.join(_project, "state_encoder_module", "src"),
    os.path.join(_project, "action_builder_module", "src"),
]
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

for pkg, path in {
    "src.ppo": os.path.join(_here, "src", "ppo"),
    "src.bc": os.path.join(_project, "bc_model_module", "src", "bc"),
    "src.encoder": os.path.join(_project, "state_encoder_module", "src", "encoder"),
}.items():
    mod = types.ModuleType(pkg)
    mod.__path__ = [path]
    mod.__package__ = pkg
    sys.modules[pkg] = mod

for sub in ("generation", "detection", "classification", "data", "pipeline"):
    pkg = f"src.{sub}"
    mod = types.ModuleType(pkg)
    mod.__path__ = [os.path.join(_project, "src", "src", sub)]
    mod.__package__ = pkg
    sys.modules[pkg] = mod

import cv2
import numpy as np
from src.bc.live_inference import (
    GameCapture, LiveConfig, PerceptionAdapter, ActionDispatcher, _NOOP_ACTION
)
from src.ppo.game_detector import GamePhaseDetector, DetectorConfig

WINDOW_TITLE = "Clash Royale - thegoodpersonplayer2"

print("=" * 60)
print("PPO Pipeline Diagnostic")
print("=" * 60)

# ---- Stage 1: Capture ----
print("\n[1] CAPTURE")
config = LiveConfig(window_title=WINDOW_TITLE)
capture = GameCapture(config)
print(f"  Monitor: {capture._monitor}")

wl, wt = capture.get_window_offset()
print(f"  Window offset: left={wl}, top={wt}")

hwnd = capture.get_window_hwnd()
print(f"  Window HWND: {hwnd}")

frame = capture.capture()
print(f"  Raw frame: {frame.shape}")

# ---- Stage 2: Crop ----
print("\n[2] CROP (game bounds detection)")
fh, fw = frame.shape[:2]
col_max = frame.max(axis=(0, 2))
row_max = frame.max(axis=(1, 2))
non_black_cols = np.where(col_max > 15)[0]
non_black_rows = np.where(row_max > 15)[0]

if len(non_black_cols) > 0 and len(non_black_rows) > 0:
    gx = int(non_black_cols[0])
    gy = int(non_black_rows[0])
    gw = int(non_black_cols[-1]) + 1 - gx
    gh = int(non_black_rows[-1]) + 1 - gy
else:
    gx, gy, gw, gh = 0, 0, fw, fh

print(f"  Game region: x={gx}, y={gy}, w={gw}, h={gh}")
print(f"  Aspect ratio: {gw/gh:.3f} (expected ~0.5625)")

cropped = frame[gy:gy+gh, gx:gx+gw]
cv2.imwrite("debug_cropped_frame.png", cropped)
print(f"  Cropped frame: {cropped.shape}")
print(f"  Saved: debug_cropped_frame.png")

# ---- Stage 3: Phase Detection ----
print("\n[3] PHASE DETECTION (on cropped frame)")
detector = GamePhaseDetector()
# Feed 5 frames for stability
for i in range(5):
    phase = detector.detect_phase(cropped)
print(f"  Phase: {phase.value}")
print(f"  Confirmed phase: {detector._confirmed_phase.value}")
if phase.value != "in_game":
    print("  *** NOT IN_GAME -- run this while in a match!")

# ---- Stage 4: Perception ----
print("\n[4] PERCEPTION")
# Update config with game bounds for perception
config.window_left = wl + gx
config.window_top = wt + gy
config.frame_w = gw
config.frame_h = gh
perception = PerceptionAdapter(config, _project)
print(f"  Perception active: {perception.perception_active}")

result = perception.process_frame(cropped)
obs = result["obs"]
mask = result["mask"]

arena = obs["arena"]
vector = obs["vector"]
if hasattr(arena, "numpy"):
    arena = arena.numpy()
if hasattr(vector, "numpy"):
    vector = vector.numpy()
if hasattr(mask, "numpy"):
    mask = mask.numpy()
if arena.ndim == 4:
    arena = arena[0]
if vector.ndim == 2:
    vector = vector[0]
if mask.ndim == 2:
    mask = mask[0]

print(f"  Arena shape: {arena.shape}, non-zero: {np.count_nonzero(arena)}")
print(f"  Vector shape: {vector.shape}")
print(f"  Vector values: elixir={vector[0]:.3f}, ally_towers={vector[9]:.3f}, enemy_towers={vector[10]:.3f}")
print(f"  Cards: present={vector[11:15]}, class={vector[15:19]}, elixir={vector[19:23]}")

# ---- Stage 5: Action Mask ----
print("\n[5] ACTION MASK")
mask_bool = mask.astype(bool)
valid_count = np.sum(mask_bool)
noop_valid = mask_bool[_NOOP_ACTION] if len(mask_bool) > _NOOP_ACTION else False
card_actions_valid = np.sum(mask_bool[:_NOOP_ACTION])  # Actions 0-2303
print(f"  Total valid actions: {valid_count} / {len(mask_bool)}")
print(f"  Card placement actions valid: {card_actions_valid}")
print(f"  Noop valid: {noop_valid}")

if card_actions_valid == 0:
    print("  *** NO CARD ACTIONS VALID -- agent can only noop!")
    print("  This means perception didn't detect any playable cards.")

# Per-card breakdown
for card_id in range(4):
    start = card_id * 576
    end = start + 576
    card_valid = np.sum(mask_bool[start:end])
    print(f"  Card {card_id}: {card_valid} valid cells")

# ---- Stage 6: Action Dispatcher ----
print("\n[6] ACTION DISPATCHER (dry run)")
dispatcher = ActionDispatcher(
    LiveConfig(
        window_title=WINDOW_TITLE,
        dry_run=True,  # Don't actually click
        window_left=wl + gx,
        window_top=wt + gy,
        frame_w=gw,
        frame_h=gh,
    ),
    game_hwnd=hwnd,
)
dispatcher._frame_w = gw
dispatcher._frame_h = gh

# Check focus
focused = dispatcher._is_game_focused()
print(f"  Game focused: {focused}")
if not focused:
    print("  *** GAME NOT FOCUSED -- clicks would be skipped!")

# Try a sample action
sample_action = 288  # Card 0, middle of grid
exec_result = dispatcher.execute(sample_action, logit_score=1.0)
print(f"  Sample action {sample_action}: reason={exec_result['reason']}")
if "card_id" in exec_result:
    card_px = int(exec_result.get("x_norm", 0) * gw) + wl + gx
    card_py = int(exec_result.get("y_norm", 0) * gh) + wt + gy
    print(f"  Would click arena at screen pixel: ({card_px}, {card_py})")

# ---- Stage 7: HWND check ----
print("\n[7] FOCUS DETECTION")
if hwnd:
    try:
        import ctypes
        fg = ctypes.windll.user32.GetForegroundWindow()
        print(f"  Game HWND:       {hwnd}")
        print(f"  Foreground HWND: {fg}")
        print(f"  Match: {fg == hwnd}")
        if fg != hwnd:
            print("  *** GAME IS NOT THE FOREGROUND WINDOW")
            print("  Clicks will be skipped by _is_game_focused() safety check.")
            print("  Make sure the game window is focused when training runs.")
    except Exception as e:
        print(f"  Focus check failed: {e}")
else:
    print("  No HWND -- focus detection disabled, clicks always attempt.")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

issues = []
if phase.value != "in_game":
    issues.append("Phase is not IN_GAME (run during a match)")
if not perception.perception_active:
    issues.append("Perception is INACTIVE (YOLO model not loaded)")
if card_actions_valid == 0:
    issues.append("No card actions in mask (only noop allowed)")
if hwnd and not focused:
    issues.append("Game window not focused (clicks would be skipped)")

if issues:
    print("ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("All stages OK!")

capture.release()
