"""Quick diagnostic: what does the capture look like and what does the detector see?

Run with the game visible and in a match (or on the main menu).
Saves the captured frame and prints detection metrics.
"""

import sys
import os

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
from src.bc.live_inference import GameCapture, LiveConfig
from src.ppo.game_detector import GamePhaseDetector, DetectorConfig

WINDOW_TITLE = "Clash Royale - thegoodpersonplayer2"

config = LiveConfig(window_title=WINDOW_TITLE)
capture = GameCapture(config)

print(f"Window title: {WINDOW_TITLE}")
print(f"Capture monitor: {capture._monitor}")
print()

frame = capture.capture()
print(f"Raw frame shape: {frame.shape}  (height x width x channels)")

# Save raw frame
cv2.imwrite("debug_raw_frame.png", frame)
print("Saved: debug_raw_frame.png")

# Detect game bounds (same as ClashRoyaleEnv.__init__)
fh, fw = frame.shape[:2]
col_max = frame.max(axis=(0, 2))
row_max = frame.max(axis=(1, 2))
non_black_cols = np.where(col_max > 15)[0]
non_black_rows = np.where(row_max > 15)[0]

if len(non_black_cols) > 0 and len(non_black_rows) > 0:
    gx = int(non_black_cols[0])
    gx_end = int(non_black_cols[-1]) + 1
    gy = int(non_black_rows[0])
    gy_end = int(non_black_rows[-1]) + 1
    gw = gx_end - gx
    gh = gy_end - gy
    print(f"\nGame bounds: x={gx}, y={gy}, w={gw}, h={gh}")
    print(f"  (offset from raw frame: left={gx}px, top={gy}px)")
    print(f"  Aspect ratio: {gw/gh:.3f}  (expected ~0.5625 for 9:16)")

    cropped = frame[gy:gy_end, gx:gx_end]
    cv2.imwrite("debug_cropped_frame.png", cropped)
    print("Saved: debug_cropped_frame.png")
else:
    print("\nWARNING: Could not detect game bounds (entire frame is dark?)")
    cropped = frame
    gx, gy, gw, gh = 0, 0, fw, fh

# Run detector on RAW frame (what wait_for_game_start sees)
cfg = DetectorConfig()
print(f"\n--- Detector metrics on RAW frame ({fh}x{fw}) ---")
card_bar_y1 = int(cfg.card_bar_y_start * fh)
card_bar_y2 = int(cfg.card_bar_y_end * fh)
arena_y1 = int(cfg.arena_y_start * fh)
arena_y2 = int(cfg.arena_y_end * fh)

card_bar = frame[card_bar_y1:card_bar_y2, :, :]
arena = frame[arena_y1:arena_y2, :, :]

card_bar_intensity = float(np.mean(card_bar))
arena_intensity = float(np.mean(arena))
arena_variance = float(np.var(arena.astype(np.float32)))

print(f"  Card bar region: y={card_bar_y1}-{card_bar_y2}")
print(f"  Card bar intensity: {card_bar_intensity:.1f}  (threshold: >{cfg.card_bar_intensity_threshold})")
print(f"  Arena region: y={arena_y1}-{arena_y2}")
print(f"  Arena intensity: {arena_intensity:.1f}")
print(f"  Arena variance: {arena_variance:.1f}  (threshold: >{cfg.arena_variance_threshold})")

card_bar_active = card_bar_intensity > cfg.card_bar_intensity_threshold
arena_active = arena_variance > cfg.arena_variance_threshold
print(f"  Card bar active: {card_bar_active}")
print(f"  Arena active: {arena_active}")
print(f"  -> Would classify as IN_GAME: {card_bar_active and arena_active}")

# Run detector on CROPPED frame (what it should see)
print(f"\n--- Detector metrics on CROPPED frame ({gh}x{gw}) ---")
card_bar_y1c = int(cfg.card_bar_y_start * gh)
card_bar_y2c = int(cfg.card_bar_y_end * gh)
arena_y1c = int(cfg.arena_y_start * gh)
arena_y2c = int(cfg.arena_y_end * gh)

card_bar_c = cropped[card_bar_y1c:card_bar_y2c, :, :]
arena_c = cropped[arena_y1c:arena_y2c, :, :]

card_bar_intensity_c = float(np.mean(card_bar_c))
arena_intensity_c = float(np.mean(arena_c))
arena_variance_c = float(np.var(arena_c.astype(np.float32)))

print(f"  Card bar region: y={card_bar_y1c}-{card_bar_y2c}")
print(f"  Card bar intensity: {card_bar_intensity_c:.1f}  (threshold: >{cfg.card_bar_intensity_threshold})")
print(f"  Arena region: y={arena_y1c}-{arena_y2c}")
print(f"  Arena intensity: {arena_intensity_c:.1f}")
print(f"  Arena variance: {arena_variance_c:.1f}  (threshold: >{cfg.arena_variance_threshold})")

card_bar_active_c = card_bar_intensity_c > cfg.card_bar_intensity_threshold
arena_active_c = arena_variance_c > cfg.arena_variance_threshold
print(f"  Card bar active: {card_bar_active_c}")
print(f"  Arena active: {arena_active_c}")
print(f"  -> Would classify as IN_GAME: {card_bar_active_c and arena_active_c}")

if (card_bar_active_c and arena_active_c) and not (card_bar_active and arena_active):
    print("\n*** DIAGNOSIS: Detection works on the cropped frame but NOT on the raw frame.")
    print("    The bug is that wait_for_game_start() uses uncropped frames.")
    print("    Fix: game_detector needs the cropped frame, not the raw capture.")

capture.release()
