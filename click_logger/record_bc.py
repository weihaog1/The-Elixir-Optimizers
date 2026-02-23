"""
BC data recording entry point.

Captures screenshots + click actions during Clash Royale gameplay.
Creates a session folder under recordings/ with screenshots, click
log, frame manifest, and session metadata.

Usage:
    python record_bc.py
    # Play a match, then press Enter to stop recording.
"""

from match_recorder import MatchRecorder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_TITLE = "Clash Royale - UnwontedTemper73"

# Card slot centers in normalized coordinates (from calibration)
CARD_POSITIONS = {
    0: (0.439, 0.889),
    1: (0.494, 0.889),
    2: (0.559, 0.889),
    3: (0.639, 0.889),
}

# Arena region bounds (normalized)
ARENA_BOUNDS = (0.05, 0.15, 0.95, 0.80)

# Screen capture settings
FPS = 2.0
JPEG_QUALITY = 85

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
recorder = MatchRecorder(
    window_title=WINDOW_TITLE,
    card_positions=CARD_POSITIONS,
    arena_bounds=ARENA_BOUNDS,
    fps=FPS,
    jpeg_quality=JPEG_QUALITY,
)

recorder.start()

try:
    input("Recording... Press Enter to stop.\n")
finally:
    recorder.stop()
