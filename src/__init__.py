"""
Clash Royale Object Detection Package
Non-embedded AI based on KataCR approach.
"""

from .detector import (
    Detection,
    ClashRoyaleDetector,
    ComboDetector,
    draw_detections,
    filter_detections
)

from .capture import (
    ScreenCapture,
    WindowCapture,
    ImageLoader,
    select_capture_region
)

from .visual_fusion import (
    UnitBelonging,
    UnitInfo,
    GameState,
    ScreenRegions,
    VisualFusion,
    ArenaAnalyzer
)

from .config import (
    IMAGE_SIZE,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    CAPTURE_FPS,
    UNIT_CATEGORIES
)

__version__ = "0.2.0"
__all__ = [
    # Detector
    "Detection",
    "ClashRoyaleDetector",
    "ComboDetector",  # Multi-model combo detection
    "draw_detections",
    "filter_detections",
    
    # Capture
    "ScreenCapture",
    "WindowCapture",
    "ImageLoader",
    "select_capture_region",
    
    # Visual Fusion
    "UnitBelonging",
    "UnitInfo",
    "GameState",
    "ScreenRegions",
    "VisualFusion",
    "ArenaAnalyzer",
    
    # Config
    "IMAGE_SIZE",
    "CONF_THRESHOLD",
    "IOU_THRESHOLD",
    "CAPTURE_FPS",
    "UNIT_CATEGORIES",
]
