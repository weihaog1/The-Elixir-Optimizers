"""Detection module for Clash Royale object detection."""

from .model import CRDetector, Detection, load_detector
from .combo_detector import ComboDetector

__all__ = [
    "CRDetector",
    "ComboDetector",
    "Detection",
    "load_detector",
]
