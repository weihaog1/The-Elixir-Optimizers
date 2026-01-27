"""Detection module for Clash Royale object detection."""

from .model import CRDetector, Detection, load_detector

__all__ = [
    "CRDetector",
    "Detection",
    "load_detector",
]
