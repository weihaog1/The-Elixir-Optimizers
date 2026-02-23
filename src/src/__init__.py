"""
Clash Royale Object Detection System.

A computer vision pipeline for detecting game elements in Clash Royale screenshots.
"""

__version__ = "0.1.0"

from . import data
from . import detection
from . import ocr
from . import pipeline

__all__ = [
    "data",
    "detection",
    "ocr",
    "pipeline",
    "__version__",
]
