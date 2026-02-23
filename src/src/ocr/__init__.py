"""OCR module for Clash Royale text extraction."""

from .text_extractor import (
    TextExtractor,
    GameTextExtractor,
    OCRResult,
    TimerResult,
    GameOCRResults,
    create_extractor,
)

__all__ = [
    "TextExtractor",
    "GameTextExtractor",
    "OCRResult",
    "TimerResult",
    "GameOCRResults",
    "create_extractor",
]
