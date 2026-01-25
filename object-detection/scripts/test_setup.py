#!/usr/bin/env python3
"""
Quick test script to verify project setup and imports.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src import data, detection, ocr, pipeline
        print("  ✓ src package imported")
    except ImportError as e:
        print(f"  ✗ Failed to import src: {e}")
        return False

    try:
        from src.data import (
            extract_frames, get_video_info,
            Region, ScreenConfig, get_default_config,
            generate_tower_annotations, write_yolo_label, read_yolo_label,
            split_dataset, generate_dataset_yaml, analyze_dataset,
            convert_extended_yolo_to_standard, prepare_external_dataset,
        )
        print("  ✓ src.data imported")
    except ImportError as e:
        print(f"  ✗ Failed to import src.data: {e}")
        return False

    try:
        from src.detection import CRDetector, Detection, load_detector
        print("  ✓ src.detection imported")
    except ImportError as e:
        print(f"  ✗ Failed to import src.detection: {e}")
        return False

    try:
        from src.ocr import TextExtractor, GameTextExtractor, OCRResult
        print("  ✓ src.ocr imported")
    except ImportError as e:
        print(f"  ✗ Failed to import src.ocr: {e}")
        return False

    try:
        from src.pipeline import StateBuilder, GameState, Tower, Unit, Card
        print("  ✓ src.pipeline imported")
    except ImportError as e:
        print(f"  ✗ Failed to import src.pipeline: {e}")
        return False

    return True


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\nTesting dependencies...")

    deps = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("yaml", "pyyaml"),
    ]

    all_ok = True
    for module, package in deps:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} not installed")
            all_ok = False

    # Optional OCR dependencies
    print("\nOptional dependencies:")
    try:
        import easyocr
        print("  ✓ easyocr")
    except ImportError:
        print("  ○ easyocr (not installed)")

    try:
        from paddleocr import PaddleOCR
        print("  ✓ paddleocr")
    except ImportError:
        print("  ○ paddleocr (not installed)")

    return all_ok


def test_config_files():
    """Test that config files exist."""
    print("\nTesting config files...")

    config_dir = Path(__file__).parent.parent / "configs"

    configs = [
        "classes.yaml",
        "full_dataset.yaml",
    ]

    all_ok = True
    for config in configs:
        path = config_dir / config
        if path.exists():
            print(f"  ✓ {config}")
        else:
            print(f"  ✗ {config} not found")
            all_ok = False

    return all_ok


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ○ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")


def main():
    print("=" * 50)
    print("Clash Royale Object Detection - Setup Test")
    print("=" * 50)

    imports_ok = test_imports()
    deps_ok = test_dependencies()
    configs_ok = test_config_files()
    test_cuda()

    print("\n" + "=" * 50)
    if imports_ok and deps_ok and configs_ok:
        print("All tests passed! ✓")
        print("\nNext steps:")
        print("  1. Run: python scripts/prepare_dataset.py")
        print("  2. Run: python -m src.detection.train --data configs/dataset.yaml")
    else:
        print("Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
