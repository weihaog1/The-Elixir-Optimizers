"""
Data processing module for Clash Royale object detection.

This module provides tools for:
- Frame extraction from gameplay videos
- Screen region configuration
- Annotation generation in YOLO format
- Dataset preparation and splitting
- Extended YOLO format conversion
"""

from .extract_frames import extract_frames, extract_specific_times, get_video_info
from .screen_regions import (
    Region,
    ScreenConfig,
    TOWER_CLASS_IDS,
    CLASS_ID_TO_NAME,
    get_default_config,
    get_config_for_image,
    detect_resolution,
)
from .annotation_helper import (
    generate_tower_annotations,
    write_yolo_label,
    read_yolo_label,
    yolo_to_bbox,
    visualize_annotations,
    process_directory,
    validate_annotations,
)
from .dataset import (
    split_dataset,
    generate_dataset_yaml,
    prepare_yolo_dataset,
    analyze_dataset,
    verify_dataset_integrity,
)
from .converter import (
    convert_extended_yolo_to_standard,
    prepare_external_dataset,
)

__all__ = [
    # Frame extraction
    "extract_frames",
    "extract_specific_times",
    "get_video_info",
    # Screen regions
    "Region",
    "ScreenConfig",
    "TOWER_CLASS_IDS",
    "CLASS_ID_TO_NAME",
    "get_default_config",
    "get_config_for_image",
    "detect_resolution",
    # Annotation
    "generate_tower_annotations",
    "write_yolo_label",
    "read_yolo_label",
    "yolo_to_bbox",
    "visualize_annotations",
    "process_directory",
    "validate_annotations",
    # Dataset
    "split_dataset",
    "generate_dataset_yaml",
    "prepare_yolo_dataset",
    "analyze_dataset",
    "verify_dataset_integrity",
    # Converter
    "convert_extended_yolo_to_standard",
    "prepare_external_dataset",
]
