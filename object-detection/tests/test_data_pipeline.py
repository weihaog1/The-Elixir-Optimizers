"""
Tests for the data processing pipeline.

Tests cover:
- Screen region configuration
- Annotation generation and YOLO format handling
- Dataset splitting and preparation
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data import (
    Region,
    ScreenConfig,
    TOWER_CLASS_IDS,
    CLASS_ID_TO_NAME,
    get_default_config,
    get_config_for_image,
    detect_resolution,
    generate_tower_annotations,
    write_yolo_label,
    read_yolo_label,
    yolo_to_bbox,
    validate_annotations,
    split_dataset,
    generate_dataset_yaml,
    analyze_dataset,
    verify_dataset_integrity,
)


class TestRegion:
    """Tests for Region dataclass."""

    def test_region_properties(self):
        region = Region(10, 20, 110, 120, "test")
        assert region.width == 100
        assert region.height == 100
        assert region.center == (60, 70)

    def test_region_as_bbox(self):
        region = Region(10, 20, 110, 120, "test")
        assert region.as_bbox() == (10, 20, 110, 120)

    def test_region_to_yolo_format(self):
        region = Region(100, 100, 200, 200, "test")
        # Image 400x400, region center at (150, 150), size 100x100
        x_c, y_c, w, h = region.to_yolo_format(400, 400)
        assert x_c == pytest.approx(0.375, abs=0.001)
        assert y_c == pytest.approx(0.375, abs=0.001)
        assert w == pytest.approx(0.25, abs=0.001)
        assert h == pytest.approx(0.25, abs=0.001)

    def test_region_contains_point(self):
        region = Region(10, 20, 110, 120, "test")
        assert region.contains_point(50, 70)
        assert not region.contains_point(5, 70)
        assert not region.contains_point(50, 130)

    def test_region_crop_image(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[50:100, 50:100] = [255, 0, 0]  # Blue region

        region = Region(50, 50, 100, 100, "test")
        crop = region.crop_image(image)

        assert crop.shape == (50, 50, 3)
        assert np.all(crop == [255, 0, 0])


class TestScreenConfig:
    """Tests for ScreenConfig dataclass."""

    def test_default_config_dimensions(self):
        config = get_default_config()
        assert config.width == 540
        assert config.height == 960

    def test_default_config_tower_regions(self):
        config = get_default_config()
        towers = config.get_tower_regions()
        assert len(towers) == 6
        assert "king_tower_player" in towers
        assert "king_tower_enemy" in towers

    def test_default_config_card_slots(self):
        config = get_default_config()
        assert "next" in config.card_slots
        assert "card_1" in config.card_slots
        assert "card_4" in config.card_slots
        assert len(config.card_slots) == 5

    def test_scale_to_resolution(self):
        config = get_default_config()
        scaled = config.scale_to_resolution(1080, 1920)

        assert scaled.width == 1080
        assert scaled.height == 1920

        # Check that regions are scaled proportionally
        original_king = config.king_tower_player
        scaled_king = scaled.king_tower_player

        assert scaled_king.x_start == original_king.x_start * 2
        assert scaled_king.y_start == original_king.y_start * 2

    def test_get_all_regions(self):
        config = get_default_config()
        regions = config.get_all_regions()

        # Should include top_bar, arena, card_bar, timer, elixir, 6 towers, 5 cards
        assert len(regions) >= 16


class TestResolutionDetection:
    """Tests for resolution detection."""

    def test_detect_resolution_rgb(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        width, height = detect_resolution(image)
        assert width == 640
        assert height == 480

    def test_detect_resolution_grayscale(self):
        image = np.zeros((480, 640), dtype=np.uint8)
        width, height = detect_resolution(image)
        assert width == 640
        assert height == 480

    def test_get_config_for_image_default(self):
        image = np.zeros((960, 540, 3), dtype=np.uint8)
        config = get_config_for_image(image)
        assert config.width == 540
        assert config.height == 960

    def test_get_config_for_image_scaled(self):
        image = np.zeros((1920, 1080, 3), dtype=np.uint8)
        config = get_config_for_image(image)
        assert config.width == 1080
        assert config.height == 1920


class TestClassMappings:
    """Tests for class ID mappings."""

    def test_tower_class_ids_count(self):
        assert len(TOWER_CLASS_IDS) == 6

    def test_tower_class_ids_values(self):
        assert TOWER_CLASS_IDS["king_tower_player"] == 0
        assert TOWER_CLASS_IDS["king_tower_enemy"] == 1

    def test_class_id_to_name_inverse(self):
        for name, id in TOWER_CLASS_IDS.items():
            assert CLASS_ID_TO_NAME[id] == name


class TestYoloFormatConversion:
    """Tests for YOLO format reading and writing."""

    def test_write_and_read_yolo_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = os.path.join(tmpdir, "test.txt")

            annotations = [
                (0, 0.5, 0.5, 0.2, 0.3),
                (1, 0.3, 0.7, 0.1, 0.15),
            ]

            write_yolo_label(label_path, annotations)
            read_annotations = read_yolo_label(label_path)

            assert len(read_annotations) == 2
            for orig, read in zip(annotations, read_annotations):
                assert orig[0] == read[0]  # class_id
                assert orig[1] == pytest.approx(read[1], abs=0.0001)
                assert orig[2] == pytest.approx(read[2], abs=0.0001)
                assert orig[3] == pytest.approx(read[3], abs=0.0001)
                assert orig[4] == pytest.approx(read[4], abs=0.0001)

    def test_read_nonexistent_label(self):
        annotations = read_yolo_label("/nonexistent/path.txt")
        assert annotations == []

    def test_yolo_to_bbox_conversion(self):
        annotation = (0, 0.5, 0.5, 0.2, 0.2)
        class_id, x1, y1, x2, y2 = yolo_to_bbox(annotation, 100, 100)

        assert class_id == 0
        assert x1 == 40
        assert y1 == 40
        assert x2 == 60
        assert y2 == 60


class TestAnnotationValidation:
    """Tests for annotation validation."""

    def test_validate_matching_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            lbl_dir = os.path.join(tmpdir, "labels")
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            # Create matching pairs
            for i in range(3):
                # Create dummy image
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(img_dir, f"img_{i}.png"), img)
                # Create label
                Path(os.path.join(lbl_dir, f"img_{i}.txt")).touch()

            result = validate_annotations(img_dir, lbl_dir)
            assert result["missing_labels"] == []
            assert result["orphan_labels"] == []

    def test_validate_missing_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            lbl_dir = os.path.join(tmpdir, "labels")
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            # Create image without label
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(img_dir, "img_0.png"), img)

            result = validate_annotations(img_dir, lbl_dir)
            assert "img_0" in result["missing_labels"]


class TestDatasetSplitting:
    """Tests for dataset splitting functionality."""

    def test_split_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            lbl_dir = os.path.join(tmpdir, "labels")
            out_dir = os.path.join(tmpdir, "output")
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            # Create 10 image-label pairs
            for i in range(10):
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(img_dir, f"img_{i}.png"), img)
                with open(os.path.join(lbl_dir, f"img_{i}.txt"), "w") as f:
                    f.write(f"0 0.5 0.5 0.1 0.1\n")

            stats = split_dataset(
                img_dir, lbl_dir, out_dir,
                train_ratio=0.8,
                val_ratio=0.2,
                seed=42,
            )

            assert stats["train"] == 8
            assert stats["val"] == 2
            assert Path(out_dir, "train", "images").exists()
            assert Path(out_dir, "val", "images").exists()

    def test_generate_dataset_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "dataset.yaml")
            generate_dataset_yaml(tmpdir, yaml_path)

            assert os.path.exists(yaml_path)

            import yaml
            with open(yaml_path) as f:
                config = yaml.safe_load(f)

            assert config["nc"] == 6
            assert "train" in config
            assert "val" in config

    def test_analyze_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = analyze_dataset(tmpdir)
            assert stats["total_images"] == 0
            assert stats["total_annotations"] == 0

    def test_verify_dataset_integrity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_img = os.path.join(tmpdir, "train", "images")
            train_lbl = os.path.join(tmpdir, "train", "labels")
            os.makedirs(train_img)
            os.makedirs(train_lbl)

            # Create valid pair
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(train_img, "valid.png"), img)
            with open(os.path.join(train_lbl, "valid.txt"), "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

            # Create orphan label
            with open(os.path.join(train_lbl, "orphan.txt"), "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

            issues = verify_dataset_integrity(tmpdir)
            assert "orphan.txt" in str(issues["missing_images"])


class TestTowerAnnotationGeneration:
    """Tests for automatic tower annotation generation."""

    def test_generate_tower_annotations_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image with some content
            img_path = os.path.join(tmpdir, "test.png")
            img = np.random.randint(0, 255, (960, 540, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)

            annotations = generate_tower_annotations(img_path, verify_visibility=False)

            # Should generate annotations for all 6 towers
            assert len(annotations) == 6

            # Check that all class IDs are present
            class_ids = {ann[0] for ann in annotations}
            assert class_ids == {0, 1, 2, 3, 4, 5}

            # Check that coordinates are normalized (0-1)
            for ann in annotations:
                _, x_c, y_c, w, h = ann
                assert 0 <= x_c <= 1
                assert 0 <= y_c <= 1
                assert 0 < w <= 1
                assert 0 < h <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
