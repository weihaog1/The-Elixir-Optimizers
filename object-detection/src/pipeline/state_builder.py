"""
State builder pipeline combining detection and OCR.

Main entry point for extracting game state from screenshots.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..detection.model import CRDetector, Detection
from ..ocr.text_extractor import GameTextExtractor, GameOCRResults
from .game_state import GameState, Tower, Unit, Card


class StateBuilder:
    """Builds game state from detection and OCR results."""

    # Mapping from detection class names to tower info
    TOWER_CLASS_MAP = {
        "king_tower_player": ("king", "center", 0),
        "king_tower_enemy": ("king", "center", 1),
        "princess_tower_left_player": ("princess", "left", 0),
        "princess_tower_left_enemy": ("princess", "left", 1),
        "princess_tower_right_player": ("princess", "right", 0),
        "princess_tower_right_enemy": ("princess", "right", 1),
    }

    def __init__(
        self,
        detector: Optional[CRDetector] = None,
        ocr_extractor: Optional[GameTextExtractor] = None,
        detection_model_path: Optional[str] = None,
        enable_ocr: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize the state builder.

        Args:
            detector: Pre-initialized detector instance.
            ocr_extractor: Pre-initialized OCR extractor instance.
            detection_model_path: Path to detection model weights.
            enable_ocr: Whether to enable OCR extraction.
            device: Device for inference ('cuda', 'cpu', or None for auto).
        """
        # Initialize detector
        if detector is not None:
            self.detector = detector
        elif detection_model_path:
            self.detector = CRDetector(
                model_path=detection_model_path,
                device=device,
            )
        else:
            self.detector = None

        # Initialize OCR
        self.enable_ocr = enable_ocr
        if enable_ocr:
            try:
                self.ocr_extractor = ocr_extractor or GameTextExtractor()
            except ImportError:
                print("Warning: OCR not available, disabling OCR extraction")
                self.enable_ocr = False
                self.ocr_extractor = None
        else:
            self.ocr_extractor = None

    def build_state(
        self,
        image: Union[str, np.ndarray],
        timestamp: float = 0.0,
        run_detection: bool = True,
        run_ocr: bool = True,
    ) -> GameState:
        """Build game state from an image.

        Args:
            image: Image path or numpy array (BGR format).
            timestamp: Frame timestamp for video processing.
            run_detection: Whether to run object detection.
            run_ocr: Whether to run OCR extraction.

        Returns:
            GameState object with extracted information.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = image

        height, width = img.shape[:2]

        # Initialize game state
        state = GameState(
            timestamp=timestamp,
            frame_width=width,
            frame_height=height,
        )

        # Run detection
        if run_detection and self.detector is not None:
            detections = self.detector.detect(img)
            self._process_detections(state, detections)

        # Run OCR
        if run_ocr and self.enable_ocr and self.ocr_extractor is not None:
            ocr_results = self.ocr_extractor.extract_game_text(img)
            self._process_ocr(state, ocr_results)

        return state

    def _process_detections(
        self,
        state: GameState,
        detections: List[Detection],
    ) -> None:
        """Process detection results and update game state.

        Args:
            state: GameState to update.
            detections: List of detections from detector.
        """
        confidences = []

        for det in detections:
            confidences.append(det.confidence)

            # Check if this is a tower
            tower_info = self.TOWER_CLASS_MAP.get(det.class_name)
            if tower_info:
                tower_type, position, belonging = tower_info
                tower = Tower(
                    tower_type=tower_type,
                    position=position,
                    belonging=belonging,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )

                # Assign to correct slot
                if tower_type == "king":
                    if belonging == 0:
                        state.player_king_tower = tower
                    else:
                        state.enemy_king_tower = tower
                elif position == "left":
                    if belonging == 0:
                        state.player_left_princess = tower
                    else:
                        state.enemy_left_princess = tower
                elif position == "right":
                    if belonging == 0:
                        state.player_right_princess = tower
                    else:
                        state.enemy_right_princess = tower

            else:
                # This is a unit
                # Determine belonging based on position or other heuristics
                belonging = self._infer_unit_belonging(det, state.frame_height)

                unit = Unit(
                    class_name=det.class_name,
                    belonging=belonging,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
                state.units.append(unit)

        # Calculate average detection confidence
        if confidences:
            state.detection_confidence = sum(confidences) / len(confidences)

    def _infer_unit_belonging(
        self,
        detection: Detection,
        frame_height: int,
    ) -> int:
        """Infer which side a unit belongs to based on position.

        This is a simple heuristic - units in top half are enemy (1),
        units in bottom half are player (0).

        Args:
            detection: Detection object.
            frame_height: Height of the frame.

        Returns:
            0 for player, 1 for enemy, -1 for unknown.
        """
        center_y = detection.center[1]
        mid_y = frame_height / 2

        # Simple position-based heuristic
        # Arena midpoint is roughly at frame_height * 0.42
        arena_mid = frame_height * 0.42

        if center_y < arena_mid:
            return 1  # Enemy
        else:
            return 0  # Player

    def _process_ocr(
        self,
        state: GameState,
        ocr_results: GameOCRResults,
    ) -> None:
        """Process OCR results and update game state.

        Args:
            state: GameState to update.
            ocr_results: OCR extraction results.
        """
        # Timer
        if ocr_results.timer:
            state.time_remaining = ocr_results.timer.total_seconds
            state.is_overtime = ocr_results.timer.is_overtime

        # Elixir
        if ocr_results.elixir is not None:
            state.elixir = ocr_results.elixir

        # Tower HP values
        if ocr_results.player_king_hp is not None and state.player_king_tower:
            state.player_king_tower.hp = ocr_results.player_king_hp

        if ocr_results.player_left_princess_hp is not None and state.player_left_princess:
            state.player_left_princess.hp = ocr_results.player_left_princess_hp

        if ocr_results.player_right_princess_hp is not None and state.player_right_princess:
            state.player_right_princess.hp = ocr_results.player_right_princess_hp

        if ocr_results.enemy_king_hp is not None and state.enemy_king_tower:
            state.enemy_king_tower.hp = ocr_results.enemy_king_hp

        if ocr_results.enemy_left_princess_hp is not None and state.enemy_left_princess:
            state.enemy_left_princess.hp = ocr_results.enemy_left_princess_hp

        if ocr_results.enemy_right_princess_hp is not None and state.enemy_right_princess:
            state.enemy_right_princess.hp = ocr_results.enemy_right_princess_hp

    def visualize(
        self,
        image: Union[str, np.ndarray],
        state: Optional[GameState] = None,
        output_path: Optional[str] = None,
        show_towers: bool = True,
        show_units: bool = True,
        show_info: bool = True,
    ) -> np.ndarray:
        """Visualize game state on image.

        Args:
            image: Image to annotate.
            state: Game state (will extract if None).
            output_path: Path to save result.
            show_towers: Draw tower detections.
            show_units: Draw unit detections.
            show_info: Draw game info overlay.

        Returns:
            Annotated image.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        if state is None:
            state = self.build_state(img)

        # Colors
        player_color = (0, 255, 0)  # Green
        enemy_color = (0, 0, 255)  # Red
        info_color = (255, 255, 255)  # White

        # Draw towers
        if show_towers:
            for tower in state.player_towers:
                if tower.bbox:
                    self._draw_bbox(img, tower.bbox, player_color, f"P-{tower.tower_type}")
                    if tower.hp:
                        self._draw_text(img, f"HP: {tower.hp}", tower.bbox, offset_y=20)

            for tower in state.enemy_towers:
                if tower.bbox:
                    self._draw_bbox(img, tower.bbox, enemy_color, f"E-{tower.tower_type}")
                    if tower.hp:
                        self._draw_text(img, f"HP: {tower.hp}", tower.bbox, offset_y=20)

        # Draw units
        if show_units:
            for unit in state.units:
                color = player_color if unit.belonging == 0 else enemy_color
                label = unit.class_name[:10]  # Truncate long names
                self._draw_bbox(img, unit.bbox, color, label)

        # Draw info overlay
        if show_info:
            info_lines = [
                f"Time: {state.time_formatted}",
                f"Elixir: {state.elixir if state.elixir is not None else '?'}",
                f"Player Towers: {state.player_tower_count}",
                f"Enemy Towers: {state.enemy_tower_count}",
                f"Units: {len(state.units)}",
            ]

            y_offset = 30
            for line in info_lines:
                cv2.putText(img, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(img, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1)
                y_offset += 25

        if output_path:
            cv2.imwrite(output_path, img)

        return img

    def _draw_bbox(
        self,
        img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        """Draw bounding box with label."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_text(
        self,
        img: np.ndarray,
        text: str,
        bbox: Tuple[int, int, int, int],
        offset_y: int = 0,
    ) -> None:
        """Draw text below bounding box."""
        x1, y1, x2, y2 = bbox
        cv2.putText(img, text, (x1, y2 + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        fps: float = 2.0,
        save_frames: bool = False,
        save_json: bool = True,
    ) -> List[GameState]:
        """Process video and extract game states.

        Args:
            video_path: Path to video file.
            output_dir: Directory for output.
            fps: Target FPS for extraction.
            save_frames: Save annotated frames.
            save_json: Save game states as JSON.

        Returns:
            List of extracted game states.
        """
        from ..data.extract_frames import get_video_info

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / fps))

        states = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps
                state = self.build_state(frame, timestamp=timestamp)
                states.append(state)

                if save_frames:
                    frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                    self.visualize(frame, state, output_path=str(frame_path))

                print(f"Frame {frame_idx}/{total_frames}: {state.summary().split(chr(10))[0]}")

            frame_idx += 1

        cap.release()

        # Save all states to JSON
        if save_json:
            json_path = output_dir / "game_states.json"
            with open(json_path, "w") as f:
                json.dump([s.to_dict() for s in states], f, indent=2)
            print(f"Saved {len(states)} states to {json_path}")

        return states


def create_pipeline(
    model_path: Optional[str] = None,
    enable_ocr: bool = True,
    device: Optional[str] = None,
) -> StateBuilder:
    """Create a StateBuilder pipeline.

    Args:
        model_path: Path to detection model weights.
        enable_ocr: Whether to enable OCR.
        device: Device for inference.

    Returns:
        StateBuilder instance.
    """
    return StateBuilder(
        detection_model_path=model_path,
        enable_ocr=enable_ocr,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build game state from Clash Royale screenshot or video"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input image or video file"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to detection model weights"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="runs/pipeline",
        help="Output directory"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR extraction"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target FPS for video processing"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualized output"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = create_pipeline(
        model_path=args.model,
        enable_ocr=not args.no_ocr,
        device=args.device,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if video or image
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    is_video = input_path.suffix.lower() in video_extensions

    if is_video:
        # Process video
        states = pipeline.process_video(
            str(input_path),
            str(output_dir),
            fps=args.fps,
            save_frames=args.visualize,
            save_json=True,
        )
        print(f"\nProcessed {len(states)} frames from video")

    else:
        # Process single image
        state = pipeline.build_state(str(input_path))
        print("\n" + state.summary())

        # Save JSON
        json_path = output_dir / f"{input_path.stem}_state.json"
        with open(json_path, "w") as f:
            f.write(state.to_json())
        print(f"\nSaved state to {json_path}")

        # Visualize
        if args.visualize:
            vis_path = output_dir / f"{input_path.stem}_annotated.jpg"
            pipeline.visualize(str(input_path), state, output_path=str(vis_path))
            print(f"Saved visualization to {vis_path}")


if __name__ == "__main__":
    main()
