"""
Main Entry Point for Clash Royale Object Detection
Non-embedded AI based on KataCR approach.
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Optional, List, Union

from config import CONF_THRESHOLD, IOU_THRESHOLD, CAPTURE_FPS, SCREEN_REGION
from capture import ScreenCapture, WindowCapture, ImageLoader, select_capture_region
from detector import ClashRoyaleDetector, ComboDetector, draw_detections, filter_detections
from visual_fusion import VisualFusion, GameState


class ClashRoyaleAI:
    """
    Main class for Clash Royale object detection and game state analysis.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, List[str]]] = None,
        conf_threshold: float = CONF_THRESHOLD,
        capture_source: str = "screen",  # "screen", "window", "image", "video"
        source_arg: Optional[str] = None,  # Window title or file path
        screen_region: Optional[tuple] = SCREEN_REGION,
        use_combo: bool = False
    ):
        """
        Initialize the Clash Royale AI.
        
        Args:
            model_path: Path to model weights, or list of paths for combo detection.
            conf_threshold: Confidence threshold for detections.
            capture_source: Type of capture source.
            source_arg: Additional argument for capture source.
            screen_region: Screen region to capture (left, top, width, height).
            use_combo: Whether to use combo detection with multiple models.
        """
        # Initialize detector
        print("Initializing object detector...")
        
        # Determine if using combo detection
        if use_combo or (isinstance(model_path, list) and len(model_path) > 1):
            # Combo detection with multiple models
            if model_path is None:
                raise ValueError("Combo detection requires model paths. Use --models to specify.")
            
            model_paths = model_path if isinstance(model_path, list) else [model_path]
            print(f"Using ComboDetector with {len(model_paths)} models")
            
            self.detector = ComboDetector(
                model_paths=model_paths,
                conf_threshold=conf_threshold
            )
            self.use_combo = True
        else:
            # Single model detection
            single_path = model_path[0] if isinstance(model_path, list) else model_path
            self.detector = ClashRoyaleDetector(
                model_path=single_path,
                conf_threshold=conf_threshold
            )
            self.use_combo = False
        
        # Initialize capture based on source type
        print(f"Initializing capture source: {capture_source}")
        if capture_source == "screen":
            if screen_region is None:
                print("No screen region specified. Please select the game window.")
                screen_region = select_capture_region()
            self.capture = ScreenCapture(region=screen_region, fps_limit=CAPTURE_FPS)
            screen_size = (screen_region[2], screen_region[3]) if screen_region else (1920, 1080)
        
        elif capture_source == "window":
            if source_arg is None:
                raise ValueError("Window title required for window capture")
            self.capture = WindowCapture(window_title=source_arg)
            # Get window size from first capture
            test_frame = self.capture.capture()
            screen_size = (test_frame.shape[1], test_frame.shape[0])
        
        elif capture_source in ["image", "video"]:
            if source_arg is None:
                raise ValueError("File path required for image/video capture")
            self.capture = ImageLoader(source=source_arg)
            test_frame = self.capture.capture()
            if test_frame is None:
                raise ValueError(f"Could not load: {source_arg}")
            screen_size = (test_frame.shape[1], test_frame.shape[0])
            # Reset for image loader
            self.capture.current_index = 0
        
        else:
            raise ValueError(f"Unknown capture source: {capture_source}")
        
        # Initialize visual fusion
        self.visual_fusion = VisualFusion(
            screen_width=screen_size[0],
            screen_height=screen_size[1]
        )
        
        # Stats tracking
        self.frame_count = 0
        self.total_time = 0
        self.fps = 0
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame.
        
        Args:
            frame: BGR image frame.
            
        Returns:
            Tuple of (detections, game_state, processing_time)
        """
        start_time = time.time()
        
        # Run detection
        detections = self.detector.detect(frame)
        
        # Extract game state
        game_state = self.visual_fusion.extract_game_state(frame, detections)
        
        processing_time = time.time() - start_time
        
        return detections, game_state, processing_time
    
    def run(
        self,
        show_display: bool = True,
        save_output: Optional[str] = None,
        max_frames: Optional[int] = None
    ):
        """
        Run the detection loop.
        
        Args:
            show_display: Whether to show the detection visualization.
            save_output: Path to save output video (optional).
            max_frames: Maximum number of frames to process.
        """
        print("\nStarting Clash Royale Object Detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        print("-" * 50)
        
        # Video writer for saving output
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Will initialize with first frame size
        
        try:
            for frame in self.capture.capture_continuous():
                if frame is None:
                    break
                
                # Process frame
                detections, game_state, proc_time = self.process_frame(frame)
                
                # Update stats
                self.frame_count += 1
                self.total_time += proc_time
                self.fps = self.frame_count / self.total_time if self.total_time > 0 else 0
                
                # Draw visualization
                vis_frame = self.visual_fusion.draw_game_state(
                    frame, game_state, show_belonging=True
                )
                
                # Add stats overlay
                stats_text = [
                    f"FPS: {self.fps:.1f}",
                    f"Detections: {len(detections)}",
                    f"Units: {len(game_state.units)}",
                    f"Process time: {proc_time*1000:.1f}ms"
                ]
                
                y_offset = 30
                for text in stats_text:
                    cv2.putText(
                        vis_frame, text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2
                    )
                    y_offset += 25
                
                # Initialize video writer with first frame
                if save_output and video_writer is None:
                    h, w = vis_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        save_output, fourcc, CAPTURE_FPS, (w, h)
                    )
                
                # Save frame to video
                if video_writer:
                    video_writer.write(vis_frame)
                
                # Show display
                if show_display:
                    cv2.imshow("Clash Royale Detection", vis_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"screenshot_{self.frame_count}.png"
                        cv2.imwrite(screenshot_path, vis_frame)
                        print(f"Saved: {screenshot_path}")
                
                # Check frame limit
                if max_frames and self.frame_count >= max_frames:
                    break
                
                # Print periodic stats
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count} frames | FPS: {self.fps:.1f}")
        
        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            self.capture.release()
            
            # Print final stats
            print("\n" + "=" * 50)
            print("Detection Complete")
            print(f"Total frames: {self.frame_count}")
            print(f"Total time: {self.total_time:.2f}s")
            print(f"Average FPS: {self.fps:.1f}")
            print("=" * 50)


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Clash Royale Object Detection - Non-Embedded AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model detection
  python main.py --model models/detector1.pt --select-region
  
  # Combo detection with multiple models (recommended)
  python main.py --models models/detector1.pt models/detector2.pt --select-region
  
  # Process video with combo detection
  python main.py --models models/detector1.pt models/detector2.pt \\
                 --source video --source-arg gameplay.mp4 --output result.mp4
  
  # Capture from emulator window
  python main.py --models models/detector1.pt models/detector2.pt \\
                 --source window --source-arg "BlueStacks"
"""
    )
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to single YOLOv8 model weights (.pt file)"
    )
    model_group.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=None,
        help="Paths to multiple YOLOv8 models for combo detection"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        choices=["screen", "window", "image", "video"],
        default="screen",
        help="Capture source type"
    )
    
    parser.add_argument(
        "--source-arg",
        type=str,
        default=None,
        help="Window title or file path for capture source"
    )
    
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        default=None,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Screen capture region"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD})"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output video"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    
    parser.add_argument(
        "--select-region",
        action="store_true",
        help="Interactively select screen capture region"
    )
    
    args = parser.parse_args()
    
    # Handle region selection
    screen_region = tuple(args.region) if args.region else SCREEN_REGION
    if args.select_region:
        screen_region = None  # Will trigger interactive selection
    
    # Determine model path(s)
    if args.models:
        model_path = args.models
        use_combo = True
    elif args.model:
        model_path = args.model
        use_combo = False
    else:
        model_path = None
        use_combo = False
    
    # Create and run AI
    try:
        ai = ClashRoyaleAI(
            model_path=model_path,
            conf_threshold=args.conf,
            capture_source=args.source,
            source_arg=args.source_arg,
            screen_region=screen_region,
            use_combo=use_combo
        )
        
        ai.run(
            show_display=not args.no_display,
            save_output=args.output,
            max_frames=args.max_frames
        )
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
