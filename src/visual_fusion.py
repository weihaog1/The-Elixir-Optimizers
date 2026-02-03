"""
Visual Fusion Module for Clash Royale
Combines detection results with game state analysis.
Based on KataCR's VisualFusion approach.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from detector import Detection, draw_detections


class UnitBelonging(Enum):
    """Unit belonging/team classification."""
    UNKNOWN = 0
    FRIENDLY = 1
    ENEMY = 2


@dataclass
class UnitInfo:
    """Information about a detected unit."""
    xy: Tuple[float, float]  # Center position
    cls: int  # Class ID
    cls_name: str  # Class name
    belonging: UnitBelonging  # Team affiliation
    confidence: float
    bbox: Tuple[float, float, float, float]  # Bounding box
    
    # Optional health bar info
    health_ratio: Optional[float] = None
    
    def to_cell(self, arena_width: int = 18, arena_height: int = 32) -> Tuple[int, int]:
        """Convert pixel position to arena cell coordinates."""
        # This is a simplified conversion - adjust based on your screen layout
        cell_x = int(self.xy[0] / arena_width)
        cell_y = int(self.xy[1] / arena_height)
        return (cell_x, cell_y)


@dataclass
class GameState:
    """Current game state extracted from visual information."""
    # Time in seconds (0-180 for regular time, 180+ for overtime)
    time: int = 0
    
    # Elixir count (0-10)
    elixir: int = 0
    
    # Cards in hand (indices 0-4: next card, card1-4)
    cards: List[str] = field(default_factory=list)
    
    # All detected units in arena
    units: List[UnitInfo] = field(default_factory=list)
    
    # Tower states
    king_tower_health: Optional[float] = None
    left_princess_health: Optional[float] = None
    right_princess_health: Optional[float] = None
    enemy_king_tower_health: Optional[float] = None
    enemy_left_princess_health: Optional[float] = None
    enemy_right_princess_health: Optional[float] = None


class ScreenRegions:
    """
    Defines regions of the Clash Royale screen.
    Based on KataCR's split_part approach.
    Adjust these values based on your screen resolution/aspect ratio.
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize screen regions based on screen dimensions.
        
        Args:
            screen_width: Width of the captured screen.
            screen_height: Height of the captured screen.
        """
        self.width = screen_width
        self.height = screen_height
        
        # Aspect ratio (height/width) - Clash Royale is typically 2.22
        self.aspect_ratio = screen_height / screen_width
        
        # Calculate region boundaries (as ratios of screen size)
        # These are based on a 1080x2400 (2.22 ratio) screen
        # Adjust if your aspect ratio is different
        
        # Part 1: Top area (time display)
        self.time_region = self._ratio_to_pixels(
            x1=0.75, y1=0.0, x2=1.0, y2=0.05
        )
        
        # Part 2: Arena area (main game area)
        self.arena_region = self._ratio_to_pixels(
            x1=0.0, y1=0.08, x2=1.0, y2=0.75
        )
        
        # Part 3: Bottom area (cards and elixir)
        self.cards_region = self._ratio_to_pixels(
            x1=0.0, y1=0.75, x2=1.0, y2=1.0
        )
        
        # Elixir bar region (within cards region)
        self.elixir_region = self._ratio_to_pixels(
            x1=0.15, y1=0.80, x2=0.85, y2=0.83
        )
        
        # Individual card positions (within cards region)
        self.card_positions = [
            self._ratio_to_pixels(0.08, 0.85, 0.18, 0.95),   # Next card
            self._ratio_to_pixels(0.20, 0.85, 0.40, 0.98),   # Card 1
            self._ratio_to_pixels(0.40, 0.85, 0.60, 0.98),   # Card 2
            self._ratio_to_pixels(0.60, 0.85, 0.80, 0.98),   # Card 3
            self._ratio_to_pixels(0.80, 0.85, 1.00, 0.98),   # Card 4
        ]
    
    def _ratio_to_pixels(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[int, int, int, int]:
        """Convert ratio coordinates to pixel coordinates."""
        return (
            int(x1 * self.width),
            int(y1 * self.height),
            int(x2 * self.width),
            int(y2 * self.height)
        )
    
    def extract_region(
        self, image: np.ndarray, region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract a region from an image."""
        x1, y1, x2, y2 = region
        return image[y1:y2, x1:x2].copy()


class VisualFusion:
    """
    Combines multiple detection results and extracts game state.
    Based on KataCR's visual_fusion.py approach.
    """
    
    def __init__(
        self,
        screen_width: int = 540,
        screen_height: int = 1200
    ):
        """
        Initialize visual fusion.
        
        Args:
            screen_width: Width of the captured screen.
            screen_height: Height of the captured screen.
        """
        self.regions = ScreenRegions(screen_width, screen_height)
        self.last_state = None
        
        # Unit categories for classification
        self.tower_units = {
            "king-tower", "queen-tower", "cannoneer-tower",
            "dagger-duchess-tower"
        }
        self.bar_units = {
            "bar1", "bar2", "king-tower-bar", "queen-tower-bar",
            "skeleton-king-bar", "bar-level"
        }
        self.spell_units = {
            "fireball", "arrows", "rage", "freeze", "lightning",
            "zap", "poison", "graveyard", "tornado", "rocket", "log"
        }
        self.building_units = {
            "cannon", "tesla", "inferno-tower", "bomb-tower",
            "goblin-cage", "tombstone", "furnace", "elixir-collector",
            "barbarian-hut", "goblin-hut"
        }
    
    def classify_belonging(
        self,
        detection: Detection,
        arena_center_y: float
    ) -> UnitBelonging:
        """
        Determine if a unit belongs to friendly or enemy team.
        Based on position in arena (top half = enemy, bottom half = friendly).
        
        Args:
            detection: Detection object.
            arena_center_y: Y-coordinate of arena center.
            
        Returns:
            UnitBelonging enum value.
        """
        # Simple heuristic: units above center are enemy, below are friendly
        if detection.center_y < arena_center_y:
            return UnitBelonging.ENEMY
        else:
            return UnitBelonging.FRIENDLY
    
    def process_detections(
        self,
        detections: List[Detection],
        image_height: int
    ) -> List[UnitInfo]:
        """
        Process raw detections into unit information.
        
        Args:
            detections: List of Detection objects.
            image_height: Height of the image for position calculation.
            
        Returns:
            List of UnitInfo objects.
        """
        arena_center_y = image_height * 0.4  # Approximate arena center
        
        units = []
        for det in detections:
            # Skip UI elements
            if det.class_name in {"small-text", "big-text", "clock", "emoji", "elixir"}:
                continue
            
            # Skip health bars (handled separately)
            if det.class_name in self.bar_units:
                continue
            
            # Classify belonging
            if det.class_name in self.tower_units:
                # Towers have fixed positions
                belonging = (
                    UnitBelonging.ENEMY if det.center_y < arena_center_y
                    else UnitBelonging.FRIENDLY
                )
            else:
                belonging = self.classify_belonging(det, arena_center_y)
            
            unit = UnitInfo(
                xy=(det.center_x, det.center_y),
                cls=det.class_id,
                cls_name=det.class_name,
                belonging=belonging,
                confidence=det.confidence,
                bbox=det.bbox
            )
            
            units.append(unit)
        
        return units
    
    def match_health_bars(
        self,
        units: List[UnitInfo],
        detections: List[Detection],
        max_distance: float = 50
    ) -> List[UnitInfo]:
        """
        Match health bars to units.
        
        Args:
            units: List of UnitInfo objects.
            detections: All detections including health bars.
            max_distance: Maximum distance to match bar to unit.
            
        Returns:
            Updated list of UnitInfo with health information.
        """
        # Get health bar detections
        bars = [d for d in detections if d.class_name in self.bar_units]
        
        for unit in units:
            # Find closest health bar above the unit
            closest_bar = None
            min_distance = max_distance
            
            for bar in bars:
                # Bar should be above the unit
                if bar.center_y > unit.xy[1]:
                    continue
                
                # Calculate horizontal distance
                dx = abs(bar.center_x - unit.xy[0])
                dy = unit.xy[1] - bar.center_y
                
                distance = (dx**2 + dy**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_bar = bar
            
            if closest_bar is not None:
                # Estimate health ratio from bar width
                # This is a simplified approximation
                unit.health_ratio = closest_bar.width / 100  # Normalize
        
        return units
    
    def extract_game_state(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> GameState:
        """
        Extract complete game state from image and detections.
        
        Args:
            image: BGR image of the game screen.
            detections: List of Detection objects.
            
        Returns:
            GameState object.
        """
        h, w = image.shape[:2]
        
        # Process units
        units = self.process_detections(detections, h)
        units = self.match_health_bars(units, detections)
        
        # Create game state
        state = GameState(
            units=units
        )
        
        # TODO: Add OCR for time and elixir
        # TODO: Add card classification
        
        self.last_state = state
        return state
    
    def draw_game_state(
        self,
        image: np.ndarray,
        state: GameState,
        show_belonging: bool = True
    ) -> np.ndarray:
        """
        Draw game state visualization on image.
        
        Args:
            image: BGR image to draw on.
            state: GameState to visualize.
            show_belonging: Whether to color-code by team.
            
        Returns:
            Image with visualization.
        """
        result = image.copy()
        
        for unit in state.units:
            # Choose color based on belonging
            if show_belonging:
                if unit.belonging == UnitBelonging.FRIENDLY:
                    color = (0, 255, 0)  # Green
                elif unit.belonging == UnitBelonging.ENEMY:
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 255, 0)  # Cyan
            else:
                color = (255, 255, 255)  # White
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, unit.bbox)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{unit.cls_name}"
            if unit.health_ratio is not None:
                label += f" HP:{unit.health_ratio:.0%}"
            
            cv2.putText(
                result, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1, cv2.LINE_AA
            )
            
            # Draw center point
            cx, cy = map(int, unit.xy)
            cv2.circle(result, (cx, cy), 3, color, -1)
        
        # Draw region boundaries (for debugging)
        # cv2.rectangle(result, 
        #     self.regions.arena_region[:2],
        #     self.regions.arena_region[2:],
        #     (128, 128, 128), 1)
        
        return result


class ArenaAnalyzer:
    """
    Analyzes the arena state for strategic decision making.
    """
    
    def __init__(self, grid_width: int = 18, grid_height: int = 32):
        """
        Initialize arena analyzer.
        
        Args:
            grid_width: Number of columns in arena grid.
            grid_height: Number of rows in arena grid.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Initialize occupancy grid
        self.occupancy = np.zeros((grid_height, grid_width), dtype=np.float32)
    
    def pixel_to_cell(
        self,
        x: float, y: float,
        screen_width: int, screen_height: int,
        arena_region: Tuple[int, int, int, int]
    ) -> Tuple[int, int]:
        """Convert pixel coordinates to arena cell coordinates."""
        ax1, ay1, ax2, ay2 = arena_region
        arena_width = ax2 - ax1
        arena_height = ay2 - ay1
        
        # Normalize to arena coordinates
        rel_x = (x - ax1) / arena_width
        rel_y = (y - ay1) / arena_height
        
        # Convert to cell indices
        cell_x = int(rel_x * self.grid_width)
        cell_y = int(rel_y * self.grid_height)
        
        # Clamp to valid range
        cell_x = max(0, min(self.grid_width - 1, cell_x))
        cell_y = max(0, min(self.grid_height - 1, cell_y))
        
        return (cell_x, cell_y)
    
    def update_occupancy(
        self,
        units: List[UnitInfo],
        screen_width: int,
        screen_height: int,
        arena_region: Tuple[int, int, int, int]
    ):
        """Update the occupancy grid based on unit positions."""
        self.occupancy = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        for unit in units:
            cell_x, cell_y = self.pixel_to_cell(
                unit.xy[0], unit.xy[1],
                screen_width, screen_height,
                arena_region
            )
            
            # Mark cell as occupied
            # Positive for friendly, negative for enemy
            if unit.belonging == UnitBelonging.FRIENDLY:
                self.occupancy[cell_y, cell_x] += 1
            elif unit.belonging == UnitBelonging.ENEMY:
                self.occupancy[cell_y, cell_x] -= 1
    
    def get_threat_zones(self) -> np.ndarray:
        """Get areas with high enemy concentration."""
        return (self.occupancy < -1).astype(np.float32)
    
    def get_defensive_gaps(self) -> np.ndarray:
        """Get areas that need defensive units."""
        # Areas in our half with low friendly presence
        our_half = np.zeros_like(self.occupancy)
        our_half[self.grid_height // 2:, :] = 1
        
        gaps = (self.occupancy <= 0) & (our_half > 0)
        return gaps.astype(np.float32)


if __name__ == "__main__":
    # Test visual fusion
    print("Visual Fusion Module Test")
    
    # Create sample screen regions
    regions = ScreenRegions(540, 1200)
    print(f"Arena region: {regions.arena_region}")
    print(f"Cards region: {regions.cards_region}")
    print(f"Card positions: {regions.card_positions}")
