"""
Utility functions for Clash Royale Object Detection.
Includes coordinate conversion, drawing helpers, and data processing.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import colorsys


def generate_colors(num_colors: int, saturation: float = 0.9, value: float = 0.9) -> List[Tuple[int, int, int]]:
    """
    Generate visually distinct colors for visualization.
    
    Args:
        num_colors: Number of colors to generate.
        saturation: Color saturation (0-1).
        value: Color value/brightness (0-1).
        
    Returns:
        List of BGR color tuples.
    """
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def pixel_to_cell(
    x: float, y: float,
    image_width: int, image_height: int,
    arena_bbox: Tuple[int, int, int, int],
    grid_width: int = 18,
    grid_height: int = 32
) -> Tuple[int, int]:
    """
    Convert pixel coordinates to arena cell coordinates.
    
    Args:
        x, y: Pixel coordinates.
        image_width, image_height: Image dimensions.
        arena_bbox: Arena bounding box (x1, y1, x2, y2).
        grid_width: Number of columns in arena grid.
        grid_height: Number of rows in arena grid.
        
    Returns:
        (cell_x, cell_y) tuple.
    """
    ax1, ay1, ax2, ay2 = arena_bbox
    arena_width = ax2 - ax1
    arena_height = ay2 - ay1
    
    # Normalize to arena coordinates
    rel_x = (x - ax1) / arena_width
    rel_y = (y - ay1) / arena_height
    
    # Convert to cell indices
    cell_x = int(rel_x * grid_width)
    cell_y = int(rel_y * grid_height)
    
    # Clamp to valid range
    cell_x = max(0, min(grid_width - 1, cell_x))
    cell_y = max(0, min(grid_height - 1, cell_y))
    
    return (cell_x, cell_y)


def cell_to_pixel(
    cell_x: int, cell_y: int,
    image_width: int, image_height: int,
    arena_bbox: Tuple[int, int, int, int],
    grid_width: int = 18,
    grid_height: int = 32
) -> Tuple[int, int]:
    """
    Convert arena cell coordinates to pixel coordinates.
    
    Args:
        cell_x, cell_y: Cell coordinates.
        image_width, image_height: Image dimensions.
        arena_bbox: Arena bounding box (x1, y1, x2, y2).
        grid_width: Number of columns in arena grid.
        grid_height: Number of rows in arena grid.
        
    Returns:
        (x, y) pixel coordinates (center of cell).
    """
    ax1, ay1, ax2, ay2 = arena_bbox
    arena_width = ax2 - ax1
    arena_height = ay2 - ay1
    
    # Calculate cell size
    cell_width = arena_width / grid_width
    cell_height = arena_height / grid_height
    
    # Calculate center of cell
    x = ax1 + (cell_x + 0.5) * cell_width
    y = ay1 + (cell_y + 0.5) * cell_height
    
    return (int(x), int(y))


def calculate_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First box (x1, y1, x2, y2).
        box2: Second box (x1, y1, x2, y2).
        
    Returns:
        IoU value (0-1).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def non_max_suppression(
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    iou_threshold: float = 0.5
) -> List[int]:
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.
    
    Args:
        boxes: List of bounding boxes (x1, y1, x2, y2).
        scores: List of confidence scores.
        iou_threshold: IoU threshold for suppression.
        
    Returns:
        List of indices to keep.
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while indices:
        current = indices[0]
        keep.append(current)
        
        remaining = []
        for idx in indices[1:]:
            iou = calculate_iou(boxes[current], boxes[idx])
            if iou < iou_threshold:
                remaining.append(idx)
        
        indices = remaining
    
    return keep


def resize_with_pad(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[float, int, int]]:
    """
    Resize image while maintaining aspect ratio with padding.
    
    Args:
        image: Input image.
        target_size: Target (width, height).
        pad_color: Padding color (BGR).
        
    Returns:
        Tuple of (resized_image, (scale, pad_x, pad_y)).
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    
    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # Place resized image
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return padded, (scale, pad_x, pad_y)


def draw_grid(
    image: np.ndarray,
    arena_bbox: Tuple[int, int, int, int],
    grid_width: int = 18,
    grid_height: int = 32,
    color: Tuple[int, int, int] = (100, 100, 100),
    thickness: int = 1
) -> np.ndarray:
    """
    Draw arena grid overlay on image.
    
    Args:
        image: Input image.
        arena_bbox: Arena bounding box (x1, y1, x2, y2).
        grid_width: Number of columns.
        grid_height: Number of rows.
        color: Grid line color (BGR).
        thickness: Line thickness.
        
    Returns:
        Image with grid overlay.
    """
    result = image.copy()
    ax1, ay1, ax2, ay2 = arena_bbox
    arena_width = ax2 - ax1
    arena_height = ay2 - ay1
    
    cell_width = arena_width / grid_width
    cell_height = arena_height / grid_height
    
    # Draw vertical lines
    for i in range(grid_width + 1):
        x = int(ax1 + i * cell_width)
        cv2.line(result, (x, ay1), (x, ay2), color, thickness)
    
    # Draw horizontal lines
    for i in range(grid_height + 1):
        y = int(ay1 + i * cell_height)
        cv2.line(result, (ax1, y), (ax2, y), color, thickness)
    
    return result


def create_heatmap(
    data: np.ndarray,
    size: Tuple[int, int],
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Create a heatmap visualization from 2D data.
    
    Args:
        data: 2D numpy array of values.
        size: Output (width, height).
        colormap: OpenCV colormap.
        
    Returns:
        BGR heatmap image.
    """
    # Normalize to 0-255
    normalized = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
    
    # Resize to target size
    resized = cv2.resize(normalized, size, interpolation=cv2.INTER_NEAREST)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(resized, colormap)
    
    return heatmap


def calculate_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: (x, y) coordinates.
        point2: (x, y) coordinates.
        
    Returns:
        Distance value.
    """
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


def smooth_positions(
    positions: List[Tuple[float, float]],
    window_size: int = 3
) -> List[Tuple[float, float]]:
    """
    Smooth a sequence of positions using moving average.
    
    Args:
        positions: List of (x, y) positions.
        window_size: Smoothing window size.
        
    Returns:
        Smoothed positions.
    """
    if len(positions) < window_size:
        return positions
    
    smoothed = []
    for i in range(len(positions)):
        start = max(0, i - window_size // 2)
        end = min(len(positions), i + window_size // 2 + 1)
        
        x_avg = sum(p[0] for p in positions[start:end]) / (end - start)
        y_avg = sum(p[1] for p in positions[start:end]) / (end - start)
        
        smoothed.append((x_avg, y_avg))
    
    return smoothed


class FPSCounter:
    """Utility class for measuring frames per second."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average.
        """
        self.window_size = window_size
        self.times = []
        self.fps = 0
    
    def update(self, current_time: float) -> float:
        """
        Update FPS measurement.
        
        Args:
            current_time: Current timestamp.
            
        Returns:
            Current FPS value.
        """
        self.times.append(current_time)
        
        # Keep only recent times
        if len(self.times) > self.window_size:
            self.times = self.times[-self.window_size:]
        
        # Calculate FPS
        if len(self.times) >= 2:
            time_diff = self.times[-1] - self.times[0]
            if time_diff > 0:
                self.fps = (len(self.times) - 1) / time_diff
        
        return self.fps
    
    def get(self) -> float:
        """Get current FPS value."""
        return self.fps
    
    def reset(self):
        """Reset FPS counter."""
        self.times = []
        self.fps = 0


class MovingAverage:
    """Simple moving average calculator."""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize moving average.
        
        Args:
            window_size: Number of values to average.
        """
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float) -> float:
        """
        Add a value and get current average.
        
        Args:
            value: New value to add.
            
        Returns:
            Current moving average.
        """
        self.values.append(value)
        
        if len(self.values) > self.window_size:
            self.values = self.values[-self.window_size:]
        
        return sum(self.values) / len(self.values)
    
    def get(self) -> float:
        """Get current average."""
        return sum(self.values) / len(self.values) if self.values else 0
    
    def reset(self):
        """Reset average."""
        self.values = []


if __name__ == "__main__":
    # Test utilities
    print("Utility functions test")
    
    # Test color generation
    colors = generate_colors(10)
    print(f"Generated {len(colors)} colors")
    
    # Test IoU calculation
    box1 = (0, 0, 10, 10)
    box2 = (5, 5, 15, 15)
    iou = calculate_iou(box1, box2)
    print(f"IoU of overlapping boxes: {iou:.2f}")
    
    # Test coordinate conversion
    arena_bbox = (50, 100, 490, 700)
    pixel = (270, 400)
    cell = pixel_to_cell(pixel[0], pixel[1], 540, 1200, arena_bbox)
    print(f"Pixel {pixel} -> Cell {cell}")
    
    pixel_back = cell_to_pixel(cell[0], cell[1], 540, 1200, arena_bbox)
    print(f"Cell {cell} -> Pixel {pixel_back}")
