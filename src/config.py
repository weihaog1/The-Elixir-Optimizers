# Clash Royale Object Detection - Configuration
# Based on KataCR approach

# Image dimensions (width, height) - optimized for YOLOv8
IMAGE_SIZE = (576, 896)

# Detection confidence threshold
CONF_THRESHOLD = 0.5

# IOU threshold for NMS
IOU_THRESHOLD = 0.4

# Screen capture settings
CAPTURE_FPS = 10  # Frames per second for screen capture

# Screen region settings (adjust based on your emulator/phone screen position)
# Set to None to capture full screen, or specify (left, top, width, height)
SCREEN_REGION = None

# Model paths
MODEL_PATH = "models/clash_royale_detector.pt"

# Clash Royale unit categories (based on KataCR label_list)
# These are the main categories the model can detect
UNIT_CATEGORIES = {
    # Base units (always detected)
    0: "king-tower",
    1: "queen-tower",
    2: "cannoneer-tower",
    3: "dagger-duchess-tower", 
    4: "king-tower-bar",
    5: "queen-tower-bar",
    6: "small-text",
    7: "big-text",
    8: "elixir",
    9: "clock",
    10: "emoji",
    11: "bar1",
    12: "bar2",
    13: "skeleton-king-bar",
    14: "bar-level",
    
    # Common troops
    15: "knight",
    16: "archers",
    17: "giant",
    18: "musketeer",
    19: "mini-pekka",
    20: "valkyrie",
    21: "hog-rider",
    22: "wizard",
    23: "witch",
    24: "skeleton-army",
    25: "minions",
    26: "minion-horde",
    27: "balloon",
    28: "prince",
    29: "baby-dragon",
    30: "skeleton",
    31: "goblin",
    32: "spear-goblin",
    33: "bomber",
    34: "giant-skeleton",
    
    # Spells
    35: "fireball",
    36: "arrows",
    37: "rage",
    38: "freeze",
    39: "lightning",
    40: "zap",
    41: "poison",
    42: "graveyard",
    43: "tornado",
    44: "rocket",
    45: "log",
    
    # Buildings
    46: "cannon",
    47: "tesla",
    48: "inferno-tower",
    49: "bomb-tower",
    50: "goblin-cage",
    51: "tombstone",
    52: "furnace",
    53: "elixir-collector",
    54: "barbarian-hut",
    55: "goblin-hut",
    
    # Add more as needed...
}

# Colors for visualization (BGR format)
COLORS = {
    "friendly": (0, 255, 0),    # Green
    "enemy": (0, 0, 255),       # Red
    "neutral": (255, 255, 0),   # Cyan
    "building": (255, 165, 0),  # Orange
    "spell": (255, 0, 255),     # Magenta
}

# Arena grid dimensions
ARENA_GRID_WIDTH = 18
ARENA_GRID_HEIGHT = 32

# Card slot positions (relative ratios)
CARD_POSITIONS = [
    (0.20, 0.92),  # Card 1
    (0.40, 0.92),  # Card 2
    (0.60, 0.92),  # Card 3
    (0.80, 0.92),  # Card 4
]

# Next card position
NEXT_CARD_POSITION = (0.10, 0.85)
