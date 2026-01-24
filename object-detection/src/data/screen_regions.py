"""
Screen region configuration for Clash Royale screenshots.

Defines regions of interest for different game elements based on
Google Play Games resolution (~540x960).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import yaml


@dataclass
class Region:
    """A rectangular region on the screen."""

    x_start: int
    y_start: int
    x_end: int
    y_end: int
    name: str = ""

    @property
    def width(self) -> int:
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.x_start + self.x_end) // 2,
            (self.y_start + self.y_end) // 2
        )

    def as_bbox(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) bounding box."""
        return (self.x_start, self.y_start, self.x_end, self.y_end)

    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert to YOLO format (x_center, y_center, width, height) normalized."""
        x_center = (self.x_start + self.x_end) / 2 / img_width
        y_center = (self.y_start + self.y_end) / 2 / img_height
        width = self.width / img_width
        height = self.height / img_height
        return (x_center, y_center, width, height)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is within this region."""
        return self.x_start <= x <= self.x_end and self.y_start <= y <= self.y_end

    def crop_image(self, image):
        """Crop an image to this region. Image should be a numpy array."""
        return image[self.y_start:self.y_end, self.x_start:self.x_end]


@dataclass
class ScreenConfig:
    """Configuration for screen regions in Clash Royale.

    Default values are calibrated for 540x960 resolution from Google Play Games.
    """

    width: int = 540
    height: int = 960

    # Main screen sections
    top_bar: Region = field(default_factory=lambda: Region(0, 0, 540, 50, "top_bar"))
    arena: Region = field(default_factory=lambda: Region(0, 50, 540, 750, "arena"))
    card_bar: Region = field(default_factory=lambda: Region(0, 750, 540, 960, "card_bar"))

    # Specific UI elements for OCR
    timer_region: Region = field(
        default_factory=lambda: Region(400, 5, 535, 45, "timer")
    )
    elixir_region: Region = field(
        default_factory=lambda: Region(90, 880, 130, 920, "elixir")
    )

    # Tower regions (approximate positions for detection ground truth)
    # Player side (bottom)
    king_tower_player: Region = field(
        default_factory=lambda: Region(210, 580, 330, 720, "king_tower_player")
    )
    princess_tower_left_player: Region = field(
        default_factory=lambda: Region(60, 520, 160, 640, "princess_tower_left_player")
    )
    princess_tower_right_player: Region = field(
        default_factory=lambda: Region(380, 520, 480, 640, "princess_tower_right_player")
    )

    # Enemy side (top)
    king_tower_enemy: Region = field(
        default_factory=lambda: Region(210, 90, 330, 230, "king_tower_enemy")
    )
    princess_tower_left_enemy: Region = field(
        default_factory=lambda: Region(60, 170, 160, 290, "princess_tower_left_enemy")
    )
    princess_tower_right_enemy: Region = field(
        default_factory=lambda: Region(380, 170, 480, 290, "princess_tower_right_enemy")
    )

    # Card slots (4 cards + next card)
    card_slots: Dict[str, Region] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize card slot regions."""
        if not self.card_slots:
            # Next card (smaller, on the left)
            self.card_slots["next"] = Region(20, 770, 90, 870, "next_card")
            # 4 playable cards
            card_width = 100
            card_start_x = 110
            card_y_start = 770
            card_y_end = 920
            for i in range(4):
                x_start = card_start_x + i * card_width
                self.card_slots[f"card_{i+1}"] = Region(
                    x_start, card_y_start, x_start + card_width, card_y_end,
                    f"card_{i+1}"
                )

    def get_tower_regions(self) -> Dict[str, Region]:
        """Return all tower regions."""
        return {
            "king_tower_player": self.king_tower_player,
            "king_tower_enemy": self.king_tower_enemy,
            "princess_tower_left_player": self.princess_tower_left_player,
            "princess_tower_left_enemy": self.princess_tower_left_enemy,
            "princess_tower_right_player": self.princess_tower_right_player,
            "princess_tower_right_enemy": self.princess_tower_right_enemy,
        }

    def get_all_regions(self) -> Dict[str, Region]:
        """Return all defined regions."""
        regions = {
            "top_bar": self.top_bar,
            "arena": self.arena,
            "card_bar": self.card_bar,
            "timer": self.timer_region,
            "elixir": self.elixir_region,
        }
        regions.update(self.get_tower_regions())
        regions.update(self.card_slots)
        return regions

    def scale_to_resolution(self, new_width: int, new_height: int) -> "ScreenConfig":
        """Create a new config scaled to a different resolution."""
        x_scale = new_width / self.width
        y_scale = new_height / self.height

        def scale_region(r: Region) -> Region:
            return Region(
                int(r.x_start * x_scale),
                int(r.y_start * y_scale),
                int(r.x_end * x_scale),
                int(r.y_end * y_scale),
                r.name
            )

        new_config = ScreenConfig(
            width=new_width,
            height=new_height,
            top_bar=scale_region(self.top_bar),
            arena=scale_region(self.arena),
            card_bar=scale_region(self.card_bar),
            timer_region=scale_region(self.timer_region),
            elixir_region=scale_region(self.elixir_region),
            king_tower_player=scale_region(self.king_tower_player),
            princess_tower_left_player=scale_region(self.princess_tower_left_player),
            princess_tower_right_player=scale_region(self.princess_tower_right_player),
            king_tower_enemy=scale_region(self.king_tower_enemy),
            princess_tower_left_enemy=scale_region(self.princess_tower_left_enemy),
            princess_tower_right_enemy=scale_region(self.princess_tower_right_enemy),
        )

        # Scale card slots
        new_config.card_slots = {
            name: scale_region(region)
            for name, region in self.card_slots.items()
        }

        return new_config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ScreenConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        screen_data = data.get("screen", {})
        config = cls(
            width=screen_data.get("width", 540),
            height=screen_data.get("height", 960),
        )

        # Load region overrides if present
        regions = screen_data.get("regions", {})
        if "top_bar" in regions:
            r = regions["top_bar"]
            config.top_bar = Region(
                0, r.get("y_start", 0), config.width, r.get("y_end", 50), "top_bar"
            )
        if "arena" in regions:
            r = regions["arena"]
            config.arena = Region(
                0, r.get("y_start", 50), config.width, r.get("y_end", 750), "arena"
            )
        if "cards" in regions:
            r = regions["cards"]
            config.card_bar = Region(
                0, r.get("y_start", 750), config.width, r.get("y_end", 960), "card_bar"
            )

        return config

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        data = {
            "screen": {
                "width": self.width,
                "height": self.height,
                "regions": {
                    "top_bar": {
                        "y_start": self.top_bar.y_start,
                        "y_end": self.top_bar.y_end
                    },
                    "arena": {
                        "y_start": self.arena.y_start,
                        "y_end": self.arena.y_end
                    },
                    "cards": {
                        "y_start": self.card_bar.y_start,
                        "y_end": self.card_bar.y_end
                    },
                }
            }
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Class ID mappings (matching classes.yaml)
TOWER_CLASS_IDS = {
    "king_tower_player": 0,
    "king_tower_enemy": 1,
    "princess_tower_left_player": 2,
    "princess_tower_left_enemy": 3,
    "princess_tower_right_player": 4,
    "princess_tower_right_enemy": 5,
}

CLASS_ID_TO_NAME = {v: k for k, v in TOWER_CLASS_IDS.items()}


def get_default_config() -> ScreenConfig:
    """Get the default screen configuration for 540x960 resolution."""
    return ScreenConfig()


def detect_resolution(image) -> Tuple[int, int]:
    """Detect image resolution from a numpy array."""
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    return width, height


def get_config_for_image(image) -> ScreenConfig:
    """Get a screen configuration scaled for the given image."""
    width, height = detect_resolution(image)
    default_config = get_default_config()
    if width == default_config.width and height == default_config.height:
        return default_config
    return default_config.scale_to_resolution(width, height)
