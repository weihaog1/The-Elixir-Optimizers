"""
GameState dataclass and related structures for Clash Royale detection.

Provides structured representation of game state from detection results.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class Tower:
    """Represents a tower in the game."""
    tower_type: str  # 'king' or 'princess'
    position: str  # 'left', 'right', or 'center' (for king)
    belonging: int  # 0=player, 1=enemy
    hp: Optional[int] = None
    max_hp: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    confidence: float = 1.0

    @property
    def is_destroyed(self) -> bool:
        """Check if tower is destroyed (HP <= 0 or missing)."""
        return self.hp is not None and self.hp <= 0

    @property
    def hp_percentage(self) -> Optional[float]:
        """Get HP as percentage of max if both values known."""
        if self.hp is not None and self.max_hp is not None and self.max_hp > 0:
            return (self.hp / self.max_hp) * 100
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Unit:
    """Represents a unit (troop, building, spell effect) in the game."""
    class_name: str
    belonging: int  # 0=player, 1=enemy, -1=unknown
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    hp: Optional[int] = None

    @property
    def center(self) -> Tuple[int, int]:
        """Get center position of unit."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def is_player(self) -> bool:
        return self.belonging == 0

    @property
    def is_enemy(self) -> bool:
        return self.belonging == 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["center"] = self.center
        return result


@dataclass
class Card:
    """Represents a card in the player's hand."""
    slot: int  # 0-3 for hand, -1 for next card
    class_name: Optional[str] = None
    elixir_cost: Optional[int] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GameState:
    """Complete game state extracted from a single frame."""
    # Timing
    timestamp: float = 0.0  # Frame timestamp in video
    time_remaining: Optional[int] = None  # Seconds remaining
    is_overtime: bool = False

    # Resources
    elixir: Optional[int] = None

    # Towers
    player_king_tower: Optional[Tower] = None
    player_left_princess: Optional[Tower] = None
    player_right_princess: Optional[Tower] = None
    enemy_king_tower: Optional[Tower] = None
    enemy_left_princess: Optional[Tower] = None
    enemy_right_princess: Optional[Tower] = None

    # Units on field
    units: List[Unit] = field(default_factory=list)

    # Cards
    cards: List[Card] = field(default_factory=list)
    next_card: Optional[Card] = None

    # Metadata
    frame_width: int = 0
    frame_height: int = 0
    detection_confidence: float = 0.0

    @property
    def player_towers(self) -> List[Tower]:
        """Get all player towers that exist."""
        towers = []
        if self.player_king_tower:
            towers.append(self.player_king_tower)
        if self.player_left_princess:
            towers.append(self.player_left_princess)
        if self.player_right_princess:
            towers.append(self.player_right_princess)
        return towers

    @property
    def enemy_towers(self) -> List[Tower]:
        """Get all enemy towers that exist."""
        towers = []
        if self.enemy_king_tower:
            towers.append(self.enemy_king_tower)
        if self.enemy_left_princess:
            towers.append(self.enemy_left_princess)
        if self.enemy_right_princess:
            towers.append(self.enemy_right_princess)
        return towers

    @property
    def all_towers(self) -> List[Tower]:
        """Get all towers."""
        return self.player_towers + self.enemy_towers

    @property
    def player_units(self) -> List[Unit]:
        """Get all player units."""
        return [u for u in self.units if u.belonging == 0]

    @property
    def enemy_units(self) -> List[Unit]:
        """Get all enemy units."""
        return [u for u in self.units if u.belonging == 1]

    @property
    def player_tower_count(self) -> int:
        """Count of player towers that are not destroyed."""
        return sum(1 for t in self.player_towers if not t.is_destroyed)

    @property
    def enemy_tower_count(self) -> int:
        """Count of enemy towers that are not destroyed."""
        return sum(1 for t in self.enemy_towers if not t.is_destroyed)

    @property
    def time_formatted(self) -> str:
        """Get time remaining as MM:SS string."""
        if self.time_remaining is None:
            return "??:??"
        minutes = self.time_remaining // 60
        seconds = self.time_remaining % 60
        return f"{minutes}:{seconds:02d}"

    def to_dict(self) -> Dict:
        """Convert game state to dictionary."""
        return {
            "timestamp": self.timestamp,
            "time_remaining": self.time_remaining,
            "time_formatted": self.time_formatted,
            "is_overtime": self.is_overtime,
            "elixir": self.elixir,
            "player_towers": [t.to_dict() for t in self.player_towers],
            "enemy_towers": [t.to_dict() for t in self.enemy_towers],
            "units": [u.to_dict() for u in self.units],
            "player_unit_count": len(self.player_units),
            "enemy_unit_count": len(self.enemy_units),
            "cards": [c.to_dict() for c in self.cards],
            "next_card": self.next_card.to_dict() if self.next_card else None,
            "frame_size": {"width": self.frame_width, "height": self.frame_height},
            "detection_confidence": self.detection_confidence,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert game state to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict) -> "GameState":
        """Create GameState from dictionary."""
        state = cls(
            timestamp=data.get("timestamp", 0.0),
            time_remaining=data.get("time_remaining"),
            is_overtime=data.get("is_overtime", False),
            elixir=data.get("elixir"),
            frame_width=data.get("frame_size", {}).get("width", 0),
            frame_height=data.get("frame_size", {}).get("height", 0),
            detection_confidence=data.get("detection_confidence", 0.0),
        )

        # Load towers
        for tower_data in data.get("player_towers", []):
            tower = Tower(**tower_data)
            if tower.tower_type == "king":
                state.player_king_tower = tower
            elif tower.position == "left":
                state.player_left_princess = tower
            elif tower.position == "right":
                state.player_right_princess = tower

        for tower_data in data.get("enemy_towers", []):
            tower = Tower(**tower_data)
            if tower.tower_type == "king":
                state.enemy_king_tower = tower
            elif tower.position == "left":
                state.enemy_left_princess = tower
            elif tower.position == "right":
                state.enemy_right_princess = tower

        # Load units
        state.units = [Unit(**u) for u in data.get("units", [])]

        # Load cards
        state.cards = [Card(**c) for c in data.get("cards", [])]
        if data.get("next_card"):
            state.next_card = Card(**data["next_card"])

        return state

    def summary(self) -> str:
        """Get human-readable summary of game state."""
        lines = [
            f"Time: {self.time_formatted}" + (" (OVERTIME)" if self.is_overtime else ""),
            f"Elixir: {self.elixir if self.elixir is not None else '?'}",
            f"Player Towers: {self.player_tower_count}/3",
            f"Enemy Towers: {self.enemy_tower_count}/3",
            f"Units on Field: {len(self.player_units)} player, {len(self.enemy_units)} enemy",
        ]

        if self.cards:
            card_names = [c.class_name or "?" for c in self.cards]
            lines.append(f"Cards: {', '.join(card_names)}")

        return "\n".join(lines)


# Default max HP values for towers
TOWER_MAX_HP = {
    "king": {
        # Level -> HP (approximate values)
        1: 2400, 9: 4008, 10: 4392, 11: 4824, 12: 5304, 13: 5832, 14: 6408
    },
    "princess": {
        1: 1400, 9: 2534, 10: 2786, 11: 3052, 12: 3346, 13: 3668, 14: 4032
    }
}


def estimate_tower_level(tower_type: str, hp: int) -> int:
    """Estimate tower level from HP value."""
    hp_table = TOWER_MAX_HP.get(tower_type, TOWER_MAX_HP["princess"])

    # Find closest match
    best_level = 9  # Default
    best_diff = float("inf")

    for level, max_hp in hp_table.items():
        # HP could be between 0 and max_hp
        if hp <= max_hp:
            diff = max_hp - hp
            if diff < best_diff:
                best_diff = diff
                best_level = level

    return best_level
