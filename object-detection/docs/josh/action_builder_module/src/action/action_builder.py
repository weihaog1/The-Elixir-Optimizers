"""
ActionBuilder: converts click logger events into Discrete(2305) action indices.

The click logger captures normalized (0-1) window coordinates. A card
placement is a two-click sequence: first click on a card slot (bottom of
screen), then click on the arena (middle of screen). ActionBuilder pairs
these clicks and encodes them as action indices for behavior cloning.
"""

from dataclasses import dataclass

from src.encoder.coord_utils import norm_to_cell, placement_to_action
from src.encoder.encoder_constants import NOOP_ACTION

from .action_constants import CARD_SLOT_REGIONS, ARENA_Y_MIN, ARENA_Y_MAX


@dataclass
class ClickEvent:
    """A single mouse click captured by the click logger."""

    timestamp: float   # Wall-clock time (seconds)
    x_norm: float      # Normalized window X (0=left, 1=right)
    y_norm: float      # Normalized window Y (0=top, 1=bottom)


@dataclass
class ActionEvent:
    """A paired card-slot + arena click encoded as a discrete action."""

    timestamp: float      # Time of the arena click
    action_idx: int       # Discrete(2305) action index
    card_id: int          # Card slot 0-3
    col: int              # Grid column 0-17
    row: int              # Grid row 0-31
    x_norm: float         # Arena click x_norm
    y_norm: float         # Arena click y_norm


class ActionBuilder:
    """Pairs click logger events into ActionEvents with discrete action indices.

    State machine for click pairing:
    - Start in 'idle' state
    - Card click -> transition to 'card_selected', remember card_id
    - Arena click while 'card_selected' -> emit ActionEvent, back to 'idle'
    - Card click while 'card_selected' -> replace card_id (changed mind)
    - Other click -> reset to 'idle'
    """

    def classify_click(self, x_norm: float, y_norm: float) -> str:
        """Classify a click by its screen position.

        Args:
            x_norm: Normalized x coordinate (0=left, 1=right).
            y_norm: Normalized y coordinate (0=top, 1=bottom).

        Returns:
            'card_0'..'card_3' if click is in a card slot,
            'arena' if click is in the arena region,
            'other' otherwise.
        """
        for i, (x_min, y_min, x_max, y_max) in enumerate(CARD_SLOT_REGIONS):
            if x_min <= x_norm <= x_max and y_min <= y_norm <= y_max:
                return f"card_{i}"

        if ARENA_Y_MIN <= y_norm <= ARENA_Y_MAX:
            return "arena"

        return "other"

    def clicks_to_actions(self, clicks: list[ClickEvent]) -> list[ActionEvent]:
        """Convert a sequence of clicks into paired ActionEvents.

        Processes clicks in order using a simple state machine:
        idle -> card click -> card_selected -> arena click -> emit + idle

        Args:
            clicks: Chronologically ordered click events.

        Returns:
            List of ActionEvents for successfully paired card+arena clicks.
        """
        actions: list[ActionEvent] = []
        state = "idle"
        pending_card_id: int = -1

        for click in clicks:
            label = self.classify_click(click.x_norm, click.y_norm)

            if label.startswith("card_"):
                card_id = int(label.split("_")[1])
                state = "card_selected"
                pending_card_id = card_id

            elif label == "arena" and state == "card_selected":
                col, row = norm_to_cell(click.x_norm, click.y_norm)
                action_idx = placement_to_action(pending_card_id, col, row)
                actions.append(ActionEvent(
                    timestamp=click.timestamp,
                    action_idx=action_idx,
                    card_id=pending_card_id,
                    col=col,
                    row=row,
                    x_norm=click.x_norm,
                    y_norm=click.y_norm,
                ))
                state = "idle"
                pending_card_id = -1

            else:
                # Arena click without card selected, or click in 'other' region
                state = "idle"
                pending_card_id = -1

        return actions

    def build_action_timeline(
        self,
        clicks: list[ClickEvent],
        frame_timestamps: list[float],
    ) -> list[int]:
        """Assign an action index to each frame timestamp.

        For each frame, checks if any ActionEvent occurred between this frame
        and the previous frame. Frames with no action get NOOP_ACTION (2304).
        No downsampling or deduplication.

        Args:
            clicks: Chronologically ordered click events.
            frame_timestamps: Monotonically increasing frame capture times.

        Returns:
            List of action indices, one per frame timestamp.
        """
        if not frame_timestamps:
            return []

        actions = self.clicks_to_actions(clicks)
        timeline: list[int] = []
        action_ptr = 0

        for i, ft in enumerate(frame_timestamps):
            # Find action window: (prev_frame_time, current_frame_time]
            if i == 0:
                window_start = float("-inf")
            else:
                window_start = frame_timestamps[i - 1]

            frame_action = NOOP_ACTION

            # Advance pointer past actions before our window
            while action_ptr < len(actions) and actions[action_ptr].timestamp <= window_start:
                action_ptr += 1

            # Check if any action falls in (window_start, ft]
            check_ptr = action_ptr
            while check_ptr < len(actions) and actions[check_ptr].timestamp <= ft:
                frame_action = actions[check_ptr].action_idx
                check_ptr += 1

            timeline.append(frame_action)

        return timeline
