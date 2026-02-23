"""
StateEncoder -- converts GameState to SB3-compatible observation tensors.

This is the bridge between perception (StateBuilder) and learning (BC / PPO).
It produces a Dict observation and an action mask, both with fixed shapes that
don't change across frames.

Usage:
    from src.encoder import StateEncoder

    encoder = StateEncoder()

    # During BC data collection or Gym env.step():
    obs = encoder.encode(game_state)
    mask = encoder.action_mask(game_state)

    # obs is {"arena": np.float32(32,18,7), "vector": np.float32(23,)}
    # mask is np.bool_(2305,)

Observation space (gymnasium.spaces.Dict):
    "arena": Box(0, max, shape=(32, 18, 7), float32)
        Channel 0: ally ground unit count per cell
        Channel 1: ally flying unit count per cell
        Channel 2: enemy ground unit count per cell
        Channel 3: enemy flying unit count per cell
        Channel 4: ally tower HP fraction (0-1) per cell
        Channel 5: enemy tower HP fraction (0-1) per cell
        Channel 6: spell effect presence per cell
    "vector": Box(0, 1, shape=(23,), float32)
        [0]     elixir / 10
        [1]     time_remaining / 300
        [2]     is_overtime (0 or 1)
        [3-5]   player tower HP fracs (king, left princess, right princess)
        [6-8]   enemy tower HP fracs (king, left princess, right princess)
        [9]     player_tower_count / 3
        [10]    enemy_tower_count / 3
        [11-14] card present (binary, 4 slots)
        [15-18] card class index / num_deck_cards (normalized)
        [19-22] card elixir cost / 10 (normalized)

Action space: Discrete(2305)
    0..2303: card_id * 576 + row * 18 + col
    2304: no-op (wait)
"""

from typing import Optional

import gymnasium as gym
import numpy as np

from src.pipeline.game_state import GameState, Tower

from .coord_utils import pixel_to_cell
from .encoder_constants import (
    ACTION_SPACE_SIZE,
    CARD_ELIXIR_COST,
    CARD_IS_SPELL,
    CH_ALLY_FLYING,
    CH_ALLY_GROUND,
    CH_ALLY_TOWER_HP,
    CH_ENEMY_FLYING,
    CH_ENEMY_GROUND,
    CH_ENEMY_TOWER_HP,
    CH_SPELL,
    DECK_CARD_TO_IDX,
    DEFAULT_KING_MAX_HP,
    DEFAULT_PRINCESS_MAX_HP,
    GRID_CELLS,
    GRID_COLS,
    GRID_ROWS,
    MAX_ELIXIR,
    MAX_TIME_SECONDS,
    NOOP_ACTION,
    NUM_ARENA_CHANNELS,
    NUM_CARD_SLOTS,
    NUM_DECK_CARDS,
    NUM_VECTOR_FEATURES,
    PLAYER_HALF_ROW_START,
    UNIT_TYPE_MAP,
)


class StateEncoder:
    """Converts GameState to fixed-shape observation tensors for SB3.

    Thread-safe: encode() and action_mask() are stateless pure functions.
    The StateEncoder holds no mutable state between calls.
    """

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "arena": gym.spaces.Box(
                low=0.0,
                high=10.0,
                shape=(GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
                dtype=np.float32,
            ),
            "vector": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(NUM_VECTOR_FEATURES,),
                dtype=np.float32,
            ),
        })
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def encode(self, state: GameState) -> dict[str, np.ndarray]:
        """Convert a GameState into the observation dict.

        Args:
            state: GameState from StateBuilder.build_state().

        Returns:
            Dict with "arena" (32, 18, 7) and "vector" (23,) arrays.
        """
        arena = self._encode_arena(state)
        vector = self._encode_vector(state)
        return {"arena": arena, "vector": vector}

    def action_mask(self, state: GameState) -> np.ndarray:
        """Build the action validity mask.

        A True value means the action is valid (can be taken).
        A False value means the action is masked out (cannot be taken).

        Masking rules:
        - If a card slot is empty (no card or "empty-slot"), all 576 cells
          for that card are masked False.
        - If elixir < card cost, all 576 cells for that card are masked False.
        - No-op (action 2304) is always valid.

        Args:
            state: GameState from StateBuilder.build_state().

        Returns:
            Boolean array of shape (2305,).
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)

        # No-op is always valid
        mask[NOOP_ACTION] = True

        current_elixir = state.elixir if state.elixir is not None else 0

        for slot_idx in range(NUM_CARD_SLOTS):
            card = self._get_card_at_slot(state, slot_idx)
            if card is None:
                continue

            card_name = card.class_name
            if card_name is None or card_name == "empty-slot":
                continue

            cost = CARD_ELIXIR_COST.get(card_name, 0)
            if current_elixir < cost:
                continue

            # This card is playable -- unmask all its grid cells
            start = slot_idx * GRID_CELLS
            end = start + GRID_CELLS
            mask[start:end] = True

        return mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_arena(self, state: GameState) -> np.ndarray:
        """Build the (32, 18, 7) spatial arena grid.

        Iterates over all units and towers in the GameState, maps each
        to a grid cell, and increments the appropriate channel.
        """
        arena = np.zeros(
            (GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS), dtype=np.float32
        )
        fw = state.frame_width or 540
        fh = state.frame_height or 960

        # --- Units ---
        for unit in state.units:
            cx, cy = unit.center
            col, row = pixel_to_cell(cx, cy, fw, fh)

            unit_type = UNIT_TYPE_MAP.get(unit.class_name, "ground")
            is_ally = unit.belonging == 0

            if unit_type == "spell":
                arena[row, col, CH_SPELL] += 1.0
            elif unit_type == "flying":
                ch = CH_ALLY_FLYING if is_ally else CH_ENEMY_FLYING
                arena[row, col, ch] += 1.0
            elif unit_type == "tower":
                # Towers are handled separately below
                pass
            elif unit_type == "other":
                # UI elements (bar, clock, emote) -- skip
                pass
            else:
                # ground (default)
                ch = CH_ALLY_GROUND if is_ally else CH_ENEMY_GROUND
                arena[row, col, ch] += 1.0

        # --- Towers ---
        tower_slots = [
            (state.player_king_tower, True),
            (state.player_left_princess, True),
            (state.player_right_princess, True),
            (state.enemy_king_tower, False),
            (state.enemy_left_princess, False),
            (state.enemy_right_princess, False),
        ]
        for tower, is_ally in tower_slots:
            if tower is None or tower.bbox is None:
                continue
            hp_frac = self._get_tower_hp_frac(tower)
            cx = (tower.bbox[0] + tower.bbox[2]) // 2
            cy = (tower.bbox[1] + tower.bbox[3]) // 2
            col, row = pixel_to_cell(cx, cy, fw, fh)
            ch = CH_ALLY_TOWER_HP if is_ally else CH_ENEMY_TOWER_HP
            arena[row, col, ch] = hp_frac

        return arena

    def _encode_vector(self, state: GameState) -> np.ndarray:
        """Build the (23,) scalar feature vector.

        All values are normalized to [0, 1]. Unknown/None values default
        to 0.0 which is a safe neutral value for neural networks.
        """
        vec = np.zeros(NUM_VECTOR_FEATURES, dtype=np.float32)

        # [0] Elixir
        if state.elixir is not None:
            vec[0] = state.elixir / MAX_ELIXIR

        # [1] Time remaining
        if state.time_remaining is not None:
            vec[1] = min(state.time_remaining / MAX_TIME_SECONDS, 1.0)

        # [2] Overtime flag
        vec[2] = 1.0 if state.is_overtime else 0.0

        # [3-5] Player tower HP fractions
        vec[3] = self._get_tower_hp_frac(state.player_king_tower)
        vec[4] = self._get_tower_hp_frac(state.player_left_princess)
        vec[5] = self._get_tower_hp_frac(state.player_right_princess)

        # [6-8] Enemy tower HP fractions
        vec[6] = self._get_tower_hp_frac(state.enemy_king_tower)
        vec[7] = self._get_tower_hp_frac(state.enemy_left_princess)
        vec[8] = self._get_tower_hp_frac(state.enemy_right_princess)

        # [9-10] Tower counts
        vec[9] = state.player_tower_count / 3.0
        vec[10] = state.enemy_tower_count / 3.0

        # [11-22] Card hand encoding (4 slots x 3 features each)
        for slot_idx in range(NUM_CARD_SLOTS):
            card = self._get_card_at_slot(state, slot_idx)
            base = 11 + slot_idx

            if card is not None and card.class_name and card.class_name != "empty-slot":
                # Card present
                vec[base] = 1.0
                # Class index (normalized)
                card_idx = DECK_CARD_TO_IDX.get(card.class_name, 0)
                vec[base + NUM_CARD_SLOTS] = card_idx / max(NUM_DECK_CARDS - 1, 1)
                # Elixir cost (normalized)
                cost = CARD_ELIXIR_COST.get(card.class_name, 0)
                vec[base + 2 * NUM_CARD_SLOTS] = cost / MAX_ELIXIR

        return vec

    def _get_card_at_slot(self, state: GameState, slot_idx: int):
        """Get the Card at a given slot index, or None."""
        for card in state.cards:
            if card.slot == slot_idx:
                return card
        return None

    def _get_tower_hp_frac(self, tower: Optional[Tower]) -> float:
        """Return tower HP as a 0-1 fraction.

        Returns 0.0 if tower is None, destroyed, or HP is unknown.
        If HP is known but max_hp is not, uses default max HP values.
        """
        if tower is None:
            return 0.0

        hp = tower.hp
        if hp is None or hp <= 0:
            return 0.0

        max_hp = tower.max_hp
        if max_hp is None or max_hp <= 0:
            if tower.tower_type == "king":
                max_hp = DEFAULT_KING_MAX_HP
            else:
                max_hp = DEFAULT_PRINCESS_MAX_HP

        return min(hp / max_hp, 1.0)
