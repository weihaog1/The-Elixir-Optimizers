"""OpenCV-based arena grid visualizer with side-by-side game frame.

Renders a composite image: game screenshot (left) + color-coded 32x18 arena
grid (center) + channel values and game state readout (right). Supports
real-time display via cv2.imshow and direct MP4 recording via cv2.VideoWriter.

Usage:
    viz = CVVisualizer(CVVisConfig(record=True, output_path="out.mp4"))
    viz.update(game_frame, obs_dict, step=0, info={})
    viz.close()
"""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.encoder.encoder_constants import (
    CH_ALLY_TOWER_HP,
    CH_ARENA_MASK,
    CH_BELONGING,
    CH_CLASS_ID,
    CH_ENEMY_TOWER_HP,
    CH_SPELL,
    CLASS_NAME_TO_ID,
    DECK_CARDS,
    GRID_COLS,
    GRID_ROWS,
    MAX_ELIXIR,
    MAX_TIME_SECONDS,
    NUM_CLASSES,
    PLAYER_HALF_ROW_START,
)

# ── Colors (BGR for OpenCV) ──────────────────────────────────────────────────

_CLR_ALLY_UNIT = (50, 200, 50)       # Green
_CLR_ENEMY_UNIT = (200, 50, 200)     # Magenta
_CLR_ALLY_TOWER = (200, 150, 50)     # Blue-ish
_CLR_ENEMY_TOWER = (50, 50, 200)     # Red
_CLR_SPELL = (50, 200, 200)          # Yellow
_CLR_EMPTY = (30, 30, 30)            # Dark gray
_CLR_RIVER = (100, 80, 40)           # Blue tint
_CLR_GRID_LINE = (60, 60, 60)        # Subtle grid
_CLR_DEPLOY_LINE = (60, 60, 180)     # Reddish deploy boundary
_CLR_TEXT = (220, 220, 220)          # Light gray
_CLR_HEADER = (180, 180, 100)        # Cyan-ish header
_CLR_BG = (20, 20, 20)              # Panel background
_CLR_ELIXIR_BAR = (200, 50, 200)     # Purple elixir bar
_CLR_ELIXIR_BG = (60, 30, 60)       # Dark purple background

# ── Reverse class lookup ─────────────────────────────────────────────────────

_ID_TO_CLASS_NAME: dict[int, str] = {v: k for k, v in CLASS_NAME_TO_ID.items()}

# ── Name abbreviation ────────────────────────────────────────────────────────

_ABBREV_OVERRIDES: dict[str, str] = {
    "knight": "kni",
    "skeleton": "skl",
    "skeletons": "skl",
    "royal-hog": "r.h",
    "royal-hogs": "r.h",
    "royal-recruit": "r.r",
    "royal-recruits": "r.r",
    "arrows": "arr",
    "barbarian-barrel": "b.b",
    "flying-machine": "f.m",
    "goblin-cage": "g.c",
    "zappies": "zap",
    "zappy": "zap",
    "ice-golem": "i.g",
    "ice-spirit": "i.s",
    "electro-spirit": "e.s",
    "eletro-spirit": "e.s",
    "fire-spirit": "f.s",
    "fireball": "fbl",
    "the-log": "log",
    "mini-pekka": "m.p",
    "mega-minion": "m.m",
    "minion-horde": "m.h",
    "musketeer": "mus",
    "valkyrie": "val",
    "wizard": "wiz",
    "witch": "wit",
    "giant": "gnt",
    "golem": "glm",
    "golemite": "glt",
    "hog-rider": "hog",
    "baby-dragon": "b.d",
    "inferno-dragon": "i.d",
    "inferno-tower": "i.t",
    "cannon": "can",
    "tesla": "tes",
    "bomb-tower": "b.t",
    "mortar": "mor",
    "x-bow": "xbw",
    "princess": "pri",
    "bandit": "bnd",
    "lumberjack": "l.j",
    "balloon": "bln",
    "goblin": "gob",
    "goblin-gang": "g.g",
    "goblin-barrel": "g.b",
    "dart-goblin": "d.g",
    "spear-goblin": "s.g",
    "barbarian": "bar",
    "elite-barbarian": "e.b",
    "battle-ram": "b.r",
    "three-musketeers": "3.m",
    "rage": "rag",
    "freeze": "frz",
    "poison": "psn",
    "tornado": "tor",
    "lightning": "ltn",
    "rocket": "rkt",
    "graveyard": "gvy",
    "mirror": "mir",
    "clone": "cln",
    "heal-spirit": "h.s",
    "elixir-golem": "e.g",
    "battle-healer": "b.h",
    "sparky": "spk",
    "pekka": "pka",
    "mega-knight": "m.k",
    "ram-rider": "r.r",
    "lava-hound": "l.h",
    "archer": "arc",
    "archers": "arc",
}


def _abbreviate(name: str) -> str:
    """Abbreviate a class name to 3 chars for grid cell display."""
    if name in _ABBREV_OVERRIDES:
        return _ABBREV_OVERRIDES[name]
    return name[:3]


def _class_id_to_name(normalized_id: float) -> str:
    """Recover class name from the normalized class_id channel value."""
    class_idx = int(round(normalized_id * NUM_CLASSES))
    return _ID_TO_CLASS_NAME.get(class_idx, f"?{class_idx}")


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class CVVisConfig:
    """Configuration for the OpenCV arena grid visualizer."""

    cell_size: int = 20
    grid_rows: int = GRID_ROWS
    grid_cols: int = GRID_COLS

    game_frame_width: int = 540
    game_frame_height: int = 960
    info_panel_width: int = 300

    record: bool = False
    output_path: str = "vis_output.mp4"
    fps: float = 2.0
    codec: str = "mp4v"

    show_window: bool = True
    window_name: str = "Clash Royale - Obs Visualizer"


# ── Visualizer ───────────────────────────────────────────────────────────────


class CVVisualizer:
    """OpenCV-based real-time observation visualizer.

    Renders a side-by-side composite: game frame (left) + arena grid
    (center) + info panel (right). Supports live display and MP4 recording.
    """

    def __init__(self, config: Optional[CVVisConfig] = None) -> None:
        self._cfg = config or CVVisConfig()

        # Layout dimensions
        self._grid_w = self._cfg.grid_cols * self._cfg.cell_size  # 360
        self._grid_h = self._cfg.grid_rows * self._cfg.cell_size  # 640
        self._target_h = self._cfg.game_frame_height

        # Video recording (lazy init)
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_initialized = False

        # Timing / performance tracking
        self._frame_count = 0
        self._start_time = time.time()
        self._last_update_time = time.time()
        self._render_ms = 0.0

    # ── Public API ───────────────────────────────────────────────────────

    def update(
        self,
        game_frame: np.ndarray,
        obs: dict[str, np.ndarray],
        step: int,
        info: Optional[dict] = None,
        detections: Optional[list] = None,
    ) -> None:
        """Render one composite frame, display it, and optionally record."""
        t0 = time.time()
        info = info or {}

        arena = obs["arena"]
        vector = obs["vector"]
        if arena.ndim == 4:
            arena = arena[0]
        if vector.ndim == 2:
            vector = vector[0]

        # 1. Render panels
        left = self._render_game_frame(game_frame, detections)
        grid = self._render_grid(arena)
        right = self._render_info_panel(arena, vector, step, info)

        # 2. Compose: stack grid + right panel, then hstack with left
        grid_plus_info = self._compose_right_panels(grid, right)
        composite = self._compose_final(left, grid_plus_info)

        # 3. Display
        if self._cfg.show_window:
            cv2.imshow(self._cfg.window_name, composite)
            cv2.waitKey(1)

        # 4. Record
        if self._cfg.record:
            if not self._writer_initialized:
                self._init_writer(composite.shape[0], composite.shape[1])
            if self._writer is not None:
                self._writer.write(composite)

        self._frame_count += 1
        self._render_ms = (time.time() - t0) * 1000
        self._last_update_time = time.time()

    def close(self) -> None:
        """Release VideoWriter and destroy display window."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._cfg.show_window and self._frame_count > 0:
            try:
                cv2.destroyWindow(self._cfg.window_name)
            except cv2.error:
                pass

    # ── Left panel: game frame ───────────────────────────────────────────

    def _render_game_frame(
        self,
        frame: np.ndarray,
        detections: Optional[list] = None,
    ) -> np.ndarray:
        """Resize game frame to target height and optionally draw detections."""
        fh, fw = frame.shape[:2]
        scale = self._target_h / fh
        new_w = int(fw * scale)
        resized = cv2.resize(frame, (new_w, self._target_h))

        if detections:
            for det in detections:
                bbox = det.get("bbox")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = [int(v * scale) for v in bbox]
                cls_name = det.get("class_name", "?")
                belonging = det.get("belonging", -1)
                color = _CLR_ALLY_UNIT if belonging == 0 else _CLR_ENEMY_UNIT
                cv2.rectangle(resized, (x1, y1), (x2, y2), color, 1)
                cv2.putText(
                    resized, cls_name[:10], (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
                )

        return resized

    # ── Center panel: arena grid ─────────────────────────────────────────

    def _render_grid(self, arena: np.ndarray) -> np.ndarray:
        """Draw the 32x18 color-coded arena grid."""
        cs = self._cfg.cell_size
        img = np.full((self._grid_h, self._grid_w, 3), _CLR_BG, dtype=np.uint8)

        # Draw river shading (rows 15-16)
        for r in (15, 16):
            y1 = r * cs
            y2 = y1 + cs
            cv2.rectangle(img, (0, y1), (self._grid_w, y2), _CLR_RIVER, cv2.FILLED)

        # Draw each cell
        for row in range(self._cfg.grid_rows):
            for col in range(self._cfg.grid_cols):
                x1 = col * cs
                y1 = row * cs
                x2 = x1 + cs
                y2 = y1 + cs

                ally_hp = arena[row, col, CH_ALLY_TOWER_HP]
                enemy_hp = arena[row, col, CH_ENEMY_TOWER_HP]
                mask = arena[row, col, CH_ARENA_MASK]
                belonging = arena[row, col, CH_BELONGING]
                class_id = arena[row, col, CH_CLASS_ID]
                spell = arena[row, col, CH_SPELL]

                color = None
                label = ""

                if ally_hp > 0:
                    color = _CLR_ALLY_TOWER
                    label = f"{ally_hp:.0%}"[:3]
                elif enemy_hp > 0:
                    color = _CLR_ENEMY_TOWER
                    label = f"{enemy_hp:.0%}"[:3]
                elif spell > 0:
                    color = _CLR_SPELL
                    label = f"S{int(spell)}"
                elif mask > 0:
                    if belonging < 0:
                        color = _CLR_ALLY_UNIT
                    else:
                        color = _CLR_ENEMY_UNIT
                    name = _class_id_to_name(class_id)
                    label = _abbreviate(name)

                if color is not None:
                    cv2.rectangle(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1),
                                  color, cv2.FILLED)

                # Grid lines
                cv2.rectangle(img, (x1, y1), (x2, y2), _CLR_GRID_LINE, 1)

                # Cell text
                if label:
                    fs = 0.28
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1,
                    )
                    tx = x1 + (cs - tw) // 2
                    ty = y1 + (cs + th) // 2
                    cv2.putText(
                        img, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )

        # Deploy boundary line at row 17
        deploy_y = PLAYER_HALF_ROW_START * cs
        cv2.line(img, (0, deploy_y), (self._grid_w, deploy_y),
                 _CLR_DEPLOY_LINE, 1)

        return img

    # ── Right panel: info readout ────────────────────────────────────────

    def _render_info_panel(
        self,
        arena: np.ndarray,
        vector: np.ndarray,
        step: int,
        info: dict,
    ) -> np.ndarray:
        """Draw channel values, game state, and performance stats."""
        w = self._cfg.info_panel_width
        h = self._target_h
        img = np.full((h, w, 3), _CLR_BG, dtype=np.uint8)
        y = 15  # Current y cursor

        def _text(text: str, x: int, color=_CLR_TEXT, scale: float = 0.38):
            nonlocal y
            cv2.putText(img, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, 1, cv2.LINE_AA)

        def _header(text: str):
            nonlocal y
            y += 8
            _text(text, 5, _CLR_HEADER, 0.42)
            y += 18

        def _line(text: str, color=_CLR_TEXT):
            nonlocal y
            _text(text, 10, color)
            y += 14

        # ── Section 1: Occupied cells readout ────────────────────────────
        _header("OCCUPIED CELLS")
        _line("(c,r)  name  bel  cls  spl")
        y += 2
        occupied_count = 0
        for row in range(self._cfg.grid_rows):
            for col in range(self._cfg.grid_cols):
                mask = arena[row, col, CH_ARENA_MASK]
                ally_hp = arena[row, col, CH_ALLY_TOWER_HP]
                enemy_hp = arena[row, col, CH_ENEMY_TOWER_HP]
                spell = arena[row, col, CH_SPELL]

                if mask <= 0 and ally_hp <= 0 and enemy_hp <= 0 and spell <= 0:
                    continue

                class_id = arena[row, col, CH_CLASS_ID]
                belonging = arena[row, col, CH_BELONGING]

                if ally_hp > 0:
                    name = "tower"
                    extra = f" hp={ally_hp:.0%}"
                elif enemy_hp > 0:
                    name = "tower"
                    extra = f" hp={enemy_hp:.0%}"
                elif spell > 0:
                    name = "spell"
                    extra = f" x{int(spell)}"
                else:
                    name = _abbreviate(_class_id_to_name(class_id))
                    extra = ""

                bel_str = f"{belonging:+.0f}"
                line_text = (
                    f"({col:2d},{row:2d}) {name:<4s} {bel_str:>3s} "
                    f"{class_id:.2f}{extra}"
                )
                _line(line_text)
                occupied_count += 1

                # Cap output to avoid overflow
                if occupied_count >= 25:
                    _line("... (truncated)")
                    break
            if occupied_count >= 25:
                break

        # ── Section 2: Game state ────────────────────────────────────────
        y += 5
        _header("GAME STATE")

        elixir = vector[0] * MAX_ELIXIR
        time_left = vector[1] * MAX_TIME_SECONDS
        overtime = vector[2] > 0.5

        _line(f"Elixir: {elixir:.0f} / {MAX_ELIXIR}")

        # Elixir bar
        bar_x = 10
        bar_w = w - 20
        bar_h = 8
        cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + bar_h),
                       _CLR_ELIXIR_BG, cv2.FILLED)
        fill_w = int(bar_w * (elixir / MAX_ELIXIR))
        if fill_w > 0:
            cv2.rectangle(img, (bar_x, y), (bar_x + fill_w, y + bar_h),
                           _CLR_ELIXIR_BAR, cv2.FILLED)
        y += bar_h + 8

        _line(f"Time:   {time_left:.0f}s / {MAX_TIME_SECONDS}s")
        _line(f"OT:     {'Yes' if overtime else 'No'}")

        # ── Section 3: Towers ────────────────────────────────────────────
        y += 5
        _header("TOWERS")

        p_king = vector[3]
        p_left = vector[4]
        p_right = vector[5]
        e_king = vector[6]
        e_left = vector[7]
        e_right = vector[8]
        p_count = vector[9] * 3
        e_count = vector[10] * 3

        _line(f"P: K={p_king:.0%}  L={p_left:.0%}  R={p_right:.0%}",
              _CLR_ALLY_TOWER)
        _line(f"E: K={e_king:.0%}  L={e_left:.0%}  R={e_right:.0%}",
              _CLR_ENEMY_TOWER)
        _line(f"Count: P={p_count:.0f}/3  E={e_count:.0f}/3")

        # ── Section 4: Card hand ─────────────────────────────────────────
        y += 5
        _header("HAND")

        for slot in range(4):
            present = vector[11 + slot]
            class_idx_norm = vector[15 + slot]
            cost_norm = vector[19 + slot]

            if present > 0.5:
                card_idx = int(round(class_idx_norm * max(len(DECK_CARDS) - 1, 1)))
                card_name = DECK_CARDS[card_idx] if card_idx < len(DECK_CARDS) else "?"
                cost = cost_norm * MAX_ELIXIR
                _line(f"[{slot + 1}] {card_name:<18s} {cost:.0f}e")
            else:
                _line(f"[{slot + 1}] (empty)")

        # ── Section 5: Performance ───────────────────────────────────────
        y = h - 60  # Anchor to bottom
        _header("PERFORMANCE")

        elapsed = time.time() - self._start_time
        avg_fps = self._frame_count / max(elapsed, 0.001)
        reward = info.get("episode_reward", 0.0)
        phase = info.get("phase", "?")

        _line(f"Step: {step}  |  Reward: {reward:+.2f}")
        _line(f"Phase: {phase}  |  FPS: {avg_fps:.1f}")
        rec_status = "ON" if self._cfg.record else "OFF"
        _line(f"Render: {self._render_ms:.1f}ms  |  Rec: {rec_status}")

        return img

    # ── Composition helpers ──────────────────────────────────────────────

    def _compose_right_panels(
        self, grid: np.ndarray, info: np.ndarray,
    ) -> np.ndarray:
        """Stack grid and info panel horizontally, then pad to target height."""
        # grid is (grid_h, grid_w, 3), info is (target_h, info_w, 3)
        # Pad grid to target_h
        gh = grid.shape[0]
        if gh < self._target_h:
            pad = np.full(
                (self._target_h - gh, grid.shape[1], 3), _CLR_BG, dtype=np.uint8,
            )
            grid_padded = np.vstack([grid, pad])
        else:
            grid_padded = grid[:self._target_h]

        return np.hstack([grid_padded, info])

    def _compose_final(
        self, left: np.ndarray, right: np.ndarray,
    ) -> np.ndarray:
        """Horizontal concat of left (game frame) and right (grid+info)."""
        lh, rh = left.shape[0], right.shape[0]
        target = max(lh, rh)

        if lh < target:
            pad = np.full((target - lh, left.shape[1], 3), _CLR_BG, dtype=np.uint8)
            left = np.vstack([left, pad])
        if rh < target:
            pad = np.full((target - rh, right.shape[1], 3), _CLR_BG, dtype=np.uint8)
            right = np.vstack([right, pad])

        return np.hstack([left, right])

    # ── Video writer ─────────────────────────────────────────────────────

    def _init_writer(self, h: int, w: int) -> None:
        """Initialize cv2.VideoWriter with actual composite dimensions."""
        fourcc = cv2.VideoWriter_fourcc(*self._cfg.codec)
        self._writer = cv2.VideoWriter(
            self._cfg.output_path, fourcc, self._cfg.fps, (w, h),
        )
        self._writer_initialized = True
        if self._writer.isOpened():
            print(f"[CVVis] Recording to {self._cfg.output_path} ({w}x{h} @ {self._cfg.fps} fps)")
        else:
            print(f"[CVVis] WARNING: Failed to open VideoWriter for {self._cfg.output_path}")
            self._writer.release()
            self._writer = None
