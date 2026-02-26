"""Live observation tensor visualization for Clash Royale PPO.

Renders the 32x18x6 arena grid as 6 heatmaps and the 23-dim vector as a
labeled bar chart. Supports live updating via matplotlib interactive mode
and optional frame saving for video assembly.

Usage:
    viz = ObsVisualizer(save_dir="vis_frames/")
    viz.update(obs_dict, step=0)
    viz.close()

Video from saved frames:
    ffmpeg -framerate 2 -i vis_frames/step_%04d.png -c:v libx264 -pix_fmt yuv420p obs_viz.mp4
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Arena channel indices and display config
_CHANNEL_CONFIG = [
    {"name": "Unit Class ID", "cmap": "tab20", "vmin": 0, "vmax": 1},
    {"name": "Belonging", "cmap": "RdBu", "vmin": -1, "vmax": 1},
    {"name": "Arena Mask", "cmap": "Greys", "vmin": 0, "vmax": 1},
    {"name": "Ally Tower HP", "cmap": "Greens", "vmin": 0, "vmax": 1},
    {"name": "Enemy Tower HP", "cmap": "Reds", "vmin": 0, "vmax": 1},
    {"name": "Spell Count", "cmap": "Purples", "vmin": 0, "vmax": 3},
]

# Vector feature labels (23 total)
_VECTOR_LABELS = [
    "Elixir",          # 0
    "Time Left",       # 1
    "Overtime",        # 2
    "P.King HP",       # 3
    "P.Left HP",       # 4
    "P.Right HP",      # 5
    "E.King HP",       # 6
    "E.Left HP",       # 7
    "E.Right HP",      # 8
    "P.Towers",        # 9
    "E.Towers",        # 10
    "Card1 Present",   # 11
    "Card2 Present",   # 12
    "Card3 Present",   # 13
    "Card4 Present",   # 14
    "Card1 Class",     # 15
    "Card2 Class",     # 16
    "Card3 Class",     # 17
    "Card4 Class",     # 18
    "Card1 Cost",      # 19
    "Card2 Cost",      # 20
    "Card3 Cost",      # 21
    "Card4 Cost",      # 22
]

# Color groups for vector bar chart
_VECTOR_COLORS = (
    ["#3b82f6"] * 3    # Game state (0-2): blue
    + ["#22c55e"] * 3   # Player towers (3-5): green
    + ["#ef4444"] * 3   # Enemy towers (6-8): red
    + ["#22c55e"]       # P.Towers (9): green
    + ["#ef4444"]       # E.Towers (10): red
    + ["#f97316"] * 12  # Cards (11-22): orange
)


class ObsVisualizer:
    """Live matplotlib visualization of observation tensors.

    Shows 6 arena channel heatmaps and a 23-feature vector bar chart,
    updated in real-time during gameplay.
    """

    def __init__(
        self,
        save_dir: str = "",
        save_every: int = 1,
    ) -> None:
        """Initialize the visualization figure.

        Args:
            save_dir: Directory to save PNG frames. Empty = no saving.
            save_every: Save every N-th frame (1 = every frame).
        """
        self._save_dir = save_dir
        self._save_every = max(1, save_every)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Create figure with interactive mode
        plt.ion()
        self._fig = plt.figure(figsize=(16, 14))
        self._fig.suptitle("Observation Tensor Visualization", fontsize=14)

        # Layout: 3 cols x 2 rows for arena + 1 full-width row for vector
        gs = gridspec.GridSpec(
            3, 3,
            height_ratios=[1, 1, 0.8],
            hspace=0.35,
            wspace=0.3,
        )

        # Create arena heatmap axes (2 rows x 3 cols)
        self._arena_axes = []
        self._arena_imgs = []
        self._arena_cbars = []

        for i, cfg in enumerate(_CHANNEL_CONFIG):
            row, col = divmod(i, 3)
            ax = self._fig.add_subplot(gs[row, col])
            ax.set_title(cfg["name"], fontsize=10)
            ax.set_xlabel("Col")
            ax.set_ylabel("Row")

            # Initialize with zeros
            data = np.zeros((32, 18), dtype=np.float32)
            img = ax.imshow(
                data,
                cmap=cfg["cmap"],
                vmin=cfg["vmin"],
                vmax=cfg["vmax"],
                aspect="auto",
                origin="upper",
            )
            cbar = self._fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

            self._arena_axes.append(ax)
            self._arena_imgs.append(img)
            self._arena_cbars.append(cbar)

        # Create vector bar chart (bottom row, spanning all 3 cols)
        self._vec_ax = self._fig.add_subplot(gs[2, :])
        self._vec_ax.set_title("Vector Features (23)", fontsize=10)
        self._vec_ax.set_xlim(0, 1.05)

        y_pos = np.arange(len(_VECTOR_LABELS))
        self._vec_bars = self._vec_ax.barh(
            y_pos,
            np.zeros(len(_VECTOR_LABELS)),
            color=_VECTOR_COLORS,
            height=0.7,
        )
        self._vec_ax.set_yticks(y_pos)
        self._vec_ax.set_yticklabels(_VECTOR_LABELS, fontsize=7)
        self._vec_ax.invert_yaxis()

        # Add value labels
        self._vec_texts = []
        for i in range(len(_VECTOR_LABELS)):
            txt = self._vec_ax.text(0.01, i, "", va="center", fontsize=6)
            self._vec_texts.append(txt)

        self._step_text = self._fig.text(
            0.5, 0.01, "Step: 0", ha="center", fontsize=10,
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        self._fig.canvas.draw()
        plt.pause(0.01)

    def update(self, obs: dict[str, np.ndarray], step: int) -> None:
        """Update all panels with new observation data.

        Args:
            obs: Dict with "arena" (32,18,6) and "vector" (23,) numpy arrays.
            step: Current step number (for title and filename).
        """
        arena = obs["arena"]  # (32, 18, 6)
        vector = obs["vector"]  # (23,)

        # Squeeze batch dim if present
        if arena.ndim == 4:
            arena = arena[0]
        if vector.ndim == 2:
            vector = vector[0]

        # Update arena heatmaps
        for i in range(6):
            self._arena_imgs[i].set_data(arena[:, :, i])

        # Update vector bars
        for i, bar in enumerate(self._vec_bars):
            val = float(vector[i])
            bar.set_width(max(val, 0.001))  # Minimum width for visibility
            self._vec_texts[i].set_text(f" {val:.2f}")
            self._vec_texts[i].set_x(max(val, 0.001) + 0.01)

        # Update step counter
        self._step_text.set_text(f"Step: {step}")

        # Redraw
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.01)

        # Save frame if configured
        if self._save_dir and step % self._save_every == 0:
            path = os.path.join(self._save_dir, f"step_{step:04d}.png")
            self._fig.savefig(path, dpi=100, bbox_inches="tight")

    def close(self) -> None:
        """Close the figure and disable interactive mode."""
        plt.ioff()
        plt.close(self._fig)
