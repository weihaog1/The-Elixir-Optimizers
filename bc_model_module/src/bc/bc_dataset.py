"""BCDataset - PyTorch Dataset for behavior cloning training data.

Loads .npz files produced by DatasetBuilder (dataset_builder_module) and
provides (arena, vector, action, mask) samples for training. Includes
file-level train/val splitting to prevent data leakage between frames
of the same game, and class weight computation for weighted cross-entropy.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.encoder.encoder_constants import (
    ACTION_SPACE_SIZE,
    GRID_CELLS,
    GRID_COLS,
    NOOP_ACTION,
)


class BCDataset(Dataset):
    """PyTorch Dataset for BC training data from .npz files.

    Loads one or more .npz files and concatenates them into a single
    dataset. Each .npz file is expected to contain:
        - obs_arena: (N, 32, 18, 6) float32
        - obs_vector: (N, 23) float32
        - actions: (N,) int64
        - masks: (N, 2305) bool

    When ``augment=True``, the dataset doubles its effective size by
    providing horizontally-flipped copies for indices >= N. The arena
    columns are reversed, and action/mask indices are remapped so that
    ``col`` becomes ``GRID_COLS - 1 - col``.

    Args:
        npz_paths: List of Path objects pointing to .npz files.
        augment: If True, enable horizontal flip augmentation (2x data).
    """

    def __init__(self, npz_paths: list[Path], augment: bool = False) -> None:
        super().__init__()
        arenas = []
        vectors = []
        actions = []
        masks = []

        for path in npz_paths:
            data = np.load(str(path))
            arenas.append(data["obs_arena"])
            vectors.append(data["obs_vector"])
            actions.append(data["actions"])
            masks.append(data["masks"])

        self.arenas = np.concatenate(arenas, axis=0)
        self.vectors = np.concatenate(vectors, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.masks = np.concatenate(masks, axis=0)

        self._augment = augment
        self._n_real = len(self.actions)

        if augment:
            # Precompute index mapping for horizontal flip.
            # For each action index i: card*576 + row*18 + col
            #   -> card*576 + row*18 + (17 - col)
            # Noop (2304) maps to itself.
            # Since flip is self-inverse, this array works as both
            # forward and inverse mapping.
            flip = np.empty(ACTION_SPACE_SIZE, dtype=np.int64)
            for i in range(ACTION_SPACE_SIZE - 1):
                card = i // GRID_CELLS
                cell = i % GRID_CELLS
                row = cell // GRID_COLS
                col = cell % GRID_COLS
                flipped_col = GRID_COLS - 1 - col
                flip[i] = card * GRID_CELLS + row * GRID_COLS + flipped_col
            flip[NOOP_ACTION] = NOOP_ACTION
            self._flip_indices = flip

    def __len__(self) -> int:
        if self._augment:
            return 2 * self._n_real
        return self._n_real

    def __getitem__(self, idx: int) -> dict:
        """Get a single training sample.

        For indices >= N (when augment=True), returns a horizontally-
        flipped version: arena columns reversed, action and mask indices
        remapped.

        Returns:
            Dict with keys:
                "arena": (32, 18, 6) float32 tensor
                "vector": (23,) float32 tensor
                "action": scalar long tensor
                "mask": (2305,) bool tensor
        """
        if self._augment and idx >= self._n_real:
            real_idx = idx - self._n_real
            # Flip arena columns (axis 1 of shape 32,18,6)
            arena = self.arenas[real_idx][:, ::-1, :].copy()
            vector = self.vectors[real_idx]
            action = int(self.actions[real_idx])
            # Remap action index
            if action != NOOP_ACTION:
                action = int(self._flip_indices[action])
            # Remap mask using the flip permutation
            mask = self.masks[real_idx][self._flip_indices].copy()
            return {
                "arena": torch.from_numpy(arena).float(),
                "vector": torch.from_numpy(vector.copy()).float(),
                "action": torch.tensor(action, dtype=torch.long),
                "mask": torch.from_numpy(mask).bool(),
            }

        return {
            "arena": torch.from_numpy(self.arenas[idx]).float(),
            "vector": torch.from_numpy(self.vectors[idx]).float(),
            "action": torch.tensor(self.actions[idx], dtype=torch.long),
            "mask": torch.from_numpy(self.masks[idx].copy()).bool(),
        }

    def action_class_counts(self) -> tuple[int, int]:
        """Count no-op vs card placement frames.

        Returns:
            (noop_count, action_count) tuple.
        """
        noop_count = int(np.sum(self.actions == NOOP_ACTION))
        action_count = len(self.actions) - noop_count
        return noop_count, action_count

    def compute_class_weights(
        self,
        noop_weight: float = 0.3,
        action_weight: float = 3.0,
    ) -> torch.Tensor:
        """Compute per-class weights for CrossEntropyLoss.

        Assigns noop_weight to the no-op action (index 2304) and
        action_weight to all 2304 card placement actions. This addresses
        the heavy class imbalance (~70% no-op even after downsampling).

        Args:
            noop_weight: Weight for the no-op class.
            action_weight: Weight for all card placement classes.

        Returns:
            (2305,) float32 tensor of per-class weights.
        """
        weights = torch.full((ACTION_SPACE_SIZE,), action_weight)
        weights[NOOP_ACTION] = noop_weight
        return weights


def load_datasets(
    npz_paths: list[Path],
    val_ratio: float = 0.2,
    seed: int = 42,
    augment: bool = False,
) -> tuple[BCDataset, BCDataset]:
    """Split .npz files into train and validation datasets.

    Performs FILE-LEVEL splitting (not frame-level) to prevent data
    leakage between consecutive frames of the same game.

    Args:
        npz_paths: List of paths to .npz files.
        val_ratio: Fraction of files for validation (default 0.2).
        seed: Random seed for reproducible splitting.
        augment: Enable horizontal flip augmentation on training set.

    Returns:
        (train_dataset, val_dataset) tuple of BCDataset instances.
    """
    rng = random.Random(seed)
    paths = list(npz_paths)
    rng.shuffle(paths)

    split_idx = max(1, len(paths) - int(len(paths) * val_ratio))
    train_paths = paths[:split_idx]
    val_paths = paths[split_idx:]

    # Ensure at least 1 file in val when we have 2+ files
    if len(val_paths) == 0 and len(paths) >= 2:
        val_paths = [train_paths.pop()]

    # Augmentation only on training set (validation stays clean)
    return BCDataset(train_paths, augment=augment), BCDataset(val_paths)
