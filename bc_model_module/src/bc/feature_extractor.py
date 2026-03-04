"""CRFeatureExtractor - Feature extractor for Clash Royale observations.

Supports optional frame stacking via the ``n_frames`` parameter.  When
``n_frames=1`` (default), the input/output shapes match the original
single-frame architecture.  When ``n_frames>1``, the arena and vector
inputs are expected to be channel-concatenated stacks of *n_frames*
consecutive observations.

Single-frame (n_frames=1):
    arena:  (B, 32, 18, 6)    vector: (B, 23)

3-frame stack (n_frames=3):
    arena:  (B, 32, 18, 18)   vector: (B, 69)

Arena branch (per-frame, shared weights):
    1. Extract CH_CLASS_ID (ch 0) -> denormalize to int -> nn.Embedding(156, 8)
    2. Concat with 5 remaining channels -> (B, 32, 18, 13)
    Repeat for each frame, then concat -> (B, 32, 18, 13*n_frames)
    3. Permute -> Conv2d -> BN -> ReLU -> MaxPool -> ... -> AdaptiveAvgPool(1) -> (B, 128)

Vector branch (per-frame, shared weights):
    1. Extract card class floats [15:19] -> denormalize -> nn.Embedding(9, 8) -> (B, 32)
    2. Concat with 19 scalar features -> (B, 51)
    Repeat for each frame, then concat -> (B, 51*n_frames)
    3. Linear -> ReLU -> Linear -> ReLU -> (B, 64)

Output: cat([arena_128, vector_64]) = (B, 192)

PPO transition:
    Save this module's state_dict() as bc_feature_extractor.pt. Load into
    MaskablePPO via:
        ppo.policy.features_extractor.load_state_dict(torch.load(...))
    Optionally freeze with param.requires_grad = False during initial PPO
    training, then unfreeze with lower LR (3e-5). Conservative PPO
    hyperparameters: lr=1e-4, clip_range=0.1.
"""

import torch
import torch.nn as nn

from src.encoder.encoder_constants import NUM_CLASSES, NUM_DECK_CARDS

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
_ARENA_EMBED_ENTRIES = NUM_CLASSES + 1  # 156: 0=empty, 1-155=unit classes
_ARENA_EMBED_DIM = 8
_CARD_EMBED_ENTRIES = NUM_DECK_CARDS + 1  # 9: 0=empty, 1-8=deck cards
_CARD_EMBED_DIM = 8

_PER_FRAME_ARENA_CH = 6  # channels per single frame
_PER_FRAME_CNN_IN = _ARENA_EMBED_DIM + (_PER_FRAME_ARENA_CH - 1)  # 8 + 5 = 13
_CARD_EMBED_TOTAL = 4 * _CARD_EMBED_DIM  # 4 card slots * 8 dims = 32
_VECTOR_SCALAR_FEATURES = 19  # 23 total - 4 card class floats = 19
_PER_FRAME_VEC_IN = _VECTOR_SCALAR_FEATURES + _CARD_EMBED_TOTAL  # 19 + 32 = 51
_PER_FRAME_VEC_FEATURES = 23

_DEFAULT_FEATURES_DIM = 192  # 128 arena + 64 vector


class CRFeatureExtractor(nn.Module):
    """Feature extractor for Clash Royale observations.

    Produces a flat feature vector from the dict observation space defined
    by StateEncoder. Designed to be SB3 BaseFeaturesExtractor-compatible
    without a hard dependency on stable_baselines3.

    Args:
        features_dim: Output feature dimension (default 192).
        n_frames: Number of stacked frames (default 1 for single-frame).
    """

    def __init__(
        self,
        features_dim: int = _DEFAULT_FEATURES_DIM,
        n_frames: int = 1,
    ) -> None:
        super().__init__()
        self.features_dim = features_dim
        self.n_frames = n_frames

        # --- Arena branch (shared embeddings across frames) ---
        self.arena_embed = nn.Embedding(_ARENA_EMBED_ENTRIES, _ARENA_EMBED_DIM)

        arena_cnn_in = n_frames * _PER_FRAME_CNN_IN  # n_frames * 13
        self.arena_cnn = nn.Sequential(
            nn.Conv2d(arena_cnn_in, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 16, 9)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 64, 8, 4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (B, 128, 1, 1)
        )

        # --- Vector branch (shared card embeddings across frames) ---
        self.card_embed = nn.Embedding(_CARD_EMBED_ENTRIES, _CARD_EMBED_DIM)

        vector_mlp_in = n_frames * _PER_FRAME_VEC_IN  # n_frames * 51
        # Scale first hidden layer with frame count for capacity
        vec_hidden = 128 if n_frames > 1 else 64
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_mlp_in, vec_hidden),
            nn.ReLU(),
            nn.Linear(vec_hidden, 64),
            nn.ReLU(),
        )

    def _embed_arena_frame(self, frame_arena: torch.Tensor) -> torch.Tensor:
        """Embed a single frame's arena channels.

        Args:
            frame_arena: (B, 32, 18, 6) single-frame arena tensor.

        Returns:
            (B, 32, 18, 13) embedded arena tensor.
        """
        class_id_float = frame_arena[:, :, :, 0]  # (B, 32, 18)
        class_id_int = torch.round(class_id_float * NUM_CLASSES).long()
        class_id_int = class_id_int.clamp(0, _ARENA_EMBED_ENTRIES - 1)
        arena_embedded = self.arena_embed(class_id_int)  # (B, 32, 18, 8)
        arena_other = frame_arena[:, :, :, 1:]  # (B, 32, 18, 5)
        return torch.cat([arena_embedded, arena_other], dim=-1)  # (B, 32, 18, 13)

    def _embed_vector_frame(self, frame_vec: torch.Tensor) -> torch.Tensor:
        """Embed a single frame's vector features.

        Args:
            frame_vec: (B, 23) single-frame vector tensor.

        Returns:
            (B, 51) embedded vector tensor.
        """
        card_class_float = frame_vec[:, 15:19]  # (B, 4)
        card_class_int = torch.round(
            card_class_float * (NUM_DECK_CARDS - 1)
        ).long()

        card_present = frame_vec[:, 11:15]  # (B, 4)
        card_embed_idx = torch.where(
            card_present > 0.5,
            card_class_int + 1,
            torch.zeros_like(card_class_int),
        )
        card_embed_idx = card_embed_idx.clamp(0, _CARD_EMBED_ENTRIES - 1)
        card_embedded = self.card_embed(card_embed_idx).flatten(1)  # (B, 32)

        vector_scalars = torch.cat(
            [frame_vec[:, :15], frame_vec[:, 19:]], dim=1
        )  # (B, 19)

        return torch.cat([vector_scalars, card_embedded], dim=1)  # (B, 51)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Process observation dict into feature vector.

        Args:
            obs: Dict with "arena" and "vector" tensors.
                 Single-frame: arena (B, 32, 18, 6), vector (B, 23)
                 N-frame stack: arena (B, 32, 18, 6*N), vector (B, 23*N)

        Returns:
            (B, features_dim) feature tensor.
        """
        arena = obs["arena"]
        vector = obs["vector"]
        n = self.n_frames

        # --- Arena branch: embed each frame, then concat ---
        arena_chunks = []
        for f in range(n):
            frame_arena = arena[:, :, :, f * _PER_FRAME_ARENA_CH:(f + 1) * _PER_FRAME_ARENA_CH]
            arena_chunks.append(self._embed_arena_frame(frame_arena))

        # (B, 32, 18, 13*n_frames) -> permute to (B, 13*n, 32, 18)
        arena_combined = torch.cat(arena_chunks, dim=-1)
        arena_combined = arena_combined.permute(0, 3, 1, 2)
        arena_features = self.arena_cnn(arena_combined).flatten(1)  # (B, 128)

        # --- Vector branch: embed each frame, then concat ---
        vec_chunks = []
        for f in range(n):
            frame_vec = vector[:, f * _PER_FRAME_VEC_FEATURES:(f + 1) * _PER_FRAME_VEC_FEATURES]
            vec_chunks.append(self._embed_vector_frame(frame_vec))

        vector_combined = torch.cat(vec_chunks, dim=1)  # (B, 51*n_frames)
        vector_features = self.vector_mlp(vector_combined)  # (B, 64)

        # --- Concatenate both branches ---
        return torch.cat([arena_features, vector_features], dim=1)  # (B, 192)
