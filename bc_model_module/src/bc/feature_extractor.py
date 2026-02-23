"""CRFeatureExtractor - Feature extractor for Clash Royale observations.

Processes a dict observation with "arena" (B, 32, 18, 6) and "vector"
(B, 23) keys into a flat (B, 192) feature vector using learned embeddings
for both arena unit class IDs and card hand class IDs.

Arena branch:
    1. Extract CH_CLASS_ID (ch 0) -> denormalize to int -> nn.Embedding(156, 8)
    2. Concat with 5 remaining channels -> (B, 32, 18, 13) -> permute to (B, 13, 32, 18)
    3. Conv2d(13, 32, 3, pad=1) + BN + ReLU + MaxPool(2) -> (B, 32, 16, 9)
    4. Conv2d(32, 64, 3, pad=1) + BN + ReLU + MaxPool(2) -> (B, 64, 8, 4)
    5. Conv2d(64, 128, 3, pad=1) + BN + ReLU + AdaptiveAvgPool(1) -> (B, 128)

Vector branch:
    1. Extract card class floats [15:19] -> denormalize -> nn.Embedding(9, 8) -> (B, 32)
    2. Concat with 19 scalar features -> (B, 51)
    3. Linear(51, 64) + ReLU -> Linear(64, 64) + ReLU -> (B, 64)

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

_ARENA_CNN_IN = _ARENA_EMBED_DIM + 5  # 8 embed dims + 5 remaining channels = 13
_CARD_EMBED_TOTAL = 4 * _CARD_EMBED_DIM  # 4 card slots * 8 dims = 32
_VECTOR_SCALAR_FEATURES = 19  # 23 total - 4 card class floats = 19
_VECTOR_MLP_IN = _VECTOR_SCALAR_FEATURES + _CARD_EMBED_TOTAL  # 19 + 32 = 51

_DEFAULT_FEATURES_DIM = 192  # 128 arena + 64 vector


class CRFeatureExtractor(nn.Module):
    """Feature extractor for Clash Royale observations.

    Produces a flat feature vector from the dict observation space defined
    by StateEncoder. Designed to be SB3 BaseFeaturesExtractor-compatible
    without a hard dependency on stable_baselines3.

    Args:
        features_dim: Output feature dimension (default 192).
    """

    def __init__(self, features_dim: int = _DEFAULT_FEATURES_DIM) -> None:
        super().__init__()
        self.features_dim = features_dim

        # --- Arena branch ---
        self.arena_embed = nn.Embedding(_ARENA_EMBED_ENTRIES, _ARENA_EMBED_DIM)

        self.arena_cnn = nn.Sequential(
            nn.Conv2d(_ARENA_CNN_IN, 32, kernel_size=3, padding=1),
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

        # --- Vector branch ---
        self.card_embed = nn.Embedding(_CARD_EMBED_ENTRIES, _CARD_EMBED_DIM)

        self.vector_mlp = nn.Sequential(
            nn.Linear(_VECTOR_MLP_IN, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Process observation dict into feature vector.

        Args:
            obs: Dict with "arena" (B, 32, 18, 6) float32 and
                 "vector" (B, 23) float32 tensors.

        Returns:
            (B, features_dim) feature tensor.
        """
        arena = obs["arena"]  # (B, 32, 18, 6)
        vector = obs["vector"]  # (B, 23)

        # --- Arena branch ---
        # Denormalize class ID from float to integer for embedding lookup.
        # Class ID is stored as class_idx / NUM_CLASSES (155). Use round()
        # to avoid off-by-one from float truncation.
        class_id_float = arena[:, :, :, 0]  # (B, 32, 18)
        class_id_int = torch.round(class_id_float * NUM_CLASSES).long()
        class_id_int = class_id_int.clamp(0, _ARENA_EMBED_ENTRIES - 1)

        # Embedding lookup: (B, 32, 18) -> (B, 32, 18, 8)
        arena_embedded = self.arena_embed(class_id_int)

        # Remaining 5 channels: belonging, arena_mask, ally_tower_hp,
        # enemy_tower_hp, spell_count
        arena_other = arena[:, :, :, 1:]  # (B, 32, 18, 5)

        # Concat along last dim: (B, 32, 18, 13)
        arena_combined = torch.cat([arena_embedded, arena_other], dim=-1)

        # Permute to (B, C, H, W) for Conv2d: (B, 13, 32, 18)
        arena_combined = arena_combined.permute(0, 3, 1, 2)

        # CNN: (B, 13, 32, 18) -> (B, 128, 1, 1) -> (B, 128)
        arena_features = self.arena_cnn(arena_combined).flatten(1)

        # --- Vector branch ---
        # Card class index is normalized by (NUM_DECK_CARDS - 1) = 7.
        # Denormalize: round(float * 7) gives card index in [0, 7].
        card_class_float = vector[:, 15:19]  # (B, 4)
        card_class_int = torch.round(
            card_class_float * (NUM_DECK_CARDS - 1)
        ).long()

        # Handle empty card slots: card_present flags at indices [11:15].
        # Present cards use embed index card_idx + 1 (1-indexed).
        # Empty slots use embed index 0 (the "empty" embedding).
        card_present = vector[:, 11:15]  # (B, 4)
        card_embed_idx = torch.where(
            card_present > 0.5,
            card_class_int + 1,
            torch.zeros_like(card_class_int),
        )
        card_embed_idx = card_embed_idx.clamp(0, _CARD_EMBED_ENTRIES - 1)

        # (B, 4) -> (B, 4, 8) -> (B, 32)
        card_embedded = self.card_embed(card_embed_idx).flatten(1)

        # Scalar features: everything except card class indices [15:19]
        # Indices [0:15] = 15 features, [19:23] = 4 features -> 19 total
        vector_scalars = torch.cat(
            [vector[:, :15], vector[:, 19:]], dim=1
        )  # (B, 19)

        # Combine scalars + card embeddings: (B, 19 + 32) = (B, 51)
        vector_combined = torch.cat([vector_scalars, card_embedded], dim=1)

        # MLP: (B, 51) -> (B, 64)
        vector_features = self.vector_mlp(vector_combined)

        # --- Concatenate both branches ---
        return torch.cat([arena_features, vector_features], dim=1)  # (B, 192)
