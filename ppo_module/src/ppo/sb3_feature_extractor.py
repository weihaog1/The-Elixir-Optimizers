"""SB3-compatible wrapper around CRFeatureExtractor.

Subclasses stable_baselines3's BaseFeaturesExtractor so the BC-pretrained
feature extractor can be plugged directly into MaskablePPO.

Usage:
    from src.ppo.sb3_feature_extractor import SB3CRFeatureExtractor

    model = MaskablePPO("MultiInputPolicy", env, policy_kwargs={
        "features_extractor_class": SB3CRFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
    })

    # Load BC pretrained weights
    bc_weights = torch.load("models/bc/bc_feature_extractor.pt")
    model.policy.features_extractor.load_bc_weights(bc_weights)
"""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.bc.feature_extractor import CRFeatureExtractor


class SB3CRFeatureExtractor(BaseFeaturesExtractor):
    """SB3 BaseFeaturesExtractor wrapping CRFeatureExtractor.

    Args:
        observation_space: Gymnasium Dict observation space.
        features_dim: Output feature dimension (default 192).
    """

    def __init__(self, observation_space, features_dim: int = 192) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self._extractor = CRFeatureExtractor(features_dim=features_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._extractor(observations)

    def load_bc_weights(self, state_dict: dict) -> None:
        """Load BC-pretrained feature extractor weights.

        Args:
            state_dict: State dict from torch.load("bc_feature_extractor.pt").
        """
        self._extractor.load_state_dict(state_dict)

    def freeze(self) -> None:
        """Freeze all parameters (for Phase 1 PPO training)."""
        for param in self._extractor.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters (for Phase 2 PPO fine-tuning)."""
        for param in self._extractor.parameters():
            param.requires_grad = True
