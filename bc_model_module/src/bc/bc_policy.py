"""BCPolicy - Behavior cloning policy for Clash Royale.

Combines CRFeatureExtractor (192-dim features) with a hierarchical action
head that decomposes the 2305-class problem into three manageable sub-problems:

1. Play head: binary (play a card vs no-op)
2. Card head: 4-way (which card slot)
3. Position head: 576-way (which grid cell)

This decomposition is critical because the flat 2305-way softmax collapses
to always-noop with imbalanced data (547 actions across 2304 classes = 0.24
examples per class on average, while noop has 3000+ examples).

The hierarchical approach splits the problem into solvable pieces:
- Play head: 547 action vs 3116 noop (manageable 6:1 ratio)
- Card head: ~137 per card (balanced)
- Position head: trained only on action frames, no noop dilution

PPO transition:
    Use get_feature_extractor_state() to extract learned feature weights.
    The action heads are discarded when transitioning to PPO -- MaskablePPO
    creates its own action_net and value_net from the 192-dim features.
    Only the feature extractor is transferred.
"""

import torch
import torch.nn as nn

from src.bc.feature_extractor import CRFeatureExtractor
from src.encoder.encoder_constants import (
    ACTION_SPACE_SIZE,
    GRID_CELLS,
    NOOP_ACTION,
    NUM_CARD_SLOTS,
)


class BCPolicy(nn.Module):
    """Behavior cloning policy with hierarchical action decomposition.

    Predicts play/noop, card selection, and grid position separately
    from the same 192-dim feature vector. This addresses the extreme
    class imbalance that makes flat 2305-way classification collapse.

    Args:
        features_dim: Feature extractor output dimension (default 192).
        hidden_dim: Shared hidden layer size (default 256).
        dropout: Dropout rate (default 0.2).
    """

    def __init__(
        self,
        features_dim: int = 192,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.features_dim = features_dim
        self.feature_extractor = CRFeatureExtractor(features_dim=features_dim)

        # Shared trunk for all heads
        self.shared_trunk = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Head 1: Play vs noop (binary)
        self.play_head = nn.Linear(hidden_dim, 2)

        # Head 2: Which card (4-way)
        self.card_head = nn.Linear(hidden_dim, NUM_CARD_SLOTS)

        # Head 3: Where to play (576-way grid position)
        self.position_head = nn.Linear(hidden_dim, GRID_CELLS)

    def forward_decomposed(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning decomposed logits.

        Args:
            obs: Dict with "arena" (B, 32, 18, 6) and "vector" (B, 23).

        Returns:
            Tuple of (play_logits, card_logits, position_logits):
                - play_logits: (B, 2) - [noop, play]
                - card_logits: (B, 4) - card slot selection
                - position_logits: (B, 576) - grid cell selection
        """
        features = self.feature_extractor(obs)
        shared = self.shared_trunk(features)
        return (
            self.play_head(shared),
            self.card_head(shared),
            self.position_head(shared),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass returning flat 2305 logits (backward compatible).

        Reconstructs the flat action space from the decomposed heads
        for compatibility with the existing inference pipeline and
        action masking.

        Args:
            obs: Dict with "arena" (B, 32, 18, 6) and "vector" (B, 23).

        Returns:
            (B, 2305) raw logits (pre-softmax).
        """
        play_logits, card_logits, position_logits = self.forward_decomposed(obs)

        B = play_logits.size(0)
        flat_logits = torch.zeros(B, ACTION_SPACE_SIZE, device=play_logits.device)

        # Noop logit = play_logits[:, 0] (the "don't play" score)
        flat_logits[:, NOOP_ACTION] = play_logits[:, 0]

        # Card placement logits = play_score + card_score + position_score
        # This additive decomposition means each placement action's logit
        # is the sum of its component scores.
        play_action_score = play_logits[:, 1]  # (B,)
        for card_id in range(NUM_CARD_SLOTS):
            card_score = card_logits[:, card_id]  # (B,)
            start_idx = card_id * GRID_CELLS
            end_idx = start_idx + GRID_CELLS
            # (B, 576) = play(B,1) + card(B,1) + position(B,576)
            flat_logits[:, start_idx:end_idx] = (
                play_action_score.unsqueeze(1)
                + card_score.unsqueeze(1)
                + position_logits
            )

        return flat_logits

    @torch.no_grad()
    def predict_action(
        self,
        obs: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> int:
        """Predict a single action using hierarchical decomposition.

        Decision flow:
        1. Play head decides play vs noop
        2. If play: mask unavailable cards, pick best card, then best position
        3. If noop: return 2304

        Args:
            obs: Dict with "arena" (1, 32, 18, 6) and "vector" (1, 23).
            mask: Boolean mask (1, 2305) or (2305,). True = valid action.

        Returns:
            Integer action index in [0, 2304].
        """
        self.eval()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        play_logits, card_logits, pos_logits = self.forward_decomposed(obs)

        # Check if ANY card placement is valid
        any_card_valid = mask[0, :NOOP_ACTION].any()

        # Play vs noop: argmax of binary head
        should_play = play_logits[0, 1] > play_logits[0, 0] and any_card_valid

        if not should_play:
            return NOOP_ACTION

        # Mask unavailable cards
        for card_id in range(NUM_CARD_SLOTS):
            start = card_id * GRID_CELLS
            if not mask[0, start:start + GRID_CELLS].any():
                card_logits[0, card_id] = float("-inf")

        best_card = card_logits[0].argmax().item()
        best_pos = pos_logits[0].argmax().item()
        return best_card * GRID_CELLS + best_pos

    @torch.no_grad()
    def predict_action_decomposed(
        self,
        obs: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> tuple[int, float, float, float]:
        """Predict action with confidence scores from each head.

        Returns:
            (action_idx, play_prob, card_prob, pos_prob) tuple.
        """
        self.eval()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        play_logits, card_logits, pos_logits = self.forward_decomposed(obs)

        play_probs = torch.softmax(play_logits[0], dim=0)
        play_prob = play_probs[1].item()  # probability of "play"

        any_card_valid = mask[0, :NOOP_ACTION].any()
        should_play = play_logits[0, 1] > play_logits[0, 0] and any_card_valid

        if not should_play:
            return NOOP_ACTION, play_prob, 0.0, 0.0

        for card_id in range(NUM_CARD_SLOTS):
            start = card_id * GRID_CELLS
            if not mask[0, start:start + GRID_CELLS].any():
                card_logits[0, card_id] = float("-inf")

        card_probs = torch.softmax(card_logits[0], dim=0)
        pos_probs = torch.softmax(pos_logits[0], dim=0)

        best_card = card_logits[0].argmax().item()
        best_pos = pos_logits[0].argmax().item()
        card_prob = card_probs[best_card].item()
        pos_prob = pos_probs[best_pos].item()

        return best_card * GRID_CELLS + best_pos, play_prob, card_prob, pos_prob

    def get_feature_extractor_state(self) -> dict:
        """Extract feature extractor weights for PPO transfer.

        Returns:
            state_dict of the CRFeatureExtractor module.
        """
        return self.feature_extractor.state_dict()

    def save(self, path: str) -> None:
        """Save full policy checkpoint."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "BCPolicy":
        """Load a saved policy checkpoint.

        Args:
            path: Path to the .pt checkpoint file.
            **kwargs: Constructor arguments (features_dim, hidden_dim, dropout).

        Returns:
            BCPolicy instance with loaded weights in eval mode.
        """
        policy = cls(**kwargs)
        policy.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        policy.eval()
        return policy
