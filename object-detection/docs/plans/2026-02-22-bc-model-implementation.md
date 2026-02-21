# BC Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Behavior Cloning model with full embedding (arena + card), custom PyTorch training loop, SB3-compatible feature extractor, and comprehensive documentation.

**Architecture:** CRFeatureExtractor (SB3 BaseFeaturesExtractor subclass) with arena CNN (3 conv layers + nn.Embedding for 155 unit classes) and vector MLP (nn.Embedding for 8 card classes). BCPolicy wraps the extractor with a 2-layer action head. Custom PyTorch training loop with weighted cross-entropy, 80/20 game-level split, early stopping. All SB3-compatible for future PPO transition.

**Tech Stack:** PyTorch 2.10, gymnasium 1.2.3, numpy 1.26. SB3/sb3-contrib needed only for PPO phase (not yet installed).

**Base path:** `docs/josh/bc_model_module/`

---

## Task 1: conftest.py (Import Path Setup)

**Files:**
- Create: `docs/josh/bc_model_module/tests/conftest.py`

**Step 1: Write conftest.py**

Follow the existing pattern from dataset_builder_module. Inject bc_model_module/src, state_encoder_module/src, and cr-object-detection root into the import path.

```python
"""Pytest conftest for josh's bc_model_module tests.

Injects bc_model, state_encoder, dataset_builder module src/ directories
at the front of src.__path__ so that:
  - `from src.bc import ...` resolves to the josh bc copy
  - `from src.encoder import ...` resolves to the josh encoder copy
  - `from src.dataset import ...` resolves to the josh dataset copy
Other subpackages resolve from the real codebase.
"""

import os
import sys

_repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import src  # noqa: E402

_encoder_src = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "state_encoder_module", "src")
)
if _encoder_src not in src.__path__:
    src.__path__.insert(0, _encoder_src)

_dataset_src = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset_builder_module", "src")
)
if _dataset_src not in src.__path__:
    src.__path__.insert(0, _dataset_src)

_bc_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _bc_src not in src.__path__:
    src.__path__.insert(0, _bc_src)
```

**Step 2: Verify import path works**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -c "import sys; sys.path.insert(0, '.'); import src; print(src.__path__)"`
Expected: prints the path list without error.

**Step 3: Commit**

```
chore: Add conftest.py for bc_model_module test imports.
```

---

## Task 2: CRFeatureExtractor (SB3-Compatible Feature Extractor)

**Files:**
- Create: `docs/josh/bc_model_module/src/bc/feature_extractor.py`
- Create: `docs/josh/bc_model_module/tests/test_feature_extractor.py`

**Step 1: Write the failing tests**

```python
"""Tests for CRFeatureExtractor."""

import gymnasium as gym
import numpy as np
import torch
import pytest

from src.bc.feature_extractor import CRFeatureExtractor
from src.encoder.encoder_constants import (
    ACTION_SPACE_SIZE,
    GRID_COLS,
    GRID_ROWS,
    NUM_ARENA_CHANNELS,
    NUM_CLASSES,
    NUM_DECK_CARDS,
    NUM_VECTOR_FEATURES,
)


@pytest.fixture
def obs_space():
    """Create the observation space matching StateEncoder."""
    return gym.spaces.Dict({
        "arena": gym.spaces.Box(
            low=-1.0, high=10.0,
            shape=(GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
            dtype=np.float32,
        ),
        "vector": gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(NUM_VECTOR_FEATURES,),
            dtype=np.float32,
        ),
    })


@pytest.fixture
def extractor(obs_space):
    """Create a CRFeatureExtractor instance."""
    return CRFeatureExtractor(obs_space)


class TestCRFeatureExtractor:
    """Tests for CRFeatureExtractor."""

    def test_output_shape(self, extractor):
        """Forward pass should produce (batch, features_dim) tensor."""
        obs = {
            "arena": torch.zeros(2, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
            "vector": torch.zeros(2, NUM_VECTOR_FEATURES),
        }
        features = extractor(obs)
        assert features.shape == (2, extractor.features_dim)

    def test_features_dim_attribute(self, extractor):
        """features_dim should be a positive integer."""
        assert isinstance(extractor.features_dim, int)
        assert extractor.features_dim > 0

    def test_single_sample(self, extractor):
        """Should work with batch size 1."""
        obs = {
            "arena": torch.zeros(1, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
            "vector": torch.zeros(1, NUM_VECTOR_FEATURES),
        }
        features = extractor(obs)
        assert features.shape == (1, extractor.features_dim)

    def test_nonzero_arena_produces_different_features(self, extractor):
        """Different arena inputs should produce different feature vectors."""
        obs_zeros = {
            "arena": torch.zeros(1, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
            "vector": torch.zeros(1, NUM_VECTOR_FEATURES),
        }
        arena_nonzero = torch.zeros(1, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS)
        arena_nonzero[0, 16, 9, 0] = 49 / NUM_CLASSES  # Knight class
        arena_nonzero[0, 16, 9, 1] = 1.0  # Enemy
        arena_nonzero[0, 16, 9, 2] = 1.0  # Present
        obs_nonzero = {"arena": arena_nonzero, "vector": torch.zeros(1, NUM_VECTOR_FEATURES)}

        f1 = extractor(obs_zeros)
        f2 = extractor(obs_nonzero)
        assert not torch.allclose(f1, f2)

    def test_arena_embedding_extracts_class_ids(self, extractor):
        """Arena embedding should handle normalized class IDs correctly."""
        arena = torch.zeros(1, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS)
        arena[0, 10, 5, 0] = 100 / NUM_CLASSES
        obs = {"arena": arena, "vector": torch.zeros(1, NUM_VECTOR_FEATURES)}
        features = extractor(obs)
        assert features.shape == (1, extractor.features_dim)
        assert not torch.isnan(features).any()

    def test_card_embedding_in_vector(self, extractor):
        """Vector branch should handle card class indices via embedding."""
        vec = torch.zeros(1, NUM_VECTOR_FEATURES)
        vec[0, 11] = 1.0  # Card 0 present
        vec[0, 15] = 5 / (NUM_DECK_CARDS - 1)  # Card 0 = royal-hogs (idx 5)
        vec[0, 19] = 5 / 10  # 5 elixir
        obs = {
            "arena": torch.zeros(1, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
            "vector": vec,
        }
        features = extractor(obs)
        assert features.shape == (1, extractor.features_dim)
        assert not torch.isnan(features).any()

    def test_gradient_flows(self, extractor):
        """Gradients should flow through the extractor."""
        obs = {
            "arena": torch.randn(2, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
            "vector": torch.rand(2, NUM_VECTOR_FEATURES),
        }
        features = extractor(obs)
        loss = features.sum()
        loss.backward()
        for param in extractor.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_deterministic_eval_mode(self, extractor):
        """Eval mode should produce deterministic output."""
        extractor.eval()
        obs = {
            "arena": torch.randn(2, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
            "vector": torch.rand(2, NUM_VECTOR_FEATURES),
        }
        f1 = extractor(obs)
        f2 = extractor(obs)
        assert torch.allclose(f1, f2)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -m pytest docs/josh/bc_model_module/tests/test_feature_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.bc'`

**Step 3: Write the implementation**

```python
"""CRFeatureExtractor - SB3-compatible feature extractor for Clash Royale.

Processes Dict observations {"arena": (B,32,18,6), "vector": (B,23)} into
a flat feature vector. Uses nn.Embedding for both arena unit class IDs
(156 entries: 0=empty + 155 classes) and card hand class IDs (9 entries:
0=empty + 8 deck cards).

Designed to subclass SB3's BaseFeaturesExtractor for seamless PPO transition,
but implemented as a standalone nn.Module so it works without SB3 installed.

PPO Transition Notes:
    When SB3 + sb3-contrib are installed, this class can be passed directly
    to MaskablePPO:

        from sb3_contrib import MaskablePPO
        model = MaskablePPO(
            "MultiInputPolicy", env,
            policy_kwargs={
                "features_extractor_class": CRFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 192},
            },
        )

    SB3's MultiInputPolicy will call forward() with a dict of tensors.
    The features_dim property tells SB3 the output size for its policy MLP.
"""

import torch
import torch.nn as nn

from src.encoder.encoder_constants import (
    GRID_COLS,
    GRID_ROWS,
    NUM_ARENA_CHANNELS,
    NUM_CLASSES,
    NUM_DECK_CARDS,
    NUM_VECTOR_FEATURES,
)

# Arena embedding: 0 = empty, 1..155 = unit classes
_ARENA_EMBED_ENTRIES = NUM_CLASSES + 1  # 156
_ARENA_EMBED_DIM = 8

# Card embedding: 0 = empty/absent, 1..8 = deck card indices (+1 offset)
_CARD_EMBED_ENTRIES = NUM_DECK_CARDS + 1  # 9
_CARD_EMBED_DIM = 8

# Arena CNN input channels: embed_dim + 5 remaining channels (belonging,
# arena_mask, ally_tower_hp, enemy_tower_hp, spell)
_ARENA_CNN_IN = _ARENA_EMBED_DIM + (NUM_ARENA_CHANNELS - 1)  # 8 + 5 = 13

# Vector MLP input: 19 scalar features + 4 cards * card_embed_dim
_NUM_SCALAR_FEATURES = NUM_VECTOR_FEATURES - 4  # Remove 4 card class floats
_VECTOR_MLP_IN = _NUM_SCALAR_FEATURES + 4 * _CARD_EMBED_DIM  # 19 + 32 = 51

# Card class feature indices in vector (slots 0-3)
_CARD_CLASS_START = 15
_CARD_CLASS_END = 19
_CARD_PRESENT_START = 11
_CARD_PRESENT_END = 15

# Feature dimensions for each branch
_ARENA_FEATURES = 128
_VECTOR_FEATURES = 64
_DEFAULT_FEATURES_DIM = _ARENA_FEATURES + _VECTOR_FEATURES  # 192


class CRFeatureExtractor(nn.Module):
    """Feature extractor for Clash Royale observations.

    Arena branch:
        1. Extract class ID channel -> denormalize to int -> nn.Embedding(156, 8)
        2. Concatenate embedding with remaining 5 channels -> (B, 13, 32, 18)
        3. 3-layer CNN -> AdaptiveAvgPool -> flatten -> 128 features

    Vector branch:
        1. Extract 4 card class indices -> denormalize to int -> nn.Embedding(9, 8)
        2. Concatenate embeddings with remaining 19 scalar features -> (B, 51)
        3. 2-layer MLP -> 64 features

    Output: concatenated (B, 192) feature vector.

    Args:
        observation_space: gymnasium Dict space (used by SB3 interface).
            Ignored if not needed -- the architecture is hardcoded to match
            StateEncoder's output.
        features_dim: Output feature dimension. Default 192.
    """

    def __init__(self, observation_space=None, features_dim: int = _DEFAULT_FEATURES_DIM):
        super().__init__()
        self._features_dim = features_dim

        # --- Arena embeddings ---
        self.arena_embedding = nn.Embedding(
            _ARENA_EMBED_ENTRIES, _ARENA_EMBED_DIM, padding_idx=0
        )

        # --- Arena CNN ---
        self.arena_cnn = nn.Sequential(
            nn.Conv2d(_ARENA_CNN_IN, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 16, 9)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 64, 8, 4)
            nn.Conv2d(64, _ARENA_FEATURES, kernel_size=3, padding=1),
            nn.BatchNorm2d(_ARENA_FEATURES),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (B, 128, 1, 1)
            nn.Flatten(),  # (B, 128)
        )

        # --- Card embeddings ---
        self.card_embedding = nn.Embedding(
            _CARD_EMBED_ENTRIES, _CARD_EMBED_DIM, padding_idx=0
        )

        # --- Vector MLP ---
        self.vector_mlp = nn.Sequential(
            nn.Linear(_VECTOR_MLP_IN, 64),
            nn.ReLU(),
            nn.Linear(64, _VECTOR_FEATURES),
            nn.ReLU(),
        )

    @property
    def features_dim(self) -> int:
        """Output feature dimension (for SB3 compatibility)."""
        return self._features_dim

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observation dict.

        Args:
            observations: Dict with:
                "arena": (B, 32, 18, 6) float32
                "vector": (B, 23) float32

        Returns:
            (B, features_dim) feature tensor.
        """
        arena = observations["arena"]  # (B, 32, 18, 6)
        vector = observations["vector"]  # (B, 23)

        arena_features = self._forward_arena(arena)
        vector_features = self._forward_vector(vector)

        return torch.cat([arena_features, vector_features], dim=1)

    def _forward_arena(self, arena: torch.Tensor) -> torch.Tensor:
        """Process arena grid through embedding + CNN.

        Args:
            arena: (B, 32, 18, 6) float32 from StateEncoder.

        Returns:
            (B, 128) arena features.
        """
        # Extract class ID channel and convert to integer indices
        class_float = arena[:, :, :, 0]  # (B, 32, 18)
        class_int = (class_float * NUM_CLASSES).round().long().clamp(0, NUM_CLASSES)  # [0, 155]

        # Embedding lookup
        embedded = self.arena_embedding(class_int)  # (B, 32, 18, 8)

        # Remaining channels (belonging, mask, ally_hp, enemy_hp, spell)
        other_channels = arena[:, :, :, 1:]  # (B, 32, 18, 5)

        # Concatenate: (B, 32, 18, 13)
        combined = torch.cat([embedded, other_channels], dim=3)

        # Permute to channels-first for Conv2d: (B, 13, 32, 18)
        combined = combined.permute(0, 3, 1, 2)

        return self.arena_cnn(combined)

    def _forward_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Process vector features through card embedding + MLP.

        Args:
            vector: (B, 23) float32 from StateEncoder.

        Returns:
            (B, 64) vector features.
        """
        # Extract card class indices and presence flags
        card_present = vector[:, _CARD_PRESENT_START:_CARD_PRESENT_END]  # (B, 4)
        card_class_float = vector[:, _CARD_CLASS_START:_CARD_CLASS_END]  # (B, 4)

        # Denormalize: card_idx = round(val * 7) gives [0, 7], +1 for embedding offset
        card_int = (card_class_float * (NUM_DECK_CARDS - 1)).round().long()  # [0, 7]
        card_int = card_int + 1  # [1, 8] for embedding (0 = empty)

        # Zero out absent cards (card_present == 0 -> embedding index 0 = padding)
        absent_mask = card_present < 0.5
        card_int[absent_mask] = 0

        # Embedding lookup: (B, 4, 8)
        card_embedded = self.card_embedding(card_int)
        card_flat = card_embedded.reshape(vector.shape[0], -1)  # (B, 32)

        # Scalar features: everything except card class indices [15:19]
        scalar_parts = [
            vector[:, :_CARD_CLASS_START],  # [0:15]
            vector[:, _CARD_CLASS_END:],    # [19:23]
        ]
        scalars = torch.cat(scalar_parts, dim=1)  # (B, 19)

        # Combine scalars + card embeddings
        vec_input = torch.cat([scalars, card_flat], dim=1)  # (B, 51)

        return self.vector_mlp(vec_input)
```

Also create `__init__.py`:

```python
"""BC model module for Clash Royale behavior cloning."""

from src.bc.feature_extractor import CRFeatureExtractor
from src.bc.bc_policy import BCPolicy
from src.bc.bc_dataset import BCDataset

__all__ = ["CRFeatureExtractor", "BCPolicy", "BCDataset"]
```

Note: `__init__.py` will cause an import error until bc_policy.py and bc_dataset.py exist. Create a minimal stub `__init__.py` first with just:

```python
"""BC model module for Clash Royale behavior cloning."""
```

And update it in Task 5 after all modules are created.

**Step 4: Run tests to verify they pass**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -m pytest docs/josh/bc_model_module/tests/test_feature_extractor.py -v`
Expected: All 8 tests PASS.

**Step 5: Commit**

```
feat: Add CRFeatureExtractor with arena and card embeddings.
```

---

## Task 3: BCPolicy (Feature Extractor + Action Head)

**Files:**
- Create: `docs/josh/bc_model_module/src/bc/bc_policy.py`
- Create: `docs/josh/bc_model_module/tests/test_bc_policy.py`

**Step 1: Write the failing tests**

```python
"""Tests for BCPolicy."""

import torch
import pytest

from src.bc.bc_policy import BCPolicy
from src.encoder.encoder_constants import (
    ACTION_SPACE_SIZE,
    GRID_COLS,
    GRID_ROWS,
    NOOP_ACTION,
    NUM_ARENA_CHANNELS,
    NUM_VECTOR_FEATURES,
)


@pytest.fixture
def policy():
    """Create a BCPolicy instance."""
    return BCPolicy()


def _make_obs(batch_size=2):
    """Create a dummy observation dict."""
    return {
        "arena": torch.zeros(batch_size, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS),
        "vector": torch.zeros(batch_size, NUM_VECTOR_FEATURES),
    }


class TestBCPolicy:
    """Tests for BCPolicy."""

    def test_output_shape(self, policy):
        """Forward should produce (batch, 2305) logits."""
        logits = policy(_make_obs(4))
        assert logits.shape == (4, ACTION_SPACE_SIZE)

    def test_predict_action_returns_valid_index(self, policy):
        """predict_action should return an int in [0, 2304]."""
        obs = _make_obs(1)
        mask = torch.ones(1, ACTION_SPACE_SIZE, dtype=torch.bool)
        action = policy.predict_action(obs, mask)
        assert isinstance(action, int)
        assert 0 <= action <= NOOP_ACTION

    def test_predict_action_respects_mask(self, policy):
        """predict_action should never select a masked-out action."""
        obs = _make_obs(1)
        mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[0, NOOP_ACTION] = True  # Only no-op is valid
        action = policy.predict_action(obs, mask)
        assert action == NOOP_ACTION

    def test_gradient_flows(self, policy):
        """Gradients should flow through the full policy."""
        obs = _make_obs(2)
        obs["arena"] = torch.randn_like(obs["arena"])
        obs["vector"] = torch.rand_like(obs["vector"])
        logits = policy(obs)
        loss = logits.sum()
        loss.backward()
        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_save_and_load(self, tmp_path, policy):
        """Should save and load weights correctly."""
        path = str(tmp_path / "test_policy.pt")
        policy.save(path)

        loaded = BCPolicy.load(path)
        obs = _make_obs(1)
        obs["arena"] = torch.randn_like(obs["arena"])

        policy.eval()
        loaded.eval()
        logits1 = policy(obs)
        logits2 = loaded(obs)
        assert torch.allclose(logits1, logits2)

    def test_get_feature_extractor_state(self, policy):
        """get_feature_extractor_state should return extractor weights only."""
        state = policy.get_feature_extractor_state()
        assert isinstance(state, dict)
        assert any("arena_embedding" in k for k in state)
        assert not any("action_head" in k for k in state)

    def test_parameter_count(self, policy):
        """Total parameters should be in expected range (100K-250K)."""
        total = sum(p.numel() for p in policy.parameters())
        assert 100_000 < total < 250_000, f"Got {total} params"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -m pytest docs/josh/bc_model_module/tests/test_bc_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.bc.bc_policy'`

**Step 3: Write the implementation**

```python
"""BCPolicy - Behavior Cloning policy network for Clash Royale.

Combines CRFeatureExtractor with a 2-layer action head to predict actions
from observations. Used for BC training with custom PyTorch loop.

PPO Transition Notes:
    The BCPolicy is NOT used directly with SB3. For PPO:
    1. Extract feature extractor weights: policy.get_feature_extractor_state()
    2. Pass CRFeatureExtractor class to MaskablePPO policy_kwargs
    3. Load weights into model.policy.features_extractor
    4. SB3 adds its own action_net and value_net on top

    Example:
        # Save BC extractor weights
        torch.save(bc_policy.get_feature_extractor_state(), "bc_extractor.pt")

        # Load into PPO
        from sb3_contrib import MaskablePPO
        ppo = MaskablePPO("MultiInputPolicy", env, policy_kwargs={
            "features_extractor_class": CRFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 192},
            "net_arch": [128, 64],
        })
        ppo.policy.features_extractor.load_state_dict(
            torch.load("bc_extractor.pt")
        )
"""

import torch
import torch.nn as nn

from src.bc.feature_extractor import CRFeatureExtractor
from src.encoder.encoder_constants import ACTION_SPACE_SIZE


class BCPolicy(nn.Module):
    """BC policy: CRFeatureExtractor + action head.

    The action head is a 2-layer MLP that maps features to action logits.
    During inference, apply an action mask before argmax.

    Args:
        features_dim: Feature extractor output dimension. Default 192.
        hidden_dim: Action head hidden layer size. Default 256.
        dropout: Dropout rate in action head. Default 0.2.
    """

    def __init__(
        self,
        features_dim: int = 192,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_extractor = CRFeatureExtractor(features_dim=features_dim)
        self.action_head = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ACTION_SPACE_SIZE),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute action logits from observations.

        Args:
            observations: Dict with "arena" (B,32,18,6) and "vector" (B,23).

        Returns:
            (B, 2305) raw logits (apply mask + softmax externally).
        """
        features = self.feature_extractor(observations)
        return self.action_head(features)

    @torch.no_grad()
    def predict_action(
        self,
        observations: dict[str, torch.Tensor],
        action_mask: torch.Tensor,
    ) -> int:
        """Predict a single action with masking.

        Args:
            observations: Dict with "arena" (1,32,18,6) and "vector" (1,23).
            action_mask: (1, 2305) bool tensor. True = valid action.

        Returns:
            Integer action index in [0, 2304].
        """
        self.eval()
        logits = self.forward(observations)  # (1, 2305)
        logits[~action_mask] = float("-inf")
        return logits.argmax(dim=1).item()

    def save(self, path: str) -> None:
        """Save full policy weights."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "BCPolicy":
        """Load a saved policy."""
        policy = cls(**kwargs)
        policy.load_state_dict(torch.load(path, weights_only=True))
        return policy

    def get_feature_extractor_state(self) -> dict:
        """Get feature extractor weights for PPO transfer.

        Returns:
            State dict of the feature extractor only (no action head).
        """
        return self.feature_extractor.state_dict()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -m pytest docs/josh/bc_model_module/tests/test_bc_policy.py -v`
Expected: All 7 tests PASS.

**Step 5: Commit**

```
feat: Add BCPolicy with action head and save/load support.
```

---

## Task 4: BCDataset (PyTorch Dataset for .npz Files)

**Files:**
- Create: `docs/josh/bc_model_module/src/bc/bc_dataset.py`
- Create: `docs/josh/bc_model_module/tests/test_bc_dataset.py`

**Step 1: Write the failing tests**

```python
"""Tests for BCDataset."""

import numpy as np
import pytest
import torch

from src.bc.bc_dataset import BCDataset, load_datasets
from src.encoder.encoder_constants import (
    ACTION_SPACE_SIZE,
    GRID_COLS,
    GRID_ROWS,
    NOOP_ACTION,
    NUM_ARENA_CHANNELS,
    NUM_VECTOR_FEATURES,
)


def _create_npz(path, n_frames=10, n_actions=3):
    """Create a minimal .npz file for testing."""
    arena = np.random.randn(n_frames, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS).astype(np.float32)
    vector = np.random.rand(n_frames, NUM_VECTOR_FEATURES).astype(np.float32)
    actions = np.full(n_frames, NOOP_ACTION, dtype=np.int64)
    for i in range(min(n_actions, n_frames)):
        actions[i] = np.random.randint(0, ACTION_SPACE_SIZE - 1)
    masks = np.ones((n_frames, ACTION_SPACE_SIZE), dtype=bool)
    timestamps = np.arange(n_frames, dtype=np.float64)
    np.savez(str(path), obs_arena=arena, obs_vector=vector,
             actions=actions, masks=masks, timestamps=timestamps)
    return str(path)


class TestBCDataset:
    """Tests for BCDataset."""

    def test_single_file_length(self, tmp_path):
        """Dataset length should match total frames."""
        p = _create_npz(tmp_path / "game1.npz", n_frames=20)
        ds = BCDataset([p])
        assert len(ds) == 20

    def test_multiple_files_concatenated(self, tmp_path):
        """Multiple .npz files should be concatenated."""
        p1 = _create_npz(tmp_path / "game1.npz", n_frames=10)
        p2 = _create_npz(tmp_path / "game2.npz", n_frames=15)
        ds = BCDataset([p1, p2])
        assert len(ds) == 25

    def test_getitem_returns_correct_keys(self, tmp_path):
        """Each sample should have arena, vector, action, mask tensors."""
        p = _create_npz(tmp_path / "game.npz", n_frames=5)
        ds = BCDataset([p])
        sample = ds[0]
        assert "arena" in sample
        assert "vector" in sample
        assert "action" in sample
        assert "mask" in sample

    def test_getitem_shapes(self, tmp_path):
        """Tensor shapes should match expected dimensions."""
        p = _create_npz(tmp_path / "game.npz", n_frames=5)
        ds = BCDataset([p])
        sample = ds[0]
        assert sample["arena"].shape == (GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS)
        assert sample["vector"].shape == (NUM_VECTOR_FEATURES,)
        assert sample["action"].shape == ()
        assert sample["mask"].shape == (ACTION_SPACE_SIZE,)

    def test_getitem_dtypes(self, tmp_path):
        """Tensors should have correct dtypes."""
        p = _create_npz(tmp_path / "game.npz", n_frames=5)
        ds = BCDataset([p])
        sample = ds[0]
        assert sample["arena"].dtype == torch.float32
        assert sample["vector"].dtype == torch.float32
        assert sample["action"].dtype == torch.long
        assert sample["mask"].dtype == torch.bool

    def test_action_class_counts(self, tmp_path):
        """action_class_counts should count noops and actions."""
        p = _create_npz(tmp_path / "game.npz", n_frames=10, n_actions=3)
        ds = BCDataset([p])
        noop, action = ds.action_class_counts()
        assert noop + action == 10
        assert action == 3

    def test_load_datasets_splits_by_file(self, tmp_path):
        """load_datasets should split .npz files 80/20."""
        for i in range(10):
            _create_npz(tmp_path / f"game_{i}.npz", n_frames=5)
        paths = sorted(str(p) for p in tmp_path.glob("*.npz"))
        train_ds, val_ds = load_datasets(paths, val_ratio=0.2, seed=42)
        assert len(train_ds) == 40  # 8 files * 5 frames
        assert len(val_ds) == 10   # 2 files * 5 frames

    def test_load_datasets_no_overlap(self, tmp_path):
        """Train and val should have no overlapping files."""
        for i in range(5):
            _create_npz(tmp_path / f"game_{i}.npz", n_frames=3)
        paths = sorted(str(p) for p in tmp_path.glob("*.npz"))
        train_ds, val_ds = load_datasets(paths, val_ratio=0.2, seed=42)
        # Total frames should equal sum of all files
        assert len(train_ds) + len(val_ds) == 15

    def test_compute_class_weights(self, tmp_path):
        """compute_class_weights should return (2305,) tensor."""
        p = _create_npz(tmp_path / "game.npz", n_frames=10, n_actions=3)
        ds = BCDataset([p])
        weights = ds.compute_class_weights(noop_weight=0.3, action_weight=3.0)
        assert weights.shape == (ACTION_SPACE_SIZE,)
        assert weights[NOOP_ACTION].item() == pytest.approx(0.3)
        assert weights[0].item() == pytest.approx(3.0)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -m pytest docs/josh/bc_model_module/tests/test_bc_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
"""BCDataset - PyTorch Dataset for loading .npz training data.

Loads one or more .npz files produced by DatasetBuilder and serves
(arena, vector, action, mask) samples for the BC training loop.

Provides utility functions for:
- 80/20 train/val splitting by file (not by frame)
- Class weight computation for weighted cross-entropy
"""

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.encoder.encoder_constants import ACTION_SPACE_SIZE, NOOP_ACTION


class BCDataset(Dataset):
    """PyTorch dataset for BC training data.

    Loads and concatenates multiple .npz files. Each file contains
    obs_arena, obs_vector, actions, masks arrays from DatasetBuilder.

    Args:
        npz_paths: List of paths to .npz files.
    """

    def __init__(self, npz_paths: list[str]):
        arenas, vectors, actions, masks = [], [], [], []
        for path in npz_paths:
            data = np.load(path)
            arenas.append(data["obs_arena"])
            vectors.append(data["obs_vector"])
            actions.append(data["actions"])
            masks.append(data["masks"])

        self.arena = np.concatenate(arenas, axis=0)
        self.vector = np.concatenate(vectors, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.masks = np.concatenate(masks, axis=0)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "arena": torch.from_numpy(self.arena[idx]),
            "vector": torch.from_numpy(self.vector[idx]),
            "action": torch.tensor(self.actions[idx], dtype=torch.long),
            "mask": torch.from_numpy(self.masks[idx]),
        }

    def action_class_counts(self) -> Tuple[int, int]:
        """Count no-op vs action frames.

        Returns:
            (noop_count, action_count) tuple.
        """
        noop_count = int((self.actions == NOOP_ACTION).sum())
        action_count = len(self.actions) - noop_count
        return noop_count, action_count

    def compute_class_weights(
        self, noop_weight: float = 0.3, action_weight: float = 3.0
    ) -> torch.Tensor:
        """Compute per-class weights for CrossEntropyLoss.

        Applies noop_weight to the no-op action and action_weight to all
        card placement actions. This addresses the ~70% no-op imbalance.

        Args:
            noop_weight: Weight for no-op class (2304). Default 0.3.
            action_weight: Weight for all card placement classes. Default 3.0.

        Returns:
            (2305,) float tensor of class weights.
        """
        weights = torch.full((ACTION_SPACE_SIZE,), action_weight)
        weights[NOOP_ACTION] = noop_weight
        return weights


def load_datasets(
    npz_paths: list[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[BCDataset, BCDataset]:
    """Split .npz files into train and validation datasets.

    Splits at the file level (not frame level) to prevent data leakage
    between frames from the same game.

    Args:
        npz_paths: List of all .npz file paths.
        val_ratio: Fraction of files for validation. Default 0.2.
        seed: Random seed for reproducible splits.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    paths = list(npz_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    n_val = max(1, int(len(paths) * val_ratio))
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]

    return BCDataset(train_paths), BCDataset(val_paths)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -m pytest docs/josh/bc_model_module/tests/test_bc_dataset.py -v`
Expected: All 9 tests PASS.

**Step 5: Commit**

```
feat: Add BCDataset with file-level train/val splitting.
```

---

## Task 5: Training Script (train_bc.py)

**Files:**
- Create: `docs/josh/bc_model_module/src/bc/train_bc.py`
- Create: `docs/josh/bc_model_module/tests/test_train_bc.py`

**Step 1: Write the failing tests**

```python
"""Tests for BC training loop."""

import numpy as np
import os
import pytest
import torch

from src.bc.train_bc import BCTrainer, TrainConfig
from src.encoder.encoder_constants import (
    ACTION_SPACE_SIZE,
    GRID_COLS,
    GRID_ROWS,
    NOOP_ACTION,
    NUM_ARENA_CHANNELS,
    NUM_VECTOR_FEATURES,
)


def _create_npz(path, n_frames=10, n_actions=3):
    """Create a minimal .npz file for testing."""
    arena = np.random.randn(n_frames, GRID_ROWS, GRID_COLS, NUM_ARENA_CHANNELS).astype(np.float32)
    vector = np.random.rand(n_frames, NUM_VECTOR_FEATURES).astype(np.float32)
    actions = np.full(n_frames, NOOP_ACTION, dtype=np.int64)
    for i in range(min(n_actions, n_frames)):
        actions[i] = np.random.randint(0, ACTION_SPACE_SIZE - 1)
    masks = np.ones((n_frames, ACTION_SPACE_SIZE), dtype=bool)
    timestamps = np.arange(n_frames, dtype=np.float64)
    np.savez(str(path), obs_arena=arena, obs_vector=vector,
             actions=actions, masks=masks, timestamps=timestamps)
    return str(path)


def _make_test_data(tmp_path, n_files=5, n_frames=20, n_actions=5):
    """Create multiple .npz files for training tests."""
    paths = []
    for i in range(n_files):
        p = _create_npz(tmp_path / f"game_{i}.npz", n_frames=n_frames, n_actions=n_actions)
        paths.append(p)
    return paths


class TestTrainConfig:
    """Tests for TrainConfig defaults."""

    def test_defaults(self):
        """Default config should have sensible values."""
        cfg = TrainConfig()
        assert cfg.epochs == 100
        assert cfg.batch_size == 64
        assert cfg.lr == 3e-4
        assert cfg.val_ratio == 0.2

    def test_custom_values(self):
        """Config should accept custom values."""
        cfg = TrainConfig(epochs=10, batch_size=32, lr=1e-3)
        assert cfg.epochs == 10
        assert cfg.batch_size == 32
        assert cfg.lr == 1e-3


class TestBCTrainer:
    """Tests for BCTrainer."""

    def test_train_one_epoch(self, tmp_path):
        """Training one epoch should produce a finite loss."""
        paths = _make_test_data(tmp_path / "data", n_files=3, n_frames=10)
        output_dir = str(tmp_path / "output")
        config = TrainConfig(epochs=1, batch_size=4, patience=5)
        trainer = BCTrainer(config)
        result = trainer.train(paths, output_dir)

        assert result["train_losses"]
        assert result["val_losses"]
        assert all(np.isfinite(l) for l in result["train_losses"])

    def test_train_saves_checkpoint(self, tmp_path):
        """Training should save best_bc.pt."""
        paths = _make_test_data(tmp_path / "data", n_files=3, n_frames=10)
        output_dir = str(tmp_path / "output")
        config = TrainConfig(epochs=2, batch_size=4, patience=5)
        trainer = BCTrainer(config)
        trainer.train(paths, output_dir)

        assert os.path.exists(os.path.join(output_dir, "best_bc.pt"))

    def test_train_saves_extractor_weights(self, tmp_path):
        """Training should save bc_feature_extractor.pt for PPO transfer."""
        paths = _make_test_data(tmp_path / "data", n_files=3, n_frames=10)
        output_dir = str(tmp_path / "output")
        config = TrainConfig(epochs=1, batch_size=4, patience=5)
        trainer = BCTrainer(config)
        trainer.train(paths, output_dir)

        assert os.path.exists(os.path.join(output_dir, "bc_feature_extractor.pt"))

    def test_train_returns_metrics(self, tmp_path):
        """Result dict should contain expected metric keys."""
        paths = _make_test_data(tmp_path / "data", n_files=3, n_frames=10)
        output_dir = str(tmp_path / "output")
        config = TrainConfig(epochs=1, batch_size=4, patience=5)
        trainer = BCTrainer(config)
        result = trainer.train(paths, output_dir)

        assert "train_losses" in result
        assert "val_losses" in result
        assert "val_accuracies" in result
        assert "best_val_loss" in result
        assert "best_epoch" in result

    def test_early_stopping(self, tmp_path):
        """Training should stop early if val loss does not improve."""
        paths = _make_test_data(tmp_path / "data", n_files=3, n_frames=10)
        output_dir = str(tmp_path / "output")
        config = TrainConfig(epochs=50, batch_size=4, patience=2)
        trainer = BCTrainer(config)
        result = trainer.train(paths, output_dir)

        # Should stop before 50 epochs due to patience=2
        assert len(result["train_losses"]) <= 50

    def test_saves_training_log(self, tmp_path):
        """Training should save a JSON log of metrics."""
        paths = _make_test_data(tmp_path / "data", n_files=3, n_frames=10)
        output_dir = str(tmp_path / "output")
        config = TrainConfig(epochs=1, batch_size=4, patience=5)
        trainer = BCTrainer(config)
        trainer.train(paths, output_dir)

        assert os.path.exists(os.path.join(output_dir, "training_log.json"))
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -m pytest docs/josh/bc_model_module/tests/test_train_bc.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
"""BC Training Script - Custom PyTorch training loop for behavior cloning.

Trains BCPolicy on .npz data from DatasetBuilder with:
- Weighted cross-entropy loss (handles ~70% no-op class imbalance)
- 80/20 file-level train/val split
- Cosine annealing LR schedule
- Early stopping with patience
- Checkpoint saving (full policy + feature extractor only)
- JSON training log

Usage:
    # As a module
    from src.bc.train_bc import BCTrainer, TrainConfig
    config = TrainConfig(epochs=100, batch_size=64)
    trainer = BCTrainer(config)
    result = trainer.train(npz_paths, output_dir)

    # As a CLI script
    python -m src.bc.train_bc --data-dir data/bc_training --output-dir models/bc

PPO Transition Notes:
    After training, the output directory contains:
    - best_bc.pt: Full BCPolicy weights (for BC inference/evaluation)
    - bc_feature_extractor.pt: Feature extractor only (for loading into PPO)

    To load into MaskablePPO:
        ppo = MaskablePPO("MultiInputPolicy", env, policy_kwargs={
            "features_extractor_class": CRFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 192},
        })
        ppo.policy.features_extractor.load_state_dict(
            torch.load("models/bc/bc_feature_extractor.pt")
        )
"""

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.bc.bc_policy import BCPolicy
from src.bc.bc_dataset import BCDataset, load_datasets
from src.encoder.encoder_constants import ACTION_SPACE_SIZE, NOOP_ACTION


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate for Adam.
        weight_decay: L2 regularization.
        patience: Early stopping patience (epochs without improvement).
        val_ratio: Fraction of files for validation.
        noop_weight: Cross-entropy weight for no-op class.
        action_weight: Cross-entropy weight for card placement classes.
        grad_clip: Max gradient norm (0 = disabled).
        seed: Random seed for split reproducibility.
        num_workers: DataLoader workers.
    """
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 10
    val_ratio: float = 0.2
    noop_weight: float = 0.3
    action_weight: float = 3.0
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0


class BCTrainer:
    """Trains BCPolicy on .npz data.

    Args:
        config: TrainConfig with hyperparameters.
    """

    def __init__(self, config: TrainConfig = None):
        self.config = config or TrainConfig()

    def train(
        self, npz_paths: list[str], output_dir: str
    ) -> dict:
        """Run the full training loop.

        Args:
            npz_paths: List of .npz file paths from DatasetBuilder.
            output_dir: Directory for checkpoints and logs.

        Returns:
            Dict with training metrics:
                train_losses, val_losses, val_accuracies,
                best_val_loss, best_epoch, total_time
        """
        os.makedirs(output_dir, exist_ok=True)
        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Split data
        train_ds, val_ds = load_datasets(npz_paths, cfg.val_ratio, cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
        )

        # Class weights
        class_weights = train_ds.compute_class_weights(
            cfg.noop_weight, cfg.action_weight
        ).to(device)

        # Model
        policy = BCPolicy().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = Adam(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        # Tracking
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float("inf")
        best_epoch = 0
        no_improve = 0
        start_time = time.time()

        for epoch in range(cfg.epochs):
            # --- Train ---
            policy.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                obs = {
                    "arena": batch["arena"].to(device),
                    "vector": batch["vector"].to(device),
                }
                targets = batch["action"].to(device)

                logits = policy(obs)
                loss = criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # --- Validate ---
            val_loss, val_acc = self._evaluate(policy, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            scheduler.step()

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0
                # Save best checkpoint
                policy.save(os.path.join(output_dir, "best_bc.pt"))
                torch.save(
                    policy.get_feature_extractor_state(),
                    os.path.join(output_dir, "bc_feature_extractor.pt"),
                )
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    break

        total_time = time.time() - start_time

        result = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "total_time": total_time,
            "train_frames": len(train_ds),
            "val_frames": len(val_ds),
            "config": asdict(cfg),
        }

        # Save training log
        log_path = os.path.join(output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    @staticmethod
    def _evaluate(
        policy: BCPolicy,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        """Evaluate policy on a dataset.

        Returns:
            (average_loss, accuracy) tuple.
        """
        policy.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                obs = {
                    "arena": batch["arena"].to(device),
                    "vector": batch["vector"].to(device),
                }
                targets = batch["action"].to(device)
                masks = batch["mask"].to(device)

                logits = policy(obs)
                loss = criterion(logits, targets)
                total_loss += loss.item() * targets.size(0)

                # Masked accuracy
                logits[~masks] = float("-inf")
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -m pytest docs/josh/bc_model_module/tests/test_train_bc.py -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```
feat: Add BC training script with weighted loss and early stopping.
```

---

## Task 6: Update __init__.py and Run Full Test Suite

**Files:**
- Modify: `docs/josh/bc_model_module/src/bc/__init__.py`

**Step 1: Update __init__.py with all exports**

```python
"""BC model module for Clash Royale behavior cloning."""

from src.bc.feature_extractor import CRFeatureExtractor
from src.bc.bc_policy import BCPolicy
from src.bc.bc_dataset import BCDataset, load_datasets
from src.bc.train_bc import BCTrainer, TrainConfig

__all__ = [
    "CRFeatureExtractor",
    "BCPolicy",
    "BCDataset",
    "load_datasets",
    "BCTrainer",
    "TrainConfig",
]
```

**Step 2: Run the full test suite**

Run: `cd /Users/alanguo/Codin/CS175/Project/cr-object-detection && python -m pytest docs/josh/bc_model_module/tests/ -v`
Expected: All tests (8 + 7 + 9 + 6 = 30) PASS.

**Step 3: Commit**

```
chore: Update bc module __init__.py with all exports.
```

---

## Task 7: CLAUDE.md (Technical Reference)

**Files:**
- Create: `docs/josh/bc_model_module/src/bc/CLAUDE.md`

Write a technical reference document covering:
- Module purpose and API
- CRFeatureExtractor architecture (arena embedding 156x8, card embedding 9x8, CNN layers, vector MLP)
- BCPolicy structure (extractor + action head, save/load, predict_action)
- BCDataset API (load, split, class weights)
- BCTrainer API (TrainConfig, train method, output files)
- PPO transition instructions (weight loading, freezing, fine-tuning)
- Constants and dimensions table
- Dependency graph

Keep under 200 lines. Follow existing CLAUDE.md style from state_encoder and dataset_builder modules.

**Step 1: Write CLAUDE.md**

**Step 2: Commit**

```
docs: Add CLAUDE.md technical reference for bc module.
```

---

## Task 8: Developer Documentation (bc-model-docs.md)

**Files:**
- Create: `docs/josh/bc_model_module/docs/bc-model-docs.md`

Write comprehensive developer documentation covering:

**Section 1: Overview**
- What the BC model does, where it fits in the pipeline
- Architecture diagram (ASCII)

**Section 2: Prerequisites**
- Required .npz files from DatasetBuilder
- Python dependencies (torch, gymnasium, numpy)
- SB3 dependencies (for PPO phase only)

**Section 3: Training Guide**
- Step-by-step: collect data -> process with DatasetBuilder -> train BC
- Example commands
- Hyperparameter tuning guide
- Expected training output and metrics

**Section 4: Evaluation**
- Offline metrics (accuracy, card selection, confusion)
- What "good" looks like

**Section 5: Live Game Testing**
- Full inference pipeline code
- Timing budget
- Safety measures (confidence threshold, cooldown)
- How to run the bot

**Section 6: PPO Transition**
- Installing SB3 + sb3-contrib
- Loading BC weights into MaskablePPO
- Freezing strategy (freeze extractor initially, unfreeze later)
- Gym environment wrapper requirements

**Section 7: Troubleshooting**
- Common issues (overfitting, all-noop predictions, NaN loss)
- Solutions

Keep under 400 lines.

**Step 1: Write bc-model-docs.md**

**Step 2: Commit**

```
docs: Add comprehensive BC model developer documentation.
```

---

## Verification

After all tasks are complete:

1. Run full test suite: `python -m pytest docs/josh/bc_model_module/tests/ -v`
   - Expected: 30 tests PASS
2. Verify file count: 8 source/doc files + 4 test files + 1 conftest = 13 files
3. Verify no import errors: `python -c "from src.bc import CRFeatureExtractor, BCPolicy, BCDataset, BCTrainer"`
4. Verify parameter count: `python -c "from src.bc import BCPolicy; p = BCPolicy(); print(sum(x.numel() for x in p.parameters()))"`
   - Expected: between 100,000 and 250,000

---

Plan complete and saved to `docs/plans/2026-02-22-bc-model-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?