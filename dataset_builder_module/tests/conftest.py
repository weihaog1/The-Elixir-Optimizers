"""Pytest conftest for josh's dataset_builder_module tests.

Injects the dataset_builder, state_encoder, and action_builder module
src/ directories at the front of src.__path__ so that:
  - `from src.dataset import ...` resolves to the josh dataset copy
  - `from src.encoder import ...` resolves to the josh encoder copy
  - `from src.action import ...` resolves to the josh action copy
Other subpackages (src.pipeline, src.generation, src.data, src.classification)
resolve from the real codebase.
"""

import os
import sys

# Ensure the cr-object-detection root is on sys.path
_repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import src  # noqa: E402

# Add state_encoder_module src so encoder imports resolve to josh's copy
_encoder_src = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "state_encoder_module", "src")
)
if _encoder_src not in src.__path__:
    src.__path__.insert(0, _encoder_src)

# Add action_builder_module src so action imports resolve to josh's copy
_action_src = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "action_builder_module", "src")
)
if _action_src not in src.__path__:
    src.__path__.insert(0, _action_src)

# Add dataset_builder_module src so dataset imports resolve to josh's copy
_dataset_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _dataset_src not in src.__path__:
    src.__path__.insert(0, _dataset_src)
