"""
Pytest conftest for josh's action_builder_module tests.

Injects both the action module and the encoder module's src/ directories
at the front of src.__path__ so that:
  - `from src.action import ...` resolves to the josh action copy
  - `from src.encoder import ...` resolves to the josh encoder copy
Other subpackages (src.pipeline, src.generation) resolve from the real codebase.
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
_action_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _action_src not in src.__path__:
    src.__path__.insert(0, _action_src)
