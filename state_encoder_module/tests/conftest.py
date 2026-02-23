"""
Pytest conftest for josh's state_encoder_module tests.

Injects the josh module's src/ directory at the front of src.__path__
so that `from src.encoder import ...` resolves to the josh copy rather
than the real codebase's encoder. Other subpackages (src.pipeline,
src.generation) still resolve from the real codebase.
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

_josh_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _josh_src not in src.__path__:
    src.__path__.insert(0, _josh_src)
