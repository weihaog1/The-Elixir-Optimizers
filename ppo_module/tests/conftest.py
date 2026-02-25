"""Test configuration for ppo_module.

Sets up namespace packages so that `from src.ppo ...`, `from src.bc ...`,
and `from src.encoder ...` resolve to module-local copies.
"""

import os
import sys
import types

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.abspath(os.path.join(_TEST_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))

_ppo_src = os.path.join(_MODULE_DIR, "src")
_bc_src = os.path.join(_PROJECT_ROOT, "bc_model_module", "src")
_encoder_src = os.path.join(_PROJECT_ROOT, "state_encoder_module", "src")
_action_src = os.path.join(_PROJECT_ROOT, "action_builder_module", "src")

_src_paths = [_ppo_src, _bc_src, _encoder_src, _action_src]

# Register src as namespace package
_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = _src_paths
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

# Register subpackages
_sub_pkgs = {
    "src.ppo": os.path.join(_ppo_src, "ppo"),
    "src.bc": os.path.join(_bc_src, "bc"),
    "src.encoder": os.path.join(_encoder_src, "encoder"),
}
for pkg_name, pkg_path in _sub_pkgs.items():
    mod = types.ModuleType(pkg_name)
    mod.__path__ = [pkg_path]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod

# Main codebase subpackages
_main_src = os.path.join(_PROJECT_ROOT, "src", "src")
for sub in ("generation", "detection", "classification", "data", "pipeline"):
    pkg_name = f"src.{sub}"
    mod = types.ModuleType(pkg_name)
    mod.__path__ = [os.path.join(_main_src, sub)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod
