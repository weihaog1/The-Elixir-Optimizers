#!/usr/bin/env python3
"""
Train the BC model on .npz demonstration data.

Entry point script with namespace package setup to properly resolve
src.bc, src.encoder, and src.generation imports.

Usage:
    python bc_model_module/train_model.py --data_dir data/bc_training/ --output_dir models/bc/
    python bc_model_module/train_model.py --data_dir data/bc_training/ --epochs 50 --batch_size 32
    python bc_model_module/train_model.py --help
"""

import os
import sys
import types as _types

# ---------------------------------------------------------------------------
# Namespace package setup for src.* imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

_bc_src = os.path.join(_SCRIPT_DIR, "src")
_encoder_src = os.path.join(PROJECT_ROOT, "state_encoder_module", "src")
_action_src = os.path.join(PROJECT_ROOT, "action_builder_module", "src")

# Pre-register src as namespace package
_src_pkg = _types.ModuleType("src")
_src_pkg.__path__ = [_bc_src, _encoder_src, _action_src]
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

# Pre-register src.bc to bypass __init__.py (eagerly imports everything)
_bc_pkg = _types.ModuleType("src.bc")
_bc_pkg.__path__ = [os.path.join(_bc_src, "bc")]
_bc_pkg.__package__ = "src.bc"
sys.modules["src.bc"] = _bc_pkg

# Pre-register src.encoder to bypass __init__.py (eagerly imports gymnasium)
_enc_pkg = _types.ModuleType("src.encoder")
_enc_pkg.__path__ = [os.path.join(_encoder_src, "encoder")]
_enc_pkg.__package__ = "src.encoder"
sys.modules["src.encoder"] = _enc_pkg

# Pre-register src.generation from main codebase (for label_list.py)
_main_src = os.path.join(PROJECT_ROOT, "src", "src")
_gen_pkg = _types.ModuleType("src.generation")
_gen_pkg.__path__ = [os.path.join(_main_src, "generation")]
_gen_pkg.__package__ = "src.generation"
sys.modules["src.generation"] = _gen_pkg


def main():
    """Delegate to BCTrainer CLI."""
    from src.bc.train_bc import main as train_main
    train_main()


if __name__ == "__main__":
    main()
