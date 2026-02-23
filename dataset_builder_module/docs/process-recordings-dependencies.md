# process_recordings.py - Dependency Report

Missing and required dependencies for `dataset_builder_module/process_recordings.py`.

---

## Status Summary

| Component | Status | Blocker |
|-----------|--------|---------|
| CLI (`--help`) | Working | - |
| Session scanning (`find_session_dirs`) | Working | - |
| `src.encoder.encoder_constants` | Blocked | `src/__init__.py` chain + missing `src.generation.label_list` |
| `src.encoder.state_encoder` | Blocked | Above + missing `gymnasium` |
| `src.dataset.dataset_builder` | Blocked | Above (needs `encoder_constants`) |
| `src.dataset.card_integration` | Blocked | Missing `src.data.screen_regions`, `src.pipeline.game_state` |
| `src.pipeline.state_builder` | Blocked | Entire `src/pipeline/` package missing |
| `src.classification.card_classifier` | Blocked | Entire `src/classification/` package missing |

---

## 1. Root Blocker: `src/__init__.py` Import Chain

**Every** `from src.* import ...` statement fails with `No module named 'config'`.

**Chain of failure:**
1. Python resolves `src` package → finds `src/__init__.py` at the repo root
2. `src/__init__.py` line 6: `from .detector import ...`
3. `src/detector.py` line 19: `from config import ...` (bare import, not `from src.config`)
4. Python looks for top-level `config` module → not found → **`ModuleNotFoundError`**

**Root cause:** `detector.py` uses a bare `from config import` instead of `from src.config import` or a relative `from .config import`. This worked when scripts ran from inside `src/` (where `config.py` was on `sys.path` implicitly), but breaks when imported as a package from outside.

**Fix options:**
- **(A)** Change `src/detector.py` line 19 from `from config import (...)` to `from .config import (...)` (and same in `src/main.py` line 13)
- **(B)** Add `src/` to `sys.path` in process_recordings.py before any `src.*` imports: `sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))`
- **(C)** Guard `src/__init__.py` imports with `try/except ImportError: pass`

---

## 2. Missing Python Packages

### Required to run (must install)

| Package | Required By | Install |
|---------|------------|---------|
| `gymnasium` | `state_encoder_module/src/encoder/state_encoder.py` | `pip install gymnasium` |

### Required for perception pipeline (optional — script degrades gracefully)

| Package | Required By | Install |
|---------|------------|---------|
| `paddleocr` | `src/pipeline/ocr.py` (OCR for timer/elixir) | `pip install paddleocr` |

### Not needed by process_recordings.py

| Package | Belongs To | Notes |
|---------|-----------|-------|
| `pynput` | `click_logger` | Recording input, not processing |
| `pygetwindow` | `click_logger` | Window capture, not processing |
| `pyautogui` | `click_logger` | Mouse automation, not processing |

### Already installed (confirmed)

| Package | Version |
|---------|---------|
| `cv2` (opencv) | 4.12.0 |
| `numpy` | 2.2.6 |
| `torch` | 2.7.1+cu118 |
| `scipy` | 1.16.3 |
| `ultralytics` | 8.4.10 |
| `mss` | Installed |

---

## 3. Missing Source Modules

These `src.*` subpackages are imported by the script or its dependencies but **do not exist** on this machine:

| Module | Used By | Contains |
|--------|---------|----------|
| `src.generation.label_list` | `encoder_constants.py` | 155 unit class names (`unit_list`, `ground_unit_list`, etc.) |
| `src.pipeline` | `process_recordings.py`, `card_integration.py` | `StateBuilder`, `GameState`, `Tower`, `Unit`, `Card` |
| `src.classification` | `process_recordings.py` | `CardPredictor` (MiniResNet card classifier) |
| `src.data` | `card_integration.py` | `ScreenConfig` (screen region definitions) |

**Source:** These modules originate from the [KataCR](https://github.com/wty-yy/KataCR) project. The `label_list.py` specifically comes from `katacr/constants/label_list.py`.

---

## 4. Model Files

| Model | Path | Status |
|-------|------|--------|
| YOLO detector v1 | `src/models/detector1_v0.7.13.pt` | Present |
| YOLO detector v2 | `src/models/detector2_v0.7.13.pt` | Present |
| Card classifier | `models/card_classifier/card_classifier.pt` | **Missing** |

The script handles missing models gracefully — `create_state_builder()` returns `None` and processing continues with zero-filled observations.

---

## 5. Resolution Steps (Priority Order)

### Step 1: Fix the `src/__init__.py` blocker

This unblocks all downstream imports. Recommended fix — change `src/detector.py`:

```python
# Line 19: Change from bare import to relative import
# Before:
from config import (IMAGE_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, CAPTURE_FPS, UNIT_CATEGORIES)

# After:
from .config import (IMAGE_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, CAPTURE_FPS, UNIT_CATEGORIES)
```

Same change in `src/main.py` line 13.

### Step 2: Provide `src/generation/label_list.py`

Without this, `encoder_constants.py` cannot define `NUM_CLASSES`, `CLASS_NAME_TO_ID`, or `UNIT_TYPE_MAP`.

Options:
- Clone from KataCR: `katacr/constants/label_list.py`
- Create a local copy with the 155 unit class name lists

### Step 3: Install gymnasium

```bash
pip install gymnasium
```

### Step 4 (Optional): Install perception pipeline dependencies

```bash
pip install paddleocr
```

And provide the missing source modules (`src.pipeline`, `src.classification`, `src.data`). Without these, the script still runs but produces zero-filled observations instead of real detections.

---

## Import Dependency Graph

```
process_recordings.py
├── find_session_dirs()           ← stdlib only (works now)
├── create_state_builder()        ← deferred imports, try/except (graceful fallback)
│   ├── src.pipeline.state_builder.StateBuilder       ← MISSING MODULE
│   ├── src.classification.card_classifier.CardPredictor ← MISSING MODULE
│   └── src.dataset.card_integration.EnhancedStateBuilder
│       ├── src.data.screen_regions.ScreenConfig      ← MISSING MODULE
│       └── src.pipeline.game_state.Card/GameState    ← MISSING MODULE
├── src.encoder.state_encoder.StateEncoder
│   ├── gymnasium                                     ← MISSING PACKAGE
│   ├── src.encoder.encoder_constants
│   │   └── src.generation.label_list                 ← MISSING MODULE
│   ├── src.encoder.coord_utils
│   └── src.encoder.position_finder (scipy)           ← OK
└── src.dataset.dataset_builder.DatasetBuilder
    ├── cv2                                           ← OK
    ├── src.encoder.coord_utils                       ← blocked by encoder_constants
    └── src.encoder.encoder_constants                 ← blocked by label_list

ALL of the above are also blocked by src/__init__.py → detector.py → config
```
