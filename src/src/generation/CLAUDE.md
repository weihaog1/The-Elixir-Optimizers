# src/generation/ - Synthetic Data Generation

Ported from KataCR's `katacr/build_dataset/` and `katacr/yolov8/`. Generates synthetic Clash Royale training images by compositing sprite cutouts onto arena backgrounds. Every training image is unique -- generated on-the-fly during training, never saved to disk.

## How It Works

1. `Generator.reset()` loads an arena background image (background15 = stone/railroad)
2. `Generator.add_tower()` places king towers and queen towers at fixed positions with randomized variants (intact, damaged, ruin)
3. `Generator.add_unit(40)` places 40 random sprite units with:
   - Inverse frequency weighting (rare units sampled more often)
   - Collision detection via occupancy grid (avoids overlapping sprites)
   - Random scaling, flipping, color augmentation per sprite
   - 25% hard negatives from background items (noise_unit_ratio)
4. `Generator.build()` composites everything and returns (image, bboxes, visualization)

## File Dependency Chain

```
synthetic_dataset.py (CRDataset)
  -> generator.py (Generator) -- 826 lines, core compositing engine
       -> datapath_manager.py (PathManager) -- finds sprite PNGs in segment/ subdirs
       -> constant.py (path_dataset) -- resolves Clash-Royale-Detection-Dataset location
       -> label_list.py (unit2idx, idx2unit) -- 155 class name/index mappings
       -> state_list.py (state2idx, idx2state) -- unit state definitions
       -> generation_config.py -- maps, tower positions, augmentation configs, scale tables
       -> plot_utils.py -- visualization only (plot_box_PIL, plot_cells_PIL)
```

## Key Files

**synthetic_dataset.py** - `CRDataset` subclasses ultralytics `YOLODataset`. For training mode (`img_path=None`), overrides `get_image_and_label()` to call the Generator. For validation mode, falls through to parent class (loads from disk). This is how the on-the-fly pipeline integrates with ultralytics.

**generator.py** - The core engine (826 lines). Ported from `katacr/build_dataset/generator.py`. Key changes from KataCR: import paths updated to `src.generation.*`, JAX dependency removed, regex strings use raw strings. No logic changes.

**generation_config.py** - All config tables: `map_fly`/`map_ground` (occupancy grids), `level2units`/`unit2level` (unit size categories), `grid_size` (18x32 cells), `background_size` (568x896), tower positions, spell lists, augmentation probabilities, scale/stretch tables, component configs (bars, HP indicators).

**label_list.py** - `unit_list` (155 class names), `unit2idx`/`idx2unit` dicts, `invalid_unit_list` (excluded from training: selected, text, mirror, tesla-evolution-shock, zap-evolution).

**constant.py** - Resolves `path_dataset` via `CR_DATASET_PATH` env var or `parents[4] / "Clash-Royale-Detection-Dataset"`. On the remote instance (vast.ai), this resolves to `/workspace/Clash-Royale-Detection-Dataset` which is a symlink to `/workspace/project-alan/cr-detection-dataset`.

**datapath_manager.py** - `PathManager` wraps the sprite dataset directory. `search(subset, part, name, regex)` finds sprite files matching criteria. Used by Generator to sample random sprites per unit class.

**plot_utils.py** - 4 PIL visualization functions extracted from KataCR's `katacr/utils/detection/__init__.py`. Only used for debug visualization (Generator.build() returns a PIL image for inspection). Not called during training. Requires `fonts/Consolas.ttf`.

## Sprite Dataset Structure

```
Clash-Royale-Detection-Dataset/images/segment/
  archer/              # PNG sprites with transparency (multiple variants per unit)
  archer-evolution/
  backgrounds/         # background01.jpg through background26.jpg (arena skins)
  background-items/    # Tower ruins, decorations (used as hard negatives)
  ...154 subdirectories total, ~4,654 sprite files
```

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `background_index` | 15 | Arena background (15 = stone/railroad, matches our gameplay video) |
| `unit_nums` | 40 | Sprites composited per image |
| `noise_unit_ratio` | 0.25 | Fraction of sprites that are hard negatives |
| `intersect_ratio_thre` | 0.5 | Occlusion NMS threshold for sprite placement |
| `TRAIN_DATASIZE` | 20000 | Virtual epoch size (images per epoch) |
| `IMG_SIZE` | (576, 896) | Generator output resolution (w, h) |

## Evaluation Findings (v12, Feb 2026)

### Domain Gap
The synthetic-to-real domain gap is the primary bottleneck. Classification loss has a 4.7x train/val gap -- the biggest contributor is visual appearance differences between sprites and real in-game units. mAP50 plateaued at 0.804 by epoch 10 of 50. More training does not help.

### Sprite Dataset Imbalance
Classes with very few sprites are not necessarily bad (tombstone has 1 sprite, AP50=0.995). However, classes with few sprites AND visual complexity fail badly:
- barbarian-barrel: 8 sprites, AP50=0.396 (most-used spell in meta, critical miss)
- arrows: 6 sprites, AP50=0.190
- zap: 9 sprites, AP50=0.245
- goblin-cage: 4 sprites, AP50=0.431

### Classes with Many Sprites but Poor Detection
- wizard: 41 sprites, AP50=0.152 (confused with ice-wizard/e-wiz)
- miner: 46 sprites, AP50=0.237 (underground emergence doesn't match static sprite)
- mother-witch: 38 sprites, AP50=0.341
- bomber-evolution: 30 sprites, AP50=0.000

### Generator Limitations
- Small units (skeleton, spirits) are inherently hard at 576x896 resolution
- Spell effects (zap, arrows, freeze) are brief visual events -- single-frame static sprites can't capture the dynamic appearance
- Visually similar class families cause confusion (wizard/barbarian/spirit families)
