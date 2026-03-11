# KataCR Synthetic Data Pipeline - Comprehensive Analysis

How the CS175 Elixir Optimizers project generates synthetic training data for Clash Royale object detection, following KataCR's approach of training exclusively on generated images.

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Sprite Dataset (The Raw Materials)](#2-sprite-dataset-the-raw-materials)
3. [The Generator (Image Factory)](#3-the-generator-image-factory)
4. [Augmentation Pipeline](#4-augmentation-pipeline)
5. [Label Generation (YOLO Format)](#5-label-generation-yolo-format)
6. [Belonging System (Ally vs Enemy)](#6-belonging-system-ally-vs-enemy)
7. [Custom YOLOv8 Modifications](#7-custom-yolov8-modifications)
8. [Training Pipeline (End to End)](#8-training-pipeline-end-to-end)
9. [Validation (Real Gameplay Data)](#9-validation-real-gameplay-data)
10. [Results and Domain Gap](#10-results-and-domain-gap)
11. [Configuration Reference](#11-configuration-reference)
12. [File Map](#12-file-map)

---

## 1. The Core Idea

KataCR's key insight: **never train on real gameplay images**. Instead:

1. Extract individual unit sprites (transparent PNGs) from game assets
2. Composite those sprites onto arena backgrounds to create fake screenshots
3. Since you placed every sprite, you know the exact bounding box and class -- labels are free
4. Train YOLOv8 on these synthetic images
5. Validate on a separate set of real human-labeled gameplay frames

**Why this works:** Clash Royale has a fixed camera angle, consistent art style, and predictable unit sizes. Sprites composited onto the correct background look close enough to real gameplay that the model generalizes.

**Why this is powerful:**
- Unlimited training data (generate as many images as you want)
- Perfect labels (no human annotation errors)
- Control over class balance (sample rare units more often)
- Easy to add new units (just add sprite PNGs)

**The tradeoff:** A domain gap exists between synthetic and real images. The model plateaus around mAP50 = 0.804 because synthetic images lack lighting, shadows, animations, compression artifacts, and spell visual effects that appear in real gameplay.

---

## 2. Sprite Dataset (The Raw Materials)

**Location:** `Froked-KataCR-Clash-Royale-Detection-Dataset/images/segment/`

### 2.1 What's in the dataset

| Component | Count | Description |
|-----------|-------|-------------|
| Class directories | 153 | One per unit/entity type |
| Total sprite PNGs | 4,232 | Transparent cutouts (was 4,785 before ally cleanup) |
| Arena backgrounds | 28 | Full arena screenshots (JPG) |
| Ally sprites (`_0_`) | 605 | Blue/bottom side (was 1,158 before removing non-deck allies) |
| Enemy sprites (`_1_`) | 3,627 | Red/top side (unchanged) |

### 2.2 Sprite naming convention

```
{class-name}_{belong}_{id}.png
```

| Field | Example | Meaning |
|-------|---------|---------|
| `class-name` | `hog-rider` | Hyphenated unit name |
| `belong` | `0` or `1` | 0 = ally (blue), 1 = enemy (red) |
| `id` | `0000003` | Numeric identifier |

Some sprites include state tags: `archer_1_attack_0000012.png`, `guard_1_shield_0000005.png`

### 2.3 Sprite characteristics

- Format: RGBA PNG with transparent background
- Size: Highly variable (28x30 px for skeletons up to 114x152 px for king towers)
- Source: Manually extracted from gameplay recordings by the KataCR team
- Quality: Clean cutouts suitable for alpha compositing

### 2.4 The ally sprite gap

Most classes only have enemy sprites. Ally sprites exist mainly for:

- **KataCR's Hog 2.6 deck** (hog-rider, musketeer, ice-spirit, ice-golem, cannon, skeleton, the-log, fireball) - well represented with 19-95 ally sprites each
- **Towers** (king-tower, queen-tower, cannoneer-tower, dagger-duchess-tower) - always have both sides
- **UI elements** (bars, clock, elixir, emote, text) - side-neutral
- **Our RR Hogs deck** (royal-hog, royal-recruit, zappy, flying-machine, goblin-cage, barbarian-barrel, electro-spirit, arrows) - we added ally sprites in our fork

127 of 153 classes have zero ally sprites (was 98 before non-deck ally cleanup). This is intentional -- only our deck's cards ever appear as allies in real gameplay.

### 2.5 Arena backgrounds

28 background images (`background01.jpg` through `background28.jpg`) representing different arena skins. The project uses **only background15** (stone/railroad texture) because that matches the arena we play on in Google Play Games.

---

## 3. The Generator (Image Factory)

**Source:** `src/generation/generator.py` (~800 lines)

### 3.1 High-level flow

```
Generator.reset()        Load fresh background image
       |
Generator.add_tower()    Place 6 fixed towers
       |
Generator.add_unit(40)   Sample and place 40 random sprites
       |
Generator.build()        Composite everything, generate labels
       |
   (image, labels)       576x896 image + YOLO bounding boxes
```

### 3.2 The arena grid

The generator works on an 18x32 cell grid overlaid on the 568x896 pixel arena:

```
Cell size: 30.9 px wide x 25 px tall
Arena pixel bounds: (6, 64) to (562, 864)

     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
  0  [                  ENEMY KING                       ]  <- Row 0-4: enemy base
  1  [                                                   ]
  ...
  7  [  QT              ARENA              QT            ]  <- Queen towers ~row 7
  ...
 15  [=================== RIVER =========================]  <- Rows 15-16
 16  [=================== RIVER =========================]
 17  [                                                   ]  <- Row 17+: player deploy zone
  ...
 26  [  QT              ARENA              QT            ]  <- Queen towers ~row 27
  ...
 30  [                  ALLY KING                        ]  <- Row 28-31: player base
 31  [                                                   ]
```

### 3.3 Occupancy maps

Two probability maps control where units spawn:

**Ground map (`map_ground`):** 32x18 grid with weighted probabilities
- `0` = forbidden (outside arena, tower positions)
- `0.5` = low probability (edges near towers)
- `1` = normal (standard playable area)
- `2` = queen tower zones
- `3` = center arena (highest probability -- units cluster here in real games)

**Flying map (`map_fly`):** All cells set to 1 (flying units can appear anywhere)

After placing a unit, the map is updated dynamically: the placed cell's probability drops by 50%, and that probability spreads to a 5x5 neighborhood. This prevents unrealistic clumping.

### 3.4 Tower placement

Towers are placed at fixed positions every image (not random):

| Tower | Position (cells) | Notes |
|-------|------------------|-------|
| Enemy king | (9, 4.7) | 95% intact, 5% ruins |
| Enemy queen left | (3.5, 7.7) | 25% each: queen/cannoneer/dagger-duchess, 25% ruins |
| Enemy queen right | (14.5, 7.7) | Same distribution |
| Ally king | (9, 30.5) | 95% intact, 5% ruins |
| Ally queen left | (3.5, 26.7) | Same distribution |
| Ally queen right | (14.5, 26.7) | Same distribution |

### 3.5 Unit placement (the main loop)

For each of the 40 units per image:

1. **Select a class** using inverse-frequency weighting (rarer sprites sampled more often)
2. **Split into real units (75%) and noise units (25%)** -- noise units are sprites from classes NOT in `avail_names` (the YOLO config's class list). They are placed on the image but their bounding boxes are dropped (`drop=True`), so the model sees them visually but gets no label. This forces the model to learn NOT to detect them, reducing false positives. With all 155 classes in `avail_names`, almost nothing qualifies as noise; this mechanism is more impactful when training on a reduced class subset (e.g., the 8-class deck-only test)
3. **Pick ally or enemy** -- selects `_0_` or `_1_` sprite variant
4. **Sample a position** from the occupancy map (ground or flying, based on unit type)
5. **Add Gaussian noise** to the position: N(0, 0.2) clipped to [-0.5, 0.5] cells
6. **Load a random sprite** from that class's available PNGs
7. **Apply per-sprite augmentation** (flip, scale, color -- see Section 4)
8. **Create Unit object** which computes pixel bounding box from cell position
9. **Add components** (HP bars, level indicators) with configurable probability
10. **Update the occupancy map** (reduce probability at placement location)

### 3.6 NMS filtering (overlap removal)

After placing all units, an iterative NMS loop removes excessive overlaps:

1. Reverse the unit list (check foreground first)
2. For each unit, compute intersection ratio against an accumulated occupancy mask
3. Remove if overlap exceeds threshold:
   - Towers: remove if > 80% occluded
   - Bars: remove if > 50% occluded
   - Regular units: remove if > 50% occluded
4. Also remove associated components (bars, levels) of filtered units
5. Repeat until no more units are removed (convergence)

### 3.7 Compositing (drawing)

Units are drawn in layer order (back to front):

| Level | Contents | Drawn |
|-------|----------|-------|
| 0 | Spells, background items | First (behind everything) |
| 1 | Ground units, towers | Middle |
| 2 | Flying units | Above ground |
| 3 | UI elements (bars, clock, text) | Last (on top) |

Each sprite is alpha-composited onto the background using `PIL.Image.alpha_composite()`. The mask (alpha channel) ensures transparent regions show the background through.

### 3.8 Component system (bars, indicators)

Units can have associated "components" drawn near them:

| Component | Attached to | Probability |
|-----------|-------------|-------------|
| HP bar + bar-level | Ground/flying units | 40% |
| tower-bar | Queen/cannoneer/dagger towers | 100% |
| king-tower-bar OR king-tower-level | King tower | 50% (one or the other) |
| skeleton-king-bar + skeleton-king-skill | Skeleton king | 100% |
| clock + elixir | Clock (UI) | 100% |

Component placement uses configurable offsets relative to the parent unit's position, with random jitter for variety.

### 3.9 Visibility filtering

After compositing, units that are too small or too occluded are dropped:

- If visible area < 30% of original sprite area: drop
- If visible width or height < 6 pixels: drop
- Dropped units get zero bounding boxes (excluded from labels)

---

## 4. Augmentation Pipeline

### 4.1 Per-sprite augmentations

Applied to each sprite before compositing:

| Augmentation | Probability | Details |
|--------------|-------------|---------|
| Horizontal flip | 50% | Mirrors the sprite left-right. Skipped for bars, text, king-tower-level |
| Scale | 100% (elixir, clock only) | Resize to 50-100% of original. Other units unaffected |
| Stretch | 0% (disabled) | Would compress width to 50-80%. Currently off |
| Color filter | ~9% total | Tints sprite red/blue/golden/white/violet |
| Transparency | 0% (disabled) | Would reduce alpha to 150/255 |

**Color filter breakdown:**

| Color | Probability | Applied to |
|-------|-------------|------------|
| Red | 2% | Towers + most non-spell units |
| Blue | 2% | Towers + most non-spell units |
| Golden | 2% | Text + non-spell units |
| White | 2% | Clock + non-spell units + towers |
| Violet | 1% | Non-spell units + towers |

Color filters simulate in-game effects like rage (red/golden), freeze (blue/white), and clone (violet).

### 4.2 Per-image augmentations (generator level)

| Augmentation | Probability | Details |
|--------------|-------------|---------|
| Red arena tint | 50% | Composites a red overlay on the top half of the arena |

### 4.3 YOLO training augmentations (ultralytics level)

Applied by the YOLOv8 training pipeline after generation:

| Augmentation | Value | Notes |
|--------------|-------|-------|
| Mosaic | 0.0 | Disabled -- generator already composites |
| Mixup | 0.0 | Disabled |
| HSV hue shift | 0.015 | Slight color variation |
| HSV saturation | 0.7 | Moderate saturation jitter |
| HSV value | 0.4 | Moderate brightness jitter |
| Rotation | 5 deg | Slight rotation |
| Translation | 0.05 | Slight shift |
| Scale | 0.5 | 50-150% zoom |
| Horizontal flip | 0.5 | 50% flip chance |
| Vertical flip | 0.0 | Disabled |
| Erasing | 0.4 | Random rectangular erasure |

---

## 5. Label Generation (YOLO Format)

### 5.1 How labels are created

Since the generator places every sprite, bounding boxes are known exactly:

1. Each `Unit` object stores its pixel bounding box `(x1, y1, x2, y2)` after clipping and visibility filtering
2. `Generator.build()` converts pixel coords to normalized center format
3. Labels include class ID, bounding box, and belonging

### 5.2 Label formats

**Standard (5-column):**
```
class_id  center_x  center_y  width  height
```

**With belonging (6-column):**
```
class_id  center_x  center_y  width  height  belonging
```

All coordinates are normalized to [0, 1] relative to image dimensions.

### 5.3 Coordinate conversion

```python
# Cell position -> pixel position
pixel = cell * cell_size + grid_offset
# where cell_size = [30.9, 25.0] and grid_offset = [6, 64]

# Pixel bounding box -> normalized center format
cx = (x1 + x2) / 2 / img_width
cy = (y1 + y2) / 2 / img_height
w  = (x2 - x1) / img_width
h  = (y2 - y1) / img_height
```

---

## 6. Belonging System (Ally vs Enemy)

### 6.1 What belonging means

Every detection in Clash Royale has a "side": ally (your units, blue, bottom half) or enemy (opponent's units, red, top half). KataCR tracks this as a binary attribute called `belonging`:

- `0` = ally (player's side)
- `1` = enemy (opponent's side)

### 6.2 How belonging flows through the system

```
Sprite filename         {class}_0_{id}.png  (0 = ally)
       |
Generator               Stores as unit.states[0]
       |
Label output            6th column in YOLO labels
       |
CRDataset               Splits into cls=(class_id, belonging)
       |
CRDetectionLoss         Target scores: channels 0-154 = class one-hot,
                         channel 155 = belonging value
       |
Model prediction        nc=156 output channels (155 classes + 1 belonging)
       |
Custom NMS              Thresholds belonging at 0.5 -> binary 0 or 1
       |
Detection output        7-column: (x1, y1, x2, y2, conf, class_id, belonging)
```

### 6.3 Ally class restriction

When `ally_classes` is set (e.g., to your deck's cards), the generator restricts which classes can appear with `_0_` (ally) sprites. Enemy sprites are unrestricted -- any class can be an enemy. This means:

- Only your 8 deck cards + towers appear as allies
- Everything else is always enemy
- The model learns: "these specific classes can be ally OR enemy; everything else is always enemy"

---

## 7. Custom YOLOv8 Modifications

**Source:** `src/yolov8_custom/`

### 7.1 Architecture change

The only architectural change: **one extra output channel**.

```
Standard YOLOv8:  nc = 155 channels  (one per class)
Custom CR model:  nc = 156 channels  (155 classes + 1 belonging)
```

The backbone (CSPDarknet), neck (PANet/FPN), and detection head structure are completely unchanged. The belonging prediction piggybacks on the existing classification head.

### 7.2 CRDetectionModel

A minimal wrapper that swaps the loss function:

```python
class CRDetectionModel(DetectionModel):
    def init_criterion(self):
        return CRDetectionLoss(self)  # Only change: custom loss
```

### 7.3 CRDetectionLoss

Extends standard YOLOv8 loss to handle 2-column class labels `(class_id, belonging)`:

**Label preprocessing:**
```
Input:  (batch_idx, class_id, belonging, x, y, w, h)  -- 7 columns
Output: gt_labels = (class_id, belonging)  -- 2 columns
        gt_bboxes = (x1, y1, x2, y2)      -- 4 columns
```

**Target score construction:**
```python
target_scores = zeros(batch, num_anchors, 156)
target_scores[..., 0:155] = one_hot(class_id)    # Standard class targets
target_scores[..., 155]   = belonging_value       # Belonging target (0 or 1)
```

Both class and belonging are supervised jointly with BCE loss. The model learns to predict all 156 channels simultaneously.

**Key design decision:** The task-aligned assigner (which matches predictions to ground truth) uses only class IDs for matching, not belonging. Spatial IoU determines which prediction is responsible for which ground truth box. Belonging is just an additional attribute that each matched prediction must also get right.

### 7.4 Custom NMS

After the model outputs 156-channel predictions:

```python
box, cls, bel = prediction.split((4, 155, 1), dim=1)

conf, class_id = cls.max(dim=1)          # Best class and its confidence
belonging = (bel > 0.5).float()          # Binary threshold

# Standard IoU-based NMS applied to (box, conf)
# Belonging does NOT affect NMS -- boxes are deduplicated by spatial overlap only

output = (x1, y1, x2, y2, confidence, class_id, belonging)  # 7 columns
```

---

## 8. Training Pipeline (End to End)

### 8.1 Two generation modes

**On-the-fly (default):** Each training batch generates fresh synthetic images. Every epoch sees 20,000 unique images that never repeat.

```
CRDataset.__getitem__(index)
  -> Generator.reset()           # Fresh background
  -> Generator.add_tower()       # Place towers
  -> Generator.add_unit(40)      # Place 40 sprites
  -> Generator.build()           # Composite + labels
  -> return (image, labels)      # To YOLO training loop
```

**Pre-generated (--pregen flag):** Generate images to disk first, then train from files. Used when on-the-fly generation is a CPU bottleneck (e.g., on GPU servers where CPU cores are limited).

```bash
# Step 1: Generate 20k images (~1.4 GB)
python scripts/generate_dataset.py --num-images 20000 --output data/synthetic/train

# Step 2: Train from disk
python scripts/train_synthetic.py --pregen --epochs 50
```

### 8.2 Training command

```bash
python scripts/train_synthetic.py \
  --model yolov8s.pt \      # Base model (YOLOv8 small)
  --epochs 50 \              # Training epochs
  --batch 16 \               # Batch size
  --imgsz 960 \              # Input image size (letterbox padded)
  --workers 4 \              # Data loading workers
  --belonging                # Enable belonging prediction (nc=156)
```

### 8.3 Key training parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Model | YOLOv8s | 3.3x capacity over nano, good speed/accuracy balance |
| Image size | 960 | Larger than default 640 for small unit detection |
| Background | 15 only | Matches our actual gameplay arena |
| Units per image | 40 | Realistic density for a typical game frame |
| Noise ratio | 0.25 | 25% hard negatives for robustness |
| Mosaic | 0.0 | Disabled (generator does its own compositing) |
| Erasing | 0.4 | Simulates partial occlusion |
| Epoch size | 20,000 | Virtual -- each image is unique |

### 8.4 What the training loop sees

Each training batch contains images like:

```
576x896 pixel image containing:
  - 1 arena background (background15)
  - 6 towers at fixed positions (with some as ruins)
  - ~30 unit sprites (after NMS filtering removes overlaps from 40 placed)
  - ~10 component sprites (HP bars, level indicators)
  - Per-sprite augmentation (flips, color tints)
  - YOLO augmentation (HSV jitter, slight rotation, random erasing)

Labels: one row per visible unit
  class_id  cx  cy  w  h  [belonging]
```

---

## 9. Validation (Real Gameplay Data)

### 9.1 Validation set

**Location:** `Froked-KataCR-Clash-Royale-Detection-Dataset/images/part2/`

- 7,380 real gameplay frames from YouTube recordings
- 6,966 with bounding box annotations (Labelme JSON + YOLO TXT)
- 1,388 frames used as the validation split
- 23,364 total annotated instances (~17 detections per frame)
- Source: 3 different players across 18 recording sessions (2021-2024)

### 9.2 Validation label format (extended YOLO)

```
class_id  cx  cy  w  h  belong  s1  s2  s3  s4  s5  s6
```

Extra fields beyond standard YOLO:
- `belong`: 0 = ally, 1 = enemy
- `s1`: Movement state (0=norm, 1=attack, 2=deploy, 3=freeze, 4=dash)
- `s2`: Shield/charge state
- `s3`: Visibility
- `s4`: Rage
- `s5`: Slow
- `s6`: Heal/clone

### 9.3 The synthetic-vs-real gap

The model never sees real gameplay during training. Validation measures how well synthetic training generalizes:

| Metric | Value |
|--------|-------|
| mAP50 | 0.804 |
| mAP50-95 | 0.567 |
| Precision | 0.822 |
| Recall | 0.771 |
| Train/val cls loss gap | 4.7x |
| Train/val box loss gap | 3.9x |

---

## 10. Results and Domain Gap

### 10.1 What works well

**73 classes (47%) achieve AP50 >= 0.90.** These are mainly:

- Large, visually distinct units (towers, buildings, large troops)
- Well-represented classes with many sprite variants
- Units with distinctive colors/shapes (fire-spirit, heal-spirit, electro-wizard)
- Our deck's units (royal-hog: 0.995, musketeer: 0.979, fireball: 0.977)

### 10.2 What fails

**Critical failures (AP50 < 0.40):**

| Class | AP50 | Root Cause |
|-------|------|-----------|
| barbarian-barrel | 0.396 | Only 8 sprites in dataset |
| miner | 0.237 | Underground emergence not in static sprites |
| zap | 0.245 | Brief flash effect, 9 sprites |
| arrows | 0.190 | Brief visual effect, 6 sprites |
| wizard | 0.152 | Confused with ice-wizard (too similar) |
| knight-evolution | 0.122 | Few sprites, confused with base knight |

**Structural reasons for the domain gap:**

1. **Static sprites vs animated units** - Real units walk, attack, deploy. Sprites are single-frame cutouts
2. **No lighting/shadows** - Synthetic images lack 3D rendering effects
3. **No spell effects** - Zap flash, arrow rain, fireball trail are dynamic, not static
4. **No compression artifacts** - Real screenshots have JPEG compression, color banding
5. **No environmental context** - Real units interact with terrain, other units cast shadows
6. **Confusion clusters** - Visually similar units (wizard/ice-wizard, barbarian variants) hard to distinguish

### 10.3 The ceiling

mAP50 plateaued at 0.804 by epoch 10 and did not improve through epoch 50. The 4.7x classification loss gap confirms the model keeps learning synthetic patterns without generalizing further to real data. **More epochs will not help.** Closing the gap requires either real training data, fine-tuning, or architectural changes.

### 10.4 Impact on the RL agent

| Capability | Status | Impact |
|------------|--------|--------|
| Detect own deck cards | Strong (>0.85 AP50) | BC training works |
| Detect major opponent troops | Good (~0.70+ AP50) | Basic game state tracking |
| Detect spells | Poor (<0.40 AP50) | Cannot react to opponent spells |
| Detect small cycle cards | Weak (0.55-0.67 AP50) | 30-40% miss rate on cheap units |
| Distinguish similar units | Poor | Wizard family, barbarian family confused |
| Count elixir from detections | Unreliable | Small unit misses affect counting |

---

## 11. Configuration Reference

### 11.1 Generator parameters

| Parameter | Default | Location |
|-----------|---------|----------|
| `background_index` | 15 | `Generator.__init__` |
| `unit_nums` | 40 | `synthetic_dataset.py` |
| `noise_unit_ratio` | 0.25 | `synthetic_dataset.py` |
| `TRAIN_DATASIZE` | 20,000 | `synthetic_dataset.py` / `$CR_TRAIN_DATASIZE` |
| `IMG_SIZE` | (576, 896) | `synthetic_dataset.py` |
| `intersect_ratio_thre` | 0.5 | `synthetic_dataset.py` |
| `map_update_mode` | 'dynamic' | `synthetic_dataset.py` |
| `ally_classes` | None | `Generator.__init__` (set in belonging training) |
| `grid_size` | (18, 32) | `generation_config.py` |
| `arena_pixel_bounds` | (6, 64, 562, 864) | `generation_config.py` |

### 11.2 YAML configs

| Config | nc | Purpose |
|--------|-----|---------|
| `synthetic_data.yaml` | 155 | Standard training (no belonging) |
| `synthetic_belonging_data.yaml` | 156 | Belonging training (155 + padding_belong) |
| `cutout_test.yaml` | 8 | Deck-only test (8 cards) |
| `cutout_test_belong.yaml` | 9 | Deck-only with belonging |
| `dataset_reduced.yaml` | 155 | Reference (validation only) |

### 11.3 Class system

155 classes organized into categories:

| Category | Count | Examples |
|----------|-------|---------|
| Ground troops | ~90 | archer, knight, barbarian, hog-rider |
| Flying units | ~20 | baby-dragon, minion, balloon |
| Spells | 16 | fireball, zap, arrows, the-log |
| Towers | 4 | king-tower, queen-tower, cannoneer-tower, dagger-duchess-tower |
| Buildings | ~15 | cannon, tesla, inferno-tower, goblin-hut |
| UI elements | ~10 | bar, clock, emote, big-text, small-text |
| Decorative | ~15 | dirt, axe, evolution-symbol, background-items |

Full class list defined in `src/generation/label_list.py`.

### 11.4 Layer system

| Level | Contents | Draw Order |
|-------|----------|------------|
| 0 | Spells, background items | First (back) |
| 1 | Ground units, towers, buildings | Middle |
| 2 | Flying units | Above ground |
| 3 | UI (bars, clock, emote, text) | Last (front) |

---

## 12. File Map

### Generation code

```
src/generation/
  generator.py            Core generator (~800 lines): Unit class, compositing, NMS
  generation_config.py    Grid, maps, towers, augmentation tables, component configs
  synthetic_dataset.py    CRDataset (extends YOLODataset): on-the-fly or pre-gen loading
  label_list.py           155 class names, ground/flying/spell/tower lists
  state_list.py           Belonging + state encoding (movement, shield, rage, etc.)
  constant.py             Dataset path resolution, image region splits
  datapath_manager.py     File discovery with regex search
  plot_utils.py           Bounding box and grid visualization
```

### Custom YOLO

```
src/yolov8_custom/
  custom_model.py         CRDetectionModel, CRDetectionLoss, CRTaskAlignedAssigner
  custom_utils.py         Custom NMS with 7-column output (adds belonging)
  custom_validator.py     CRDetectionValidator for 6-column labels
  custom_predictor.py     CRDetectionPredictor, CRBoxes with side field
```

### Training scripts

```
scripts/
  train_synthetic.py      Primary training script (SyntheticTrainer class)
  train_belonging.py      Belonging-specific training with ally_classes
  train_cutout_test.py    8-class deck-only test training
  generate_dataset.py     Pre-generate synthetic images to disk
```

### Configs

```
configs/
  synthetic_data.yaml             Active training config (nc=155)
  synthetic_belonging_data.yaml   Belonging config (nc=156)
  cutout_test.yaml                8-class test config
  cutout_test_belong.yaml         8-class + belonging test config
  dataset_reduced.yaml            Validation-only reference
```

### Sprite dataset

```
Froked-KataCR-Clash-Royale-Detection-Dataset/
  images/
    segment/                      ~4,300 sprite PNGs across 153 classes
      backgrounds/                28 arena backgrounds
      {class-name}/               Transparent unit sprites
    part2/                        7,380 real gameplay frames (validation)
      ClashRoyale_detection.yaml  201-class YOLO config
      {video-name}/{episode}/     JPG + JSON + TXT per frame
    card_classification/          227 card images (Hog 2.6 only)
    elixir_classification/        526 elixir count images
  version_info/                   Sprite/annotation version history
```
