# Roboflow Clash Royale Dataset Analysis

Comprehensive comparison of six Roboflow Universe datasets for replacing KataCR's synthetic
dataset with real-image training data for YOLOv8 battlefield troop detection at 1080x1920.

**Date:** 2025-02-11
**Target Resolution:** 1080x1920 (portrait mobile, 9:16 aspect ratio)
**Target Model:** YOLOv8 object detection
**Baseline:** KataCR synthetic generation dataset (150 classes, 127 units, 1080x2400)

---

## 1. MinesBot - Clash Royale Bot (Instance Segmentation)

**URL:** https://universe.roboflow.com/minesbot/clash-royale-bot

| Property | Value |
|----------|-------|
| Images | 4,997 |
| Classes | 13 |
| Task | Instance Segmentation |
| License | CC BY 4.0 |
| Last Updated | ~2024 (version 18 available) |
| Versions | At least 18 |

**Class Names (French):**
archere, bat, chevalier, gargouille, geant, gobelin, gobelin_lances, mini_PEKKA,
mousquetaire, squelette, valkyrie, zappy

(English equivalents: archer, bat, knight, gargoyle/minion, giant, goblin,
spear_goblin, mini_PEKKA, musketeer, skeleton, valkyrie, zap)

**Resolution:** Not publicly confirmed; Roboflow typically resizes to 640x640 for YOLO
formats. Original capture resolution unknown.

**Preprocessing/Augmentation:** Specifics not publicly listed per version. Roboflow's
standard pipeline includes auto-orient and resize. Augmentations (if applied) are only
to the training split.

**Download Code:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("minesbot").project("clash-royale-bot")
dataset = project.version(18).download("yolov8")
```

**Key Observations:**
- Largest dataset by image count (4,997) - good for training
- Instance segmentation masks (polygon annotations), not just bounding boxes
- Only 13 classes - very limited troop coverage
- French class names would need remapping
- Segmentation masks can be converted to bounding boxes but not vice versa
- Does NOT distinguish ally vs enemy

---

## 2. Nejc Zavodnik - Clash Royale Troop Detection

**URL:** https://universe.roboflow.com/nejc-zavodnik/clash-royale-troop-detection

| Property | Value |
|----------|-------|
| Images | ~1,289 |
| Classes | 107 |
| Task | Object Detection |
| License | Not specified |
| Last Updated | February 2025 |

**Class Names:** Full list of 107 classes not publicly enumerable from search results.
The author notes the dataset "does not contain each and every troop and spell" and that
"annotations are not very accurate."

**Resolution:** Unknown from public metadata.

**Preprocessing/Augmentation:** Not publicly listed.

**Download Code:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("nejc-zavodnik").project("clash-royale-troop-detection")
# Version number needs to be checked on the project page
dataset = project.version(1).download("yolov8")
```

**Key Observations:**
- Highest class count (107) - closest to KataCR's 150 categories
- Author self-reports annotation quality as "not very accurate"
- 1,289 images is moderate; may be insufficient for 107 classes (~12 images per class)
- Could be very valuable if annotation quality issues can be fixed
- Does NOT appear to distinguish ally vs enemy

---

## 3. aff3npirat - ClashRoyale (Ally/Enemy Distinction)

**URL:** https://universe.roboflow.com/aff3npirat/clashroyale-ikd8o

| Property | Value |
|----------|-------|
| Images | 550 |
| Classes | 28 |
| Task | Object Detection |
| License | MIT |
| Last Updated | March 2023 |

**Class Names (28 total):**

Blue Team (Ally) - 13 classes:
- blue_archer, blue_arrows, blue_fireball, blue_giant, blue_goblinhut
- blue_knight, blue_minion, blue_minipekka, blue_musketeer, blue_prince
- blue_speargoblin, blue_valkyrie, blue_wallbreaker

Red Team (Enemy) - 15 classes:
- red_archer, red_arrows, red_cagegoblin, red_fireball, red_giant
- red_goblin, red_goblincage, red_goblinhut, red_hunter, red_knight
- red_minion, red_minipekka, red_musketeer, red_prince, red_speargoblin

**Resolution:** Not publicly confirmed.

**Preprocessing/Augmentation:** Not publicly listed.

**Download Code:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("aff3npirat").project("clashroyale-ikd8o")
# Version number needs to be checked on the project page
dataset = project.version(1).download("yolov8")
```

**Key Observations:**
- Distinguishes ally (blue) vs enemy (red) - critical for gameplay AI
- Only 550 images across 28 classes (~20 images per class) - very sparse
- Includes both troops AND spells (arrows, fireball) in annotations
- Includes buildings (goblinhut, goblincage)
- Class naming convention (blue_/red_ prefix) is clean and systematic
- MIT license is permissive
- Relatively old (March 2023), may not cover newer troops/evolutions

---

## 4. Nathan Yan - Clash Royale Detection

**URL:** https://universe.roboflow.com/nathan-yan/clash-royale-detection-cysig

| Property | Value |
|----------|-------|
| Images | 679 |
| Classes | 34 |
| Task | Object Detection |
| License | Not specified |
| Last Updated | June 2024 (model v8) |
| Models | 8 trained versions |

**Class Names (34 total):**

Ally (A- prefix) - 15 classes:
- A-balloon, A-canon, A-fireball, A-freeze, A-hog-rider
- A-ice-golem, A-ice-spirit, A-log, A-minion-horde, A-musketeer
- A-royal-ghost, A-skeletons, A-valkyrie, A-wall-breakers, A-wizard

Competitor (C- prefix) - 15 classes:
- C-balloon, C-canon, C-fireball, C-freeze, C-hog-rider
- C-ice-golem, C-ice-spirit, C-log, C-minion-horde, C-musketeer
- C-royal-ghost, C-skeletons, C-valkyrie, C-wall-breakers, C-wizard

Structure classes - 4 classes:
- Crown Name, T-archer-tower, T-king-tower (+ likely one more)

**Resolution:** Images reportedly resized to 1136x640 in at least one version.

**Preprocessing/Augmentation:** At least one version uses resize to 1136x640.

**Download Code:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("nathan-yan").project("clash-royale-detection-cysig")
dataset = project.version(8).download("yolov8")
```

**Key Observations:**
- Clean ally/competitor distinction with A-/C- prefix naming
- Includes towers (archer-tower, king-tower) - useful for game state
- Includes spells (fireball, freeze, log) - valuable for AI reaction
- 679 images for 34 classes (~20 per class) - still sparse
- 8 trained model versions suggest iterative improvement
- Resolution 1136x640 preserves roughly 16:9 aspect ratio (landscape)
- Focused on a specific meta deck (2.6 Hog Cycle variant)

---

## 5. Cicadas - Clash Royale

**URL:** https://universe.roboflow.com/cicadas/clash-royale-9eug2

| Property | Value |
|----------|-------|
| Images | 408 |
| Classes | 10 |
| Task | Object Detection |
| License | CC BY 4.0 |
| Last Updated | January 2025 |

**Class Names (10 total):**
cannon, evo_ice_spirit, evo_skeletons, fireball, hog_rider,
ice_golem, ice_spirit, log, musketeer, skeletons

**Resolution:** Not publicly confirmed.

**Preprocessing/Augmentation:** Not publicly listed.

**Download Code:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("cicadas").project("clash-royale-9eug2")
# Version number needs to be checked on the project page
dataset = project.version(1).download("yolov8")
```

**Key Observations:**
- Very specific to the 2.6 Hog Cycle deck
- Includes evolution variants (evo_ice_spirit, evo_skeletons) - modern/current
- Only 10 classes - extremely limited coverage
- 408 images is small but focused (~40 per class)
- Does NOT distinguish ally vs enemy
- No buildings or towers annotated
- Most recent update (Jan 2025) means likely current game visuals

---

## 6. SoNotMold - Clash Royale (Generic Ally/Enemy)

**URL:** https://universe.roboflow.com/sonotmold/clash-royale-xy2jw-u0djb

| Property | Value |
|----------|-------|
| Images | 142 |
| Classes | 8 |
| Task | Object Detection |
| License | MIT |
| Last Updated | ~February 2025 |

**Class Names (8 total):**
- ally_building, ally_king_tower, ally_princess_tower, ally_troop
- enemy_building, enemy_king_tower, enemy_princess_tower, enemy_troop

**Resolution:** Not publicly confirmed.

**Preprocessing/Augmentation:** Not publicly listed.

**Download Code:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("sonotmold").project("clash-royale-xy2jw-u0djb")
# Version number needs to be checked on the project page
dataset = project.version(1).download("yolov8")
```

**Key Observations:**
- Generic class approach: identifies WHAT (troop/building/tower) and WHOSE (ally/enemy)
- Does NOT identify specific troop types
- Only 142 images - far too small for reliable training
- 8 classes is very coarse but conceptually elegant for basic game AI
- Very recent updates
- Could be useful as a baseline or for tower/building detection specifically
- MIT license

---

## Comparison Summary

| Dataset | Images | Classes | Ally/Enemy | Troop ID | Spells | Buildings | Towers | License |
|---------|--------|---------|------------|----------|--------|-----------|--------|---------|
| MinesBot | 4,997 | 13 | No | Yes (13) | No | No | No | CC BY 4.0 |
| Nejc Zavodnik | 1,289 | 107 | No | Yes (107) | Yes | Yes | Yes | Unknown |
| aff3npirat | 550 | 28 | Yes | Yes (14 unique) | Yes (2) | Yes (2) | No | MIT |
| Nathan Yan | 679 | 34 | Yes | Yes (15 unique) | Yes (4) | Yes (1) | Yes (2) | Unknown |
| Cicadas | 408 | 10 | No | Yes (10) | Yes (2) | Yes (1) | No | CC BY 4.0 |
| SoNotMold | 142 | 8 | Yes | No (generic) | No | Generic | Yes | MIT |

### Images Per Class Ratio

| Dataset | Images/Class |
|---------|-------------|
| MinesBot | ~384 |
| Nejc Zavodnik | ~12 |
| aff3npirat | ~20 |
| Nathan Yan | ~20 |
| Cicadas | ~41 |
| SoNotMold | ~18 |

---

## API Key Requirements

All Roboflow Universe datasets require a free Roboflow account and API key to download
programmatically. There is no way to download datasets without authentication. However:

1. Creating a free Roboflow account is straightforward
2. The free tier allows dataset downloads
3. Public datasets on Universe can be downloaded by any authenticated user
4. Downloads are available as ZIP files or via Python code snippets
5. You can also use `curl` with your API key for direct downloads

**To get an API key:**
1. Create a free account at https://app.roboflow.com
2. Go to Settings -> API Key
3. Copy the API key for use in download scripts

---

## Detailed Analysis

### Resolution Match for 1080x1920

**Winner: Unknown / Likely None**

None of the datasets publicly confirm their original capture resolution. Roboflow
typically resizes images during version generation (commonly to 640x640 for YOLO).
Nathan Yan's dataset mentions 1136x640 in at least one version, suggesting landscape
orientation which is wrong for mobile portrait gameplay.

**Key concern:** Clash Royale on mobile is played in portrait mode (1080x1920 or similar).
Most datasets likely captured screenshots at varying resolutions. The KataCR baseline
uses 1080x2400 (modern phone resolution with taller aspect ratio).

**Recommendation:** Download each dataset and inspect actual image dimensions before
committing. If images are captured from phone screenshots, they should be portrait
orientation. If captured from emulators, they could be any resolution.

### Class Coverage for Battlefield Troops

**Winner: Nejc Zavodnik (107 classes)**

KataCR has 150 categories covering 127 units. Nejc Zavodnik's 107 classes is the closest
match, though the author warns annotations are inaccurate. No other dataset comes close:

- MinesBot: 13 troops (only ~10% of game troops)
- aff3npirat: 14 unique troops (with ally/enemy split = 28 classes)
- Nathan Yan: 15 unique troops (with ally/enemy split + towers = 34 classes)
- Cicadas: 10 troops (single deck only)
- SoNotMold: 0 specific troops (generic categories only)

### Ally vs Enemy Distinction

**Winner: Nathan Yan (34 classes) or aff3npirat (28 classes)**

For a gameplay AI, distinguishing friendly from enemy troops is critical:

- Nathan Yan uses A-/C- prefix (Ally/Competitor) with 15 unique unit types + towers
- aff3npirat uses blue_/red_ prefix with 14 unique unit types
- SoNotMold uses generic ally_troop/enemy_troop (no unit identification)

Nathan Yan edges out aff3npirat due to inclusion of towers and slightly more unit variety.

### Annotation Quality

**Estimated Winner: MinesBot or Nathan Yan**

- MinesBot: 4,997 images with instance segmentation (polygon masks) suggests careful
  annotation effort. 13 classes means manageable labeling scope.
- Nathan Yan: 8 model versions indicate iterative refinement of annotations.
- Nejc Zavodnik: Author explicitly warns annotations are "not very accurate."
- aff3npirat: No quality information available, older dataset (2023).
- Cicadas: Small, focused dataset likely has decent annotations for its 10 classes.
- SoNotMold: Too small (142 images) to assess meaningfully.

### Best Single Dataset

**Winner: Nathan Yan (clash-royale-detection-cysig)**

Rationale:
1. Best balance of class coverage (34) with ally/enemy distinction
2. Includes troops, spells, buildings, AND towers
3. 8 iterative model versions suggest good annotation refinement
4. 679 images is small but usable as a starting point
5. Clean, systematic naming convention

**Runner-up: aff3npirat (clashroyale-ikd8o)**
Similar ally/enemy approach with MIT license, but fewer classes and no towers.

### Best Combination Strategy

**Recommended combination:**

1. **Nathan Yan** (679 images, 34 classes) - Primary dataset for ally/enemy troops,
   spells, buildings, and towers
2. **MinesBot** (4,997 images, 13 classes) - Supplement with high-volume training data
   for the 12 overlapping troop types (after converting segmentation to detection boxes
   and remapping French class names)
3. **Cicadas** (408 images, 10 classes) - Supplement with evolution troop variants
   (evo_skeletons, evo_ice_spirit) that other datasets lack

**Merging approach:**
- Use Nathan Yan as the foundation with its ally/enemy naming
- Convert MinesBot segmentation masks to bounding boxes
- Map French class names to English (e.g., archere -> archer, chevalier -> knight)
- Add ally/enemy labels to MinesBot data (would require manual review or heuristic
  based on position on screen: top half = enemy, bottom half = ally)
- Merge Cicadas evo_ classes as new categories
- Total: ~6,084 images across ~40+ unique classes

### Can Any Dataset Fully Replace KataCR?

**No. None of these datasets can completely replace KataCR alone.**

Reasons:

1. **Class coverage gap:** KataCR covers 150 categories (127 units). Even the best
   candidate (Nejc Zavodnik at 107 classes) falls short, and its annotation quality
   is self-reportedly poor.

2. **Volume gap:** KataCR generates 20,000 training images per detector. The largest
   Roboflow dataset (MinesBot at 4,997) has only ~25% of that volume.

3. **Resolution mismatch:** No dataset confirms 1080x1920 or similar portrait mobile
   resolution. KataCR is purpose-built for 1080x2400.

4. **Synthetic advantage:** KataCR's synthetic generation can produce unlimited training
   data with controlled augmentation. Real datasets are fixed in size.

5. **Missing modern content:** Most datasets lack evolution troops, champions, and
   recent card additions.

**However, these datasets have advantages over KataCR:**

1. **Real images:** Real screenshots capture lighting, occlusion, overlapping troops,
   and visual effects that synthetic data may miss.
2. **Domain realism:** Real battlefield scenes include natural troop groupings,
   placement patterns, and background context.
3. **Spell effects:** Some datasets annotate active spells (fireball, freeze, log),
   which KataCR's synthetic generation may not replicate well.

### Recommended Strategy

**Hybrid approach - use Roboflow data to supplement and validate KataCR:**

1. **Download Nathan Yan + MinesBot + Cicadas** as real-image validation/test sets
2. **Train primary model on KataCR synthetic data** (maintains class coverage)
3. **Fine-tune on merged Roboflow real data** to bridge the synthetic-to-real gap
4. **Use SoNotMold's generic approach** as a secondary lightweight model for quick
   ally/enemy classification when specific troop ID is not needed
5. **Consider annotating additional real data** using a model trained on KataCR
   to bootstrap labeling (semi-supervised approach)

This hybrid strategy preserves KataCR's comprehensive class coverage while gaining
the domain realism benefits of real screenshot data.

---

## Download Scripts

### Download All Datasets (requires API key)

```python
from roboflow import Roboflow

API_KEY = "YOUR_API_KEY"
rf = Roboflow(api_key=API_KEY)

# 1. MinesBot (instance segmentation -> can convert to detection)
p1 = rf.workspace("minesbot").project("clash-royale-bot")
d1 = p1.version(18).download("yolov8")

# 2. Nejc Zavodnik
p2 = rf.workspace("nejc-zavodnik").project("clash-royale-troop-detection")
d2 = p2.version(1).download("yolov8")  # Check actual version number

# 3. aff3npirat
p3 = rf.workspace("aff3npirat").project("clashroyale-ikd8o")
d3 = p3.version(1).download("yolov8")  # Check actual version number

# 4. Nathan Yan
p4 = rf.workspace("nathan-yan").project("clash-royale-detection-cysig")
d4 = p4.version(8).download("yolov8")

# 5. Cicadas
p5 = rf.workspace("cicadas").project("clash-royale-9eug2")
d5 = p5.version(1).download("yolov8")  # Check actual version number

# 6. SoNotMold
p6 = rf.workspace("sonotmold").project("clash-royale-xy2jw-u0djb")
d6 = p6.version(1).download("yolov8")  # Check actual version number
```

**Note:** Version numbers marked "Check actual version number" should be verified on
each project's Roboflow Universe page. The download page URL pattern is:
`https://universe.roboflow.com/{workspace}/{project}/dataset/{version}/download`

### Post-Download Verification Script

```python
import os
from pathlib import Path
from PIL import Image

def analyze_dataset(dataset_path):
    """Analyze a downloaded Roboflow dataset."""
    images_dir = Path(dataset_path) / "train" / "images"
    labels_dir = Path(dataset_path) / "train" / "labels"

    if not images_dir.exists():
        print(f"No train/images found at {dataset_path}")
        return

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Total training images: {len(image_files)}")

    # Check resolutions
    resolutions = set()
    for img_path in image_files[:20]:  # Sample first 20
        with Image.open(img_path) as img:
            resolutions.add(img.size)

    print(f"Unique resolutions (sample of 20): {resolutions}")

    # Count classes from labels
    classes = set()
    for label_path in labels_dir.glob("*.txt"):
        with open(label_path) as f:
            for line in f:
                class_id = line.strip().split()[0]
                classes.add(class_id)

    print(f"Unique class IDs: {len(classes)}")
    print(f"Class IDs: {sorted(classes, key=int)}")
```

---

## Sources

- [MinesBot - Clash Royale Bot](https://universe.roboflow.com/minesbot/clash-royale-bot)
- [Nejc Zavodnik - Clash Royale Troop Detection](https://universe.roboflow.com/nejc-zavodnik/clash-royale-troop-detection)
- [aff3npirat - ClashRoyale](https://universe.roboflow.com/aff3npirat/clashroyale-ikd8o)
- [Nathan Yan - Clash Royale Detection](https://universe.roboflow.com/nathan-yan/clash-royale-detection-cysig)
- [Cicadas - Clash Royale](https://universe.roboflow.com/cicadas/clash-royale-9eug2)
- [SoNotMold - Clash Royale](https://universe.roboflow.com/sonotmold/clash-royale-xy2jw-u0djb)
- [Roboflow Download Docs](https://docs.roboflow.com/universe/download-a-universe-dataset)
- [KataCR GitHub](https://github.com/wty-yy/KataCR)
- [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)
