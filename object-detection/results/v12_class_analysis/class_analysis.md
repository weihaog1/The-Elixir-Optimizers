# Per-Class Detection Performance and Meta Deck Coverage Analysis

Model: `best_yolov8s_50epochs_fixed_pregen_set.pt` (YOLOv8s, 11.2M params, 50 epochs synthetic training)
Validation set: 1,388 human-labeled images from KataCR dataset
Overall: **mAP50 = 0.8056**, mAP50-95 = 0.5675

---

## 1. Per-Class AP50 Results (All 155 Classes)

### Tier 1: Strong Detection (AP50 >= 0.90) - 73 classes

| Idx | Class | AP50 | Sprites |
|-----|-------|------|---------|
| 0 | king-tower | 0.9948 | 76 |
| 1 | queen-tower | 0.9940 | 54 |
| 2 | cannoneer-tower | 0.9929 | 20 |
| 3 | dagger-duchess-tower | 0.9950 | 18 |
| 5 | tower-bar | 0.9910 | 46 |
| 6 | king-tower-bar | 0.9855 | 17 |
| 9 | clock | 0.9847 | 30 |
| 12 | elixir | 0.9378 | 32 |
| 14 | skeleton-king-bar | 0.9950 | 9 |
| 20 | heal-spirit | 0.9510 | 21 |
| 22 | spear-goblin | 0.9238 | 91 |
| 23 | bomber | 0.9950 | 29 |
| 24 | bat | 0.9385 | 95 |
| 25 | bat-evolution | 0.9200 | 59 |
| 27 | giant-snowball | 0.9950 | 11 |
| 32 | wall-breaker | 0.9950 | 20 |
| 33 | rage | 0.9050 | 9 |
| 37 | knight | 0.9162 | 53 |
| 40 | cannon | 0.9866 | 25 |
| 41 | skeleton-barrel | 0.9950 | 12 |
| 42 | firecracker | 0.9459 | 26 |
| 43 | firecracker-evolution | 0.9950 | 17 |
| 44 | royal-delivery | 0.9950 | 9 |
| 47 | tombstone | 0.9950 | 1 |
| 48 | mega-minion | 0.9818 | 25 |
| 49 | dart-goblin | 0.9950 | 21 |
| 51 | elixir-golem-big | 0.9950 | 11 |
| 52 | elixir-golem-mid | 0.9721 | 12 |
| 53 | elixir-golem-small | 0.9950 | 6 |
| 54 | goblin-barrel | 0.9817 | 21 |
| 57 | tornado | 0.9078 | 7 |
| 59 | dirt | 0.9950 | 30 |
| 64 | fisherman | 0.9533 | 30 |
| 65 | skeleton-dragon | 0.9786 | 101 |
| 66 | mortar | 0.9950 | 10 |
| 69 | fireball | 0.9766 | 37 |
| 71 | musketeer | 0.9787 | 87 |
| 74 | valkyrie | 0.9950 | 22 |
| 75 | battle-ram | 0.9950 | 21 |
| 76 | battle-ram-evolution | 0.9950 | 11 |
| 81 | battle-healer | 0.9950 | 31 |
| 84 | baby-dragon | 0.9201 | 35 |
| 85 | dark-prince | 0.9950 | 34 |
| 87 | poison | 0.9270 | 14 |
| 88 | hunter | 0.9315 | 22 |
| 91 | inferno-dragon | 0.9950 | 27 |
| 93 | phoenix-egg | 0.9950 | 7 |
| 94 | phoenix-small | 0.9193 | 14 |
| 96 | lumberjack | 0.9915 | 22 |
| 97 | night-witch | 0.9155 | 36 |
| 99 | hog | 0.9248 | 16 |
| 100 | golden-knight | 0.9684 | 45 |
| 101 | skeleton-king | 0.9950 | 56 |
| 102 | mighty-miner | 0.9769 | 16 |
| 103 | rascal-boy | 0.9690 | 52 |
| 104 | rascal-girl | 0.9636 | 98 |
| 105 | giant | 0.9950 | 23 |
| 106 | goblin-hut | 0.9950 | 3 |
| 107 | inferno-tower | 0.9483 | 9 |
| 109 | royal-hog | 0.9950 | 27 |
| 110 | witch | 0.9950 | 17 |
| 111 | balloon | 0.9950 | 8 |
| 112 | prince | 0.9043 | 58 |
| 113 | electro-dragon | 0.9950 | 25 |
| 116 | axe | 0.9950 | 18 |
| 118 | ram-rider | 0.9950 | 18 |
| 125 | rocket | 0.9950 | 21 |
| 126 | barbarian-hut | 0.9950 | 1 |
| 127 | elixir-collector | 0.9950 | 16 |
| 128 | giant-skeleton | 0.9950 | 15 |
| 129 | lightning | 0.9950 | 7 |
| 130 | goblin-giant | 0.9950 | 20 |
| 131 | x-bow | 0.9877 | 4 |
| 132 | sparky | 0.9903 | 53 |
| 134 | electro-giant | 0.9950 | 13 |
| 139 | golemite | 0.9924 | 26 |
| 141 | royal-guardian | 0.9950 | 26 |
| 144 | valkyrie-evolution | 0.9617 | 35 |
| 147 | evolution-symbol | 0.9950 | 29 |
| 148 | mirror | 0.9950 | 0 |
| 150 | goblin-ball | 0.9950 | 8 |
| 143 | ice-spirit-evolution | 0.9236 | 11 |
| 120 | archer-queen | 0.9189 | 32 |

### Tier 2: Adequate Detection (AP50 0.70-0.89) - 32 classes

| Idx | Class | AP50 | Sprites |
|-----|-------|------|---------|
| 18 | fire-spirit | 0.8780 | 29 |
| 34 | the-log | 0.8018 | 34 |
| 35 | archer | 0.8279 | 44 |
| 28 | ice-golem | 0.8097 | 54 |
| 39 | minion | 0.8083 | 44 |
| 48 | mega-minion | 0.8818 | 25 |
| 50 | earthquake | 0.8376 | 10 |
| 55 | guard | 0.7478 | 69 |
| 56 | clone | 0.8259 | 2 |
| 60 | princess | 0.8803 | 17 |
| 62 | royal-ghost | 0.7021 | 21 |
| 63 | bandit | 0.7238 | 36 |
| 70 | mini-pekka | 0.7561 | 26 |
| 73 | goblin-brawler | 0.8421 | 39 |
| 78 | bomb | 0.8514 | 10 |
| 80 | hog-rider | 0.8693 | 68 |
| 83 | zappy | 0.7092 | 30 |
| 86 | freeze | 0.8735 | 5 |
| 89 | goblin-drill | 0.7380 | 8 |
| 90 | electro-wizard | 0.8873 | 17 |
| 95 | magic-archer | 0.7629 | 26 |
| 114 | bowler | 0.8214 | 16 |
| 117 | cannon-cart | 0.7885 | 18 |
| 119 | graveyard | 0.8036 | 3 |
| 122 | royal-giant | 0.8588 | 10 |
| 133 | pekka | 0.8702 | 22 |
| 135 | mega-knight | 0.7752 | 60 |
| 136 | lava-hound | 0.8549 | 22 |
| 138 | golem | 0.7688 | 30 |
| 140 | little-prince | 0.7370 | 53 |
| 142 | archer-evolution | 0.8874 | 57 |
| 10 | emote | 0.7277 | 25 |
| 21 | goblin | 0.7139 | 63 |

### Tier 3: Weak Detection (AP50 0.40-0.69) - 20 classes

| Idx | Class | AP50 | Sprites |
|-----|-------|------|---------|
| 15 | skeleton | 0.6699 | 148 |
| 16 | skeleton-evolution | 0.5277 | 46 |
| 17 | electro-spirit | 0.5601 | 20 |
| 19 | ice-spirit | 0.6437 | 66 |
| 30 | barbarian | 0.4723 | 42 |
| 31 | barbarian-evolution | 0.4736 | 55 |
| 45 | royal-recruit | 0.6774 | 67 |
| 46 | royal-recruit-evolution | 0.7362 | 150 |
| 61 | ice-wizard | 0.5077 | 60 |
| 67 | mortar-evolution | 0.6409 | 11 |
| 68 | tesla | 0.4203 | 12 |
| 72 | goblin-cage | 0.4306 | 4 |
| 79 | flying-machine | 0.6187 | 25 |
| 82 | furnace | 0.6852 | 2 |
| 92 | phoenix-big | 0.4972 | 19 |
| 115 | executioner | 0.6653 | 20 |
| 123 | royal-giant-evolution | 0.5087 | 16 |
| 124 | elite-barbarian | 0.6651 | 34 |
| 137 | lava-pup | 0.6886 | 36 |
| 4 | dagger-duchess-tower-bar | 0.4245 | 22 |

### Tier 4: Very Weak / Failing Detection (AP50 < 0.40) - 14 classes

| Idx | Class | AP50 | Sprites |
|-----|-------|------|---------|
| 7 | bar | 0.3465 | 136 |
| 8 | bar-level | 0.0492 | 22 |
| 11 | text | 0.0000 | 0 |
| 13 | selected | 0.0000 | 0 |
| 26 | zap | 0.2451 | 9 |
| 29 | barbarian-barrel | 0.3957 | 8 |
| 36 | arrows | 0.1899 | 6 |
| 38 | knight-evolution | 0.1215 | 21 |
| 58 | miner | 0.2373 | 46 |
| 77 | bomb-tower | 0.3288 | 14 |
| 98 | mother-witch | 0.3413 | 38 |
| 108 | wizard | 0.1516 | 41 |
| 145 | bomber-evolution | 0.0000 | 30 |
| 146 | wall-breaker-evolution | 0.3686 | 7 |
| 149 | tesla-evolution | 0.0000 | 12 |

### Not evaluated (no validation instances) - 4 classes

| Idx | Class | Sprites |
|-----|-------|---------|
| 151 | skeleton-king-skill | 4 |
| 152 | tesla-evolution-shock | 0 |
| 153 | ice-spirit-evolution-symbol | 6 |
| 154 | zap-evolution | 0 |

---

## 2. Current Meta Deck Analysis (February 2026)

Sources: Dexerto, LDShop, LootBar (compiled top 20 unique archetypes)

### Top 20 Meta Decks and Their Troop/Spell Classes

| # | Deck Archetype | Cards (mapped to model classes) |
|---|---------------|--------------------------------|
| 1 | **2.6 Hog Cycle** | hog-rider, musketeer, ice-golem, skeleton, ice-spirit, cannon, fireball, the-log |
| 2 | **Royal Giant Fisherman** | royal-giant (+ evo), fisherman, royal-ghost, hunter, phoenix-big/small, electro-spirit, fireball, the-log |
| 3 | **Miner Balloon Control** | balloon, miner, musketeer, skeleton, ice-golem, bomb-tower, giant-snowball, barbarian-barrel |
| 4 | **LavaLoon** | lava-hound, balloon, mega-minion, baby-dragon, valkyrie, tombstone, fireball, zap |
| 5 | **Miner Rocket Control** | miner, rocket, knight, tesla, royal-delivery, the-log, electro-spirit, skeleton |
| 6 | **PEKKA Bridge Spam** | pekka, battle-ram (+ evo), bandit, royal-ghost, electro-wizard, magic-archer, zap, fireball |
| 7 | **Classic Log Bait** | goblin-barrel, princess, knight, inferno-tower, guard, rocket, the-log, ice-spirit |
| 8 | **Hog Earthquake** | hog-rider, mighty-miner, firecracker (+ evo), earthquake, tesla, the-log, ice-spirit, skeleton |
| 9 | **Golem Beatdown** | golem, night-witch, lumberjack, baby-dragon, mega-minion, tornado, lightning, barbarian-barrel |
| 10 | **Giant Double Prince** | giant, prince, dark-prince, electro-wizard, mega-minion, zap, fireball, phoenix-big/small |
| 11 | **Goblin Drill Poison** | goblin-drill, giant-snowball, knight, musketeer, poison, ice-spirit, guard, bomb-tower |
| 12 | **Royal Hogs Recruits (Fireball)** | royal-recruit, royal-hog, zappy, fireball, flying-machine, goblin-cage, arrows, barbarian-barrel |
| 13 | **Evo Hogs 3M** | royal-hog, royal-ghost, ice-golem, musketeer, fireball, minion, barbarian-barrel, heal-spirit |
| 14 | **Splashyard** | knight, graveyard, ice-wizard, baby-dragon, tombstone, poison, tornado, barbarian-barrel |
| 15 | **Miner Poison** | skeleton-king, miner, barbarian-barrel, poison, minion, wall-breaker, skeleton, bomb-tower |
| 16 | **Mortar Bait** | mortar, miner, archer, barbarian-barrel, skeleton, fireball, goblin, ice-spirit |
| 17 | **Hog EQ (Furnace)** | cannon, electro-spirit, earthquake, skeleton, mighty-miner, hog-rider, cannon, furnace |
| 18 | **E-Giant** | electro-giant, baby-dragon, dark-prince, bomber, goblin-hut, tornado, barbarian-barrel, bowler |
| 19 | **Lumberloon Freeze** | lumberjack, balloon, baby-dragon/electro-dragon, inferno-dragon, bowler, freeze, tornado, barbarian-barrel |
| 20 | **Mega Knight Sparky** | mega-knight, sparky, royal-ghost, princess, fisherman, the-log, electro-wizard, tornado |

---

## 3. Cross-Reference: Meta-Critical Classes with Detection Quality

### Frequency of appearance across top 20 meta decks (gameplay-relevant classes only)

| Class | Deck Appearances | AP50 | Status |
|-------|-----------------|------|--------|
| fireball | 8 | 0.977 | STRONG |
| barbarian-barrel | 8 | 0.396 | **FAILING** |
| skeleton | 7 | 0.670 | WEAK |
| ice-spirit | 6 | 0.644 | WEAK |
| the-log | 5 | 0.802 | ADEQUATE |
| knight | 5 | 0.916 | STRONG |
| tornado | 5 | 0.908 | STRONG |
| musketeer | 4 | 0.979 | STRONG |
| baby-dragon | 4 | 0.920 | STRONG |
| balloon | 4 | 0.995 | STRONG |
| electro-wizard | 3 | 0.887 | ADEQUATE |
| mega-minion | 3 | 0.882 | ADEQUATE |
| miner | 3 | 0.237 | **FAILING** |
| royal-ghost | 3 | 0.702 | ADEQUATE |
| electro-spirit | 3 | 0.560 | WEAK |
| zap | 3 | 0.245 | **FAILING** |
| hog-rider | 3 | 0.869 | ADEQUATE |
| poison | 3 | 0.927 | STRONG |
| bomb-tower | 3 | 0.329 | **FAILING** |
| lumberjack | 2 | 0.992 | STRONG |
| ice-golem | 2 | 0.810 | ADEQUATE |
| pekka | 2 | 0.870 | ADEQUATE |
| bandit | 2 | 0.724 | ADEQUATE |
| royal-hog | 2 | 0.995 | STRONG |
| guard | 2 | 0.748 | ADEQUATE |
| bowler | 2 | 0.821 | ADEQUATE |
| giant-snowball | 2 | 0.995 | STRONG |
| golem | 2 | 0.769 | ADEQUATE |
| night-witch | 2 | 0.916 | STRONG |
| mighty-miner | 2 | 0.977 | STRONG |
| earthquake | 2 | 0.838 | ADEQUATE |
| tesla | 2 | 0.420 | WEAK |
| goblin-drill | 1 | 0.638 | WEAK |
| flying-machine | 1 | 0.619 | WEAK |
| royal-recruit | 1 | 0.677 | WEAK |
| goblin-cage | 1 | 0.431 | WEAK |
| arrows | 1 | 0.190 | **FAILING** |
| ice-wizard | 1 | 0.508 | WEAK |
| furnace | 1 | 0.685 | WEAK |
| zappy | 1 | 0.709 | ADEQUATE |
| magic-archer | 1 | 0.763 | ADEQUATE |

### CRITICAL FAILURES: High-meta classes with AP50 < 0.40

These classes appear frequently in top meta decks but our model essentially cannot detect them:

1. **barbarian-barrel** (AP50=0.396, 8 decks) - Appears in nearly half of all meta decks. Only 8 sprites in dataset. This is one of the most-used defensive spells in the game. Critical miss.
2. **miner** (AP50=0.237, 3 decks) - Win condition in multiple archetypes. 46 sprites but still failing - likely a domain gap issue (miner pops up from underground, hard to distinguish from dirt).
3. **zap** (AP50=0.245, 3 decks) - Very common lightweight spell. Only 9 sprites. Spell visual is brief flash, hard to capture.
4. **arrows** (AP50=0.190, 1 deck) - Key in Royal Hogs Recruits deck. Only 6 sprites. Spell visual is also very brief.
5. **bomb-tower** (AP50=0.329, 3 decks) - Used in Miner/Drill control decks. 14 sprites but still failing.

### WEAK BUT META-IMPORTANT: Classes with AP50 0.40-0.70

1. **skeleton** (AP50=0.670, 7 decks) - The single most common card in the meta. Tiny sprites are hard to detect. 148 sprites but low AP suggests size is the real issue.
2. **ice-spirit** (AP50=0.644, 6 decks) - Very common cycle card. Small sprite.
3. **electro-spirit** (AP50=0.560, 3 decks) - Similar issues to ice-spirit.
4. **tesla** (AP50=0.420, 2 decks) - Building that hides underground between attacks. Only 12 sprites.
5. **goblin-cage** (AP50=0.431, 1 deck) - Building with only 4 sprites.
6. **flying-machine** (AP50=0.619, 1 deck) - Small aerial unit.
7. **ice-wizard** (AP50=0.508, 1 deck) - Important in Splashyard.
8. **royal-recruit** (AP50=0.677, 1 deck) - Core of RR decks.
9. **goblin-drill** (AP50=0.638, 1 deck) - Win condition for drill poison.
10. **furnace** (AP50=0.685, 1 deck) - Building with only 2 sprites.

---

## 4. Sprite Dataset Imbalance Analysis

### Classes with extremely few sprites (<= 5)

| Class | Sprites | AP50 | Meta? |
|-------|---------|------|-------|
| tombstone | 1 | 0.995 | Yes (2 decks) |
| barbarian-hut | 1 | 0.995 | No |
| clone | 2 | 0.826 | No |
| furnace | 2 | 0.685 | Yes (1 deck) |
| graveyard | 3 | 0.804 | Yes (1 deck) |
| goblin-hut | 3 | 0.995 | Yes (1 deck) |
| goblin-cage | 4 | 0.431 | Yes (1 deck) |
| x-bow | 4 | 0.888 | No |
| skeleton-king-skill | 4 | N/A | No |
| freeze | 5 | 0.874 | Yes (1 deck) |

Note: Some classes with very few sprites still achieve high AP50 (tombstone, barbarian-hut). This is because the synthetic generator can use even a single sprite with augmentation to generate many training samples. The issue is more about *visual distinctiveness* than raw sprite count.

### Classes with many sprites but poor detection

| Class | Sprites | AP50 | Notes |
|-------|---------|------|-------|
| skeleton | 148 | 0.670 | Very small unit, hard to distinguish from background noise |
| ice-wizard | 60 | 0.508 | Possibly confused with other wizard variants |
| mega-knight | 60 | 0.775 | Large unit but complex visual states (jump/land) |
| barbarian | 42 | 0.472 | Confused with barbarian-evolution and elite-barbarian |
| miner | 46 | 0.237 | Underground emergence makes detection difficult |
| wizard | 41 | 0.152 | Possibly confused with ice-wizard, electro-wizard |
| mother-witch | 38 | 0.341 | Unique visual but still failing |
| barbarian-evolution | 55 | 0.474 | Confused with base barbarian |

---

## 5. Confusion Clusters: Visually Similar Classes

Several class groups have poor detection likely due to inter-class confusion:

### Barbarian family
- barbarian: AP50=0.472 (42 sprites)
- barbarian-evolution: AP50=0.474 (55 sprites)
- elite-barbarian: AP50=0.665 (34 sprites)
- All look similar at small scale. The model struggles to distinguish variants.

### Wizard family
- wizard: AP50=0.152 (41 sprites) - CRITICAL FAILURE
- ice-wizard: AP50=0.508 (60 sprites)
- electro-wizard: AP50=0.887 (17 sprites)
- Wizard and ice-wizard are likely confused. E-wiz has distinctive visual (dual lightning).

### Spirit family
- ice-spirit: AP50=0.644 (66 sprites)
- electro-spirit: AP50=0.560 (20 sprites)
- fire-spirit: AP50=0.878 (29 sprites)
- heal-spirit: AP50=0.951 (21 sprites)
- All are very small. Fire and heal spirits have more distinctive colors.

### Knight family
- knight: AP50=0.916 (53 sprites)
- knight-evolution: AP50=0.122 (21 sprites) - CRITICAL FAILURE
- Knight-evolution has a very different visual (armor upgrade) but only 21 sprites and still fails badly.

### Skeleton family
- skeleton: AP50=0.670 (148 sprites)
- skeleton-evolution: AP50=0.528 (46 sprites)
- Both are very small. Hard to distinguish from each other.

---

## 6. Ally vs Enemy Distinction

The model does NOT classify ally vs enemy - this is by design. The system uses Y-coordinate on screen to determine side:
- Top half of arena = enemy units
- Bottom half of arena = ally units

This is a sound approach because:
1. Ally and enemy sprites look identical (same unit, same sprites)
2. The arena is symmetric, so position reliably indicates side
3. Reducing class count by half (no ally/enemy split) helps detection accuracy

No changes needed here.

---

## 7. Coverage Assessment for Our Fixed Deck

Based on context (likely hog rider / royal hogs "pigs" deck), our deck would use cards like:
- royal-hog: AP50=0.995 -- STRONG
- hog-rider: AP50=0.869 -- ADEQUATE
- musketeer/firecracker: AP50=0.979/0.946 -- STRONG
- fireball: AP50=0.977 -- STRONG
- the-log: AP50=0.802 -- ADEQUATE
- skeleton/ice-spirit: AP50=0.670/0.644 -- WEAK (cycle cards)

Our own deck detection is mostly adequate. The bigger issue is detecting **opponent** cards from any meta deck.

---

## 8. Priority Improvement Recommendations

### Priority 1 - Critical (high meta frequency, AP50 < 0.40)
1. **barbarian-barrel** - Add more sprite variants (currently 8). This is the #1 most urgent fix.
2. **miner** - Investigate synthetic-to-real domain gap. Miner emerges from ground differently than sprite shows.
3. **zap** - Add more sprites (currently 9). Brief spell effect is hard to capture.

### Priority 2 - Important (high meta frequency, AP50 0.40-0.70)
4. **skeleton** - Consider larger detection resolution for small units, or a dedicated small-unit model.
5. **ice-spirit** - Same small-unit problem as skeleton.
6. **electro-spirit** - Same.
7. **tesla** - Add more sprites (currently 12). The underground/above-ground states need both represented.
8. **barbarian** / **barbarian-evolution** - Need better visual distinction in training data.

### Priority 3 - Secondary (lower meta frequency, still weak)
9. **bomb-tower** - Add sprites (14 currently), investigate confusion source.
10. **wizard** / **mother-witch** - Investigate cross-class confusion.
11. **knight-evolution** - Only 21 sprites, AP50=0.122 is catastrophic.
12. **goblin-cage** - Only 4 sprites.
13. **arrows** - Only 6 sprites, very brief visual.

### Priority 4 - Not urgent for gameplay
14. **bomber-evolution** (AP50=0.000) - Not in any meta deck.
15. **tesla-evolution** (AP50=0.000) - Rare in meta.
16. **bar** / **bar-level** / **text** / **selected** - UI elements, less critical for gameplay decisions.

### Structural Recommendations
- **Small unit detection**: Skeletons, spirits, and goblins are the most common swarm units. A dedicated small-object detection head or a second model at higher resolution for the arena center could help.
- **Spell detection**: Many spells (zap, arrows, freeze) are brief visual effects. Frame timing and temporal context (detecting the spell landing zone) may be more effective than single-frame detection.
- **Sprite augmentation**: For classes with < 10 sprites, create additional synthetic variants (color shifts, scale variations, state variations).
- **Confusion resolution**: For visually similar classes (barbarian family, wizard family), consider training with hard-negative mining or adding distinguishing visual features to the loss function.

---

## 9. Summary Statistics

| Metric | Value |
|--------|-------|
| Overall mAP50 | 0.8056 |
| Overall mAP50-95 | 0.5675 |
| Classes with AP50 >= 0.90 | 73 (47%) |
| Classes with AP50 0.70-0.89 | 32 (21%) |
| Classes with AP50 0.40-0.69 | 20 (13%) |
| Classes with AP50 < 0.40 | 14 (9%) |
| Classes not evaluated | 4 (3%) |
| Meta-critical classes with AP50 < 0.40 | 5 (barbarian-barrel, miner, zap, arrows, bomb-tower) |
| Meta-critical classes with AP50 0.40-0.70 | 10 |
| Total meta deck coverage (>= 0.70 AP50) | ~70% of unique meta cards are adequately detected |
