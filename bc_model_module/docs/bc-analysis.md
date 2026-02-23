# Behavior Cloning Model - Complete Analysis & Decision Guide

This document covers everything you need to understand to build a Behavior Cloning (BC) model for Clash Royale. Read it top to bottom - each section builds on the previous one.

---

## Table of Contents

1. [What Is Behavior Cloning?](#1-what-is-behavior-cloning)
2. [Your Input: The Observation Space](#2-your-input-the-observation-space)
3. [Your Output: The Action Space](#3-your-output-the-action-space)
4. [The nn.Embedding Question](#4-the-nnembedding-question)
5. [Model Architecture Options](#5-model-architecture-options)
6. [SB3 Integration Approaches](#6-sb3-integration-approaches)
7. [Training Considerations](#7-training-considerations)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Live Game Testing Pipeline](#9-live-game-testing-pipeline)
10. [Decision Summary](#10-decision-summary)

---

## 1. What Is Behavior Cloning?

Behavior cloning is **supervised learning on expert demonstrations**. You recorded yourself playing Clash Royale. The DatasetBuilder processed those recordings into (observation, action) pairs stored in .npz files. Now you train a neural network to predict: "Given this game state, what would the expert do?"

It's a classification problem. The model sees a game state (arena grid + scalar features) and outputs a probability distribution over 2,305 possible actions. During training, you minimize cross-entropy loss between the model's predicted distribution and the expert's actual action.

### Why BC Before PPO?

PPO (Proximal Policy Optimization) learns by trial and error - it plays games, gets rewards, and improves. But with 2,305 actions, random exploration is hopeless. The agent would need millions of random games before accidentally discovering that playing Royal Hogs at the bridge when the opponent's elixir is low is a good idea.

BC gives PPO a warm start. Instead of random exploration, the PPO agent starts with a policy that already plays like a human (roughly). PPO then fine-tunes from there. This is the standard approach for complex action spaces.

### What BC Can and Cannot Do

**Can do:**
- Learn general patterns (don't play cards at 0 elixir, defend when opponent pushes)
- Learn card preferences (when to use arrows vs barbarian-barrel)
- Learn basic spatial placement (put troops in front of towers, not behind them)

**Cannot do:**
- Adapt to novel situations the expert never demonstrated
- Improve beyond the expert's skill level
- Handle compounding errors (one bad move leads to an unfamiliar state, which leads to another bad move)

That last point is called **distribution shift** - the model only trains on states the expert visited. In a live game, one wrong move puts you in a state the model has never seen, making the next prediction worse, creating a cascade. PPO fine-tuning fixes this by training on the agent's own states.

---

## 2. Your Input: The Observation Space

Your model receives a dictionary with two keys:

### Arena Grid: shape `(32, 18, 6)`, float32

A spatial representation of the Clash Royale arena. Think of it as a tiny 32x18 "image" with 6 channels. The arena (the playing field between the two kings) is divided into an 18-column, 32-row grid. Each cell is about 30x22 pixels at 540x960 resolution.

```
Row 0  = top of arena (enemy king tower)
Row 14 = enemy side boundary
Row 15-16 = river
Row 17 = player side boundary (troops deploy here and below)
Row 31 = bottom of arena (your king tower)

Col 0  = left edge
Col 17 = right edge
```

#### Channel-by-channel breakdown:

| Channel | Name | Range | What It Tells the Model |
|---------|------|-------|------------------------|
| 0 | CH_CLASS_ID | 0.0-1.0 | **What** is in this cell. 0.0 = empty. Otherwise `class_index / 155`. There are 155 possible unit types (knight, skeleton, hog-rider, etc.) |
| 1 | CH_BELONGING | -1, 0, +1 | **Whose** unit. -1.0 = yours (ally), +1.0 = theirs (enemy), 0.0 = empty |
| 2 | CH_ARENA_MASK | 0 or 1 | **Is** there anything here? Binary presence flag |
| 3 | CH_ALLY_TOWER_HP | 0.0-1.0 | Your tower's HP fraction at this cell (0 = no tower or destroyed) |
| 4 | CH_ENEMY_TOWER_HP | 0.0-1.0 | Enemy tower's HP fraction at this cell |
| 5 | CH_SPELL | 0.0+ | Spell effects active at this cell (can stack: 2.0 means two spells) |

**Key insight:** Channels 1-5 are clean signals - binary flags, continuous fractions, or counts. A CNN can process these naturally. Channel 0 is the problematic one. More on this in Section 4.

#### How units get placed on the grid

The `PositionFinder` algorithm ensures each cell has at most one ground/flying unit. When two units map to the same cell (e.g., two skeletons clumped together), the second one gets pushed to the nearest empty cell. This means the grid is a sparse representation - in a typical frame, maybe 10-20 of the 576 cells are occupied.

Spells bypass PositionFinder - they go directly into Channel 5 as an additive count. Towers are handled separately through Channels 3-4.

### Vector Features: shape `(23,)`, float32

Scalar features about the overall game state, all normalized to [0, 1]:

```
Index  Feature                              Normalization
-----  -------                              -------------
[0]    Elixir (your current mana)           / 10 (max elixir)
[1]    Time remaining in match              / 300 (5 min max)
[2]    Is overtime?                         Binary (0 or 1)

[3]    Your king tower HP                   Fraction of max (6408 HP)
[4]    Your left princess tower HP          Fraction of max (4032 HP)
[5]    Your right princess tower HP         Fraction of max (4032 HP)
[6]    Enemy king tower HP                  Fraction of max
[7]    Enemy left princess tower HP         Fraction of max
[8]    Enemy right princess tower HP        Fraction of max

[9]    Your towers alive                    / 3 (you have 3 towers)
[10]   Enemy towers alive                   / 3

[11]   Card present in slot 0?              Binary (0 or 1)
[12]   Card present in slot 1?              Binary
[13]   Card present in slot 2?              Binary
[14]   Card present in slot 3?              Binary

[15]   Card class index in slot 0           Normalized: card_idx / 7
[16]   Card class index in slot 1           Normalized: card_idx / 7
[17]   Card class index in slot 2           Normalized: card_idx / 7
[18]   Card class index in slot 3           Normalized: card_idx / 7

[19]   Card elixir cost in slot 0           / 10
[20]   Card elixir cost in slot 1           / 10
[21]   Card elixir cost in slot 2           / 10
[22]   Card elixir cost in slot 3           / 10
```

**The deck (8 cards, indices 0-7):**

| Index | Card | Elixir | Type |
|-------|------|--------|------|
| 0 | arrows | 3 | Spell |
| 1 | barbarian-barrel | 2 | Spell |
| 2 | eletro-spirit | 1 | Troop |
| 3 | flying-machine | 4 | Troop |
| 4 | goblin-cage | 4 | Troop |
| 5 | royal-hogs | 5 | Troop |
| 6 | royal-recruits | 7 | Troop |
| 7 | zappies | 4 | Troop |

So if slot 0 has "royal-hogs" (index 5), then:
- `vector[11] = 1.0` (card is present)
- `vector[15] = 5/7 = 0.714` (card class index normalized)
- `vector[19] = 5/10 = 0.5` (5 elixir cost normalized)

---

## 3. Your Output: The Action Space

### Discrete(2305)

The model outputs logits (raw scores) for 2,305 possible actions. After applying the action mask and softmax, it picks the highest-probability action.

**Action encoding:**
```
action_idx = card_id * 576 + row * 18 + col

Where:
  card_id in [0, 3]    (4 card slots in your hand)
  row     in [0, 31]   (32 rows in the grid)
  col     in [0, 17]   (18 columns in the grid)
  576 = 18 * 32        (total grid cells per card)
```

**Action ranges:**
```
Actions 0-575:      Play card in slot 0 at each of 576 grid cells
Actions 576-1151:   Play card in slot 1 at each of 576 grid cells
Actions 1152-1727:  Play card in slot 2 at each of 576 grid cells
Actions 1728-2303:  Play card in slot 3 at each of 576 grid cells
Action 2304:        No-op (wait, do nothing this frame)
```

**Example:** Action index 1200 means:
```
card_id = 1200 // 576 = 2      -> Play card in slot 2
cell    = 1200 % 576 = 48
row     = 48 // 18 = 2         -> Row 2 (near enemy king)
col     = 48 % 18 = 12         -> Column 12 (right side)
```

### Action Masking

A boolean array `(2305,)` where `True` = valid action. Rules:
- **Empty card slot**: All 576 actions for that card are `False`
- **Not enough elixir**: All 576 actions for that card are `False`
- **No-op**: Always `True` (you can always wait)

During inference, the model's logits for invalid actions get set to `-infinity` before softmax. This guarantees the model never picks an impossible action.

**Important for training:** During BC training, we do NOT apply the mask to the loss function. The expert's action is the ground truth regardless of what the mask says (the mask might be wrong due to perception errors). The mask is only for inference.

### Why 2,305 is a Challenge

Most RL benchmarks have tiny action spaces (Atari: 18 actions, CartPole: 2 actions). With 2,305 actions, random chance gives 0.04% accuracy. The model must learn meaningful patterns from limited data.

However, most actions are structurally similar. Placing Royal Hogs at cell (9, 20) vs cell (10, 20) is nearly identical - the model just needs to learn the right "region" to place cards. This structural similarity means a CNN that understands spatial patterns can generalize across nearby cells.

---

## 4. The nn.Embedding Question

This is the most important architectural decision. Let me explain it from first principles.

### The Problem: Categorical Variables Encoded as Floats

Channel 0 of the arena grid encodes **which unit type** is in each cell. There are 155 possible unit types. The StateEncoder normalizes this as `class_index / 155`, producing a float between 0 and 1.

**Example:**
- Knight = class index 49, encoded as `49/155 = 0.316`
- Mini-PEKKA = class index 50, encoded as `50/155 = 0.323`
- Skeleton = class index 27, encoded as `27/155 = 0.174`

The problem: treating these as floats implies an **ordering** that doesn't exist. The model would think Mini-PEKKA (0.323) is "closer to" Knight (0.316) than Skeleton (0.174) is. In reality, these are arbitrary index numbers - Knight and Skeleton might be more strategically similar (both cheap ground troops) than Knight and Mini-PEKKA (light troop vs heavy hitter).

The same issue exists for card class indices in vector features [15-18]. "arrows" = 0/7 = 0.0, "barbarian-barrel" = 1/7 = 0.143. The model would think arrows is more similar to barbarian-barrel than to zappies (4/7 = 0.571), when in reality arrows and zappies have nothing in common.

### What nn.Embedding Does

`nn.Embedding(num_classes, embed_dim)` is a lookup table. Each class gets its own learnable vector of size `embed_dim`. During training, the model learns what vector best represents each class.

```python
# Without embedding:
input = 0.316  # Knight, but model just sees "a float near 0.3"

# With embedding:
embedding = nn.Embedding(156, 8)  # 156 entries (0=empty, 1-155=classes), 8 dims each
input = 49  # Knight (integer index)
output = embedding(input)  # tensor([0.23, -0.41, 0.87, ...])  (8 learned values)
```

After training, the 8-dimensional vector for "knight" might be similar to the vector for "valkyrie" (both ground melee troops) and very different from "arrows" (a spell). The model discovers these relationships itself.

### Arena Unit Embedding: Detailed Analysis

**How it would work:**

1. Extract channel 0 from arena: `class_ids = arena[:, :, 0]` -> shape `(32, 18)`
2. Denormalize to integer: `class_ints = round(class_ids * 155)` -> shape `(32, 18)`, values in [0, 155]
3. Embedding lookup: `embedded = embedding(class_ints)` -> shape `(32, 18, embed_dim)`
4. Take remaining 5 channels: `other = arena[:, :, 1:]` -> shape `(32, 18, 5)`
5. Concatenate: `combined = cat([embedded, other], dim=-1)` -> shape `(32, 18, embed_dim + 5)`
6. Permute for Conv2D: `(batch, embed_dim + 5, 32, 18)` -> feed into CNN

**Arguments FOR arena embedding:**
- Correctly represents unit identity as categorical, not ordinal
- Model can learn strategic relationships between unit types
- KataCR-style encoding was designed with this in mind (the encoder CLAUDE.md explicitly mentions it)

**Arguments AGAINST arena embedding:**
- **Data sparsity**: With 30-40 games, maybe 2,000-3,000 frames. Each frame has ~10-20 occupied cells. Across all frames, you might see 30-50 of the 155 unit types, with most appearing in fewer than 100 cells. That's not enough to learn good 8-dimensional embeddings for rare classes.
- **Most classes never appear**: Your deck is 8 cards. Your opponent's deck is 8 cards. In any given game, at most ~16 of 155 classes show up. Many classes will have zero training examples.
- **The CNN can compensate**: A CNN doesn't need to know "this is a knight" vs "this is a skeleton." For BC, what matters more is "there's an enemy unit here" (channel 2 + channel 1), "it's near my tower" (spatial position), and "I have enough elixir" (vector[0]). The unit type is secondary.
- **More parameters = more overfitting**: An Embedding(156, 8) adds 1,248 parameters. With limited data, these won't train well and may overfit.

### Card Hand Embedding: Detailed Analysis

**How it would work:**

1. Extract card class indices from vector: `card_idx_floats = vector[15:19]` -> shape `(4,)`
2. Denormalize: `card_ints = round(card_idx_floats * 7)` -> shape `(4,)`, values in [0, 7]
3. Add 1 for empty handling: if `vector[11+i] == 0`, use index 0 (empty), else use `card_ints[i] + 1`
4. Embedding lookup: `card_embeds = embedding(card_ints)` -> shape `(4, card_embed_dim)`
5. Flatten: `(4 * card_embed_dim,)`
6. Replace the 4 scalar card-class features in the vector with the flattened embeddings

**Arguments FOR card embedding:**
- Only 8 classes + 1 empty = 9 entries. Very trainable.
- Card identity is the single most important feature for "which card to play." Knowing you have "royal-hogs" vs "arrows" completely changes what actions are reasonable.
- Each card appears in ~25% of frames (4 cards in hand, 8 in deck, each rotates through). With 2,000 frames, each card appears in ~500 frames. Plenty of data for 8-dim embedding.
- Elixir cost is already encoded separately, so the embedding can learn strategic properties beyond cost (spell vs troop, splash vs single target, etc.)

**Arguments AGAINST card embedding:**
- Only 8 classes - the model might learn the distinction anyway from the float + elixir cost
- Adds a small amount of complexity to the vector branch
- The float encoding isn't terrible when there are only 8 values (the model only needs to learn 8 decision boundaries)

### The Verdict

| Feature | Embedding? | Confidence | Reasoning |
|---------|-----------|------------|-----------|
| Arena unit class ID (155 classes) | **No** (for now) | High | Too sparse for dataset size. CNN relies more on channels 1-5. |
| Card hand class ID (8 classes) | **Yes** | High | Critical feature, plenty of data, small overhead. |

**Design for flexibility:** Build the architecture so arena embedding can be toggled on with a flag. If you later collect 200+ games or fine-tune with PPO (which generates unlimited data), you can enable it.

---

## 5. Model Architecture Options

### Option A: Simple CNN + MLP (No Embedding Anywhere)

```
Arena (32,18,6) -> Conv2D layers -> flatten -> 128 features
Vector (23,)    -> Linear layers -> 64 features
Concatenate     -> 192 features -> MLP -> 2305 logits
```

**Pros:** Simplest. Fewest parameters. Hardest to mess up.
**Cons:** Treats card class ID as ordinal float. May struggle to learn "which card to play."
**Params:** ~120K
**Best for:** Quick baseline to see if anything works at all.

### Option B: CNN + MLP with Card Embedding (Recommended)

```
Arena (32,18,6)    -> Conv2D layers -> flatten -> 128 features
Vector (23,)       -> card slots get Embedding(9, 8) lookup
                   -> remaining features stay as floats
                   -> combined through Linear layers -> 64 features
Concatenate        -> 192 features -> MLP -> 2305 logits
```

**Pros:** Card identity properly handled. SB3-compatible architecture. Moderate complexity.
**Cons:** Slightly more complex vector processing.
**Params:** ~130K
**Best for:** Primary BC model. This is what I recommend.

### Option C: Full Embedding (Arena + Card)

```
Arena (32,18,6)    -> extract class channel -> Embedding(156, 8) -> (32,18,8)
                   -> concat with channels 1-5 -> (32,18,13)
                   -> Conv2D layers -> flatten -> 128 features
Vector (23,)       -> card embedding + linear layers -> 64 features
Concatenate        -> 192 features -> MLP -> 2305 logits
```

**Pros:** Maximum expressiveness. Theoretically best if data is sufficient.
**Cons:** Arena embedding likely undertrained with 30-40 games. Overfitting risk.
**Params:** ~140K
**Best for:** After collecting 200+ games, or after PPO generates tons of self-play data.

### CNN Architecture Details (Shared Across All Options)

The arena is a small (32, 18) grid. Standard Atari CNNs use 8x8 kernels with stride 4, which would be way too aggressive here. We need smaller kernels:

```
Layer 1: Conv2d(in_channels, 32, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool(2)
  Input:  (batch, C, 32, 18)
  Output: (batch, 32, 16, 9)

Layer 2: Conv2d(32, 64, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool(2)
  Input:  (batch, 32, 16, 9)
  Output: (batch, 64, 8, 4)

Layer 3: Conv2d(64, 128, kernel=3, padding=1) + BatchNorm + ReLU + AdaptiveAvgPool(1)
  Input:  (batch, 64, 8, 4)
  Output: (batch, 128, 1, 1) -> flatten -> (batch, 128)
```

**Why 3x3 kernels:** Each 3x3 kernel sees a 3-cell neighborhood. After 3 layers with pooling, the receptive field covers the entire arena. This lets the model detect spatial patterns like "enemy push on the left side" or "units clumped at the bridge."

**Why BatchNorm:** Normalizes activations between layers. Helps with training stability, especially with limited data.

**Why AdaptiveAvgPool instead of flatten:** If the input dimensions change (e.g., you change GRID_ROWS/COLS), the model still works. `AdaptiveAvgPool2d(1)` always produces a 1x1 output regardless of input size.

### Vector Branch Details

```
Input: (batch, 23)
  |
  [Card class indices extracted, passed through Embedding(9, 8)]
  [Remaining 19 features stay as-is]
  [Concatenated: 19 + 4*8 = 51 features]
  |
Linear(51, 64) + ReLU
Linear(64, 64) + ReLU
  |
Output: (batch, 64)
```

The card embedding expands 4 scalar card-class features into 4x8 = 32 features. Combined with the 19 non-card features, the vector branch input grows from 23 to 51.

### BC Head (Action Prediction)

```
Input: (batch, 192)  [128 arena + 64 vector]
  |
Linear(192, 256) + ReLU
Dropout(0.2)
Linear(256, 2305)
  |
Output: (batch, 2305)  [raw logits]
```

The BC head is separate from the feature extractor. When transitioning to PPO, you keep the feature extractor and replace this head with SB3's policy/value networks.

---

## 6. SB3 Integration Approaches

### The Goal

You want to:
1. Train a BC model on recorded gameplay
2. Later, use the same model architecture with PPO for online reinforcement learning
3. PPO should start from the BC-pretrained weights (warm start)

This means the model architecture must be compatible with SB3's `MaskablePPO`.

### How SB3 Policies Work Internally

```
Observation (dict)
       |
       v
[Feature Extractor]     <-- This is what we're building (CRFeatureExtractor)
  "arena" -> CNN -> flat
  "vector" -> MLP -> flat
  concatenate -> features_dim
       |
       v
[Shared MLP layers]     <-- Controlled by net_arch param
       |
       +-- [action_net] -> logits (2305,) -> masked softmax -> action
       |
       +-- [value_net] -> scalar value (for PPO advantage estimation)
```

SB3's `MultiInputPolicy` handles dict observations by applying a "combined extractor" that processes each key separately, then concatenates. The default combined extractor uses `NatureCNN` for image-like inputs and `nn.Flatten()` for vector inputs. But NatureCNN expects Atari-sized images (84x84x3), not our tiny 32x18x6 grid. So we need a custom extractor.

### Approach 1: Custom PyTorch BC Loop + Load into SB3 (Recommended)

**Training phase:**
```python
# 1. Define CRFeatureExtractor (subclasses SB3's BaseFeaturesExtractor)
# 2. Define BCPolicy (CRFeatureExtractor + BC head)
# 3. Train with custom PyTorch loop on .npz data
# 4. Save feature extractor weights
```

**PPO transition phase:**
```python
# 1. Create MaskablePPO with same CRFeatureExtractor class
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class": CRFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
    },
)
# 2. Load BC-pretrained feature extractor weights
model.policy.features_extractor.load_state_dict(bc_weights)
# 3. Fine-tune with PPO
model.learn(total_timesteps=1_000_000)
```

**Pros:**
- Full control over BC training (custom loss, logging, early stopping)
- Action masking handled exactly how we want
- Clean weight transfer to PPO

**Cons:**
- Must write training loop from scratch
- Must ensure architecture compatibility with SB3 internals

### Approach 2: Use `imitation` Library

The `imitation` library (by the SB3 team) provides `BC` class for behavior cloning.

```python
from imitation.algorithms.bc import BC
from imitation.data import types

bc = BC(
    observation_space=obs_space,
    action_space=action_space,
    demonstrations=demo_transitions,
    policy=policy,
)
bc.train(n_epochs=100)
```

**Pros:**
- Less code to write
- Built by the SB3 team, good integration

**Cons:**
- Does NOT natively support action masking
- Requires transitions in a specific format (not .npz)
- Less control over training details (class weighting, logging)
- Extra dependency

### Approach 3: Train MaskablePPO on Offline Data (Hack)

Create a dummy gym env that replays the recorded data, then run MaskablePPO with `n_steps` set to consume the full dataset.

**Pros:**
- No custom training loop
- Native SB3 masking

**Cons:**
- PPO is designed for online learning, not offline. The value function and advantage estimation are meaningless for static data.
- Much slower convergence than supervised learning
- Wasteful (computes things you don't need)

### Recommendation: Approach 1

Write a custom PyTorch BC training loop that uses the exact same `CRFeatureExtractor` that SB3 will use. This gives you full control now and clean integration later.

---

## 7. Training Considerations

### Your Dataset Size

Assuming 30-40 games of 3 minutes each at 2 FPS:

```
Raw frames per game:     360
Actions per game:        ~20-25 card placements
After no-op downsample:  ~22 action frames + ~50 no-op frames = ~72 frames/game

Total for 30 games:      ~2,160 frames
Total for 40 games:      ~2,880 frames
```

This is a small dataset. For reference, Atari BC typically uses 500K-1M frames. You need to be careful about overfitting.

### Class Imbalance

Even after no-op downsampling (keeping 15% of no-ops), about 70% of frames are still no-ops. Without addressing this, the model learns to always predict "do nothing."

**Solution 1: Weighted Cross-Entropy Loss**

```python
weights = torch.ones(2305)
weights[2304] = 0.3    # No-op gets low weight
weights[:2304] = 3.0   # All card placements get high weight

criterion = nn.CrossEntropyLoss(weight=weights)
```

This tells the model: "Getting a card placement right is 10x more important than getting a no-op right."

**Solution 2: Focal Loss (Alternative)**

Focal loss automatically down-weights "easy" examples (the model quickly learns to predict no-op) and up-weights "hard" examples (rare card placements).

```
FL(p) = -(1-p)^gamma * log(p)

gamma=0: standard cross-entropy
gamma=2: "hard" examples get 4x more weight than "easy" ones
```

**Recommendation:** Start with weighted cross-entropy (simpler, proven). Try focal loss if results plateau.

### Hyperparameters

| Parameter | Recommended Value | Why |
|-----------|------------------|-----|
| Learning rate | 3e-4 | Standard for Adam optimizer |
| Batch size | 64 | Small enough for ~2,000 frames, large enough for stable gradients |
| Epochs | 50-100 | With early stopping on validation loss |
| LR schedule | Cosine annealing | Smooth decay, no manual tuning |
| Dropout | 0.2 (BC head only) | Prevents overfitting with small dataset |
| Weight decay | 1e-4 | Additional regularization |
| No-op weight | 0.3 | Reduces no-op dominance |
| Action weight | 3.0 | Emphasizes card placement learning |

### Data Splitting

**Critical rule:** Split by game, NOT by frame.

If you split by frame, frames from the same game end up in both train and validation. Since consecutive frames are nearly identical, the model "memorizes" specific games rather than learning general patterns.

```
30 games total:
  Train: games 1-24 (80%)  -> ~1,728 frames
  Val:   games 25-30 (20%) -> ~432 frames
```

### Augmentation

Limited options since observations are already encoded tensors (not raw images):

- **No-op target noise:** With 10% probability, change a no-op frame's label to a random valid action. This prevents the model from being too confident about no-op.
- **Card slot shuffling:** Randomly permute the 4 card slots and their corresponding action ranges. This teaches the model that card identity matters, not slot position.
- **Arena noise:** Add small Gaussian noise (std=0.01) to the arena grid. Simulates perception errors.

These are optional and should only be tried if the model overfits.

### Training Loop Pseudocode

```python
model = BCPolicy(arena_channels=6, vector_dim=23, action_dim=2305)
optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
criterion = CrossEntropyLoss(weight=class_weights)

best_val_loss = float("inf")
patience = 10
no_improve = 0

for epoch in range(100):
    # Train
    model.train()
    for arena, vector, action, mask in train_loader:
        logits = model(arena, vector)         # (batch, 2305)
        loss = criterion(logits, action)       # CE loss against expert action
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Validate
    model.eval()
    val_loss = evaluate(model, val_loader)
    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, "best_bc.pt")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break
```

---

## 8. Evaluation Metrics

### Primary Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Overall accuracy** | % of frames where model predicts exact expert action | 25-40% |
| **Action-only accuracy** | Among frames where expert DID something, % correct | 15-30% |
| **Card selection accuracy** | Does model pick the right card slot? (ignore placement) | 45-65% |
| **No-op F1** | Balance of precision/recall on no-op prediction | 75%+ |
| **Top-5 accuracy** | Is expert action in model's top 5 predictions? | 50-70% |

### Why Accuracy Seems Low

2,305 classes is enormous. Random chance = 0.04%. Even 25% accuracy means the model is ~625x better than random. And for card placement, being off by one cell is often equally valid gameplay - the metric is harsh.

**What matters more than exact accuracy:** Does the model play reasonable cards at reasonable positions at reasonable times? This is best evaluated by watching it play.

### How to Evaluate (Offline)

```python
# Per-batch evaluation
logits = model(arena, vector)           # (batch, 2305)
logits[~mask] = float("-inf")           # Apply action mask
predictions = logits.argmax(dim=1)      # (batch,)

# Overall accuracy
correct = (predictions == targets).float().mean()

# Card selection accuracy (compare card_id, ignore grid cell)
pred_cards = predictions // 576         # Card slot (0-3, or 4+ for no-op)
true_cards = targets // 576
card_acc = (pred_cards == true_cards).float().mean()

# Action-only accuracy (exclude no-op frames)
action_mask = targets != 2304
action_correct = (predictions[action_mask] == targets[action_mask]).float().mean()
```

### Confusion Analysis

After training, build a confusion matrix of:
- Predicted card vs actual card (5x5: cards 0-3 + no-op)
- Predicted region vs actual region (e.g., divide arena into 4 quadrants)

This tells you where the model is confused. Common patterns:
- Always predicts no-op -> class imbalance not handled
- Correct card but wrong position -> spatial learning is weak
- Correct position but wrong card -> card identity learning is weak

---

## 9. Live Game Testing Pipeline

### Architecture

```
Screen Capture (mss at 2 FPS)
       |
       v
EnhancedStateBuilder
  |-- YOLO detection (~65ms)
  |-- OCR for timer/elixir (~50ms)
  |-- CardPredictor for hand (~10ms)
       |
       v
GameState
       |
       v
StateEncoder
  |-- encode() -> obs dict {arena: (32,18,6), vector: (23,)}
  |-- action_mask() -> mask (2305,)
       |
       v
BC Model
  |-- forward(obs) -> logits (2305,)
  |-- Apply mask -> masked logits
  |-- argmax -> action_idx
       |
       v
Action Execution
  |-- action_to_placement(action_idx)
  |-- If not no-op: cell_to_norm(col, row) -> (x_norm, y_norm)
  |-- play_card(card_id, x_norm, y_norm) via PyAutoGUI
```

### Timing Budget

At 2 FPS, you have 500ms per frame. The pipeline must complete within this:

| Component | Estimated Time | Cumulative |
|-----------|---------------|------------|
| Screen capture (mss) | ~5ms | 5ms |
| YOLO inference (YOLOv8s, imgsz=960) | ~65ms | 70ms |
| OCR (PaddleOCR, 3 regions) | ~50ms | 120ms |
| Card classifier (4 crops) | ~10ms | 130ms |
| StateEncoder.encode() | ~2ms | 132ms |
| BC model forward pass | ~5ms | 137ms |
| PyAutoGUI click execution | ~100ms | 237ms |
| **Total** | **~237ms** | **Well within 500ms** |

### Safety Measures

- **Confidence threshold:** Only act if max logit exceeds a threshold. Below threshold, default to no-op. This prevents low-confidence random actions.
- **Action cooldown:** After playing a card, wait at least 0.5s before the next action. Prevents rapid-fire plays that waste elixir.
- **Manual override:** Keyboard shortcut to pause/resume the bot.
- **Logging:** Save every observation, prediction, and action to a log file for post-game analysis.

### Live Testing Script Structure

```python
def main():
    # 1. Initialize components
    yolo_model = YOLO("models/best.pt")
    state_builder = StateBuilder(yolo_model)
    card_predictor = CardPredictor("models/card_classifier/")
    enhanced_sb = EnhancedStateBuilder(state_builder, card_predictor)
    encoder = StateEncoder()
    bc_model = BCPolicy.load("models/bc_best.pt")
    bc_model.eval()

    # 2. Find game window
    window = find_window("Clash Royale")

    # 3. Main loop (2 FPS)
    while running:
        screenshot = capture_screen(window)
        state = enhanced_sb.build_state(screenshot, time.time())
        obs = encoder.encode(state)
        mask = encoder.action_mask(state)

        with torch.no_grad():
            logits = bc_model(obs)
            logits[~mask] = float("-inf")
            action_idx = logits.argmax().item()

        result = action_to_placement(action_idx)
        if result is not None:
            card_id, col, row = result
            x_norm, y_norm = cell_to_norm(col, row)
            play_card(card_id, x_norm, y_norm)

        time.sleep(0.5)  # 2 FPS
```

---

## 10. Decision Summary

### Decisions That Need Your Input

| # | Decision | Options | My Recommendation | Your Choice? |
|---|----------|---------|-------------------|-------------|
| 1 | Arena unit embedding | A: No embedding (float) / B: nn.Embedding(156, 8) | **A** - data too sparse for 155 classes | |
| 2 | Card hand embedding | A: No embedding (float) / B: nn.Embedding(9, 8) | **B** - only 8 classes, critical feature | |
| 3 | Training framework | A: Custom PyTorch / B: `imitation` lib / C: SB3 PPO hack | **A** - full control, mask support | |
| 4 | Loss function | A: Weighted cross-entropy / B: Focal loss | **A** - simpler, proven | |
| 5 | Architecture | A: Simple CNN+MLP / B: CNN+MLP+CardEmbed / C: Full embed | **B** - best balance | |
| 6 | Arena embedding toggle | Build toggle for future? Yes/No | **Yes** - minimal overhead | |

### Decisions Already Made (By Prior Modules)

These are fixed by the existing codebase and cannot change without rewriting prior modules:

| Decision | Value | Set By |
|----------|-------|--------|
| Observation space | Dict: arena (32,18,6) + vector (23,) | StateEncoder |
| Action space | Discrete(2305) | encoder_constants |
| Dataset format | .npz with obs_arena, obs_vector, actions, masks | DatasetBuilder |
| Deck | 8-card Royal Hogs/Recruits | encoder_constants |
| Grid dimensions | 18 cols x 32 rows | encoder_constants |
| No-op action index | 2304 | encoder_constants |
| No-op downsample ratio | 0.15 default | DatasetBuilder |

### File Structure (What Will Be Built)

```
docs/josh/bc_model_module/
  src/bc/
    __init__.py                # Exports
    feature_extractor.py       # CRFeatureExtractor (SB3 BaseFeaturesExtractor)
    bc_policy.py               # BCPolicy (feature extractor + action head)
    bc_dataset.py              # PyTorch Dataset loading .npz files
    train_bc.py                # Training script with logging/checkpoints
    CLAUDE.md                  # Technical reference
  tests/
    conftest.py                # Import path setup
    test_feature_extractor.py
    test_bc_policy.py
    test_bc_dataset.py
    test_train_bc.py
  docs/
    bc-analysis.md             # This document
    bc-model-docs.md           # Usage guide + live testing notes
```

---

## Appendix A: SB3 Weight Transfer Details

When transitioning from BC to PPO, you need to transfer the feature extractor weights. Here's how:

```python
# After BC training:
bc_model = BCPolicy.load("models/bc_best.pt")
feature_extractor_state = bc_model.feature_extractor.state_dict()
torch.save(feature_extractor_state, "models/bc_feature_extractor.pt")

# When starting PPO:
from sb3_contrib import MaskablePPO

model = MaskablePPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class": CRFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
        "net_arch": [128, 64],  # Shared MLP layers after extractor
    },
    learning_rate=1e-4,
)

# Load BC weights into the feature extractor
bc_weights = torch.load("models/bc_feature_extractor.pt")
model.policy.features_extractor.load_state_dict(bc_weights)

# Freeze extractor initially (optional - let PPO focus on policy head first)
for param in model.policy.features_extractor.parameters():
    param.requires_grad = False

# Train PPO
model.learn(total_timesteps=500_000)

# Unfreeze and continue with lower LR
for param in model.policy.features_extractor.parameters():
    param.requires_grad = True
model.learning_rate = 3e-5
model.learn(total_timesteps=500_000)
```

## Appendix B: Full Observation Space Reference

### Arena Grid `(32, 18, 6)` Values

```
Channel 0: class_id / 155
  0.000 = empty cell
  0.006 = class 1 (first in unit_list)
  ...
  1.000 = class 155 (last in unit_list)

Channel 1: belonging
  -1.0 = ally (your units)
   0.0 = empty
  +1.0 = enemy (their units)

Channel 2: arena_mask
   0.0 = no unit
   1.0 = unit present (ground or flying)

Channel 3: ally_tower_hp
   0.0 = no ally tower / destroyed
   0.0-1.0 = HP fraction (HP / max_HP)

Channel 4: enemy_tower_hp
   0.0 = no enemy tower / destroyed
   0.0-1.0 = HP fraction

Channel 5: spell_count
   0.0 = no spell
   1.0 = one spell active
   2.0+ = multiple spells stacked
```

### Vector `(23,)` Values

```
[0]     elixir / 10                    -> 0.0 = no elixir, 1.0 = full (10)
[1]     time_remaining / 300           -> 1.0 = just started, 0.0 = time's up
[2]     is_overtime                    -> 0.0 or 1.0
[3]     player_king_hp / 6408         -> 1.0 = full HP, 0.0 = destroyed
[4]     player_left_princess / 4032
[5]     player_right_princess / 4032
[6]     enemy_king_hp / 6408
[7]     enemy_left_princess / 4032
[8]     enemy_right_princess / 4032
[9]     player_towers_alive / 3        -> 1.0 = all 3, 0.33 = 1 left
[10]    enemy_towers_alive / 3
[11]    card_0_present                 -> 0 or 1
[12]    card_1_present
[13]    card_2_present
[14]    card_3_present
[15]    card_0_class / 7               -> 0.0 = arrows, 0.143 = barb-barrel, ...
[16]    card_1_class / 7
[17]    card_2_class / 7
[18]    card_3_class / 7
[19]    card_0_elixir / 10             -> 0.3 = 3 elixir, 0.7 = 7 elixir
[20]    card_1_elixir / 10
[21]    card_2_elixir / 10
[22]    card_3_elixir / 10
```

## Appendix C: Action Space Full Reference

```
Action index = card_id * 576 + row * 18 + col

card_id 0: actions 0-575      (card in slot 0)
card_id 1: actions 576-1151   (card in slot 1)
card_id 2: actions 1152-1727  (card in slot 2)
card_id 3: actions 1728-2303  (card in slot 3)
no-op:     action 2304        (wait)

Grid layout (row, col):
  Row 0, Col 0..17  = top of arena (enemy side)
  Row 15-16         = river
  Row 17, Col 0..17 = start of player's deployable half
  Row 31, Col 0..17 = bottom of arena (your king tower)

Decoding example:
  action_idx = 1200
  card_id = 1200 // 576 = 2          -> card slot 2
  remainder = 1200 % 576 = 48
  row = 48 // 18 = 2                 -> row 2 (enemy side)
  col = 48 % 18 = 12                -> column 12 (right-center)
```

## Appendix D: Known Perception Limitations

These affect BC training data quality:

| Limitation | Impact on BC | Severity |
|-----------|-------------|----------|
| No belonging output from YOLO | Units near river may be mislabeled ally/enemy | Medium |
| CardPredictor confidence varies | Occasional wrong card identification | Low |
| OCR accuracy ~85-90% | Elixir/timer values sometimes wrong | Medium |
| Domain gap (mAP50=0.804) | Some units missed, some misclassified | Medium |
| Single-frame observations | No temporal context (can't see troop movement) | Low (for BC) |

**Mitigation:** The BC model learns from many frames across many games. Occasional perception errors average out. The model learns the overall pattern, not individual frame details.
