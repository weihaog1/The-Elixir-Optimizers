# How the RL Agent Makes Decisions

## Overview: Two-Stage Learning

Our Clash Royale agent uses a **two-stage learning pipeline**:

1. **Behavior Cloning (BC)** — Learn from expert human gameplay recordings. The model observes game states and learns to imitate the actions a human player took in those situations. This gives the agent a reasonable starting policy.

2. **Proximal Policy Optimization (PPO)** — Fine-tune the BC policy by playing actual games and receiving rewards. The agent keeps actions that lead to good outcomes (destroying towers, winning) and avoids actions that lead to bad outcomes (losing towers, wasting elixir).

Your understanding is correct: BC provides the foundation ("what would a human do here?"), and PPO refines it through trial and error ("which of these human-like actions actually win games?").

```
Human Gameplay Recordings
        |
        v
  Behavior Cloning (BC)
  "Learn to imitate expert actions"
        |
        v
  Pre-trained Policy (best_bc.pt)
        |
        v
  PPO Fine-Tuning (Phase 1: frozen, Phase 2: unfrozen)
  "Optimize for winning through self-play"
        |
        v
  Final RL Agent
```

---

## 1. Perception Pipeline: Screenshot to Numbers

Before the agent can make decisions, it needs to understand what's happening on screen. Every 0.5 seconds (2 FPS), it runs this pipeline:

```
Screenshot (1080x1920 pixels)
    |
    v
Crop to game region (609x1077)
    |
    v
YOLO Object Detection (ComboDetector: 2 YOLOv8m models)
    |   Outputs: bounding boxes, class names, confidence, belonging (ally/enemy)
    |
    +---> Card Classification (MiniResNet on 4 card slots)
    |       Outputs: which cards are in hand
    |
    +---> OCR (elixir number)
    |       Outputs: current elixir count (0-10)
    |
    v
GameState (structured data: units, towers, cards, elixir, time)
    |
    v
StateEncoder (GameState -> fixed-size tensors)
    |
    v
Observation Tensors (arena grid + scalar vector)
```

### Object Detection Details

We use two YOLOv8m models running in parallel (the "ComboDetector"):
- **D1**: Detects small sprites (troops, spells)
- **D2**: Detects large sprites (towers, buildings)

Each model outputs **belonging** directly (ally=0, enemy=1), so the agent knows which units are friendly and which are hostile without relying on position heuristics. The 13 classes shared between both detectors are deduplicated using cross-detector NMS (Non-Maximum Suppression).

---

## 2. Observation Space: What the Agent Sees

The agent receives two tensors every frame:

### Arena Grid: `(32, 18, 6)` — A Spatial Map of the Battlefield

The arena is divided into a 32-row x 18-column grid. Each cell has 6 channels:

| Channel | Name | What It Represents |
|---------|------|--------------------|
| 0 | Class ID | What type of unit is here (normalized 0-1, 155 possible classes) |
| 1 | Belonging | Whose unit is it? (-1.0 = ally, +1.0 = enemy, 0 = empty) |
| 2 | Arena Mask | Is there a unit here? (1.0 = yes, 0.0 = no) |
| 3 | Ally Tower HP | Health of friendly tower at this position (0-1 fraction) |
| 4 | Enemy Tower HP | Health of enemy tower at this position (0-1 fraction) |
| 5 | Spell Count | Number of active spell effects on this cell |

This is like a top-down heatmap of the battlefield. The neural network processes this grid with convolutional layers (CNNs) to understand spatial patterns like "enemy troops are pushing left lane" or "my tower is under attack."

### Scalar Vector: `(23,)` — Non-Spatial Game Info

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Elixir | Current elixir (0-10, normalized) |
| 1 | Time | Time remaining (0-300 seconds, normalized) |
| 2 | Overtime | Is it overtime? (0 or 1) |
| 3-5 | Ally Tower HP | King tower, left princess, right princess (fractions) |
| 6-8 | Enemy Tower HP | King tower, left princess, right princess (fractions) |
| 9 | Ally Tower Count | How many of our towers are standing (0-3) |
| 10 | Enemy Tower Count | How many enemy towers are standing (0-3) |
| 11-14 | Card Present | Is each card slot occupied? (4 binary flags) |
| 15-18 | Card Class | Which card is in each slot (8 possible cards) |
| 19-22 | Card Cost | Elixir cost of each card |

### Frame Stacking

For temporal context, we stack the last 3 frames together. This means the agent actually sees:
- Arena: `(32, 18, 18)` — 3 frames x 6 channels
- Vector: `(69,)` — 3 frames x 23 features

This helps the agent understand movement direction and elixir generation rate without explicit velocity features.

---

## 3. Action Space: What the Agent Can Do

The agent chooses from **2,305 possible actions** each step:

```
Action Index = card_slot * 576 + row * 18 + col

Where:
  card_slot: 0-3 (which card to play)
  row: 0-31 (vertical position on the grid)
  col: 0-17 (horizontal position on the grid)
  576 = 32 rows x 18 columns (one full grid)

Total: 4 cards x 576 positions = 2,304 placement actions
       + 1 no-op (action 2304 = "do nothing / wait")
       = 2,305 actions
```

### Action Masking

Not all 2,305 actions are valid at any time. The agent uses an **action mask** (2,305 booleans) to filter out illegal moves:

- **Card not in hand** -> all 576 positions for that card are masked out
- **Not enough elixir** -> cards that cost more than current elixir are masked
- **Invalid placement zone** -> cells outside the player's half (rows 0-16) are masked for most cards
- **No-op is always valid** -> the agent can always choose to wait

MaskablePPO (from the sb3-contrib library) respects these masks during both action selection and policy updates, ensuring the agent never learns to take impossible actions.

---

## 4. Behavior Cloning: Learning from Expert Play

### The Problem with Naive BC

A flat 2,305-way classification would fail because:
- ~85% of frames are no-ops (the human is waiting for elixir)
- The remaining ~15% of actions are spread across 2,304 classes
- Average: 0.24 examples per placement class — far too sparse

### The Solution: Hierarchical Decomposition

Instead of one massive classifier, we decompose the decision into three manageable sub-problems:

```
Step 1: PLAY or WAIT?
  Binary classification: "Should I play a card right now?"
  Play head: Linear(256, 2) -> [noop_score, play_score]
  Trained on ALL frames with weighted loss (play_weight=10x)

Step 2: WHICH CARD? (only if play)
  4-way classification: "Which card slot should I use?"
  Card head: Linear(256, 4) -> [slot_0, slot_1, slot_2, slot_3]
  Trained only on action frames

Step 3: WHERE? (only if play, per card)
  576-way classification: "Where on the grid should I place it?"
  Position head: FiLM-conditioned Linear(256, 576)
  Each card gets its own position distribution via learned modulation
  Trained only on action frames
```

### FiLM Conditioning (Position Head)

Different cards should be placed in different positions (e.g., tanks in front, ranged behind). The position head uses **Feature-wise Linear Modulation (FiLM)**:

```
card_embedding = Embedding(4, 16)[card_id]
gamma = Linear(16, 128)(card_embedding)    # scale
beta = Linear(16, 128)(card_embedding)     # shift
position_features = gamma * shared_features + beta
position_logits = Linear(128, 576)(position_features)
```

This means each card slot modulates the shared trunk differently, producing card-specific placement preferences without needing 4 separate position networks.

### Neural Network Architecture

```
                    Screenshot
                        |
                        v
              [CRFeatureExtractor]
               /                \
        Arena Branch          Vector Branch
        (32,18,6)              (23,)
           |                      |
     Embedding(156,8)       Embedding(9,8) for card classes
     for class IDs          + 19 scalar features
           |                      |
     3-layer CNN              2-layer MLP
     (13->32->64->128)       (51->64->64)
           |                      |
     AdaptiveAvgPool             |
           \                    /
            \                  /
             Concatenate: 192-dim
                    |
              Shared Trunk
              Linear(192, 256)
              ReLU + Dropout(0.2)
                    |
         +---------+---------+
         |         |         |
     Play Head  Card Head  Position Head
     (2-way)    (4-way)    (576-way x 4 cards)
```

### BC Training Details

| Setting | Value | Why |
|---------|-------|-----|
| Loss | Weighted CE (play) + CE (card) + Label-smoothed CE (position) | Balance sparse action classes |
| Play weight | 10.0 | Counteract 85% no-op imbalance |
| Label smoothing | 0.1 | Prevent position head mode collapse |
| Entropy bonus | 0.01 | Encourage diverse placements |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) | Stable training with regularization |
| LR schedule | Cosine annealing over 100 epochs | Gradual refinement |
| Early stopping | Patience=25, metric=Action F1 | Stop when validation stops improving |
| Data split | 80/20 file-level (game-level) | Prevent data leakage |
| No-op downsample | Keep all actions + 15% of no-ops | Reduce class imbalance |

### BC Results

| Metric | Value |
|--------|-------|
| Best Action F1 | 0.384 |
| Action Recall | 53.5% |
| Action Precision | 30.0% |
| Training Data | 79 games, ~9,300 frames |

The BC model catches about half of the situations where a human would play a card, with 30% of its "play" decisions being correct. This is a reasonable starting point for PPO to refine.

---

## 5. PPO Fine-Tuning: Learning from Experience

### Why PPO After BC?

BC has limitations:
- It only imitates — it doesn't understand *why* an action is good
- It can't improve beyond the expert's skill level
- It doesn't learn from mistakes

PPO addresses these by giving the agent a **reward signal** based on actual game outcomes. The agent learns: "when I played Royal Hogs in the left lane, I destroyed a tower — do more of that."

### How PPO Works (Simplified)

1. **Play a game** using the current policy (with action masking)
2. **Collect rewards** at each step (tower destruction, elixir management, win/loss)
3. **Compute advantage**: "Was this action better or worse than expected?"
4. **Update policy**: Increase probability of above-average actions, decrease below-average
5. **Clip updates**: Don't change too much at once (the "Proximal" in PPO)

### Reward Structure

The agent receives rewards at each step based on what changed:

| Event | Reward | Signal |
|-------|--------|--------|
| Destroy enemy tower | +10.0 | Offensive progress |
| Lose a tower | -10.0 | Defensive failure |
| **Win the game** | **+30.0** | Ultimate goal |
| **Lose the game** | **-30.0** | Ultimate penalty |
| Draw | -5.0 | Slightly negative (we want wins) |
| Survive each step | +0.02 | Stay alive, keep playing |
| Waste elixir (at max) | -0.1 | Don't sit at 10 elixir doing nothing |

All rewards are scaled by 0.1x for numerical stability. The win/loss reward (+/-30) is by far the strongest signal.

### Two-Phase Training Schedule

**Phase 1: Frozen Feature Extractor (~15 games)**

```
[BC Feature Extractor] ---- FROZEN (no gradient updates)
         |
    [Policy Head]  ---- TRAINING (learns when to play)
    [Value Head]   ---- TRAINING (learns state values)
```

- Only ~148K parameters train (policy + value heads)
- The BC-learned visual features are protected from destruction
- LR = 1e-4, entropy annealing 0.02 -> 0.005
- Goal: Learn a value function and basic reward-driven play/noop behavior

**Phase 2: Full Fine-Tuning (~25+ games)**

```
[BC Feature Extractor] ---- UNFROZEN (adapts to RL signal)
         |
    [Policy Head]  ---- TRAINING (continues improving)
    [Value Head]   ---- TRAINING (continues improving)
```

- All ~300K parameters train together
- LR reduced to 3e-5 (10x lower) to protect learned features
- The feature extractor gradually adapts to reward-relevant patterns
- Goal: End-to-end optimization for winning

### MaskablePPO

We use MaskablePPO from `sb3-contrib`, which is standard PPO with action masking support. During both action selection and gradient updates, impossible actions (masked out) are ignored. This is critical because:

- Without masking, the agent might learn to "play" a card it doesn't have
- Masking focuses learning on actually available decisions
- Dramatically reduces the effective action space each step

---

## 6. Live Inference: Playing a Real Game

During actual gameplay, the agent runs this loop at 2 FPS (~500ms per decision):

```
Every 500ms:
  1. Capture screenshot (mss screen capture)           ~150ms
  2. Run dual YOLO detection + card classification      ~70ms
  3. Encode to observation tensors                      ~2ms
  4. Neural network forward pass (policy inference)     ~10ms
  5. Select action from masked logits (argmax)          ~1ms
  6. Execute action via PyAutoGUI (two-click sequence)  ~7ms
     - Click 1: Card slot position
     - Click 2: Arena placement position
     - 150ms delay between clicks for UI responsiveness
```

### Hierarchical Decision at Inference Time

```python
# Step 1: Should I play or wait?
play_logits = play_head(features)    # [noop_score, play_score]
if argmax(play_logits) == 0:
    return NOOP  # Wait

# Step 2: Which card?
card_logits = card_head(features)
card_logits[masked_cards] = -inf     # Remove unaffordable/missing cards
best_card = argmax(card_logits)

# Step 3: Where to place?
pos_logits = position_head(features, card=best_card)  # FiLM-conditioned
pos_logits[invalid_cells] = -inf     # Remove cells outside deployable zone
best_pos = argmax(pos_logits)

return best_card * 576 + best_pos
```

---

## 7. Summary: The Full Learning Journey

```
Stage 1: DATA COLLECTION
  Human plays Clash Royale while recording screenshots + clicks
  Output: session directories with screenshots/ + actions.jsonl

Stage 2: DATASET BUILDING
  Each frame -> YOLO detection -> GameState -> StateEncoder -> obs tensors
  Each click -> nearest frame assignment -> action index
  Output: .npz files (obs_arena, obs_vector, actions, masks)

Stage 3: BEHAVIOR CLONING
  Train hierarchical policy (play/card/position heads) on expert data
  Output: best_bc.pt (full policy) + bc_feature_extractor.pt (for PPO)

Stage 4: PPO PHASE 1 (Frozen, ~15 games)
  Load BC feature extractor, freeze it
  Train policy + value heads on game rewards
  Output: models/ppo/latest_ppo.zip

Stage 5: PPO PHASE 2 (Unfrozen, ~25+ games)
  Resume from Phase 1, unfreeze feature extractor
  Fine-tune everything with lower LR
  Output: models/ppo/final_ppo.zip

Result: An agent that combines human-like game sense (from BC)
        with reward-optimized decision making (from PPO)
```
