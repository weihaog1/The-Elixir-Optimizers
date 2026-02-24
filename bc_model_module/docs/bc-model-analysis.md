# BC Model Analysis — State, Assumptions, & Improvement Plan

**Date:** 2026-02-23
**Model:** BCPolicy (304K params, hierarchical play/card/position heads)
**Dataset:** 40 games, 5,366 frames (943 actions, 4,423 no-ops)
**Sessions Analyzed:** 23 live inference sessions

---

## 1. Current Model Performance

### Training Metrics (Best Checkpoint — Epoch 4)

| Metric | Value | Assessment |
|--------|-------|------------|
| Best F1 | 0.338 | Low — barely above random |
| Action Recall | 51.1% | Misses half of valid placements |
| Action Precision | 25.2% | 3 of 4 predicted actions are wrong |
| Card Accuracy | 19.8% | Below random (25%) — card head is failing |
| Noop Accuracy | 75.0% | Reasonable but not great |
| Best Epoch | 4/30 | Very early convergence then degrades |
| Position Loss | 4.45 | Dominates total loss (5.71) |

### Live Session Behavior (23 sessions)

| Phase | Sessions | Behavior |
|-------|----------|----------|
| Initial (11:33-11:44) | 3 | Good execution, 33-37% action rate |
| Dry-run testing (11:57-12:01) | 3 | Testing mode, 0 actions |
| Degradation (12:03-12:20) | 5 | below_confidence appears, rate_limited bursts |
| Afternoon (13:55-16:35) | 12 | Mixed, persistent below_confidence |

**Key patterns from logs:**
- Cards 2-3 account for 70-80% of all executed actions (card bias)
- Position collapse: almost all placements at row=22, col=8-9
- Detection counts vary wildly (1-35 per frame)
- Logit score range: 0.3-3.5 (actions with score < 1.0 frequently blocked)
- Burst-then-rate-limited pattern: 20 actions in ~40 seconds, then 25+ frames blocked

---

## 2. Current Assumptions

| Assumption | Reality | Impact |
|------------|---------|--------|
| Single-frame obs sufficient | No temporal context for troop movement or elixir timing | Medium — misses strategic timing |
| YOLO detection is consistent | 1-35 detections per frame, highly variable | High — unreliable arena encoding |
| 4 cards always visible | Card slots sometimes obscured or in transition | Low — fallback handles it |
| Elixir always ~5 | No OCR, hardcoded to 0.5 | High — plays cards it can't afford |
| All 6 towers alive at full HP | No OCR, hardcoded to 1.0 | Medium — misses strategic context |
| Enemy detection reliable for spell masking | YOLO often detects 0 enemies when present | High — spell masking too aggressive |

---

## 3. Current Limitations

### Architecture
1. **No temporal context** — single frame obs, no memory of previous states
2. **Hierarchical logit decomposition** — play/card/position heads share trunk but learned independently; card head performs below random
3. **Position head has 576 outputs** — for only 943 training actions, severe data sparsity
4. **No elixir gating** — action mask is always all-ones (every card considered playable)

### Data
5. **Small dataset** — 943 action frames is insufficient for 2305-way classification
6. **Class imbalance** — 82% no-ops even after downsampling to 15%
7. **Position distribution** — training placements cluster around row=22, cols 8-10
8. **Card distribution** — some cards over-represented in training data

### Inference
9. **Confidence threshold** — not calibrated; logit magnitudes vary with model retraining
10. **Repeat penalty too aggressive** — 2.0 per action in a 0.3-3.5 logit range
11. **Temperature too high** — 1.5 flattens distribution, samples low-probability (low-score) actions
12. **Spell masking on enemy_count==0** — unreliable when YOLO detection is poor

### Perception
13. **No belonging output from YOLO** — uses Y-position heuristic (fails when troops cross river)
14. **Detection domain gap** — mAP50=0.804 on synthetic data, lower on real screenshots
15. **Hardcoded game state** — elixir, tower HP, time, overtime all use fixed defaults

---

## 4. Why "Below Confidence" Occurs

The confidence check fires when `logit_score < confidence_threshold`.

**Three compounding factors reduce logit scores:**

1. **Retrained model outputs lower logits.** `play_weight=10.0` makes play_logits[:,1] (play confidence) low. `label_smoothing=0.1` and `entropy_coeff=0.01` spread position logits. Combined: placement logits = play + card + position, all with lower peaks.

2. **Repeat penalty (2.0 per action)** subtracts up to 10.0 from recent actions' logits. Model logits range 0.3-3.5 — a single penalty pushes most actions negative.

3. **Spell masking** when YOLO sees 0 enemies (frequent due to detection variance) eliminates 2 of 4 card slots from consideration, concentrating sampling on fewer options.

**Result:** Even with `confidence_threshold=0.0`, sampled actions frequently have negative logit scores from repeat penalty. With `--confidence 1.0`, virtually nothing executes.

---

## 5. Fixes Applied

### Fix 1: Confidence threshold stays at 0.0
- Default is already 0.0; document that `--confidence` should not be set

### Fix 2: Repeat penalty 2.0 → 0.5
- Proportional to logit range (0.3-3.5)
- With 5 recent actions: max penalty = 2.5, not 10.0

### Fix 3: Spell masking uses total detection count
- Only mask spells when arena is truly empty (0 total detections, not 0 enemies)
- Prevents false masking on poor YOLO frames

### Fix 4: Temperature 1.5 → 1.0
- Keeps learned distribution intact while still sampling (not argmax)
- Higher-probability actions are selected, which also have higher logit scores

---

## 6. Recommendations for Future Improvement

### Short-term (inference tuning, no retraining)
- Lower `--max-apm` to 10-12 for more strategic play pacing
- Try `--noop-frames 3` if auto-calculated 5 is too conservative
- Add per-card cooldown to prevent same-card spam

### Medium-term (retraining)
- **More data** — target 100+ games with 2000+ action frames
- **Fix card head** — try per-card class weighting, or separate card MLP
- **Reduce label smoothing** to 0.05 — current 0.1 may be too aggressive
- **Position constraint** — mask rows 0-16 during training (not valid deployment zone)
- **Lower play_weight** to 5.0-8.0 — 10.0 makes model too conservative

### Long-term (architecture)
- **Temporal features** — LSTM/GRU over frame sequences, or just delta from previous frame
- **Elixir OCR** — read elixir bar from screen capture
- **Tower HP OCR** — read tower health values
- **PPO fine-tuning** — use BC-trained feature extractor as initialization for RL

---

## 7. Questions for Team Discussion

1. Should we prioritize data collection (more games) or model architecture changes?
2. Is the 2 FPS capture rate sufficient, or should we go to 1 FPS for more deliberate play?
3. Should we implement a simple elixir tracker (count up from 0 at 2.8 elixir/sec, subtract card costs)?
4. Would it help to separate the position head into row + column predictions (32-way + 18-way) instead of flat 576-way?
5. Should we transition to PPO with the current BC weights, or wait for better BC performance?
