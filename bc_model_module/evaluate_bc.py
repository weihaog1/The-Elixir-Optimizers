#!/usr/bin/env python3
"""Comprehensive BC model evaluation with quantitative and qualitative results.

Generates:
  1. Training curves (loss, F1, recall/precision)
  2. Dataset statistics (class distribution, action heatmaps)
  3. Per-head analysis (play/card/position breakdown)
  4. Confusion-style analysis on validation set
  5. Live inference log analysis
  6. Summary tables printed to console

Usage:
    python bc_model_module/evaluate_bc.py

Outputs saved to: eval_results/
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Namespace package setup (same pattern as train_model.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

import types as _types

_bc_src = os.path.join(_SCRIPT_DIR, "src")
_encoder_src = os.path.join(PROJECT_ROOT, "state_encoder_module", "src")
_action_src = os.path.join(PROJECT_ROOT, "action_builder_module", "src")

_src_pkg = _types.ModuleType("src")
_src_pkg.__path__ = [_bc_src, _encoder_src, _action_src]
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

for pkg_name, pkg_path in {
    "src.bc": os.path.join(_bc_src, "bc"),
    "src.encoder": os.path.join(_encoder_src, "encoder"),
}.items():
    mod = _types.ModuleType(pkg_name)
    mod.__path__ = [pkg_path]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod

_main_src = os.path.join(PROJECT_ROOT, "src", "src")
for sub in ("generation", "detection", "classification", "data", "pipeline"):
    pkg_name = f"src.{sub}"
    mod = _types.ModuleType(pkg_name)
    mod.__path__ = [os.path.join(_main_src, sub)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader

from src.bc.bc_dataset import BCDataset, load_datasets
from src.bc.bc_policy import BCPolicy
from src.encoder.encoder_constants import (
    ACTION_SPACE_SIZE,
    DECK_CARDS,
    GRID_CELLS,
    GRID_COLS,
    GRID_ROWS,
    NOOP_ACTION,
    NUM_CARD_SLOTS,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "eval_results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "bc_training")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bc", "best_bc.pt")
LOG_PATH = os.path.join(PROJECT_ROOT, "models", "bc", "training_log.json")
LIVE_LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "live")


def load_training_log():
    with open(LOG_PATH) as f:
        return json.load(f)


def load_live_logs():
    """Load all JSONL live inference logs."""
    entries = []
    log_dir = Path(LIVE_LOG_DIR)
    if not log_dir.exists():
        return entries
    for jsonl_path in sorted(log_dir.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Training Curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(log: dict):
    """4-panel training curves: loss, decomposed losses, F1, recall/precision."""
    epochs = list(range(1, len(log["train_losses"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BC Training Curves", fontsize=16, fontweight="bold")

    # Panel 1: Total loss
    ax = axes[0, 0]
    ax.plot(epochs, log["train_losses"], "b-", label="Train", linewidth=1.5)
    ax.plot(epochs, log["val_losses"], "r-", label="Val", linewidth=1.5)
    best_ep = log["best_epoch"] + 1
    ax.axvline(best_ep, color="green", linestyle="--", alpha=0.7, label=f"Best (ep {best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss (Train vs Val)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Decomposed train losses
    ax = axes[0, 1]
    ax.plot(epochs, log["train_play_losses"], "g-", label="Play (binary)", linewidth=1.5)
    ax.plot(epochs, log["train_card_losses"], "orange", label="Card (4-way)", linewidth=1.5)
    ax.plot(epochs, log["train_pos_losses"], "purple", label="Position (576-way)", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Decomposed Train Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Action F1 score
    ax = axes[1, 0]
    ax.plot(epochs, log["val_action_f1s"], "b-o", markersize=3, label="Action F1", linewidth=1.5)
    ax.axhline(log["best_action_f1"], color="green", linestyle="--", alpha=0.7,
               label=f"Best F1 = {log['best_action_f1']:.3f}")
    ax.axvline(best_ep, color="green", linestyle=":", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("Action F1 Score (Early Stopping Criterion)")
    ax.set_ylim(0, 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Recall, Precision, Card Accuracy
    ax = axes[1, 1]
    ax.plot(epochs, log["val_action_recalls"], "b-", label="Action Recall", linewidth=1.5)
    ax.plot(epochs, log["val_action_precisions"], "r-", label="Action Precision", linewidth=1.5)
    ax.plot(epochs, log["val_card_accuracies"], "orange", label="Card Accuracy", linewidth=1.5)
    ax.plot(epochs, log["val_noop_accuracies"], "gray", label="Noop Accuracy", linewidth=1.0, alpha=0.7)
    ax.axvline(best_ep, color="green", linestyle="--", alpha=0.5, label=f"Best epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rate")
    ax.set_title("Validation Metrics")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════════

def plot_dataset_stats(all_actions: np.ndarray, all_vectors: np.ndarray):
    """Dataset analysis: action distribution, elixir histogram, action heatmap."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Dataset Statistics", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Noop vs Action pie chart
    ax = fig.add_subplot(gs[0, 0])
    n_noop = int(np.sum(all_actions == NOOP_ACTION))
    n_action = len(all_actions) - n_noop
    ax.pie([n_noop, n_action],
           labels=[f"No-op\n({n_noop})", f"Action\n({n_action})"],
           colors=["#95a5a6", "#e74c3c"],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"Action Distribution\n(N={len(all_actions)} frames)")

    # Panel 2: Card slot distribution (among actions only)
    ax = fig.add_subplot(gs[0, 1])
    action_mask = all_actions != NOOP_ACTION
    if action_mask.any():
        card_ids = all_actions[action_mask] // GRID_CELLS
        card_counts = [int(np.sum(card_ids == i)) for i in range(NUM_CARD_SLOTS)]
        colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
        bars = ax.bar(range(NUM_CARD_SLOTS), card_counts, color=colors)
        ax.set_xticks(range(NUM_CARD_SLOTS))
        ax.set_xticklabels([f"Slot {i}" for i in range(NUM_CARD_SLOTS)])
        ax.set_ylabel("Count")
        ax.set_title(f"Card Slot Usage\n({n_action} actions)")
        for bar, count in zip(bars, card_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Elixir distribution at action time
    ax = fig.add_subplot(gs[0, 2])
    elixir_at_action = all_vectors[action_mask, 0] * 10 if action_mask.any() else np.array([])
    elixir_at_noop = all_vectors[~action_mask, 0] * 10
    if len(elixir_at_action) > 0:
        ax.hist(elixir_at_action, bins=np.arange(0, 11.5, 1), alpha=0.7,
                color="#e74c3c", label="Action frames", edgecolor="black")
    ax.hist(elixir_at_noop, bins=np.arange(0, 11.5, 1), alpha=0.5,
            color="#95a5a6", label="No-op frames", edgecolor="black")
    ax.set_xlabel("Elixir")
    ax.set_ylabel("Count")
    ax.set_title("Elixir at Decision Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Action placement heatmap (all cards combined)
    ax = fig.add_subplot(gs[1, 0])
    heatmap = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    if action_mask.any():
        for act in all_actions[action_mask]:
            cell = act % GRID_CELLS
            row = cell // GRID_COLS
            col = cell % GRID_COLS
            heatmap[row, col] += 1
    im = ax.imshow(heatmap, cmap="hot", aspect="auto", origin="upper")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Placement Heatmap\n(all cards)")
    ax.axhline(15.5, color="cyan", linewidth=1, linestyle="--", alpha=0.7, label="River")
    ax.axhline(16.5, color="cyan", linewidth=1, linestyle="--", alpha=0.7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    # Panel 5-6: Per-card heatmaps (2x2 sub-grid)
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 1:])
    for card_id in range(NUM_CARD_SLOTS):
        ax = fig.add_subplot(inner_gs[card_id // 2, card_id % 2])
        card_heatmap = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        if action_mask.any():
            card_acts = all_actions[action_mask]
            card_mask = card_acts // GRID_CELLS == card_id
            for act in card_acts[card_mask]:
                cell = act % GRID_CELLS
                row = cell // GRID_COLS
                col = cell % GRID_COLS
                card_heatmap[row, col] += 1
        im = ax.imshow(card_heatmap, cmap="hot", aspect="auto", origin="upper")
        count = int(card_heatmap.sum())
        ax.set_title(f"Card {card_id} ({count})", fontsize=9)
        ax.axhline(15.5, color="cyan", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axhline(16.5, color="cyan", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "dataset_stats.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: Model Evaluation on Validation Set
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model_on_val(npz_paths: list[Path]):
    """Run model inference on validation set, generate confusion & analysis plots."""
    print("  Loading model and validation data...")
    policy = BCPolicy.load(MODEL_PATH)
    policy.eval()

    _, val_dataset = load_datasets(npz_paths, val_ratio=0.2, seed=42, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    all_true_play = []
    all_pred_play = []
    all_true_card = []
    all_pred_card = []
    all_true_actions = []
    all_pred_actions = []
    all_play_probs = []

    with torch.no_grad():
        for batch in val_loader:
            obs = {"arena": batch["arena"], "vector": batch["vector"]}
            actions = batch["action"]

            play_l, card_l, pos_per_card = policy.forward_decomposed(obs)

            # Play predictions
            play_pred = play_l.argmax(dim=1)
            play_true = (actions != NOOP_ACTION).long()
            play_probs = torch.softmax(play_l, dim=1)[:, 1]

            all_true_play.extend(play_true.numpy())
            all_pred_play.extend(play_pred.numpy())
            all_play_probs.extend(play_probs.numpy())

            # Card predictions (for action frames only)
            is_action = actions != NOOP_ACTION
            if is_action.any():
                true_cards = actions[is_action] // GRID_CELLS
                pred_cards = card_l[is_action].argmax(dim=1)
                all_true_card.extend(true_cards.numpy())
                all_pred_card.extend(pred_cards.numpy())

            # Full action predictions
            pred_card_all = card_l.argmax(dim=1)
            pred_pos = torch.zeros_like(pred_card_all)
            for card_id in range(NUM_CARD_SLOTS):
                mask = pred_card_all == card_id
                if mask.any():
                    pred_pos[mask] = pos_per_card[card_id][mask].argmax(dim=1)
            full_preds = torch.where(
                play_pred == 1,
                pred_card_all * GRID_CELLS + pred_pos,
                torch.full_like(actions, NOOP_ACTION),
            )
            all_true_actions.extend(actions.numpy())
            all_pred_actions.extend(full_preds.numpy())

    all_true_play = np.array(all_true_play)
    all_pred_play = np.array(all_pred_play)
    all_true_card = np.array(all_true_card)
    all_pred_card = np.array(all_pred_card)
    all_true_actions = np.array(all_true_actions)
    all_pred_actions = np.array(all_pred_actions)
    all_play_probs = np.array(all_play_probs)

    # --- Generate plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("BC Model Evaluation (Validation Set)", fontsize=16, fontweight="bold")

    # Panel 1: Play/Noop confusion matrix
    ax = axes[0, 0]
    tp = int(np.sum((all_pred_play == 1) & (all_true_play == 1)))
    fp = int(np.sum((all_pred_play == 1) & (all_true_play == 0)))
    fn = int(np.sum((all_pred_play == 0) & (all_true_play == 1)))
    tn = int(np.sum((all_pred_play == 0) & (all_true_play == 0)))
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Noop", "Pred Play"])
    ax.set_yticklabels(["True Noop", "True Play"])
    ax.set_title("Play/Noop Confusion Matrix")
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    ax.set_xlabel(f"Recall={recall:.2%}  Prec={precision:.2%}  F1={f1:.3f}")

    # Panel 2: Card confusion matrix (action frames only)
    ax = axes[0, 1]
    if len(all_true_card) > 0:
        card_cm = np.zeros((NUM_CARD_SLOTS, NUM_CARD_SLOTS), dtype=int)
        for t, p in zip(all_true_card, all_pred_card):
            card_cm[t, p] += 1
        im = ax.imshow(card_cm, cmap="Oranges")
        for i in range(NUM_CARD_SLOTS):
            for j in range(NUM_CARD_SLOTS):
                ax.text(j, i, str(card_cm[i, j]), ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if card_cm[i, j] > card_cm.max() / 2 else "black")
        ax.set_xticks(range(NUM_CARD_SLOTS))
        ax.set_yticks(range(NUM_CARD_SLOTS))
        ax.set_xticklabels([f"Pred {i}" for i in range(NUM_CARD_SLOTS)], fontsize=8)
        ax.set_yticklabels([f"True {i}" for i in range(NUM_CARD_SLOTS)], fontsize=8)
        card_acc = np.sum(all_true_card == all_pred_card) / max(len(all_true_card), 1)
        ax.set_title(f"Card Confusion (acc={card_acc:.1%})")
        ax.set_xlabel(f"N={len(all_true_card)} action frames")
    else:
        ax.text(0.5, 0.5, "No action frames", ha="center", va="center")
        ax.set_title("Card Confusion")

    # Panel 3: Play probability distribution
    ax = axes[0, 2]
    action_probs = all_play_probs[all_true_play == 1]
    noop_probs = all_play_probs[all_true_play == 0]
    if len(action_probs) > 0:
        ax.hist(action_probs, bins=30, alpha=0.7, color="#e74c3c",
                label=f"True Action (n={len(action_probs)})", density=True)
    if len(noop_probs) > 0:
        ax.hist(noop_probs, bins=30, alpha=0.5, color="#3498db",
                label=f"True Noop (n={len(noop_probs)})", density=True)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_xlabel("P(play)")
    ax.set_ylabel("Density")
    ax.set_title("Play Head Probability Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Predicted placement heatmap
    ax = axes[1, 0]
    pred_heatmap = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    pred_actions_only = all_pred_actions[all_pred_actions != NOOP_ACTION]
    for act in pred_actions_only:
        cell = act % GRID_CELLS
        row = cell // GRID_COLS
        col = cell % GRID_COLS
        pred_heatmap[row, col] += 1
    im = ax.imshow(pred_heatmap, cmap="hot", aspect="auto", origin="upper")
    ax.set_title(f"Predicted Placements\n({len(pred_actions_only)} actions)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.axhline(15.5, color="cyan", linewidth=1, linestyle="--", alpha=0.7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 5: True placement heatmap (for comparison)
    ax = axes[1, 1]
    true_heatmap = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    true_actions_only = all_true_actions[all_true_actions != NOOP_ACTION]
    for act in true_actions_only:
        cell = act % GRID_CELLS
        row = cell // GRID_COLS
        col = cell % GRID_COLS
        true_heatmap[row, col] += 1
    im = ax.imshow(true_heatmap, cmap="hot", aspect="auto", origin="upper")
    ax.set_title(f"True Placements (Ground Truth)\n({len(true_actions_only)} actions)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.axhline(15.5, color="cyan", linewidth=1, linestyle="--", alpha=0.7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 6: Summary metrics table
    ax = axes[1, 2]
    ax.axis("off")
    n_val = len(all_true_actions)
    overall_acc = np.sum(all_true_actions == all_pred_actions) / max(n_val, 1)
    noop_acc = tn / max(tn + fp, 1)
    card_acc = np.sum(all_true_card == all_pred_card) / max(len(all_true_card), 1) if len(all_true_card) > 0 else 0

    table_data = [
        ["Metric", "Value"],
        ["Val samples", f"{n_val}"],
        ["Overall accuracy", f"{overall_acc:.1%}"],
        ["Action F1", f"{f1:.3f}"],
        ["Action recall", f"{recall:.1%}"],
        ["Action precision", f"{precision:.1%}"],
        ["Card accuracy", f"{card_acc:.1%}"],
        ["Noop accuracy", f"{noop_acc:.1%}"],
        ["True positives", f"{tp}"],
        ["False positives", f"{fp}"],
        ["False negatives", f"{fn}"],
        ["True negatives", f"{tn}"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    # Style header row
    for j in range(2):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Validation Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "model_evaluation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return {
        "n_val": n_val, "overall_acc": overall_acc,
        "f1": f1, "recall": recall, "precision": precision,
        "card_acc": card_acc, "noop_acc": noop_acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: Live Inference Analysis
# ═══════════════════════════════════════════════════════════════════════════

def plot_live_inference(entries: list[dict]):
    """Analyze live inference logs: execution rates, timing, action distribution."""
    if not entries:
        print("  No live inference logs found, skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Live Inference Analysis", fontsize=16, fontweight="bold")

    # Panel 1: Execution reason breakdown
    ax = axes[0, 0]
    reasons = {}
    for e in entries:
        r = e.get("reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    labels = list(reasons.keys())
    counts = [reasons[k] for k in labels]
    colors_map = {
        "played": "#2ecc71", "rate_limited": "#e67e22",
        "below_confidence": "#e74c3c", "noop": "#95a5a6",
    }
    bar_colors = [colors_map.get(l, "#3498db") for l in labels]
    bars = ax.barh(labels, counts, color=bar_colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{count} ({count / len(entries) * 100:.1f}%)",
                va="center", fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"Execution Reasons\n({len(entries)} total frames)")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: Logit score distribution
    ax = axes[0, 1]
    scores = [e.get("logit_score", 0) for e in entries if "logit_score" in e]
    played_scores = [e.get("logit_score", 0) for e in entries
                     if e.get("reason") == "played" and "logit_score" in e]
    blocked_scores = [e.get("logit_score", 0) for e in entries
                      if e.get("reason") != "played" and e.get("reason") != "noop"
                      and "logit_score" in e]
    if scores:
        ax.hist(played_scores, bins=20, alpha=0.7, color="#2ecc71",
                label=f"Played ({len(played_scores)})")
        ax.hist(blocked_scores, bins=20, alpha=0.5, color="#e74c3c",
                label=f"Blocked ({len(blocked_scores)})")
        ax.set_xlabel("Logit Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    ax.set_title("Logit Score Distribution")
    ax.grid(True, alpha=0.3)

    # Panel 3: Step time distribution
    ax = axes[1, 0]
    step_times = [e.get("step_time_ms", 0) for e in entries if "step_time_ms" in e]
    if step_times:
        ax.hist(step_times, bins=30, color="#3498db", edgecolor="black", alpha=0.7)
        median_time = np.median(step_times)
        ax.axvline(median_time, color="red", linestyle="--",
                   label=f"Median: {median_time:.0f}ms")
        ax.set_xlabel("Step Time (ms)")
        ax.set_ylabel("Count")
        ax.legend()
    ax.set_title("Inference Latency")
    ax.grid(True, alpha=0.3)

    # Panel 4: Placement positions from live play
    ax = axes[1, 1]
    live_heatmap = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    for e in entries:
        if e.get("reason") == "played":
            col = e.get("col")
            row = e.get("row")
            if col is not None and row is not None:
                row = min(max(int(row), 0), GRID_ROWS - 1)
                col = min(max(int(col), 0), GRID_COLS - 1)
                live_heatmap[row, col] += 1
    im = ax.imshow(live_heatmap, cmap="hot", aspect="auto", origin="upper")
    played_count = int(live_heatmap.sum())
    ax.set_title(f"Live Placement Heatmap\n({played_count} actions executed)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.axhline(15.5, color="cyan", linewidth=1, linestyle="--", alpha=0.7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "live_inference.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5: Architecture & Model Summary
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_summary():
    """Generate a visual model architecture summary."""
    policy = BCPolicy()
    total_params = sum(p.numel() for p in policy.parameters())

    # Count params per component
    components = {
        "Feature Extractor\n(CNN + MLP)": sum(
            p.numel() for p in policy.feature_extractor.parameters()
        ),
        "Shared Trunk\n(Linear 192→256)": sum(
            p.numel() for p in policy.shared_trunk.parameters()
        ),
        "Play Head\n(Binary)": sum(
            p.numel() for p in policy.play_head.parameters()
        ),
        "Card Head\n(4-way)": sum(
            p.numel() for p in policy.card_head.parameters()
        ),
        "Position Head\n(FiLM + 576-way)": sum(
            p.numel() for n, p in policy.named_parameters()
            if any(k in n for k in ["position", "film", "card_position"])
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"BCPolicy Architecture ({total_params:,} parameters)", fontsize=14, fontweight="bold")

    # Panel 1: Parameter distribution
    ax = axes[0]
    names = list(components.keys())
    counts = list(components.values())
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#e67e22", "#9b59b6"]
    bars = ax.barh(names, counts, color=colors)
    for bar, count in zip(bars, counts):
        pct = count / total_params * 100
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("Parameters")
    ax.set_title("Parameter Distribution by Component")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: Architecture diagram as text
    ax = axes[1]
    ax.axis("off")
    arch_text = (
        "Input\n"
        "├─ arena: (B, 32, 18, 6)  ──┐\n"
        "│  └─ Embed(156,8) + 3×Conv2d  │\n"
        "│    (13→32→64→128, BN, ReLU)  │\n"
        "│    → (B, 128)  ──────────────┤\n"
        "│                               ├─ concat → (B, 192)\n"
        "└─ vector: (B, 23)  ───────────┤\n"
        "   └─ CardEmbed(9,8) + 2×Linear │\n"
        "     (51→64→64, ReLU)           │\n"
        "     → (B, 64)  ───────────────┘\n"
        "\n"
        "Shared Trunk: Linear(192→256) + ReLU + Drop(0.2)\n"
        "                         ↓\n"
        "    ┌────────────────────┼────────────────────┐\n"
        "    ↓                    ↓                    ↓\n"
        "Play Head            Card Head         Position Head\n"
        "Linear(256→2)     Linear(256→4)    FiLM(card_embed)\n"
        "→ play/noop        → slot 0-3       × Linear(256→128)\n"
        "                                     → Linear(128→576)\n"
        "                                     per card → (B,576)\n"
    )
    ax.text(0.05, 0.95, arch_text, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))
    ax.set_title("Architecture Diagram", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUTPUT_DIR, "model_architecture.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Console Summary Tables
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_tables(log: dict, val_metrics: dict, n_total: int,
                         n_action: int, live_entries: list):
    """Print formatted summary tables to console."""
    print("\n" + "=" * 72)
    print("  BC MODEL EVALUATION SUMMARY")
    print("=" * 72)

    # Dataset
    print("\n-- DATASET " + "-" * 52)
    print(f"  Total frames:        {n_total:>6}")
    print(f"  Action frames:       {n_action:>6} ({n_action / n_total * 100:.1f}%)")
    print(f"  No-op frames:        {n_total - n_action:>6} ({(n_total - n_action) / n_total * 100:.1f}%)")
    print(f"  Training files:      {40:>6}")
    print(f"  Grid:                32x18 (576 cells)")
    print(f"  Action space:        Discrete(2305)")
    print(f"  Augmentation:        Horizontal flip (2x train data)")

    # Model
    policy = BCPolicy()
    total_params = sum(p.numel() for p in policy.parameters())
    print("\n-- MODEL " + "-" * 53)
    print(f"  Architecture:        Hierarchical 3-head BCPolicy")
    print(f"  Parameters:          {total_params:>7,}")
    print(f"  Feature dim:         192 (128 arena + 64 vector)")
    print(f"  Heads:               Play(2) + Card(4) + Position(576)")
    print(f"  Position head:       FiLM-conditioned per card")

    # Training
    best_ep = log["best_epoch"] + 1
    total_ep = len(log["train_losses"])
    print("\n-- TRAINING " + "-" * 50)
    print(f"  Epochs run:          {total_ep:>3} / 100 (early stopped)")
    print(f"  Best epoch:          {best_ep:>3} (by action F1)")
    print(f"  Best F1:             {log['best_action_f1']:.4f}")
    print(f"  Final train loss:    {log['train_losses'][-1]:.3f}")
    print(f"  Final val loss:      {log['val_losses'][-1]:.3f}")
    print(f"  Optimizer:           AdamW (lr=3e-4, wd=1e-4)")
    print(f"  Play weight:         10.0 (vs 1.0 for noop)")
    print(f"  Label smoothing:     0.1 (position head)")

    # Validation metrics
    print("\n-- VALIDATION RESULTS " + "-" * 40)
    print(f"  Overall accuracy:    {val_metrics['overall_acc']:>6.1%}")
    print(f"  Action F1:           {val_metrics['f1']:>6.3f}")
    print(f"  Action recall:       {val_metrics['recall']:>6.1%}  (catches real actions)")
    print(f"  Action precision:    {val_metrics['precision']:>6.1%}  (many false positives)")
    print(f"  Card accuracy:       {val_metrics['card_acc']:>6.1%}  (among true actions)")
    print(f"  Noop accuracy:       {val_metrics['noop_acc']:>6.1%}")
    print(f"  Confusion:  TP={val_metrics['tp']:>4}  FP={val_metrics['fp']:>4}  FN={val_metrics['fn']:>4}  TN={val_metrics['tn']:>4}")

    # Live inference
    if live_entries:
        played = sum(1 for e in live_entries if e.get("reason") == "played")
        rate_lim = sum(1 for e in live_entries if e.get("reason") == "rate_limited")
        below_conf = sum(1 for e in live_entries if e.get("reason") == "below_confidence")
        step_times = [e.get("step_time_ms", 0) for e in live_entries if "step_time_ms" in e]
        median_ms = np.median(step_times) if step_times else 0

        print(f"\n-- LIVE INFERENCE ({len(live_entries)} frames, 23 sessions) " + "-" * 15)
        print(f"  Actions played:      {played:>4} ({played / len(live_entries) * 100:.1f}%)")
        print(f"  Rate limited:        {rate_lim:>4} ({rate_lim / len(live_entries) * 100:.1f}%)")
        print(f"  Below confidence:    {below_conf:>4} ({below_conf / len(live_entries) * 100:.1f}%)")
        print(f"  Median latency:      {median_ms:>6.0f} ms")

    print("\n" + "=" * 72)
    print(f"  All plots saved to: {OUTPUT_DIR}/")
    print("=" * 72 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("BC Model Evaluation")
    print("=" * 50)

    # Load data
    print("\n[1/5] Loading training log...")
    log = load_training_log()

    print("[2/5] Loading dataset...")
    npz_paths = sorted(Path(DATA_DIR).glob("*.npz"))
    print(f"  Found {len(npz_paths)} .npz files")

    # Load all data for stats
    all_actions = []
    all_vectors = []
    for p in npz_paths:
        data = np.load(str(p))
        all_actions.append(data["actions"])
        all_vectors.append(data["obs_vector"])
    all_actions = np.concatenate(all_actions)
    all_vectors = np.concatenate(all_vectors)
    n_total = len(all_actions)
    n_action = int(np.sum(all_actions != NOOP_ACTION))
    print(f"  Total: {n_total} frames ({n_action} action, {n_total - n_action} noop)")

    # Load live logs
    live_entries = load_live_logs()
    print(f"  Live inference logs: {len(live_entries)} entries")

    # Generate plots
    print("\n[3/5] Generating training curves...")
    plot_training_curves(log)

    print("[4/5] Generating dataset stats...")
    plot_dataset_stats(all_actions, all_vectors)

    print("[5/5] Running model evaluation on validation set...")
    val_metrics = evaluate_model_on_val(npz_paths)

    print("\nGenerating additional plots...")
    plot_live_inference(live_entries)
    plot_model_summary()

    # Console summary
    print_summary_tables(log, val_metrics, n_total, n_action, live_entries)


if __name__ == "__main__":
    main()
