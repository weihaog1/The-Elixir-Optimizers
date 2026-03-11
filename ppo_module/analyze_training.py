#!/usr/bin/env python3
"""Analyze PPO training logs and generate summary statistics + plots.

Parses training_log.jsonl and action_log.jsonl, handles episode counter
resets across training sessions, and produces:
- Overall and per-session win rate, reward, and card stats
- Reward/win-rate progression plots
- Card usage breakdown

Usage:
    python ppo_module/analyze_training.py --log logs/ppo/training_log.jsonl
    python ppo_module/analyze_training.py --log logs/ppo/training_log.jsonl --action-log logs/ppo/action_log.jsonl --save-plots plots/
"""

import argparse
import json
import os
import sys
from collections import Counter


def load_jsonl(path):
    """Load a JSONL file, returning list of dicts."""
    entries = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {line_num}: {e}")
    return entries


def assign_global_episodes(entries):
    """Detect session boundaries (episode resets to 1) and assign global episode numbers."""
    if not entries:
        return entries, []

    session_boundaries = [0]  # first entry is always session start
    for i in range(1, len(entries)):
        # Session reset: episode goes back to 1 (or drops significantly)
        if entries[i]["episode"] <= entries[i - 1]["episode"] and entries[i]["episode"] == 1:
            session_boundaries.append(i)

    # Assign global episode numbers
    global_ep = 0
    session_id = 0
    for i, entry in enumerate(entries):
        if session_id + 1 < len(session_boundaries) and i >= session_boundaries[session_id + 1]:
            session_id += 1
        global_ep = sum(
            entries[b - 1]["episode"] if b > 0 else 0
            for b in session_boundaries[:session_id]
        ) + entry["episode"] if session_id > 0 else entry["episode"]
        entry["global_episode"] = global_ep
        entry["session"] = session_id + 1

    # Simpler approach: just increment globally
    for i, entry in enumerate(entries):
        entry["global_episode"] = i + 1

    return entries, session_boundaries


def print_overall_summary(entries):
    """Print overall training statistics."""
    n = len(entries)
    wins = sum(1 for e in entries if e["outcome"] == "win")
    losses = sum(1 for e in entries if e["outcome"] == "loss")
    draws = sum(1 for e in entries if e["outcome"] == "draw")
    unknown = n - wins - losses - draws

    rewards = [e["reward"] for e in entries]
    cards = [e["cards_played"] for e in entries]
    lengths = [e["episode_length"] for e in entries]
    noops = [e["noop_ratio"] for e in entries]
    costs = [e["card_cost_avg"] for e in entries if e.get("card_cost_avg", 0) > 0]
    anomalies = sum(1 for e in entries if e.get("anomaly_detected", False))

    print(f"\n{'=' * 65}")
    print(f"  OVERALL TRAINING SUMMARY  ({n} episodes)")
    print(f"{'=' * 65}")
    print(f"  Outcomes:       {wins}W / {losses}L / {draws}D" +
          (f" / {unknown} unknown" if unknown else ""))
    print(f"  Win rate:       {wins}/{n} = {wins / n:.1%}")
    print(f"  Avg reward:     {sum(rewards) / n:+.2f}  "
          f"(min={min(rewards):+.2f}, max={max(rewards):+.2f})")
    print(f"  Avg cards/game: {sum(cards) / n:.1f}  "
          f"(min={min(cards)}, max={max(cards)})")
    print(f"  Avg length:     {sum(lengths) / n:.0f} steps  "
          f"(min={min(lengths)}, max={max(lengths)})")
    print(f"  Avg noop ratio: {sum(noops) / n:.1%}")
    if costs:
        print(f"  Avg card cost:  {sum(costs) / len(costs):.1f} elixir")
    if anomalies:
        print(f"  Anomalies:      {anomalies}")
    print(f"{'=' * 65}")


def print_session_summary(entries, session_boundaries):
    """Print per-session (training run) statistics."""
    if len(session_boundaries) <= 1:
        return

    sessions = {}
    for e in entries:
        s = e["session"]
        if s not in sessions:
            sessions[s] = []
        sessions[s].append(e)

    print(f"\n{'=' * 65}")
    print(f"  PER-SESSION BREAKDOWN  ({len(sessions)} sessions)")
    print(f"{'=' * 65}")
    print(f"  {'Session':>7}  {'Games':>5}  {'W/L/D':>9}  {'Win%':>5}  "
          f"{'AvgRwd':>7}  {'AvgCards':>8}  {'Noop%':>5}")
    print(f"  {'-------':>7}  {'-----':>5}  {'---------':>9}  {'-----':>5}  "
          f"{'-------':>7}  {'--------':>8}  {'-----':>5}")

    for s_id in sorted(sessions.keys()):
        eps = sessions[s_id]
        n = len(eps)
        w = sum(1 for e in eps if e["outcome"] == "win")
        l = sum(1 for e in eps if e["outcome"] == "loss")
        d = sum(1 for e in eps if e["outcome"] == "draw")
        avg_r = sum(e["reward"] for e in eps) / n
        avg_c = sum(e["cards_played"] for e in eps) / n
        avg_noop = sum(e["noop_ratio"] for e in eps) / n
        print(f"  {s_id:>7}  {n:>5}  {w}W/{l}L/{d}D  {w / n:>5.0%}  "
              f"{avg_r:>+7.2f}  {avg_c:>8.1f}  {avg_noop:>4.0%}")
    print()


def print_progression(entries, window=5):
    """Print rolling win rate and reward progression."""
    if len(entries) < window:
        window = len(entries)

    print(f"\n{'=' * 65}")
    print(f"  PROGRESSION (rolling window = {window})")
    print(f"{'=' * 65}")
    print(f"  {'Episode':>7}  {'Outcome':>7}  {'Reward':>7}  "
          f"{'RollWin%':>8}  {'RollRwd':>8}  {'Cards':>5}")
    print(f"  {'-------':>7}  {'-------':>7}  {'-------':>7}  "
          f"{'--------':>8}  {'--------':>8}  {'-----':>5}")

    for i, e in enumerate(entries):
        start = max(0, i - window + 1)
        window_entries = entries[start:i + 1]
        roll_wr = sum(1 for x in window_entries if x["outcome"] == "win") / len(window_entries)
        roll_rwd = sum(x["reward"] for x in window_entries) / len(window_entries)

        icon = {"win": "WIN", "loss": "LOSS", "draw": "DRAW"}.get(e["outcome"], "?")
        print(f"  {e['global_episode']:>7}  {icon:>7}  {e['reward']:>+7.2f}  "
              f"{roll_wr:>7.0%}   {roll_rwd:>+7.2f}  {e['cards_played']:>5}")
    print()


def analyze_actions(action_entries):
    """Analyze action log for card usage patterns."""
    if not action_entries:
        return

    card_counts = Counter()
    card_rows = {}
    card_cols = {}
    card_elixir = {}

    for a in action_entries:
        name = a.get("card_name", "")
        if not name:
            continue
        card_counts[name] += 1

        if name not in card_rows:
            card_rows[name] = []
            card_cols[name] = []
            card_elixir[name] = []

        card_rows[name].append(a.get("row", 0))
        card_cols[name].append(a.get("col", 0))
        if a.get("elixir", 0) > 0:
            card_elixir[name].append(a["elixir"])

    total_actions = sum(card_counts.values())

    print(f"\n{'=' * 65}")
    print(f"  CARD USAGE ANALYSIS  ({total_actions} total card plays)")
    print(f"{'=' * 65}")
    print(f"  {'Card':>20}  {'Plays':>5}  {'%':>5}  {'AvgRow':>6}  "
          f"{'AvgCol':>6}  {'AvgElixir':>9}")
    print(f"  {'----':>20}  {'-----':>5}  {'-----':>5}  {'------':>6}  "
          f"{'------':>6}  {'---------':>9}")

    for name, count in card_counts.most_common():
        pct = count / total_actions
        avg_row = sum(card_rows[name]) / len(card_rows[name])
        avg_col = sum(card_cols[name]) / len(card_cols[name])
        avg_elix = sum(card_elixir[name]) / len(card_elixir[name]) if card_elixir[name] else 0
        print(f"  {name:>20}  {count:>5}  {pct:>4.0%}   {avg_row:>6.1f}  "
              f"{avg_col:>6.1f}  {avg_elix:>9.1f}")

    # Placement heatmap summary (which half of the grid)
    left_plays = sum(1 for a in action_entries if a.get("col", 9) < 9)
    right_plays = total_actions - left_plays
    print(f"\n  Lane preference: Left={left_plays} ({left_plays / total_actions:.0%}) | "
          f"Right={right_plays} ({right_plays / total_actions:.0%})")

    # Average elixir when playing cards
    all_elixir = [a["elixir"] for a in action_entries if a.get("elixir", 0) > 0]
    if all_elixir:
        print(f"  Avg elixir at play: {sum(all_elixir) / len(all_elixir):.1f}")
    print()


def generate_plots(entries, action_entries, save_dir):
    """Generate matplotlib plots for training progression."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed, skipping plots.")
        return

    os.makedirs(save_dir, exist_ok=True)

    episodes = [e["global_episode"] for e in entries]
    rewards = [e["reward"] for e in entries]
    noops = [e["noop_ratio"] for e in entries]
    cards = [e["cards_played"] for e in entries]
    lengths = [e["episode_length"] for e in entries]

    # Rolling window calculations
    window = min(5, len(entries))
    roll_wr = []
    roll_rwd = []
    for i in range(len(entries)):
        start = max(0, i - window + 1)
        w = entries[start:i + 1]
        roll_wr.append(sum(1 for x in w if x["outcome"] == "win") / len(w))
        roll_rwd.append(sum(x["reward"] for x in w) / len(w))

    # Color outcomes
    outcome_colors = []
    for e in entries:
        if e["outcome"] == "win":
            outcome_colors.append("green")
        elif e["outcome"] == "loss":
            outcome_colors.append("red")
        else:
            outcome_colors.append("gray")

    # --- Figure 1: Reward progression ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Training Analysis", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(episodes, rewards, color=outcome_colors, alpha=0.7, width=0.8)
    ax.plot(episodes, roll_rwd, color="blue", linewidth=2, label=f"Rolling avg ({window})")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Figure 2: Win rate progression ---
    ax = axes[0, 1]
    ax.plot(episodes, roll_wr, color="green", linewidth=2, marker="o", markersize=4)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.set_title(f"Rolling Win Rate (window={window})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # --- Figure 3: Noop ratio over time ---
    ax = axes[1, 0]
    ax.plot(episodes, noops, color="orange", linewidth=1.5, marker="s", markersize=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Noop Ratio")
    ax.set_title("Noop Ratio per Episode")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # --- Figure 4: Cards played per episode ---
    ax = axes[1, 1]
    ax.bar(episodes, cards, color="steelblue", alpha=0.7, width=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cards Played")
    ax.set_title("Cards Played per Episode")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_progression.png")
    plt.savefig(path, dpi=150)
    print(f"[Plots] Saved: {path}")
    plt.close()

    # --- Figure 5: Card usage pie chart (if action log available) ---
    if action_entries:
        card_counts = Counter(a.get("card_name", "") for a in action_entries if a.get("card_name"))
        if card_counts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("Card Usage Analysis", fontsize=16, fontweight="bold")

            # Pie chart
            names = [n for n, _ in card_counts.most_common()]
            counts = [c for _, c in card_counts.most_common()]
            colors = plt.cm.Set3(range(len(names)))
            ax1.pie(counts, labels=names, autopct="%1.0f%%", colors=colors, startangle=90)
            ax1.set_title("Card Play Distribution")

            # Placement scatter (col vs row)
            rows = [a.get("row", 0) for a in action_entries if a.get("card_name")]
            cols = [a.get("col", 0) for a in action_entries if a.get("card_name")]
            card_names_list = [a.get("card_name", "") for a in action_entries if a.get("card_name")]

            unique_cards = list(card_counts.keys())
            color_map = {name: plt.cm.Set1(i / max(len(unique_cards), 1))
                         for i, name in enumerate(unique_cards)}
            scatter_colors = [color_map.get(n, "gray") for n in card_names_list]

            ax2.scatter(cols, rows, c=scatter_colors, alpha=0.4, s=15)
            ax2.axhline(y=17, color="red", linewidth=1, linestyle="--", label="Deploy line")
            ax2.set_xlabel("Column (0=left, 17=right)")
            ax2.set_ylabel("Row (0=enemy, 31=player)")
            ax2.set_title("Card Placement Positions")
            ax2.set_xlim(-0.5, 17.5)
            ax2.set_ylim(-0.5, 31.5)
            ax2.invert_yaxis()
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            path = os.path.join(save_dir, "card_analysis.png")
            plt.savefig(path, dpi=150)
            print(f"[Plots] Saved: {path}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze PPO training logs")
    parser.add_argument("--log", type=str, required=True, help="Path to training_log.jsonl")
    parser.add_argument("--action-log", type=str, default="", help="Path to action_log.jsonl (optional)")
    parser.add_argument("--save-plots", type=str, default="", help="Directory to save plots (optional)")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Error: File not found: {args.log}")
        return 1

    # Load training log
    print(f"[Analyze] Loading {args.log}...")
    entries = load_jsonl(args.log)
    if not entries:
        print("Error: No entries found in training log.")
        return 1
    print(f"[Analyze] Loaded {len(entries)} episodes.")

    # Assign global episode numbers (handles session resets)
    entries, session_boundaries = assign_global_episodes(entries)
    print(f"[Analyze] Detected {len(session_boundaries)} training session(s).")

    # Load action log (optional)
    action_entries = []
    if args.action_log and os.path.exists(args.action_log):
        print(f"[Analyze] Loading {args.action_log}...")
        action_entries = load_jsonl(args.action_log)
        print(f"[Analyze] Loaded {len(action_entries)} card actions.")

    # Print analyses
    print_overall_summary(entries)
    print_session_summary(entries, session_boundaries)
    print_progression(entries, window=args.window)

    if action_entries:
        analyze_actions(action_entries)

    # Generate plots
    if args.save_plots:
        generate_plots(entries, action_entries, args.save_plots)

    return 0


if __name__ == "__main__":
    sys.exit(main())
