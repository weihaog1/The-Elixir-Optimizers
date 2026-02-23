"""BCTrainer - Custom PyTorch training loop for behavior cloning.

Trains a BCPolicy on .npz demonstration data produced by DatasetBuilder.
Uses focal loss to handle extreme class imbalance (85%+ no-op frames),
AdamW optimizer, cosine annealing LR schedule, gradient clipping, and
early stopping based on action recall (not val loss).

Outputs:
    best_bc.pt               - Full BC policy checkpoint (best action recall)
    bc_feature_extractor.pt  - Feature extractor weights only (for PPO transfer)
    training_log.json        - Per-epoch training metrics

PPO transition:
    After training, load bc_feature_extractor.pt into MaskablePPO:

        from sb3_contrib import MaskablePPO
        model = MaskablePPO("MultiInputPolicy", env, policy_kwargs={
            "features_extractor_class": CRFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 192},
            "net_arch": [128, 64],
        }, learning_rate=1e-4, clip_range=0.1)

        bc_weights = torch.load("bc_feature_extractor.pt")
        model.policy.features_extractor.load_state_dict(bc_weights)

        # Freeze extractor initially:
        for param in model.policy.features_extractor.parameters():
            param.requires_grad = False
        model.learn(total_timesteps=500_000)

        # Unfreeze with lower LR:
        for param in model.policy.features_extractor.parameters():
            param.requires_grad = True
        model.learning_rate = 3e-5
        model.learn(total_timesteps=500_000)
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.bc.bc_dataset import load_datasets
from src.bc.bc_policy import BCPolicy
from src.encoder.encoder_constants import GRID_CELLS, NOOP_ACTION, NUM_CARD_SLOTS


@dataclass
class TrainConfig:
    """Configuration for BC training.

    Attributes:
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        lr: Initial learning rate for AdamW.
        weight_decay: L2 regularization coefficient.
        patience: Early stopping patience (epochs without improvement).
        val_ratio: Fraction of .npz files held out for validation.
        play_weight: Weight for the "play" class in the binary play/noop
            loss. Higher values push the model to predict actions more
            aggressively. The noop class implicitly gets weight 1.0.
        grad_clip: Maximum gradient norm for clipping.
        seed: Random seed for reproducibility.
    """

    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15
    val_ratio: float = 0.2
    play_weight: float = 5.0
    grad_clip: float = 1.0
    seed: int = 42


class BCTrainer:
    """Trains a BCPolicy on .npz demonstration data.

    Args:
        config: TrainConfig instance (uses defaults if None).
    """

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()

    @staticmethod
    def _decompose_action(actions: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Decompose flat action indices into play/card/position targets.

        Args:
            actions: (B,) action indices in [0, 2304].

        Returns:
            play_targets: (B,) long - 0=noop, 1=play
            card_targets: (B,) long - card slot 0-3 (valid only when play=1)
            pos_targets: (B,) long - grid cell 0-575 (valid only when play=1)
            is_action: (B,) bool - True for card placement frames
        """
        is_action = actions != NOOP_ACTION
        play_targets = is_action.long()

        # Decompose placement actions: action = card * 576 + cell
        card_targets = torch.zeros_like(actions)
        pos_targets = torch.zeros_like(actions)
        if is_action.any():
            placement = actions[is_action]
            card_targets[is_action] = placement // GRID_CELLS
            pos_targets[is_action] = placement % GRID_CELLS

        return play_targets, card_targets, pos_targets, is_action

    def train_loop(self, npz_paths: list[Path], output_dir: str) -> dict:
        """Run the full training loop with hierarchical decomposition.

        Uses three separate losses:
        1. Play loss: weighted binary CE (play vs noop) on ALL frames
        2. Card loss: CE on action frames only (which card)
        3. Position loss: CE on action frames only (which cell)

        Args:
            npz_paths: List of paths to .npz training files.
            output_dir: Directory to save checkpoints and logs.

        Returns:
            Dict with per-epoch metrics and best checkpoint info.
        """
        cfg = self.config
        os.makedirs(output_dir, exist_ok=True)

        # Reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Datasets (file-level split)
        train_dataset, val_dataset = load_datasets(
            npz_paths, val_ratio=cfg.val_ratio, seed=cfg.seed
        )
        noop_count, action_count = train_dataset.action_class_counts()
        val_noop, val_action = val_dataset.action_class_counts()
        print(
            f"Train: {len(train_dataset)} frames "
            f"({noop_count} noop, {action_count} action, "
            f"{action_count / len(train_dataset) * 100:.1f}% action)"
        )
        print(
            f"Val:   {len(val_dataset)} frames "
            f"({val_noop} noop, {val_action} action, "
            f"{val_action / len(val_dataset) * 100:.1f}% action)"
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        # Model
        policy = BCPolicy().to(device)
        param_count = sum(p.numel() for p in policy.parameters())
        print(f"Model parameters: {param_count:,}")

        # --- Losses ---
        # Play head: weighted binary CE to address noop:action imbalance
        play_weights = torch.tensor(
            [1.0, cfg.play_weight], device=device
        )
        play_criterion = nn.CrossEntropyLoss(weight=play_weights)

        # Card head: standard CE (balanced across 4 cards)
        card_criterion = nn.CrossEntropyLoss()

        # Position head: standard CE
        position_criterion = nn.CrossEntropyLoss()

        print(
            f"Loss: Decomposed (play_weight={cfg.play_weight}, "
            f"card=CE, position=CE)"
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )

        # Tracking
        best_f1 = -1.0
        best_epoch = 0
        no_improve = 0
        history: dict[str, list] = {
            "train_losses": [],
            "train_play_losses": [],
            "train_card_losses": [],
            "train_pos_losses": [],
            "val_losses": [],
            "val_accuracies": [],
            "val_action_recalls": [],
            "val_action_precisions": [],
            "val_action_f1s": [],
            "val_noop_accuracies": [],
            "val_card_accuracies": [],
        }

        print(f"\nStarting training for up to {cfg.epochs} epochs...")
        print(f"Early stopping: patience={cfg.patience}, criterion=action_f1")
        print("-" * 100)

        for epoch in range(cfg.epochs):
            # --- Train ---
            policy.train()
            train_loss_sum = 0.0
            play_loss_sum = 0.0
            card_loss_sum = 0.0
            pos_loss_sum = 0.0
            train_batches = 0

            for batch in train_loader:
                obs = {
                    "arena": batch["arena"].to(device),
                    "vector": batch["vector"].to(device),
                }
                actions = batch["action"].to(device)

                # Decompose actions
                play_tgt, card_tgt, pos_tgt, is_action = (
                    self._decompose_action(actions)
                )

                # Forward through decomposed heads
                play_logits, card_logits, pos_logits = (
                    policy.forward_decomposed(obs)
                )

                # Loss 1: play/noop on ALL frames
                loss_play = play_criterion(play_logits, play_tgt)

                # Loss 2 & 3: card/position ONLY on action frames
                loss_card = torch.tensor(0.0, device=device)
                loss_pos = torch.tensor(0.0, device=device)
                if is_action.any():
                    loss_card = card_criterion(
                        card_logits[is_action], card_tgt[is_action]
                    )
                    loss_pos = position_criterion(
                        pos_logits[is_action], pos_tgt[is_action]
                    )

                loss = loss_play + loss_card + loss_pos

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), cfg.grad_clip
                )
                optimizer.step()

                train_loss_sum += loss.item()
                play_loss_sum += loss_play.item()
                card_loss_sum += loss_card.item()
                pos_loss_sum += loss_pos.item()
                train_batches += 1

            avg_train_loss = train_loss_sum / max(train_batches, 1)
            avg_play_loss = play_loss_sum / max(train_batches, 1)
            avg_card_loss = card_loss_sum / max(train_batches, 1)
            avg_pos_loss = pos_loss_sum / max(train_batches, 1)

            # --- Validate ---
            policy.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            val_batches = 0

            # Action-specific metrics
            action_true_pos = 0
            action_false_pos = 0
            action_false_neg = 0
            noop_correct = 0
            noop_total = 0
            action_total = 0
            card_correct = 0  # correct card among correctly-predicted actions

            with torch.no_grad():
                for batch in val_loader:
                    obs = {
                        "arena": batch["arena"].to(device),
                        "vector": batch["vector"].to(device),
                    }
                    actions = batch["action"].to(device)
                    masks = batch["mask"].to(device)

                    # Decomposed forward pass
                    play_tgt, card_tgt, pos_tgt, is_action = (
                        self._decompose_action(actions)
                    )
                    play_l, card_l, pos_l = policy.forward_decomposed(obs)

                    # Decomposed validation loss
                    loss = play_criterion(play_l, play_tgt)
                    if is_action.any():
                        loss = loss + card_criterion(
                            card_l[is_action], card_tgt[is_action]
                        ) + position_criterion(
                            pos_l[is_action], pos_tgt[is_action]
                        )

                    # Predictions using decomposed heads directly
                    # Play decision: argmax of binary head
                    play_preds = play_l.argmax(dim=1)  # 0=noop, 1=play
                    is_action_pred_play = play_preds == 1

                    # Full action predictions
                    pred_card = card_l.argmax(dim=1)   # (B,)
                    pred_pos = pos_l.argmax(dim=1)     # (B,)
                    preds = torch.where(
                        is_action_pred_play,
                        pred_card * GRID_CELLS + pred_pos,
                        torch.full_like(actions, NOOP_ACTION),
                    )
                    val_correct += (preds == actions).sum().item()
                    val_total += actions.size(0)

                    # Action-specific tracking
                    is_action_true = actions != NOOP_ACTION

                    action_total += is_action_true.sum().item()
                    noop_total += (~is_action_true).sum().item()

                    action_true_pos += (
                        is_action_pred_play & is_action_true
                    ).sum().item()
                    action_false_pos += (
                        is_action_pred_play & ~is_action_true
                    ).sum().item()
                    action_false_neg += (
                        ~is_action_pred_play & is_action_true
                    ).sum().item()
                    noop_correct += (
                        ~is_action_pred_play & ~is_action_true
                    ).sum().item()

                    # Card accuracy for correctly predicted actions
                    both_action = is_action_pred_play & is_action_true
                    if both_action.any():
                        card_correct += (
                            pred_card[both_action] == card_tgt[both_action]
                        ).sum().item()

                    val_loss_sum += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss_sum / max(val_batches, 1)
            val_accuracy = val_correct / max(val_total, 1)

            action_recall = action_true_pos / max(action_total, 1)
            action_precision = (
                action_true_pos / max(action_true_pos + action_false_pos, 1)
            )
            # F1 = harmonic mean of precision and recall
            if action_precision + action_recall > 0:
                action_f1 = (
                    2 * action_precision * action_recall
                    / (action_precision + action_recall)
                )
            else:
                action_f1 = 0.0
            noop_accuracy = noop_correct / max(noop_total, 1)
            card_accuracy = card_correct / max(action_true_pos, 1)
            current_lr = scheduler.get_last_lr()[0]

            scheduler.step()

            # Record history
            history["train_losses"].append(avg_train_loss)
            history["train_play_losses"].append(avg_play_loss)
            history["train_card_losses"].append(avg_card_loss)
            history["train_pos_losses"].append(avg_pos_loss)
            history["val_losses"].append(avg_val_loss)
            history["val_accuracies"].append(val_accuracy)
            history["val_action_recalls"].append(action_recall)
            history["val_action_precisions"].append(action_precision)
            history["val_action_f1s"].append(action_f1)
            history["val_noop_accuracies"].append(noop_accuracy)
            history["val_card_accuracies"].append(card_accuracy)

            # Print epoch summary
            print(
                f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
                f"Loss: {avg_train_loss:.3f} "
                f"(p={avg_play_loss:.3f} c={avg_card_loss:.3f} "
                f"g={avg_pos_loss:.3f}) | "
                f"F1: {action_f1:.3f} "
                f"R/P: {action_recall:.3f}/{action_precision:.3f} "
                f"Card: {card_accuracy:.3f} | "
                f"Noop: {noop_accuracy:.3f} | "
                f"LR: {current_lr:.2e}"
            )

            # Early stopping on F1 score (balances precision and recall)
            if action_f1 > best_f1:
                best_f1 = action_f1
                best_epoch = epoch
                no_improve = 0

                # Save best checkpoint
                policy.save(os.path.join(output_dir, "best_bc.pt"))
                torch.save(
                    policy.get_feature_extractor_state(),
                    os.path.join(output_dir, "bc_feature_extractor.pt"),
                )
                print(
                    f"  -> Saved best (F1={action_f1:.3f}, "
                    f"R={action_recall:.3f}, P={action_precision:.3f}, "
                    f"tp={action_true_pos}, fp={action_false_pos}, "
                    f"fn={action_false_neg})"
                )
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    print(
                        f"\nEarly stopping at epoch {epoch + 1} "
                        f"(no improvement for {cfg.patience} epochs)"
                    )
                    break

        print("-" * 100)
        print(
            f"Training complete. Best action F1: {best_f1:.4f} "
            f"at epoch {best_epoch + 1}"
        )

        # Save training log
        history["best_action_f1"] = best_f1
        history["best_epoch"] = best_epoch
        log_path = os.path.join(output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training log saved to {log_path}")

        return history


def main() -> None:
    """CLI entry point for BC training.

    Usage:
        python bc_model_module/train_model.py --data_dir data/bc_training/ --output_dir models/bc/
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train BC policy for Clash Royale")
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing .npz files"
    )
    parser.add_argument(
        "--output_dir", default="models/bc/", help="Output directory for checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--play_weight", type=float, default=5.0,
        help="Weight for the 'play' class in play/noop head (default: 5.0)",
    )
    args = parser.parse_args()

    npz_paths = sorted(Path(args.data_dir).glob("*.npz"))
    if not npz_paths:
        print(f"No .npz files found in {args.data_dir}")
        return

    print(f"Found {len(npz_paths)} .npz files in {args.data_dir}")

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        play_weight=args.play_weight,
    )
    trainer = BCTrainer(config)
    trainer.train_loop(npz_paths, args.output_dir)


if __name__ == "__main__":
    main()
