"""
Small ResNet card classifier for Clash Royale hand detection.

Classifies card crops (188x235) into one of the deck's card types.
Trained from 8 reference images with heavy augmentation.

Usage:
    # Train
    python card_classifier.py train --data <card_crops_dir> --epochs 50

    # Predict
    python card_classifier.py predict --model <weights.pt> --image <crop.png>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import argparse
from pathlib import Path


# --- Model ---

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class MiniResNet(nn.Module):
    """Tiny ResNet for card classification. ~25K parameters."""

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = ResBlock(16, 16)
        self.layer2 = ResBlock(16, 32, stride=2)
        self.layer3 = ResBlock(32, 64, stride=2)
        self.layer4 = ResBlock(64, 64, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# --- Dataset ---

INPUT_SIZE = (80, 64)  # (H, W) - same as KataCR


# Per-class grey-out probability. Royal Recruits (7 elixir) is greyed out
# most often in gameplay since it's the most expensive card in the deck.
GREYOUT_PROB = {
    "royal-recruits": 0.7,
}
GREYOUT_PROB_DEFAULT = 0.4


def apply_greyout(img):
    """Apply Clash Royale greyed-out card effect to a PIL image."""
    import PIL.ImageEnhance as IE
    sat_factor = torch.empty(1).uniform_(0.0, 0.15).item()
    img = IE.Color(img).enhance(sat_factor)
    bright_factor = torch.empty(1).uniform_(0.6, 0.85).item()
    img = IE.Brightness(img).enhance(bright_factor)
    return img


class CardAugDataset(Dataset):
    """
    Dataset that generates augmented samples from reference card images.
    Each epoch produces `samples_per_class` augmented versions of each card.
    """

    def __init__(self, card_dir, samples_per_class=500, grayscale=False):
        self.card_dir = Path(card_dir)
        self.grayscale = grayscale
        self.samples_per_class = samples_per_class

        # Load reference images
        self.classes = []
        self.images = []
        for f in sorted(self.card_dir.glob("*.png")):
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.classes.append(f.stem)
            self.images.append(img)

        self.num_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        print(f"Loaded {self.num_classes} card classes: {self.classes}")

        # Augmentation pipeline
        in_channels = 1 if grayscale else 3
        aug_list = []
        if grayscale:
            aug_list.append(transforms.Grayscale(num_output_channels=1))
        aug_list.extend([
            transforms.RandomAffine(
                degrees=5, translate=(0.08, 0.08), scale=(0.85, 1.15), shear=3
            ),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] * in_channels,
                std=[0.5] * in_channels,
            ),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
        self.transform = transforms.Compose(aug_list)

    def __len__(self):
        return self.num_classes * self.samples_per_class

    def __getitem__(self, idx):
        class_idx = idx % self.num_classes
        img = self.images[class_idx]
        pil_img = transforms.ToPILImage()(img)
        # Per-class grey-out probability
        class_name = self.classes[class_idx]
        p = GREYOUT_PROB.get(class_name, GREYOUT_PROB_DEFAULT)
        if torch.rand(1).item() < p:
            pil_img = apply_greyout(pil_img)
        tensor = self.transform(pil_img)
        return tensor, class_idx


class CardInferDataset(Dataset):
    """Dataset for inference - just resize and normalize, no augmentation."""

    def __init__(self, images, grayscale=False):
        self.images = images
        self.grayscale = grayscale
        in_channels = 1 if grayscale else 3
        t_list = []
        if grayscale:
            t_list.append(transforms.Grayscale(num_output_channels=1))
        t_list.extend([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] * in_channels,
                std=[0.5] * in_channels,
            ),
        ])
        self.transform = transforms.Compose(t_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        pil_img = transforms.ToPILImage()(img)
        return self.transform(pil_img)


# --- Training ---

def train(card_dir, epochs=50, batch_size=32, lr=1e-3, samples_per_class=500,
          grayscale=False, output_dir=None):
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    dataset = CardAugDataset(card_dir, samples_per_class=samples_per_class,
                             grayscale=grayscale)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    in_channels = 1 if grayscale else 3
    model = MiniResNet(num_classes=dataset.num_classes, in_channels=in_channels)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "models" / "card_classifier"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        acc = correct / total
        avg_loss = total_loss / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc

    # Save model
    save_dict = {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "num_classes": dataset.num_classes,
        "input_size": INPUT_SIZE,
        "grayscale": grayscale,
        "in_channels": in_channels,
    }
    save_path = output_dir / "card_classifier.pt"
    torch.save(save_dict, save_path)
    print(f"\nSaved model to {save_path}")
    print(f"Best training accuracy: {best_acc:.4f}")
    return model, dataset


# --- Inference ---

class CardPredictor:
    """Load a trained card classifier and predict card names from crops."""

    def __init__(self, weights_path):
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        checkpoint = torch.load(weights_path, map_location=self.device,
                                weights_only=False)
        self.classes = checkpoint["classes"]
        self.class_to_idx = checkpoint["class_to_idx"]
        self.grayscale = checkpoint.get("grayscale", False)
        in_channels = checkpoint.get("in_channels", 3)

        self.model = MiniResNet(
            num_classes=checkpoint["num_classes"],
            in_channels=in_channels,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        t_list = []
        if self.grayscale:
            t_list.append(transforms.Grayscale(num_output_channels=1))
        t_list.extend([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] * in_channels,
                std=[0.5] * in_channels,
            ),
        ])
        self.transform = transforms.Compose(t_list)

    def predict(self, bgr_image):
        """Predict card name from a BGR OpenCV image."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = transforms.ToPILImage()(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(1)
        return self.classes[idx.item()], conf.item()

    def predict_hand(self, screenshot):
        """Predict all 4 cards from a full 1080x1920 screenshot."""
        from crop_cards import CARD_SLOTS
        results = []
        for x1, y1, x2, y2 in CARD_SLOTS:
            crop = screenshot[y1:y2, x1:x2]
            name, conf = self.predict(crop)
            results.append((name, conf))
        return results


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Card classifier")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True, help="Directory with card PNGs")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--samples-per-class", type=int, default=500)
    p_train.add_argument("--grayscale", action="store_true")
    p_train.add_argument("--output", default=None)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model", required=True, help="Path to .pt weights")
    p_pred.add_argument("--image", required=True, help="Card crop image")

    args = parser.parse_args()

    if args.command == "train":
        train(
            card_dir=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            samples_per_class=args.samples_per_class,
            grayscale=args.grayscale,
            output_dir=args.output,
        )
    elif args.command == "predict":
        predictor = CardPredictor(args.model)
        img = cv2.imread(args.image)
        name, conf = predictor.predict(img)
        print(f"{name} ({conf:.3f})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
