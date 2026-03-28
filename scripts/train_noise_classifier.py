from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models

from src.noise_classifier.dataset import LABELS, NoiseDataset
from src.noise_classifier.transforms import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, get_train_transforms, get_val_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 noise classifier")
    parser.add_argument("--data_dir", type=str, default="dataset/synthetic")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(num_classes: int) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def split_indices(n_items: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = np.arange(n_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_size = max(1, int(n_items * val_ratio))
    if val_size >= n_items:
        val_size = n_items - 1

    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return train_indices, val_indices


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = NoiseDataset(root_dir=data_dir, labels=LABELS, transform=None)
    train_indices, val_indices = split_indices(len(base_dataset), args.val_ratio, args.seed)

    train_dataset = NoiseDataset(root_dir=data_dir, labels=LABELS, transform=get_train_transforms())
    val_dataset = NoiseDataset(root_dir=data_dir, labels=LABELS, transform=get_val_transforms())

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    checkpoint_path = out_dir / "noise_classifier_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * targets.size(0)
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += targets.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                val_correct += (logits.argmax(dim=1) == targets).sum().item()
                val_total += targets.size(0)

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": list(LABELS),
                    "input_size": INPUT_SIZE,
                    "normalize_mean": list(IMAGENET_MEAN),
                    "normalize_std": list(IMAGENET_STD),
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
