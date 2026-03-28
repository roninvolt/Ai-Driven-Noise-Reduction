from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models

from src.noise_classifier.dataset import LABELS, NoiseDataset
from src.noise_classifier.transforms import get_val_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained noise classifier")
    parser.add_argument("--checkpoint", type=str, default="models/noise_classifier_best.pt")
    parser.add_argument("--data_dir", type=str, default="dataset/synthetic")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def create_model(num_classes: int):
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    labels = checkpoint.get("labels", list(LABELS))

    dataset = NoiseDataset(root_dir=Path(args.data_dir), labels=labels, transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    n_classes = len(labels)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            t = targets.cpu().numpy()
            p = preds.cpu().numpy()
            for ti, pi in zip(t, p):
                confusion[int(ti), int(pi)] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / max(1, total)

    print("Confusion matrix:")
    print(confusion)
    print("Per-class accuracy:")
    for idx, label in enumerate(labels):
        row_total = int(confusion[idx].sum())
        class_acc = confusion[idx, idx] / row_total if row_total > 0 else 0.0
        print(f"- {label}: {class_acc:.4f}")

    print(f"Overall accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
