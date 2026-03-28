"""Noise classification utilities with model-first and heuristic fallback behavior."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models

from src.noise_classifier.dataset import LABELS
from src.noise_classifier.transforms import build_inference_transform

NOISE_LABELS = tuple(LABELS)


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _fft_peak_ratio(image_rgb: np.ndarray) -> float:
    ratios: list[float] = []
    channels = image_rgb.shape[2] if image_rgb.ndim == 3 else 1

    if channels == 1:
        work_channels = [image_rgb.astype(np.float32)]
    else:
        work_channels = [image_rgb[..., i].astype(np.float32) for i in range(channels)]

    for channel in work_channels:
        fft = np.fft.fftshift(np.fft.fft2(channel))
        power = np.abs(fft)
        power_mean = float(np.mean(power))
        power_max = float(np.max(power))
        if power_mean <= 0:
            ratios.append(0.0)
        else:
            ratios.append(power_max / power_mean)

    return float(np.mean(ratios))


def _classify_noise_heuristic(image: np.ndarray) -> str:
    """Heuristic placeholder used when a trained classifier is unavailable."""
    rgb = _to_rgb_uint8(image)

    p01 = np.percentile(rgb, 1)
    p99 = np.percentile(rgb, 99)
    extreme_ratio = float(np.mean((rgb <= p01) | (rgb >= p99)))
    if extreme_ratio > 0.18:
        return "salt_pepper"

    if _fft_peak_ratio(rgb) > 30:
        return "periodic"

    intensity = np.mean(rgb, axis=2).astype(np.float32)
    mean = float(np.mean(intensity)) + 1e-6
    std = float(np.std(intensity))
    if (std / mean) > 0.55:
        return "speckle"

    return "gaussian"


def _create_resnet18(num_classes: int):
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_trained_classifier(
    checkpoint_path: str,
    device: str,
) -> tuple[torch.nn.Module, list[str], Callable[[Image.Image], torch.Tensor]]:
    """Load trained classifier and its preprocessing metadata from a checkpoint."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    torch_device = torch.device(device)
    checkpoint = torch.load(path, map_location=torch_device)

    labels = checkpoint.get("labels", list(LABELS))
    input_size = int(checkpoint.get("input_size", 224))
    mean = tuple(checkpoint.get("normalize_mean", (0.485, 0.456, 0.406)))
    std = tuple(checkpoint.get("normalize_std", (0.229, 0.224, 0.225)))

    model = _create_resnet18(num_classes=len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch_device)
    model.eval()

    preprocess = build_inference_transform(input_size=input_size, mean=mean, std=std)
    return model, labels, preprocess


def classify_noise_model(
    image: np.ndarray,
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[str, float]:
    """Classify noise using the trained model and return label with confidence."""
    model, labels, preprocess = load_trained_classifier(checkpoint_path=checkpoint_path, device=device)
    rgb = _to_rgb_uint8(image)

    pil_image = Image.fromarray(rgb)
    tensor = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    index = int(pred_idx.item())
    confidence = float(conf.item())
    if index < 0 or index >= len(labels):
        return "gaussian", confidence

    label = str(labels[index])
    return label, confidence


def classify_noise(
    image: np.ndarray,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> str:
    """Classify image noise using model if available; fallback to heuristic otherwise."""
    candidate_checkpoint = checkpoint_path or os.getenv("NOISE_CLASSIFIER_CHECKPOINT")
    candidate_device = os.getenv("NOISE_CLASSIFIER_DEVICE", device)

    if candidate_checkpoint:
        try:
            label, _ = classify_noise_model(
                image=image,
                checkpoint_path=candidate_checkpoint,
                device=candidate_device,
            )
            if label in NOISE_LABELS:
                return label
        except Exception:
            pass

    return _classify_noise_heuristic(image)
