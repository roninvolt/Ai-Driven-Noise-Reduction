"""Colab + Google Drive runner for training/evaluation workflows."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch

try:
    from google.colab import drive  # type: ignore
except ImportError:  # pragma: no cover
    drive = None


# User-editable roots
DRIVE_ROOT = "/content/drive/MyDrive/AI_Noise_Reduction_Project"
REPO_ROOT = "/content/ai-driven-noise-reduction"

# Auto-managed paths
DATA_SYN = f"{DRIVE_ROOT}/dataset/synthetic"
DATA_CLEAN = f"{DRIVE_ROOT}/dataset/clean"
MODELS_DIR = f"{DRIVE_ROOT}/models"
OUT_DIR = f"{DRIVE_ROOT}/outputs/experiments/exp_001"


def mount_drive() -> None:
    """Mount Google Drive in Colab."""
    if drive is None:
        raise RuntimeError("google.colab is not available. Run this in Google Colab.")
    drive.mount("/content/drive")


def _ensure_dirs() -> None:
    for path in (DATA_SYN, DATA_CLEAN, MODELS_DIR, OUT_DIR):
        os.makedirs(path, exist_ok=True)


def _resolve_repo_root() -> str:
    """Use REPO_ROOT if present, otherwise default to /content clone path."""
    if os.path.isdir(REPO_ROOT):
        return REPO_ROOT

    fallback = "/content/ai-driven-noise-reduction"
    if os.path.isdir(fallback):
        return fallback

    return REPO_ROOT


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        return "cpu"
    return requested


def _run(cmd: list[str], cwd: str) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def run_train(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Run classifier training script."""
    _ensure_dirs()
    repo_root = _resolve_repo_root()
    selected_device = _resolve_device(device)

    cmd = [
        "python3",
        "scripts/train_classifier.py",
        "--data_dir",
        DATA_SYN,
        "--out_dir",
        MODELS_DIR,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--device",
        selected_device,
    ]
    _run(cmd, cwd=repo_root)


def run_eval(device: str = "cuda") -> None:
    """Run classifier evaluation script."""
    _ensure_dirs()
    repo_root = _resolve_repo_root()
    selected_device = _resolve_device(device)

    checkpoint = f"{MODELS_DIR}/noise_classifier_best.pt"
    cmd = [
        "python3",
        "scripts/eval_classifier.py",
        "--checkpoint",
        checkpoint,
        "--data_dir",
        DATA_SYN,
        "--device",
        selected_device,
    ]
    _run(cmd, cwd=repo_root)


def run_experiment(num_per_class: int = 25, device: str = "cuda") -> None:
    """Run experiment script when available."""
    _ensure_dirs()
    repo_root = _resolve_repo_root()
    selected_device = _resolve_device(device)

    script_path = Path(repo_root) / "scripts" / "run_experiments.py"
    if not script_path.exists():
        print(
            "Warning: scripts/run_experiments.py not found. "
            "Create it first to use run_experiment()."
        )
        return

    cmd = [
        "python3",
        "scripts/run_experiments.py",
        "--data_dir",
        DATA_SYN,
        "--clean_dir",
        DATA_CLEAN,
        "--models_dir",
        MODELS_DIR,
        "--out_dir",
        OUT_DIR,
        "--num_per_class",
        str(num_per_class),
        "--device",
        selected_device,
    ]
    _run(cmd, cwd=repo_root)


if __name__ == "__main__":
    mount_drive()
    run_train()
    run_eval()
    run_experiment()
