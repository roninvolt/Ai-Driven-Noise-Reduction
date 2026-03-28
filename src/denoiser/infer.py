"""Inference helpers for denoiser and pipeline integration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.pipeline import PipelineOutput, run_pipeline


def run_inference(
    image: np.ndarray,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> PipelineOutput:
    """Run in-memory inference through the unified pipeline."""
    return run_pipeline(image=image, checkpoint_path=checkpoint_path, device=device)


def run_inference_from_path(
    image_path: str | Path,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> PipelineOutput:
    """Read an image from disk and run denoising pipeline inference."""
    path = Path(image_path).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return run_inference(image=image, checkpoint_path=checkpoint_path, device=device)
