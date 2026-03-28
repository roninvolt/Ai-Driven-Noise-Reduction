from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.denoiser.router import DENOISERS, get_denoiser
from src.noise_classifier.classifier import classify_noise


@dataclass
class PipelineOutput:
    predicted_noise: str
    denoised_image: np.ndarray


def run_pipeline(
    image: np.ndarray,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> PipelineOutput:
    """Run classification + routed denoising with AI-first architecture."""
    predicted_noise = classify_noise(image, checkpoint_path=checkpoint_path, device=device)

    # Temporary compatibility layer until AI denoisers are trained.
    denoiser = get_denoiser(predicted_noise)
    denoised = denoiser(image)

    output_noise = predicted_noise if predicted_noise in DENOISERS else "gaussian"
    return PipelineOutput(predicted_noise=output_noise, denoised_image=denoised)
