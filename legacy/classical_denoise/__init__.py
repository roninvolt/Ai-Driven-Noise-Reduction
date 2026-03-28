"""Classical denoising baselines kept as legacy implementations."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .gaussian import denoise_gaussian
from .periodic import denoise_periodic
from .salt_pepper import denoise_salt_pepper
from .speckle import denoise_speckle

DENOISERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gaussian": denoise_gaussian,
    "salt_pepper": denoise_salt_pepper,
    "speckle": denoise_speckle,
    "periodic": denoise_periodic,
}


def get_denoiser(noise: str) -> Callable[[np.ndarray], np.ndarray]:
    return DENOISERS.get(noise, denoise_gaussian)


__all__ = [
    "denoise_gaussian",
    "denoise_salt_pepper",
    "denoise_speckle",
    "denoise_periodic",
    "DENOISERS",
    "get_denoiser",
]
