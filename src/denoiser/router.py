"""Routing for noise-specific denoisers.

Temporary compatibility layer until AI denoisers are trained.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from legacy.classical_denoise import DENOISERS as LEGACY_DENOISERS
from legacy.classical_denoise import denoise_gaussian

DENOISERS: dict[str, Callable[[np.ndarray], np.ndarray]] = dict(LEGACY_DENOISERS)


def get_denoiser(noise: str) -> Callable[[np.ndarray], np.ndarray]:
    return DENOISERS.get(noise, denoise_gaussian)
