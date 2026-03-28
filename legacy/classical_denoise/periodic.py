from __future__ import annotations

import cv2
import numpy as np


def denoise_periodic(image: np.ndarray, notch_radius: int = 10) -> np.ndarray:
    """Simple FFT notch around center frequencies as a baseline periodic denoiser."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    mask = np.ones_like(gray, dtype=np.float32)
    mask[cy - notch_radius : cy + notch_radius, cx - notch_radius : cx + notch_radius] = 0.0

    filtered = f * mask
    rec = np.fft.ifft2(np.fft.ifftshift(filtered))
    rec = np.abs(rec)
    rec = np.clip(rec, 0, 255).astype(np.uint8)

    if image.ndim == 3:
        return cv2.cvtColor(rec, cv2.COLOR_GRAY2BGR)
    return rec
