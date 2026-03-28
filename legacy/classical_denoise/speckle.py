from __future__ import annotations

import cv2
import numpy as np


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def denoise_speckle(image: np.ndarray) -> np.ndarray:
    """Denoise speckle-like noise using a bilateral filter baseline."""
    image_u8 = _to_uint8(image)

    if image_u8.ndim == 2:
        return cv2.bilateralFilter(image_u8, d=9, sigmaColor=75, sigmaSpace=75)

    if image_u8.ndim == 3 and image_u8.shape[2] == 1:
        denoised = cv2.bilateralFilter(image_u8[:, :, 0], d=9, sigmaColor=75, sigmaSpace=75)
        return denoised[:, :, None]

    return cv2.bilateralFilter(image_u8, d=9, sigmaColor=75, sigmaSpace=75)
