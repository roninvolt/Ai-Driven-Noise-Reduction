from __future__ import annotations

import cv2
import numpy as np


def denoise_salt_pepper(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    return cv2.medianBlur(image, kernel_size)
