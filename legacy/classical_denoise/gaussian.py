from __future__ import annotations

import cv2
import numpy as np


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def denoise_gaussian(image: np.ndarray) -> np.ndarray:
    """Denoise Gaussian noise with OpenCV Non-Local Means."""
    image_u8 = _to_uint8(image)

    if image_u8.ndim == 2:
        return cv2.fastNlMeansDenoising(image_u8, None, h=12, templateWindowSize=7, searchWindowSize=21)

    if image_u8.ndim == 3 and image_u8.shape[2] == 1:
        denoised = cv2.fastNlMeansDenoising(
            image_u8[:, :, 0], None, h=12, templateWindowSize=7, searchWindowSize=21
        )
        return denoised[:, :, None]

    return cv2.fastNlMeansDenoisingColored(
        image_u8, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )
