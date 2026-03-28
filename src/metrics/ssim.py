from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def ssim_score(target: np.ndarray, prediction: np.ndarray) -> float:
    channel_axis = -1 if target.ndim == 3 else None
    return float(structural_similarity(target, prediction, data_range=255, channel_axis=channel_axis))
