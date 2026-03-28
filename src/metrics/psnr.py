from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def psnr_score(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, prediction, data_range=255))
