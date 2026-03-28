from __future__ import annotations

import numpy as np


def _as_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    base = _as_float01(image)
    noisy = np.clip(base + np.random.normal(0.0, sigma, size=base.shape), 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    noisy = np.copy(image)
    num_pixels = int(amount * image.shape[0] * image.shape[1])
    if num_pixels == 0:
        return noisy.astype(np.uint8)

    ys = np.random.randint(0, image.shape[0], num_pixels)
    xs = np.random.randint(0, image.shape[1], num_pixels)
    half = num_pixels // 2
    noisy[ys[:half], xs[:half]] = 255
    noisy[ys[half:], xs[half:]] = 0
    return noisy.astype(np.uint8)


def add_speckle_noise(image: np.ndarray, intensity: float = 0.15) -> np.ndarray:
    base = _as_float01(image)
    noisy = np.clip(base + base * np.random.normal(0.0, intensity, size=base.shape), 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def add_periodic_noise(image: np.ndarray, amplitude: float = 0.15, frequency: float = 8.0) -> np.ndarray:
    base = _as_float01(image)
    h, w = base.shape[:2]
    x = np.linspace(0, 2 * np.pi * frequency, w, dtype=np.float32)
    sinusoid = amplitude * np.sin(x)
    pattern = np.tile(sinusoid, (h, 1))
    if base.ndim == 3:
        pattern = pattern[..., None]
    noisy = np.clip(base + pattern, 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)
