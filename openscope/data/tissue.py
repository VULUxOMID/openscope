"""Tissue detection.

H&E-stained tissue is strongly colored, while bare glass is close to white or gray.
Saturation separates them well, and Otsu's method picks the cut automatically
for each slide or camera frame.
"""

from __future__ import annotations

import numpy as np


def rgb_to_saturation(rgb: np.ndarray) -> np.ndarray:
    """Return HSV saturation in [0, 1] for an HxWx3 uint8 RGB image."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {rgb.shape}")
    img = rgb.astype(np.float32) / 255.0
    high = img.max(axis=-1)
    low = img.min(axis=-1)
    return np.where(high > 0, (high - low) / np.maximum(high, 1e-8), 0.0).astype(np.float32)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold for values in [0, 1]. Values >= the result form the upper class."""
    hist, edges = np.histogram(values.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2
    weight_low = np.cumsum(hist)
    weight_high = total - weight_low
    sum_low = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_low = sum_low / weight_low
        mean_high = (sum_low[-1] - sum_low) / weight_high
        between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between = np.nan_to_num(between)
    return float(edges[int(np.argmax(between)) + 1])


def tissue_mask(rgb: np.ndarray, min_saturation: float = 0.08) -> np.ndarray:
    """Boolean mask of pixels that look like stained tissue.

    ``min_saturation`` is a floor for the Otsu cut, so a frame of empty glass
    is not split into "tissue" and "background" by noise alone.
    0.08 is about 20 on a 0-255 scale, a common floor for H&E tissue detection.
    """
    saturation = rgb_to_saturation(rgb)
    threshold = max(otsu_threshold(saturation), min_saturation)
    return saturation >= threshold
