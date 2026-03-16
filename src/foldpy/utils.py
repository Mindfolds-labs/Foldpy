"""Low-level numeric helpers."""

from __future__ import annotations

import numpy as np


def clamp_int(value: float, low: int, high: int) -> int:
    """Clamp a floating-point value into integer bounds."""
    return max(low, min(high, int(round(value))))


def sample_nearest(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """Nearest-neighbor sample from HW/HWC array."""
    h, w = image.shape[:2]
    xi = clamp_int(x, 0, w - 1)
    yi = clamp_int(y, 0, h - 1)
    return image[yi, xi]


def safe_norm(x: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray:
    """Compute stable Euclidean norm with epsilon."""
    return np.sqrt(np.sum(np.square(x), axis=axis, keepdims=keepdims) + 1e-12)
