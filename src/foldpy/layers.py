"""Layer extraction and region decomposition primitives."""

from __future__ import annotations

import numpy as np

from .image import ImageGrid


def concentric_layers(img: ImageGrid) -> list[np.ndarray]:
    """Extract concentric crops from border to center."""
    arr = img.as_hwc()
    h, w = arr.shape[:2]
    layers: list[np.ndarray] = []
    for k in range(min(h, w) // 2 + 1):
        y0, y1 = k, h - k
        x0, x1 = k, w - k
        if y0 >= y1 or x0 >= x1:
            break
        layers.append(arr[y0:y1, x0:x1])
    return layers


def border_series(layers: list[np.ndarray]) -> list[np.ndarray]:
    """Compute border-only tensors for each layer."""
    borders: list[np.ndarray] = []
    for layer in layers:
        border = np.zeros_like(layer)
        border[0] = layer[0]
        border[-1] = layer[-1]
        border[:, 0] = layer[:, 0]
        border[:, -1] = layer[:, -1]
        borders.append(border)
    return borders


def shell_series(layers: list[np.ndarray]) -> list[np.ndarray]:
    """Alias for border-series shell extraction."""
    return border_series(layers)


def area_series(layers: list[np.ndarray], binary_threshold: float | None = None) -> np.ndarray:
    """Compute area profile for each layer.

    Args:
        layers: Concentric layer list.
        binary_threshold: Optional threshold to count active pixels.

    Returns:
        1D float32 vector with one value per layer.
    """
    areas: list[float] = []
    for layer in layers:
        if binary_threshold is None:
            areas.append(float(layer.shape[0] * layer.shape[1]))
        else:
            gray = layer.mean(axis=2) if layer.ndim == 3 else layer
            areas.append(float(np.sum(gray > binary_threshold)))
    return np.asarray(areas, dtype=np.float32)
