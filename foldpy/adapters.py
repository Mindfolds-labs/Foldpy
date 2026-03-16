from __future__ import annotations

import numpy as np

from .image import ImageGrid


def concentric_layers(img: ImageGrid) -> list[np.ndarray]:
    layers: list[np.ndarray] = []
    arr = img.as_hwc() if img.data.ndim == 3 else img.data
    h, w = arr.shape[:2]
    max_k = min(h, w) // 2

    for k in range(max_k + 1):
        y0, y1 = k, h - k
        x0, x1 = k, w - k
        if y0 >= y1 or x0 >= x1:
            break
        layers.append(arr[y0:y1, x0:x1])

    return layers


def border_series(layers: list[np.ndarray]) -> list[np.ndarray]:
    borders: list[np.ndarray] = []

    for layer in layers:
        border = np.zeros_like(layer)
        if layer.ndim == 2:
            border[0, :] = layer[0, :]
            border[-1, :] = layer[-1, :]
            border[:, 0] = layer[:, 0]
            border[:, -1] = layer[:, -1]
        else:
            border[0, :, :] = layer[0, :, :]
            border[-1, :, :] = layer[-1, :, :]
            border[:, 0, :] = layer[:, 0, :]
            border[:, -1, :] = layer[:, -1, :]
        borders.append(border)

    return borders


def shell_series(layers: list[np.ndarray]) -> list[np.ndarray]:
    return border_series(layers)


def area_series(layers: list[np.ndarray], binary_threshold: float | None = None) -> np.ndarray:
    areas: list[float] = []

    for layer in layers:
        if binary_threshold is None:
            areas.append(float(layer.shape[0] * layer.shape[1]))
        else:
            if layer.ndim == 3:
                gray = layer.mean(axis=2)
            else:
                gray = layer
            areas.append(float(np.sum(gray > binary_threshold)))

    return np.asarray(areas, dtype=np.float32)