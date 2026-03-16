"""Pixel graph representation."""

from __future__ import annotations

import numpy as np

from .exceptions import ValidationError
from .image import ImageGrid


def projection_graph(img: ImageGrid, connectivity: int = 4) -> dict[str, np.ndarray]:
    """Build weighted grid graph from grayscale similarities."""
    if connectivity not in {4, 8}:
        raise ValidationError("connectivity must be 4 or 8")
    gray = img.grayscale().data.astype(np.float32, copy=False)
    h, w = gray.shape
    nodes = np.arange(h * w).reshape(h, w)
    edges, weights = [], []

    def add(a_y: int, a_x: int, b_y: int, b_x: int) -> None:
        a, b = nodes[a_y, a_x], nodes[b_y, b_x]
        edges.append((a, b))
        weights.append(float(np.exp(-abs(gray[a_y, a_x] - gray[b_y, b_x]))))

    for y in range(h):
        for x in range(w):
            if x + 1 < w:
                add(y, x, y, x + 1)
            if y + 1 < h:
                add(y, x, y + 1, x)
            if connectivity == 8 and y + 1 < h:
                if x + 1 < w:
                    add(y, x, y + 1, x + 1)
                if x - 1 >= 0:
                    add(y, x, y + 1, x - 1)
    return {"edges": np.asarray(edges, dtype=np.int32), "weights": np.asarray(weights, dtype=np.float32), "shape": np.asarray([h, w], dtype=np.int32)}
