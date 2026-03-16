from __future__ import annotations

import numpy as np

from .image import ImageGrid


def projection_graph(img: ImageGrid, connectivity: int = 4) -> dict[str, np.ndarray]:
    gray = img.grayscale().data.astype(np.float32, copy=False)
    h, w = gray.shape

    nodes = np.arange(h * w).reshape(h, w)
    edges = []
    weights = []

    def add_edge(a_y: int, a_x: int, b_y: int, b_x: int) -> None:
        a = nodes[a_y, a_x]
        b = nodes[b_y, b_x]
        wa = gray[a_y, a_x]
        wb = gray[b_y, b_x]
        weight = float(np.exp(-abs(wa - wb)))
        edges.append((a, b))
        weights.append(weight)

    for y in range(h):
        for x in range(w):
            if x + 1 < w:
                add_edge(y, x, y, x + 1)
            if y + 1 < h:
                add_edge(y, x, y + 1, x)
            if connectivity == 8:
                if x + 1 < w and y + 1 < h:
                    add_edge(y, x, y + 1, x + 1)
                if x - 1 >= 0 and y + 1 < h:
                    add_edge(y, x, y + 1, x - 1)

    return {
        "edges": np.asarray(edges, dtype=np.int32),
        "weights": np.asarray(weights, dtype=np.float32),
        "shape": np.asarray([h, w], dtype=np.int32),
    }