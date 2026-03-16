from __future__ import annotations

import numpy as np

from .generators import angular_perspective_matrix
from .image import ImageGrid


def structural_depth_axis(img: ImageGrid) -> np.ndarray:
    N = img.min_side
    max_k = N // 2

    z_values = []
    for k in range(max_k + 1):
        s_k = N - 2 * k
        if s_k <= 0:
            break
        z_values.append(1.0 - (s_k / N))

    return np.asarray(z_values, dtype=np.float32)


def depth_order_map(img: ImageGrid) -> np.ndarray:
    h, w = img.height, img.width
    zmap = np.zeros((h, w), dtype=np.int32)
    max_k = img.min_side // 2

    for y in range(h):
        for x in range(w):
            k = min(x, y, w - 1 - x, h - 1 - y)
            zmap[y, x] = min(k, max_k)

    return zmap


def depth_hint(img: ImageGrid, diag_weight: float = 0.35, border_weight: float = 0.65) -> np.ndarray:
    z_ord = depth_order_map(img).astype(np.float32)
    z_ord = z_ord / (z_ord.max() + 1e-6)

    G = angular_perspective_matrix(img, grayscale=True)
    ne, nw, sw, se = G

    max_len = G.shape[1]
    asym = np.zeros(max_len, dtype=np.float32)

    for i in range(max_len):
        vals = []
        if not np.isnan(ne[i]) and not np.isnan(sw[i]):
            vals.append(abs(ne[i] - sw[i]))
        if not np.isnan(nw[i]) and not np.isnan(se[i]):
            vals.append(abs(nw[i] - se[i]))
        asym[i] = float(np.mean(vals)) if vals else 0.0

    asym = asym / (asym.max() + 1e-6)

    h, w = img.height, img.width
    cx, cy = img.center
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rr = rr / (rr.max() + 1e-6)

    idx = np.minimum((rr * (max_len - 1)).astype(np.int32), max_len - 1)
    diag_map = asym[idx]

    z = border_weight * z_ord + diag_weight * diag_map
    z = z / (z.max() + 1e-6)
    return z.astype(np.float32)