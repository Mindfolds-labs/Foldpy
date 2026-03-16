from __future__ import annotations

import numpy as np


def volume_lift(zmap: np.ndarray, depth_bins: int = 16, threshold: float = 0.05) -> np.ndarray:
    if zmap.ndim != 2:
        raise ValueError("zmap must be a 2D array")

    z = zmap.astype(np.float32, copy=False)
    z = z / (z.max() + 1e-6)

    h, w = z.shape
    volume = np.zeros((h, w, depth_bins), dtype=np.float32)

    for d in range(depth_bins):
        level = d / max(1, depth_bins - 1)
        volume[:, :, d] = (z >= (level - threshold)).astype(np.float32)

    return volume