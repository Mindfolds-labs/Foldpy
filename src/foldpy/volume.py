"""Volumetric lifting utilities."""

from __future__ import annotations

import numpy as np

from .exceptions import ValidationError


def volume_lift(zmap: np.ndarray, depth_bins: int = 16, threshold: float = 0.05) -> np.ndarray:
    """Lift a 2D depth map to a binary occupancy volume."""
    z = np.asarray(zmap, dtype=np.float32)
    if z.ndim != 2:
        raise ValidationError("zmap must be 2D")
    z = z / (z.max() + 1e-6)
    h, w = z.shape
    out = np.zeros((h, w, depth_bins), dtype=np.float32)
    for d in range(depth_bins):
        level = d / max(1, depth_bins - 1)
        out[:, :, d] = (z >= (level - threshold)).astype(np.float32)
    return out
