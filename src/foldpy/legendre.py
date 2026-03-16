"""Legendre signatures for structural profiling."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legfit, legval

from .image import ImageGrid
from .layers import area_series, concentric_layers
from .projection import depth_hint


def legendre_encode(series: np.ndarray, order: int = 6) -> np.ndarray:
    """Encode a 1D series into Legendre coefficients."""
    y = np.asarray(series, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, len(y), dtype=np.float32)
    return legfit(x, y, deg=min(order, len(y) - 1)).astype(np.float32)


def legendre_decode(coeffs: np.ndarray, size: int) -> np.ndarray:
    """Reconstruct a sequence from Legendre coefficients."""
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return legval(x, np.asarray(coeffs, dtype=np.float32)).astype(np.float32)


def legendre_signature(img: ImageGrid, source: str = "area", order: int = 6) -> dict[str, np.ndarray]:
    """Compute FoldPy Legendre signatures from area/depth/radial source."""
    if source == "area":
        series = area_series(concentric_layers(img))
    elif source == "depth":
        series = depth_hint(img).mean(axis=1).astype(np.float32)
    else:
        gray = img.grayscale().data
        cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
        r = min(cy, cx)
        series = np.asarray([gray[cy, max(0, cx - k): min(gray.shape[1], cx + k + 1)].mean() for k in range(1, r + 1)], dtype=np.float32)
    coeffs = legendre_encode(series, order=order)
    return {"series": series, "coeffs": coeffs, "reconstruction": legendre_decode(coeffs, len(series))}
