from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legfit, legval

from .depth import depth_hint
from .image import ImageGrid
from .layers import concentric_layers, area_series


def legendre_encode(series: np.ndarray, order: int = 6) -> np.ndarray:
    y = np.asarray(series, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("series must be 1D")
    if len(y) < 2:
        raise ValueError("series must have length >= 2")

    x = np.linspace(-1.0, 1.0, len(y), dtype=np.float32)
    coeffs = legfit(x, y, deg=min(order, len(y) - 1))
    return coeffs.astype(np.float32)


def legendre_decode(coeffs: np.ndarray, size: int) -> np.ndarray:
    c = np.asarray(coeffs, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return legval(x, c).astype(np.float32)


def legendre_signature(img: ImageGrid, source: str = "area", order: int = 6) -> dict[str, np.ndarray]:
    if source == "area":
        layers = concentric_layers(img)
        series = area_series(layers, binary_threshold=None)
    elif source == "depth":
        z = depth_hint(img)
        series = z.mean(axis=1).astype(np.float32)
    elif source == "radial":
        gray = img.grayscale().data.astype(np.float32, copy=False)
        cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
        radius = min(cy, cx)
        series = np.array([gray[cy, max(0, cx - r):min(gray.shape[1], cx + r + 1)].mean() for r in range(radius)], dtype=np.float32)
    else:
        raise ValueError("source must be one of: area, depth, radial")

    coeffs = legendre_encode(series, order=order)
    recon = legendre_decode(coeffs, size=len(series))

    return {
        "series": series.astype(np.float32),
        "coeffs": coeffs,
        "reconstruction": recon,
    }