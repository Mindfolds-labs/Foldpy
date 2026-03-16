"""Directional, angular and depth projection operators."""

from __future__ import annotations

from collections.abc import Iterable
import numpy as np

from .exceptions import ValidationError
from .image import ImageGrid
from .utils import sample_nearest


def directional_field(img: ImageGrid, angles: Iterable[float], radius: int | None = None, grayscale: bool = True) -> dict[float, np.ndarray]:
    """Sample rays from image center along provided angles."""
    source = img.grayscale().data if grayscale else img.as_hwc()
    cx, cy = img.center
    out: dict[float, np.ndarray] = {}
    for theta in angles:
        ux, uy = np.cos(theta), np.sin(theta)
        rmax = radius if radius is not None else int(np.floor(img.min_side / 2))
        values = [sample_nearest(source, cx + r * ux, cy + r * uy) for r in range(rmax + 1)]
        out[float(theta)] = np.asarray(values)
    return out


def angular_perspective_matrix(img: ImageGrid, grayscale: bool = True) -> np.ndarray:
    """Build 4-direction angular perspective matrix (NE/NW/SW/SE)."""
    angles = [-np.pi / 4, -3 * np.pi / 4, 3 * np.pi / 4, np.pi / 4]
    field = directional_field(img, angles, grayscale=grayscale)
    max_len = max(len(v) for v in field.values())
    mat = np.full((4, max_len), np.nan, dtype=np.float32)
    for i, theta in enumerate(angles):
        seq = np.asarray(field[float(theta)], dtype=np.float32).reshape(-1, np.asarray(field[float(theta)]).shape[-1] if np.asarray(field[float(theta)]).ndim > 1 else 1).mean(axis=1)
        mat[i, : len(seq)] = seq
    return mat


def depth_order_map(img: ImageGrid) -> np.ndarray:
    """Generate depth order map using distance to nearest border."""
    h, w = img.height, img.width
    yy, xx = np.mgrid[0:h, 0:w]
    return np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype(np.float32)


def depth_hint(img: ImageGrid, diag_weight: float = 0.35, border_weight: float = 0.65) -> np.ndarray:
    """Estimate normalized structural depth map by border and angular asymmetry."""
    if not np.isclose(diag_weight + border_weight, 1.0, atol=1e-3):
        raise ValidationError("diag_weight + border_weight must be close to 1.0")
    border = depth_order_map(img)
    border = border / (border.max() + 1e-6)
    A = angular_perspective_matrix(img)
    asym = np.nanmean(np.abs(A[0] - A[2]) + np.abs(A[1] - A[3])) / 2.0
    z = border_weight * border + diag_weight * float(asym)
    return (z / (z.max() + 1e-6)).astype(np.float32)
