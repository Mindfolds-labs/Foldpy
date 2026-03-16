"""Geometric reconstruction and triangulation primitives."""

from __future__ import annotations

import numpy as np

from .exceptions import ValidationError


def triangulate_points(pts1: np.ndarray, pts2: np.ndarray, baseline: float = 1.0, focal: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    """Triangulate stereo correspondences using disparity."""
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)
    if pts1.shape != pts2.shape or pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValidationError("pts1 and pts2 must have shape [N, 2]")
    disparity = pts1[:, 0] - pts2[:, 0]
    z = (focal * baseline) / (np.abs(disparity) + eps)
    x = pts1[:, 0] * z / focal
    y = pts1[:, 1] * z / focal
    return np.stack([x, y, z], axis=1).astype(np.float32)


def surface_mesh_hint(zmap: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a 2D height field into vertices and triangular faces."""
    z = np.asarray(zmap, dtype=np.float32)
    if z.ndim != 2:
        raise ValidationError("zmap must be a 2D array")
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w]
    vertices = np.stack([xx.ravel(), yy.ravel(), z.ravel()], axis=1).astype(np.float32)
    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            i0, i1, i2, i3 = y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1
            faces.extend([(i0, i1, i2), (i1, i3, i2)])
    return {"vertices": vertices, "faces": np.asarray(faces, dtype=np.int32)}
