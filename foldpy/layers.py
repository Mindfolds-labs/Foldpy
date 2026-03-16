from __future__ import annotations

from typing import Iterable

import numpy as np

from .image import ImageGrid
from .utils import sample_nearest


def _ray_length_from_center(img: ImageGrid, theta: float) -> int:
    cx, cy = img.center
    ux, uy = np.cos(theta), np.sin(theta)
    candidates = []

    if abs(ux) > 1e-8:
        candidates.append((0 - cx) / ux)
        candidates.append(((img.width - 1) - cx) / ux)

    if abs(uy) > 1e-8:
        candidates.append((0 - cy) / uy)
        candidates.append(((img.height - 1) - cy) / uy)

    positive = [t for t in candidates if t >= 0]
    if not positive:
        return 0
    return int(np.floor(min(positive)))


def directional_field(
    img: ImageGrid,
    angles: Iterable[float],
    radius: int | None = None,
    grayscale: bool = True,
) -> dict[float, np.ndarray]:
    source = img.grayscale().data if grayscale else img.as_hwc()
    cx, cy = img.center
    field: dict[float, np.ndarray] = {}

    for theta in angles:
        rmax = _ray_length_from_center(img, theta) if radius is None else radius
        samples = []
        ux, uy = np.cos(theta), np.sin(theta)

        for r in range(rmax + 1):
            x = cx + r * ux
            y = cy + r * uy
            samples.append(sample_nearest(source, x, y))

        field[theta] = np.asarray(samples)

    return field


def diagonal_generators(img: ImageGrid, grayscale: bool = True) -> dict[str, np.ndarray]:
    angles = {
        "NE": -np.pi / 4,
        "NW": -3 * np.pi / 4,
        "SW": 3 * np.pi / 4,
        "SE": np.pi / 4,
    }
    field = directional_field(img, angles.values(), grayscale=grayscale)
    return {name: field[theta] for name, theta in angles.items()}


def lateral_generators(img: ImageGrid, grayscale: bool = True) -> dict[str, np.ndarray]:
    angles = {"E": 0.0, "N": -np.pi / 2, "W": np.pi, "S": np.pi / 2}
    field = directional_field(img, angles.values(), grayscale=grayscale)
    return {name: field[theta] for name, theta in angles.items()}


def angular_perspective_matrix(img: ImageGrid, grayscale: bool = True) -> np.ndarray:
    gens = diagonal_generators(img, grayscale=grayscale)
    seqs = [gens["NE"], gens["NW"], gens["SW"], gens["SE"]]
    max_len = max(len(s) for s in seqs)

    G = np.full((4, max_len), np.nan, dtype=np.float32)
    for i, seq in enumerate(seqs):
        flat = np.asarray(seq, dtype=np.float32).reshape(len(seq), -1).mean(axis=1)
        G[i, : len(flat)] = flat
    return G