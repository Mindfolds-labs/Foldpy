"""Tensor-oriented structural operators."""

from __future__ import annotations

import numpy as np

from .image import ImageGrid
from .projection import directional_field


def angular_tensor(img: ImageGrid, num_angles: int = 8, radius: int | None = None) -> np.ndarray:
    """Build angular tensor over uniformly sampled directions."""
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False, dtype=np.float32)
    field = directional_field(img, angles=angles, radius=radius, grayscale=True)
    max_len = max(len(v) for v in field.values())
    tensor = np.full((num_angles, max_len), np.nan, dtype=np.float32)
    for i, angle in enumerate(angles):
        seq = np.asarray(field[float(angle)], dtype=np.float32)
        seq = seq.reshape(seq.shape[0], -1).mean(axis=1)
        tensor[i, : len(seq)] = seq
    return tensor
