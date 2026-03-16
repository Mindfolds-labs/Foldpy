from __future__ import annotations

import numpy as np

from .generators import directional_field
from .image import ImageGrid


def angular_tensor(img: ImageGrid, num_angles: int = 8, radius: int | None = None) -> np.ndarray:
    """
    Retorna tensor angular local/global simplificado:
    shape = [num_angles, max_len]
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False, dtype=np.float32)
    field = directional_field(img, angles=angles, radius=radius, grayscale=True)

    max_len = max(len(v) for v in field.values())
    T = np.full((num_angles, max_len), np.nan, dtype=np.float32)

    for i, theta in enumerate(angles):
        seq = np.asarray(field[float(theta)], dtype=np.float32).reshape(len(field[float(theta)]), -1).mean(axis=1)
        T[i, : len(seq)] = seq

    return T