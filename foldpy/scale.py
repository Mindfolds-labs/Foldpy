from __future__ import annotations

from .image import ImageGrid
from .transforms import resize_nearest


def scale_pyramid(img: ImageGrid, levels: int = 4, scale_factor: float = 0.5) -> list[ImageGrid]:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if not (0.0 < scale_factor < 1.0):
        raise ValueError("scale_factor must be in (0,1)")

    pyramid = [img]
    current = img

    for _ in range(1, levels):
        new_h = max(1, int(round(current.height * scale_factor)))
        new_w = max(1, int(round(current.width * scale_factor)))
        current = resize_nearest(current, new_h, new_w)
        pyramid.append(current)

    return pyramid