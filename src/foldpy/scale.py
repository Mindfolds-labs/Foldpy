"""Scale-space utilities."""

from __future__ import annotations

from .image import ImageGrid
from .transforms import resize_nearest


def scale_pyramid(img: ImageGrid, levels: int = 4, scale_factor: float = 0.5) -> list[ImageGrid]:
    """Build image pyramid with nearest downsampling."""
    pyr = [img]
    cur = img
    for _ in range(1, levels):
        new_h = max(1, int(cur.height * scale_factor))
        new_w = max(1, int(cur.width * scale_factor))
        cur = resize_nearest(cur, new_h, new_w)
        pyr.append(cur)
    return pyr
