"""Region primitives and region graph generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .image import ImageGrid


@dataclass(slots=True, frozen=True)
class Region:
    """Axis-aligned image region."""

    y0: int
    x0: int
    y1: int
    x1: int


def region(img: ImageGrid, y0: int, x0: int, y1: int, x1: int) -> tuple[Region, ImageGrid]:
    """Extract sub-region and return metadata + cropped ImageGrid."""
    out = img.as_hwc()[y0:y1, x0:x1]
    return Region(y0, x0, y1, x1), ImageGrid(out, domain=img.domain, layout="HWC" if out.ndim == 3 else "HW")


def region_graph(img: ImageGrid, patch_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Build adjacency graph among regular patches."""
    ph, pw = patch_size
    nh, nw = img.height // ph, img.width // pw
    ids = np.arange(nh * nw).reshape(nh, nw)
    edges = [(ids[y, x], ids[y, x + 1]) for y in range(nh) for x in range(nw - 1)]
    edges.extend((ids[y, x], ids[y + 1, x]) for y in range(nh - 1) for x in range(nw))
    return {"edges": np.asarray(edges, dtype=np.int32), "grid_shape": np.asarray([nh, nw], dtype=np.int32)}
