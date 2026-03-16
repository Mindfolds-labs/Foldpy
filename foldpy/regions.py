from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .image import ImageGrid


@dataclass(slots=True)
class Region:
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)

    @property
    def area(self) -> int:
        h, w = self.shape
        return max(0, h) * max(0, w)


def region(img: ImageGrid, y0: int, x0: int, y1: int, x1: int) -> tuple[Region, ImageGrid]:
    arr = img.as_hwc() if img.data.ndim == 3 else img.data
    out = arr[y0:y1, x0:x1]
    layout = "HWC" if out.ndim == 3 else "HW"
    return Region(y0, x0, y1, x1), ImageGrid(out, domain=img.domain, layout=layout)


def region_graph(img: ImageGrid, patch_size: tuple[int, int]) -> dict[str, np.ndarray]:
    ph, pw = patch_size
    h, w = img.height, img.width
    nh = h // ph
    nw = w // pw

    region_ids = np.arange(nh * nw).reshape(nh, nw)
    edges = []

    for iy in range(nh):
        for ix in range(nw):
            a = region_ids[iy, ix]
            if ix + 1 < nw:
                edges.append((a, region_ids[iy, ix + 1]))
            if iy + 1 < nh:
                edges.append((a, region_ids[iy + 1, ix]))

    return {
        "edges": np.asarray(edges, dtype=np.int32),
        "grid_shape": np.asarray([nh, nw], dtype=np.int32),
        "patch_size": np.asarray([ph, pw], dtype=np.int32),
    }