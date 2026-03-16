from __future__ import annotations

import numpy as np

from .image import ImageGrid


def normalize(img: ImageGrid, mean=None, std=None) -> ImageGrid:
    arr = img.as_hwc().astype(np.float32, copy=False) if img.data.ndim == 3 else img.data.astype(np.float32, copy=False)

    if mean is None:
        mean = arr.mean(axis=(0, 1), keepdims=True) if arr.ndim == 3 else arr.mean()
    if std is None:
        std = arr.std(axis=(0, 1), keepdims=True) if arr.ndim == 3 else arr.std()

    out = (arr - mean) / (std + 1e-6)
    layout = "HWC" if out.ndim == 3 else "HW"
    return ImageGrid(out, domain=img.domain, layout=layout)


def to_chw(img: ImageGrid) -> ImageGrid:
    if img.data.ndim == 2:
        return img
    return ImageGrid(img.as_chw(), domain=img.domain, layout="CHW")


def to_hwc(img: ImageGrid) -> ImageGrid:
    if img.data.ndim == 2:
        return img
    return ImageGrid(img.as_hwc(), domain=img.domain, layout="HWC")


def crop(img: ImageGrid, y0: int, x0: int, y1: int, x1: int) -> ImageGrid:
    arr = img.as_hwc() if img.data.ndim == 3 else img.data
    out = arr[y0:y1, x0:x1]
    layout = "HWC" if out.ndim == 3 else "HW"
    return ImageGrid(out, domain=img.domain, layout=layout)


def resize_nearest(img: ImageGrid, new_h: int, new_w: int) -> ImageGrid:
    arr = img.as_hwc() if img.data.ndim == 3 else img.data
    h, w = arr.shape[:2]

    y_idx = np.clip(np.round(np.linspace(0, h - 1, new_h)).astype(np.int32), 0, h - 1)
    x_idx = np.clip(np.round(np.linspace(0, w - 1, new_w)).astype(np.int32), 0, w - 1)

    if arr.ndim == 2:
        out = arr[y_idx][:, x_idx]
        return ImageGrid(out, domain=img.domain, layout="HW")

    out = arr[y_idx][:, x_idx, :]
    return ImageGrid(out, domain=img.domain, layout="HWC")


def patchify(img: ImageGrid, patch_size: tuple[int, int]) -> np.ndarray:
    arr = img.as_hwc() if img.data.ndim == 3 else img.data
    ph, pw = patch_size
    h, w = arr.shape[:2]

    nh = h // ph
    nw = w // pw

    arr = arr[: nh * ph, : nw * pw]

    patches = []
    for iy in range(nh):
        for ix in range(nw):
            y0, y1 = iy * ph, (iy + 1) * ph
            x0, x1 = ix * pw, (ix + 1) * pw
            patches.append(arr[y0:y1, x0:x1])

    return np.asarray(patches)