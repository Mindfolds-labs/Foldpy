"""Image transform operators."""

from __future__ import annotations

import numpy as np

from .image import ImageGrid


def normalize(img: ImageGrid, mean: float | np.ndarray | None = None, std: float | np.ndarray | None = None) -> ImageGrid:
    """Normalize image tensor with optional mean/std."""
    arr = img.as_hwc().astype(np.float32, copy=False)
    mean = arr.mean(axis=(0, 1), keepdims=True) if mean is None and arr.ndim == 3 else (arr.mean() if mean is None else mean)
    std = arr.std(axis=(0, 1), keepdims=True) if std is None and arr.ndim == 3 else (arr.std() if std is None else std)
    out = (arr - mean) / (std + 1e-6)
    return ImageGrid(out, domain=img.domain, layout="HWC" if out.ndim == 3 else "HW")


def resize_nearest(img: ImageGrid, new_h: int, new_w: int) -> ImageGrid:
    """Resize using nearest-neighbor interpolation."""
    arr = img.as_hwc()
    h, w = arr.shape[:2]
    y_idx = np.clip(np.round(np.linspace(0, h - 1, new_h)).astype(np.int32), 0, h - 1)
    x_idx = np.clip(np.round(np.linspace(0, w - 1, new_w)).astype(np.int32), 0, w - 1)
    out = arr[y_idx][:, x_idx] if arr.ndim == 2 else arr[y_idx][:, x_idx, :]
    return ImageGrid(out, domain=img.domain, layout="HWC" if out.ndim == 3 else "HW")


def patchify(img: ImageGrid, patch_size: tuple[int, int]) -> np.ndarray:
    """Split image into non-overlapping patches."""
    arr = img.as_hwc()
    ph, pw = patch_size
    h, w = arr.shape[:2]
    nh, nw = h // ph, w // pw
    patches = [arr[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] for i in range(nh) for j in range(nw)]
    return np.asarray(patches)
