from __future__ import annotations

import numpy as np

from .image import ImageGrid


_RGB_TO_LMS = np.array(
    [
        [0.31399022, 0.63951294, 0.04649755],
        [0.15537241, 0.75789446, 0.08670142],
        [0.01775239, 0.10944209, 0.87256922],
    ],
    dtype=np.float32,
)

_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS).astype(np.float32)


def rgb_to_lms(img: ImageGrid) -> ImageGrid:
    arr = img.as_hwc().astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("rgb_to_lms expects HWC image with 3 channels")

    flat = arr.reshape(-1, 3)
    out = flat @ _RGB_TO_LMS.T
    out = out.reshape(arr.shape)

    return ImageGrid(out, domain="lms", layout="HWC")


def lms_to_rgb(img: ImageGrid) -> ImageGrid:
    arr = img.as_hwc().astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("lms_to_rgb expects HWC image with 3 channels")

    flat = arr.reshape(-1, 3)
    out = flat @ _LMS_TO_RGB.T
    out = np.clip(out, 0.0, 1.0 if out.max() <= 1.5 else 255.0)
    out = out.reshape(arr.shape)

    return ImageGrid(out, domain="visual", layout="HWC")