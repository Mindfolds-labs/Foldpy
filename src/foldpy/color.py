"""Color space conversion helpers."""

from __future__ import annotations

import numpy as np

from .exceptions import ValidationError
from .image import ImageGrid

_RGB_TO_LMS = np.array(
    [[0.31399022, 0.63951294, 0.04649755], [0.15537241, 0.75789446, 0.08670142], [0.01775239, 0.10944209, 0.87256922]],
    dtype=np.float32,
)
_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS).astype(np.float32)


def _convert(img: ImageGrid, matrix: np.ndarray, domain: str) -> ImageGrid:
    arr = img.as_hwc().astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValidationError("Color conversion expects HWC image with 3 channels")
    out = arr.reshape(-1, 3) @ matrix.T
    return ImageGrid(out.reshape(arr.shape), domain=domain, layout="HWC")


def rgb_to_lms(img: ImageGrid) -> ImageGrid:
    """Convert RGB ImageGrid to LMS."""
    return _convert(img, _RGB_TO_LMS, domain="lms")


def lms_to_rgb(img: ImageGrid) -> ImageGrid:
    """Convert LMS ImageGrid to RGB."""
    return _convert(img, _LMS_TO_RGB, domain="visual")
