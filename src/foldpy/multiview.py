"""Minimal multi-view alignment and correspondence helpers."""

from __future__ import annotations

import numpy as np

from .exceptions import EmptyInputError
from .image import ImageGrid


def multi_view_align(images: list[ImageGrid]) -> dict[str, np.ndarray | int]:
    """Center-crop all images to smallest common shape and stack."""
    if not images:
        raise EmptyInputError("images list is empty")
    h = min(i.height for i in images)
    w = min(i.width for i in images)
    aligned = []
    for img in images:
        arr = img.as_hwc()
        y0 = (arr.shape[0] - h) // 2
        x0 = (arr.shape[1] - w) // 2
        aligned.append(arr[y0:y0 + h, x0:x0 + w])
    stack = np.asarray(aligned, dtype=np.float32)
    return {"aligned": stack, "views": len(images), "height": h, "width": w}
