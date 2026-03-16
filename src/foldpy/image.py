"""Core image container abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .exceptions import ValidationError


@dataclass(slots=True)
class ImageGrid:
    """Represents a 2D/3D image tensor.

    Args:
        data: Image array in HW, HWC, or CHW format.
        domain: Semantic domain label (e.g., ``visual``, ``lms``).
        layout: Tensor layout identifier: ``HW``, ``HWC``, or ``CHW``.
    """

    data: np.ndarray
    domain: str = "visual"
    layout: str = "HWC"

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            self.data = np.asarray(self.data)
        if self.data.ndim not in (2, 3):
            raise ValidationError("ImageGrid only supports 2D/3D arrays")
        if self.data.ndim == 2:
            self.layout = "HW"
        if self.layout not in {"HW", "HWC", "CHW"}:
            raise ValidationError("layout must be one of HW/HWC/CHW")

    @property
    def height(self) -> int:
        return int(self.as_hwc().shape[0])

    @property
    def width(self) -> int:
        return int(self.as_hwc().shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.data.ndim == 2 else int(self.as_hwc().shape[2])

    @property
    def min_side(self) -> int:
        return min(self.height, self.width)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.width - 1) / 2.0, (self.height - 1) / 2.0)

    def as_hwc(self) -> np.ndarray:
        """Return image in HWC or HW format."""
        if self.data.ndim == 2:
            return self.data
        return self.data if self.layout == "HWC" else np.transpose(self.data, (1, 2, 0))

    def as_chw(self) -> np.ndarray:
        """Return image in CHW or HW format."""
        if self.data.ndim == 2:
            return self.data
        return self.data if self.layout == "CHW" else np.transpose(self.data, (2, 0, 1))

    def grayscale(self) -> "ImageGrid":
        """Return grayscale version as ``HW`` float32 image."""
        if self.data.ndim == 2:
            gray = self.data.astype(np.float32, copy=False)
        else:
            gray = self.as_hwc().astype(np.float32, copy=False).mean(axis=2)
        return ImageGrid(gray, domain=self.domain, layout="HW")

    def summary(self) -> dict[str, Any]:
        """Return metadata summary dictionary."""
        return {
            "shape": tuple(self.data.shape),
            "layout": self.layout,
            "domain": self.domain,
            "dtype": str(self.data.dtype),
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
        }
