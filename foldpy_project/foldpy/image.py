from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class ImageGrid:
    """
    Representa uma imagem como grade discreta.

    Layouts suportados:
    - HW   : imagem 2D em escala de cinza
    - HWC  : altura, largura, canais
    - CHW  : canais, altura, largura
    """

    data: np.ndarray
    domain: str = "visual"
    layout: str = "HWC"

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            self.data = np.asarray(self.data)

        if self.data.ndim not in (2, 3):
            raise ValueError("ImageGrid only supports 2D or 3D arrays")

        if self.data.ndim == 2 and self.layout != "HW":
            self.layout = "HW"

        if self.layout not in ("HW", "HWC", "CHW"):
            raise ValueError("layout must be one of: HW, HWC, CHW")

        if self.data.ndim == 3:
            if self.layout == "HWC" and self.data.shape[2] <= 0:
                raise ValueError("invalid HWC image")
            if self.layout == "CHW" and self.data.shape[0] <= 0:
                raise ValueError("invalid CHW image")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def height(self) -> int:
        if self.data.ndim == 2:
            return int(self.data.shape[0])

        if self.layout == "HWC":
            return int(self.data.shape[0])

        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        if self.data.ndim == 2:
            return int(self.data.shape[1])

        if self.layout == "HWC":
            return int(self.data.shape[1])

        return int(self.data.shape[2])

    @property
    def channels(self) -> int:
        if self.data.ndim == 2:
            return 1

        if self.layout == "HWC":
            return int(self.data.shape[2])

        return int(self.data.shape[0])

    @property
    def min_side(self) -> int:
        return min(self.height, self.width)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.width - 1) / 2.0, (self.height - 1) / 2.0)

    def copy(self) -> "ImageGrid":
        return ImageGrid(
            self.data.copy(),
            domain=self.domain,
            layout=self.layout,
        )

    def astype(self, dtype) -> "ImageGrid":
        return ImageGrid(
            self.data.astype(dtype, copy=False),
            domain=self.domain,
            layout=self.layout,
        )

    def to_float32(self) -> "ImageGrid":
        return self.astype(np.float32)

    def to_uint8(self) -> "ImageGrid":
        arr = self.data
        if arr.dtype.kind == "f":
            arr = np.clip(arr, 0.0, 255.0)
        return ImageGrid(
            arr.astype(np.uint8, copy=False),
            domain=self.domain,
            layout=self.layout,
        )

    def as_hwc(self) -> np.ndarray:
        """
        Retorna a imagem em layout HWC ou HW.
        """
        if self.data.ndim == 2:
            return self.data

        if self.layout == "HWC":
            return self.data

        return np.transpose(self.data, (1, 2, 0))

    def as_chw(self) -> np.ndarray:
        """
        Retorna a imagem em layout CHW ou HW.
        """
        if self.data.ndim == 2:
            return self.data

        if self.layout == "CHW":
            return self.data

        return np.transpose(self.data, (2, 0, 1))

    def as_hw(self) -> np.ndarray:
        """
        Retorna a imagem em escala de cinza (HW).
        """
        if self.data.ndim == 2:
            return self.data

        return self.as_hwc().mean(axis=2)

    def to_hwc(self) -> "ImageGrid":
        if self.data.ndim == 2:
            return ImageGrid(self.data, domain=self.domain, layout="HW")

        return ImageGrid(self.as_hwc(), domain=self.domain, layout="HWC")

    def to_chw(self) -> "ImageGrid":
        if self.data.ndim == 2:
            return ImageGrid(self.data, domain=self.domain, layout="HW")

        return ImageGrid(self.as_chw(), domain=self.domain, layout="CHW")

    def grayscale(self) -> "ImageGrid":
        gray = self.as_hw().astype(np.float32, copy=False)
        return ImageGrid(gray, domain=self.domain, layout="HW")

    def normalize_01(self) -> "ImageGrid":
        arr = self.data.astype(np.float32, copy=False)
        arr_min = float(arr.min())
        arr_max = float(arr.max())

        if arr_max - arr_min < 1e-12:
            out = np.zeros_like(arr, dtype=np.float32)
        else:
            out = (arr - arr_min) / (arr_max - arr_min)

        return ImageGrid(out, domain=self.domain, layout=self.layout)

    def channel(self, idx: int) -> "ImageGrid":
        if self.data.ndim == 2:
            if idx != 0:
                raise IndexError("grayscale image has only one channel")
            return ImageGrid(self.data, domain=self.domain, layout="HW")

        if self.layout == "HWC":
            return ImageGrid(self.data[:, :, idx], domain=self.domain, layout="HW")

        return ImageGrid(self.data[idx, :, :], domain=self.domain, layout="HW")

    def summary(self) -> dict:
        return {
            "shape": self.shape,
            "layout": self.layout,
            "domain": self.domain,
            "dtype": str(self.dtype),
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "min_side": self.min_side,
            "center": self.center,
        }

    def __repr__(self) -> str:
        return (
            f"ImageGrid(shape={self.shape}, layout={self.layout}, "
            f"domain={self.domain}, dtype={self.dtype})"
        )