from __future__ import annotations

import numpy as np

from .image import ImageGrid


def edge_map(img: ImageGrid, normalize: bool = True) -> np.ndarray:
    """
    Mapa de bordas simples via gradiente finito.
    Retorna magnitude do gradiente.
    """
    gray = img.grayscale().data.astype(np.float32, copy=False)

    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)

    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    mag = np.sqrt(gx * gx + gy * gy)

    if normalize:
        mag = mag / (mag.max() + 1e-6)

    return mag.astype(np.float32)


def bbox_from_mask(mask: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    """
    Retorna bbox como (y0, x0, y1, x1).
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    ys, xs = np.where(mask > threshold)
    if len(ys) == 0:
        return None

    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return (y0, x0, y1, x1)


def mask_region(img: ImageGrid, threshold: float = 0.5) -> dict[str, np.ndarray | tuple[int, int, int, int] | None]:
    """
    Constrói máscara simples a partir de grayscale e bbox correspondente.
    """
    gray = img.grayscale().data.astype(np.float32, copy=False)
    mask = (gray > threshold).astype(np.float32)
    bbox = bbox_from_mask(mask, threshold=0.5)

    return {
        "mask": mask,
        "bbox": bbox,
    }