"""Utilities to build FoldPy structural channels for neural networks."""

from __future__ import annotations

import numpy as np

from .edge import edge_map, mask_region
from .image import ImageGrid
from .projection import depth_hint


def _to_image_grid(image_np: np.ndarray) -> ImageGrid:
    arr = np.asarray(image_np)
    if arr.ndim == 2:
        return ImageGrid(arr.astype(np.float32, copy=False), layout="HW")
    if arr.ndim == 3:
        # accept HWC by default for user convenience
        return ImageGrid(arr.astype(np.float32, copy=False), layout="HWC")
    raise ValueError("image_np must be 2D (H, W) or 3D (H, W, C)")


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(channel.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    cmin = float(x.min())
    cmax = float(x.max())
    if cmax - cmin < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - cmin) / (cmax - cmin)
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def build_structural_channels(
    image_np: np.ndarray,
    include_original: bool = True,
    use_edge: bool = True,
    use_depth: bool = True,
    use_mask: bool = True,
) -> np.ndarray:
    """Build stacked structural channels for a single image.

    Args:
        image_np: Input image in ``(H, W)`` or ``(H, W, C)`` format.
        include_original: Include normalized grayscale image as first channel.
        use_edge: Include ``foldpy.edge.edge_map`` channel.
        use_depth: Include ``foldpy.projection.depth_hint`` channel.
        use_mask: Include foreground mask channel from ``foldpy.edge.mask_region``.

    Returns:
        Numpy array in ``(C, H, W)`` format using ``float32`` values in ``[0, 1]``.
    """
    img = _to_image_grid(image_np)

    channels: list[np.ndarray] = []
    if include_original:
        channels.append(_normalize_channel(img.grayscale().data))
    if use_edge:
        channels.append(_normalize_channel(edge_map(img, normalize=True)))
    if use_depth:
        channels.append(_normalize_channel(depth_hint(img)))
    if use_mask:
        mask_dict = mask_region(img, threshold=0.5)
        channels.append(_normalize_channel(np.asarray(mask_dict["mask"], dtype=np.float32)))

    if not channels:
        raise ValueError("At least one channel must be enabled")

    return np.stack(channels, axis=0).astype(np.float32, copy=False)
