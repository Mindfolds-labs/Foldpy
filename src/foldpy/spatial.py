"""Approximate 2D-to-spatial inference pipeline utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edge import edge_map, mask_region
from .image import ImageGrid
from .projection import angular_perspective_matrix, depth_hint, depth_order_map
from .volume import volume_lift


@dataclass(slots=True)
class SpatialApproximation:
    """Structured outputs produced from a 2D image.

    All fields are approximate structural cues, not metric 3D geometry.
    """

    edge_map: np.ndarray
    object_mask: np.ndarray
    background_mask: np.ndarray
    plane_mask: np.ndarray
    depth_hint: np.ndarray
    depth_order: np.ndarray
    occupancy_volume: np.ndarray
    occupancy_projection: np.ndarray
    surface_hint: np.ndarray
    angular_features: np.ndarray


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    return arr / (arr.max() + 1e-6)


def _surface_hint_from_depth(z: np.ndarray) -> np.ndarray:
    """Surface orientation hint from depth gradients.

    Returns a 2-channel map (dx, dy) normalized to [0, 1].
    """
    z = z.astype(np.float32, copy=False)
    gx = np.zeros_like(z, dtype=np.float32)
    gy = np.zeros_like(z, dtype=np.float32)
    gx[:, 1:-1] = z[:, 2:] - z[:, :-2]
    gy[1:-1, :] = z[2:, :] - z[:-2, :]
    gx = (gx + 1.0) / 2.0
    gy = (gy + 1.0) / 2.0
    return np.stack([np.clip(gx, 0.0, 1.0), np.clip(gy, 0.0, 1.0)], axis=0)


def infer_spatial_approximation(
    img: ImageGrid,
    object_threshold: float = 0.1,
    plane_quantile: float = 0.4,
    depth_bins: int = 8,
) -> SpatialApproximation:
    """Build an approximate spatial representation from one 2D image.

    Args:
        img: Input image as ``ImageGrid``.
        object_threshold: Threshold used to estimate foreground/object mask.
        plane_quantile: Quantile over depth hint used as coarse plane/background split.
        depth_bins: Number of bins for occupancy lift.
    """
    edge = edge_map(img, normalize=True)

    masks = mask_region(img, threshold=object_threshold)
    object_mask = masks["mask"].astype(np.float32, copy=False)
    background_mask = (1.0 - object_mask).astype(np.float32, copy=False)

    z_hint = depth_hint(img)
    z_order = _normalize_map(depth_order_map(img))

    plane_thr = float(np.quantile(z_hint, plane_quantile))
    plane_mask = (z_hint <= plane_thr).astype(np.float32)

    occupancy = volume_lift(z_hint, depth_bins=depth_bins, threshold=0.08)
    occupancy_proj = occupancy.mean(axis=2).astype(np.float32)

    surface_hint = _surface_hint_from_depth(z_hint)
    angular = angular_perspective_matrix(img)
    angular = np.nan_to_num(angular, nan=0.0).astype(np.float32)

    return SpatialApproximation(
        edge_map=edge,
        object_mask=object_mask,
        background_mask=background_mask,
        plane_mask=plane_mask,
        depth_hint=z_hint,
        depth_order=z_order,
        occupancy_volume=occupancy,
        occupancy_projection=occupancy_proj,
        surface_hint=surface_hint,
        angular_features=angular,
    )


def structural_channels(img: ImageGrid, depth_bins: int = 8) -> np.ndarray:
    """Return CNN-ready structural channels for one image.

    Output shape: ``(8, H, W)``.
    """
    spatial = infer_spatial_approximation(img, depth_bins=depth_bins)
    return np.stack(
        [
            spatial.edge_map,
            spatial.plane_mask,
            spatial.object_mask,
            spatial.depth_hint,
            spatial.depth_order,
            spatial.occupancy_projection,
            spatial.surface_hint[0],
            spatial.surface_hint[1],
        ],
        axis=0,
    ).astype(np.float32)


def image_plus_structural_channels(img: ImageGrid, depth_bins: int = 8) -> np.ndarray:
    """Return raw grayscale image + structural channels.

    Output shape: ``(9, H, W)``.
    """
    gray = img.grayscale().data.astype(np.float32, copy=False)
    gray = _normalize_map(gray)
    return np.concatenate([gray[None, ...], structural_channels(img, depth_bins=depth_bins)], axis=0)
