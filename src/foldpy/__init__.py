"""FoldPy public API."""

from .color import lms_to_rgb, rgb_to_lms
from .config import RuntimeConfig, set_seed
from .geometry import surface_mesh_hint, triangulate_points
from .graph import projection_graph
from .image import ImageGrid
from .nn_features import build_structural_channels
from .layers import area_series, border_series, concentric_layers, shell_series
from .legendre import legendre_decode, legendre_encode, legendre_signature
from .multiview import multi_view_align
from .projection import angular_perspective_matrix, depth_hint, depth_order_map, directional_field
from .edge import edge_map, mask_region
from .regions import Region, region, region_graph
from .scale import scale_pyramid
from .spatial import SpatialApproximation, image_plus_structural_channels, infer_spatial_approximation, structural_channels
from .svg_bridge import from_svg, svg_to_raster
from .tensors import angular_tensor
from .transforms import normalize, patchify, resize_nearest
from .volume import volume_lift

__all__ = [
    "ImageGrid",
    "build_structural_channels",
    "RuntimeConfig",
    "set_seed",
    "concentric_layers",
    "border_series",
    "shell_series",
    "area_series",
    "directional_field",
    "edge_map",
    "mask_region",
    "angular_perspective_matrix",
    "depth_order_map",
    "depth_hint",
    "projection_graph",
    "region",
    "Region",
    "region_graph",
    "normalize",
    "resize_nearest",
    "patchify",
    "multi_view_align",
    "rgb_to_lms",
    "lms_to_rgb",
    "scale_pyramid",
    "SpatialApproximation",
    "infer_spatial_approximation",
    "structural_channels",
    "image_plus_structural_channels",
    "legendre_encode",
    "legendre_decode",
    "legendre_signature",
    "from_svg",
    "svg_to_raster",
    "angular_tensor",
    "triangulate_points",
    "surface_mesh_hint",
    "volume_lift",
]
