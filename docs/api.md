# FoldPy API Overview

## Core Container
- `ImageGrid`: normalized image abstraction supporting `HW`, `HWC`, `CHW`.

## Layers and Signatures
- `concentric_layers`, `border_series`, `shell_series`, `area_series`
- `legendre_encode`, `legendre_decode`, `legendre_signature`

## Projection and Geometry
- `directional_field`, `angular_perspective_matrix`
- `depth_order_map`, `depth_hint`
- `triangulate_points`, `surface_mesh_hint`

## Graph and Regions
- `projection_graph`
- `Region`, `region`, `region_graph`

## Color and Scale
- `rgb_to_lms`, `lms_to_rgb`
- `scale_pyramid`

## Tensor and Volume
- `angular_tensor`
- `volume_lift`

## Reproducibility
- `RuntimeConfig`
- `set_seed`
