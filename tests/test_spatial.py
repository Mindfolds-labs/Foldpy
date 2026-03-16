import numpy as np

from foldpy import ImageGrid, image_plus_structural_channels, infer_spatial_approximation, structural_channels


def test_spatial_pipeline_shapes() -> None:
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[8:20, 10:18] = 1.0
    img = ImageGrid(arr, layout="HW")

    spatial = infer_spatial_approximation(img, depth_bins=6)

    assert spatial.edge_map.shape == (28, 28)
    assert spatial.object_mask.shape == (28, 28)
    assert spatial.background_mask.shape == (28, 28)
    assert spatial.plane_mask.shape == (28, 28)
    assert spatial.depth_hint.shape == (28, 28)
    assert spatial.depth_order.shape == (28, 28)
    assert spatial.occupancy_volume.shape == (28, 28, 6)
    assert spatial.occupancy_projection.shape == (28, 28)
    assert spatial.surface_hint.shape == (2, 28, 28)
    assert spatial.angular_features.shape[0] == 4


def test_structural_channels_stack() -> None:
    arr = np.random.default_rng(0).random((16, 16), dtype=np.float32)
    img = ImageGrid(arr, layout="HW")

    s = structural_channels(img, depth_bins=4)
    x = image_plus_structural_channels(img, depth_bins=4)

    assert s.shape == (8, 16, 16)
    assert x.shape == (9, 16, 16)
    assert np.all((x >= 0.0) & (x <= 1.0 + 1e-5))
