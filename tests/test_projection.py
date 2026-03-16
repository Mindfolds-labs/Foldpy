import numpy as np

from foldpy import ImageGrid, angular_perspective_matrix, depth_hint


def test_projection_outputs_shape() -> None:
    img = ImageGrid(np.random.default_rng(1).random((16, 16, 3), dtype=np.float32), layout="HWC")
    mat = angular_perspective_matrix(img)
    z = depth_hint(img)
    assert mat.shape[0] == 4
    assert z.shape == (16, 16)
