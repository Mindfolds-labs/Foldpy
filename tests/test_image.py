import numpy as np

from foldpy import ImageGrid


def test_imagegrid_summary() -> None:
    img = ImageGrid(np.zeros((8, 8, 3), dtype=np.float32), layout="HWC")
    s = img.summary()
    assert s["height"] == 8
    assert s["channels"] == 3
