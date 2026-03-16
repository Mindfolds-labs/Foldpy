import numpy as np

from foldpy import ImageGrid, area_series, concentric_layers


def test_concentric_layers_and_area() -> None:
    img = ImageGrid(np.ones((10, 10), dtype=np.float32), layout="HW")
    layers = concentric_layers(img)
    areas = area_series(layers)
    assert len(layers) > 0
    assert areas[0] == 100
