import numpy as np

from foldpy import surface_mesh_hint, triangulate_points


def test_triangulate_points() -> None:
    p1 = np.array([[10, 10], [20, 20]], dtype=np.float32)
    p2 = np.array([[9, 10], [19, 20]], dtype=np.float32)
    p3d = triangulate_points(p1, p2)
    assert p3d.shape == (2, 3)


def test_surface_mesh_hint() -> None:
    mesh = surface_mesh_hint(np.zeros((4, 4), dtype=np.float32))
    assert mesh["vertices"].shape[0] == 16
