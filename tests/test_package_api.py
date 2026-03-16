from __future__ import annotations

import importlib


def test_import_foldpy_package() -> None:
    fp = importlib.import_module("foldpy")
    assert hasattr(fp, "ImageGrid")
    assert hasattr(fp, "build_structural_channels")


def test_public_all_is_stable() -> None:
    fp = importlib.import_module("foldpy")
    expected = {
        "ImageGrid",
        "build_structural_channels",
        "edge_map",
        "mask_region",
        "depth_hint",
        "depth_order_map",
        "angular_perspective_matrix",
        "volume_lift",
        "projection_graph",
    }
    assert expected.issubset(set(fp.__all__))
