from __future__ import annotations

import numpy as np

from foldpy.nn_features import build_structural_channels


def test_import_nn_features_module() -> None:
    assert callable(build_structural_channels)


def test_build_structural_channels_shape_and_range() -> None:
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[10:18, 10:18] = 1.0
    channels = build_structural_channels(arr)

    assert channels.shape == (4, 28, 28)
    assert channels.dtype == np.float32
    assert np.all(np.isfinite(channels))
    assert np.min(channels) >= 0.0
    assert np.max(channels) <= 1.0 + 1e-6


def test_build_structural_channels_empty_image_is_stable() -> None:
    arr = np.zeros((16, 16), dtype=np.float32)
    channels = build_structural_channels(arr)
    assert channels.shape == (4, 16, 16)
    assert np.all(np.isfinite(channels))


def test_build_structural_channels_with_no_enabled_channel_raises() -> None:
    arr = np.zeros((8, 8), dtype=np.float32)
    try:
        build_structural_channels(arr, include_original=False, use_edge=False, use_depth=False, use_mask=False)
    except ValueError as exc:
        assert "At least one channel" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty channel selection")
