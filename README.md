# FoldPy

[![CI](https://img.shields.io/github/actions/workflow/status/Mindfolds-labs/Foldpy/ci.yml?branch=main)](https://github.com/Mindfolds-labs/Foldpy/actions)
[![PyPI](https://img.shields.io/pypi/v/foldpy)](https://pypi.org/project/foldpy/)
[![Python](https://img.shields.io/pypi/pyversions/foldpy)](https://pypi.org/project/foldpy/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

FoldPy is a **structural visual transformation engine** for tensors, geometry, scale-space, projection, and approximate lifting.

## Key Features

- Concentric layers + border/shell series
- Directional generators and angular perspective matrices
- Structural depth hint and approximate lifting primitives
- RGB ↔ LMS color conversion
- Scale pyramid generation
- Legendre structural signatures
- SVG → raster bridge
- Region and projection graph primitives
- Triangulation and surface mesh hints
- Angular tensor and projective score foundations

## Installation

```bash
pip install foldpy
```

Development install:

```bash
pip install -e .[dev]
```

## Quickstart

```python
import numpy as np
from foldpy import ImageGrid, concentric_layers, area_series, depth_hint

arr = np.random.default_rng(42).random((64, 64, 3), dtype=np.float32)
img = ImageGrid(arr, layout="HWC")

layers = concentric_layers(img)
areas = area_series(layers)
z = depth_hint(img)

print(len(layers), areas.shape, z.shape)
```

## Documentation

- API guide: [`docs/api.md`](docs/api.md)
- Changelog: GitHub Releases
- Source repository: <https://github.com/Mindfolds-labs/Foldpy>

## Contributing

1. Fork the project and create a feature branch.
2. Install with `pip install -e .[dev]`.
3. Run `pytest` and linting before opening a PR.
4. Submit concise PRs with tests and docs.

## Release / PyPI

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

For TestPyPI, replace the upload URL:

```bash
python -m twine upload --repository testpypi dist/*
```
