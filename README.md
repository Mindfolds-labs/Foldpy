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
- Approximate 2D spatial inference pipeline (edge/mask/depth/occupancy features)
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
- MNIST spatial experiment: [`docs/spatial_mnist_pipeline.md`](docs/spatial_mnist_pipeline.md)
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

## Experimentos com PyTorch

Esta base agora inclui um experimento mínimo de classificação no MNIST para comparar:

1. **Baseline** com entrada original `[1, 28, 28]`.
2. **FoldPy** com entrada estrutural enriquecida `[4, 28, 28]`.

### Canais estruturais usados no experimento

Implementados em `foldpy.build_structural_channels(...)`:

1. imagem original em grayscale normalizada
2. `edge_map`
3. `depth_hint`
4. máscara simples via `mask_region`

Saída da função: `numpy.ndarray` em formato `[C, H, W]`, `float32`, faixa `[0, 1]`.

### Limitações e aproximações

- O canal de máscara é uma segmentação simples por limiar de intensidade (não é segmentação semântica).
- `depth_hint` é um indício estrutural aproximado, não reconstrução métrica 3D.
- Resultados variam com hardware/seed/subconjunto escolhido.

### Instalação para desenvolvimento/experimentos

```bash
pip install -e .[dev]
```

### Como rodar

Treino baseline:

```bash
python -m experiments.train_baseline --epochs 2 --subset 20000
```

Treino FoldPy (canais estruturais):

```bash
python -m experiments.train_foldpy --epochs 2 --subset 20000
```

Comparar resultados:

```bash
python -m experiments.compare_results \
  --baseline experiments/outputs/baseline_metrics.json \
  --foldpy experiments/outputs/foldpy_metrics.json
```

Arquivos de saída são gravados em `experiments/outputs/` (`*.json`, `results_summary.csv`, `comparison.csv`).
