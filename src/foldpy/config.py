"""Reproducibility utilities and runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime controls to ensure reproducibility.

    Attributes:
        seed: Random seed used by NumPy and Python random modules.
        eps: Numerical stability epsilon used by numerical routines.
        version_tag: Parameter version tag for experiment tracking.
    """

    seed: int = 42
    eps: float = 1e-6
    version_tag: str = "v1"


def set_seed(seed: int) -> None:
    """Set deterministic seed for built-in random generators.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
