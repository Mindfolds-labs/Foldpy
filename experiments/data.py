from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from foldpy import build_structural_channels


class FoldPyMNISTDataset(Dataset):
    """MNIST dataset wrapper that expands each sample into structural channels."""

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, target = self.base_dataset[idx]
        if isinstance(image, torch.Tensor):
            gray = image.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            gray = np.array(image, dtype=np.float32) / 255.0
        channels = build_structural_channels(gray)
        return torch.from_numpy(channels), int(target)


def load_mnist(root: Path, subset: int = 0) -> tuple[Dataset, Dataset]:
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(root), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)
    if subset > 0:
        train_n = min(subset, len(train_ds))
        test_n = min(max(1, subset // 5), len(test_ds))
        train_ds = Subset(train_ds, list(range(train_n)))
        test_ds = Subset(test_ds, list(range(test_n)))
    return train_ds, test_ds
