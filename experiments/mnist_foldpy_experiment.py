"""MNIST baseline vs FoldPy structural-channel experiment.

Usage:
    python experiments/mnist_foldpy_experiment.py --epochs 2 --subset 20000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from foldpy import ImageGrid, image_plus_structural_channels


@dataclass
class Metrics:
    model: str
    train_loss: float
    test_accuracy: float
    train_seconds: float
    inference_ms_per_batch: float


class FoldPyMNISTDataset(Dataset):
    """MNIST wrapper that augments samples with FoldPy channels."""

    def __init__(self, base_dataset: Dataset, depth_bins: int = 8) -> None:
        self.base_dataset = base_dataset
        self.depth_bins = depth_bins

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, target = self.base_dataset[idx]
        if isinstance(image, torch.Tensor):
            gray = image.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            gray = np.array(image, dtype=np.float32) / 255.0
        fold_img = ImageGrid(gray, layout="HW")
        channels = image_plus_structural_channels(fold_img, depth_bins=self.depth_bins)
        return torch.from_numpy(channels), int(target)


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * x.shape[0]
    return running / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    n = 0
    timings: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            start = perf_counter()
            logits = model(x)
            timings.append((perf_counter() - start) * 1000.0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            n += int(y.numel())
    mean_batch_ms = float(np.mean(timings)) if timings else 0.0
    return correct / max(n, 1), mean_batch_ms


def run_training(
    train_ds: Dataset,
    test_ds: Dataset,
    in_channels: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Metrics:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = TinyCNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = perf_counter()
    train_loss = 0.0
    for _ in range(epochs):
        train_loss = train_one_epoch(model, train_loader, device, optimizer)
    train_seconds = perf_counter() - start
    acc, inf_ms = evaluate(model, test_loader, device)

    return Metrics(
        model=f"cnn_{in_channels}ch",
        train_loss=train_loss,
        test_accuracy=acc,
        train_seconds=train_seconds,
        inference_ms_per_batch=inf_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST baseline vs FoldPy structural channels")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset", type=int, default=0, help="use only first N samples from train/test")
    parser.add_argument("--depth-bins", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mnist_foldpy_metrics.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.ToTensor()

    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_base = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    if args.subset > 0:
        train_base = Subset(train_base, list(range(min(args.subset, len(train_base)))))
        test_base = Subset(test_base, list(range(min(args.subset // 5 or 1, len(test_base)))))

    baseline_train = train_base
    baseline_test = test_base

    foldpy_train = FoldPyMNISTDataset(train_base, depth_bins=args.depth_bins)
    foldpy_test = FoldPyMNISTDataset(test_base, depth_bins=args.depth_bins)

    baseline_metrics = run_training(
        baseline_train,
        baseline_test,
        in_channels=1,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    foldpy_metrics = run_training(
        foldpy_train,
        foldpy_test,
        in_channels=9,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    summary = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "subset": args.subset,
            "depth_bins": args.depth_bins,
            "lr": args.lr,
            "device": str(device),
        },
        "baseline": baseline_metrics.__dict__,
        "foldpy_structural": foldpy_metrics.__dict__,
        "delta_accuracy": foldpy_metrics.test_accuracy - baseline_metrics.test_accuracy,
        "delta_train_seconds": foldpy_metrics.train_seconds - baseline_metrics.train_seconds,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
