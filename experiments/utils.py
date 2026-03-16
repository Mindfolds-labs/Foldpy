from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import SmallCNN


@dataclass
class RunMetrics:
    model_name: str
    in_channels: int
    epochs: int
    train_loss: float
    accuracy: float
    f1_macro: float
    train_seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
    criterion = nn.CrossEntropyLoss()
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * x.shape[0]
    return running / max(1, len(loader.dataset))


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    acc = float((y_true_arr == y_pred_arr).mean()) if y_true_arr.size else 0.0
    f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro")) if y_true_arr.size else 0.0
    return acc, f1


def train_model(
    train_ds: Dataset,
    test_ds: Dataset,
    in_channels: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> RunMetrics:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = SmallCNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = perf_counter()
    train_loss = 0.0
    for _ in range(epochs):
        train_loss = _train_one_epoch(model, train_loader, device, optimizer)
    train_seconds = perf_counter() - start

    accuracy, f1_macro = _evaluate(model, test_loader, device)
    return RunMetrics(
        model_name="small_cnn",
        in_channels=in_channels,
        epochs=epochs,
        train_loss=float(train_loss),
        accuracy=accuracy,
        f1_macro=f1_macro,
        train_seconds=float(train_seconds),
    )


def save_metrics_json(metrics: RunMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metrics), indent=2))


def append_summary_csv(metrics: RunMetrics, csv_path: Path, tag: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = "run_tag,model_name,in_channels,epochs,train_loss,accuracy,f1_macro,train_seconds\n"
    row = (
        f"{tag},{metrics.model_name},{metrics.in_channels},{metrics.epochs},"
        f"{metrics.train_loss:.6f},{metrics.accuracy:.6f},{metrics.f1_macro:.6f},{metrics.train_seconds:.6f}\n"
    )
    if not csv_path.exists():
        csv_path.write_text(header + row)
    else:
        with csv_path.open("a", encoding="utf-8") as f:
            f.write(row)
