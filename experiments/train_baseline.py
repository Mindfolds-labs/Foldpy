from __future__ import annotations

import argparse
from pathlib import Path

from experiments.data import load_mnist
from experiments.utils import append_summary_csv, save_metrics_json, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline CNN on MNIST with original channel")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/outputs/baseline_metrics.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("experiments/outputs/results_summary.csv"))
    args = parser.parse_args()

    train_ds, test_ds = load_mnist(args.data_dir, subset=args.subset)
    metrics = train_model(
        train_ds=train_ds,
        test_ds=test_ds,
        in_channels=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    save_metrics_json(metrics, args.output_json)
    append_summary_csv(metrics, args.summary_csv, tag="baseline")

    print(f"accuracy={metrics.accuracy:.4f}")
    print(f"f1_macro={metrics.f1_macro:.4f}")
    print(f"train_seconds={metrics.train_seconds:.2f}")


if __name__ == "__main__":
    main()
