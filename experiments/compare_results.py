from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and FoldPy MNIST results")
    parser.add_argument("--baseline", type=Path, default=Path("experiments/outputs/baseline_metrics.json"))
    parser.add_argument("--foldpy", type=Path, default=Path("experiments/outputs/foldpy_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("experiments/outputs/comparison.csv"))
    args = parser.parse_args()

    baseline = _load_json(args.baseline)
    foldpy = _load_json(args.foldpy)

    rows = [
        {
            "run": "baseline",
            "accuracy": baseline["accuracy"],
            "f1_macro": baseline["f1_macro"],
            "train_seconds": baseline["train_seconds"],
        },
        {
            "run": "foldpy",
            "accuracy": foldpy["accuracy"],
            "f1_macro": foldpy["f1_macro"],
            "train_seconds": foldpy["train_seconds"],
        },
        {
            "run": "delta_foldpy_minus_baseline",
            "accuracy": foldpy["accuracy"] - baseline["accuracy"],
            "f1_macro": foldpy["f1_macro"] - baseline["f1_macro"],
            "train_seconds": foldpy["train_seconds"] - baseline["train_seconds"],
        },
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "accuracy", "f1_macro", "train_seconds"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['run']}: accuracy={row['accuracy']:.4f}, "
            f"f1_macro={row['f1_macro']:.4f}, train_seconds={row['train_seconds']:.2f}"
        )


if __name__ == "__main__":
    main()
