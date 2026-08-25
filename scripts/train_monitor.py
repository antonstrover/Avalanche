"""Train the learned process monitor on the labelled traces.

The script splits the rows by scenario family, trains the perceptron, and
saves the weights and the metadata under `outputs/models/`.

The training part and the validation part are the only parts it reads.
The test part stays for the final evaluation.

Usage:
    python scripts/train_monitor.py outputs/datasets/monitor-training.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from avalanche.config.run_identity import REPO_ROOT
from avalanche.monitors.perceptron import TrainingConfig, save_model, train_perceptron
from avalanche.monitors.splits import split_by_family

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "models" / "monitor-perceptron.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train_monitor")
    parser.add_argument("rows", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = pd.read_parquet(args.rows)
    parts, assignment = split_by_family(frame, seed=args.seed)
    model = train_perceptron(
        parts["train"],
        parts["validation"],
        TrainingConfig(seed=args.seed, epochs=args.epochs),
    )
    model.metadata["split"] = assignment.as_dict()
    path = save_model(model, args.output)
    print(f"Wrote the model to {path}")
    print(json.dumps(model.metadata["validation_scores"], indent=2, sort_keys=True))
    print(json.dumps(model.metadata["constant_baseline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
