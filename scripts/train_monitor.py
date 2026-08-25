"""Train, calibrate, gate, and lock the principal process monitor.

Usage:
    python scripts/train_monitor.py outputs/datasets/monitor-training.parquet \
        outputs/audit/shortcut-audit.json
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from avalanche.config.run_identity import REPO_ROOT
from avalanche.monitors.perceptron import TrainingConfig
from avalanche.monitors.training import train_locked_monitor

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "models" / "monitor-principal"


def build_parser() -> argparse.ArgumentParser:
    """Build the locked training command arguments."""
    parser = argparse.ArgumentParser(prog="train_monitor")
    parser.add_argument("rows", type=Path)
    parser.add_argument("shortcut_report", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train the locked principal monitor from fixed dataset splits."""
    args = build_parser().parse_args(argv)
    frame = pd.read_parquet(args.rows)
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("the monitor dataset needs training and validation rows")
    result = train_locked_monitor(
        train,
        validation,
        args.shortcut_report,
        args.output,
        config=TrainingConfig(seed=args.seed, epochs=args.epochs),
        dataset_checksums={"dataset_sha256": _checksum(args.rows)},
    )
    print(f"Wrote the locked monitor to {args.output}")
    print(f"Selected the {result['metadata']['model_kind']} model.")
    return 0


def _checksum(path: Path) -> str:
    """Return one full SHA-256 checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
