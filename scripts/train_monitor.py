"""Train, calibrate, gate, and lock one process monitor.

Usage:
    python scripts/train_monitor.py outputs/datasets/monitor-training.parquet \
        outputs/audit/shortcut-audit.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import InformationProfile
from avalanche.monitors.dataset import validate_generated_dataset
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--information-profile",
        choices=[profile.value for profile in InformationProfile],
        default=InformationProfile.PRINCIPAL.value,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train one locked monitor from the fixed dataset splits."""
    args = build_parser().parse_args(argv)
    frame = pd.read_parquet(args.rows)
    checksums = validate_generated_dataset(
        args.rows,
        frame,
        args.information_profile,
    )
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("the monitor dataset needs training and validation rows")
    result = train_locked_monitor(
        train,
        validation,
        args.shortcut_report,
        args.output,
        config=TrainingConfig(
            seed=args.seed,
            epochs=args.epochs,
            information_profile=args.information_profile,
        ),
        dataset_checksums=checksums,
    )
    print(f"Wrote the locked monitor to {args.output}")
    print(f"Selected the {result['metadata']['model_kind']} model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
