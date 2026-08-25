"""Make the labelled development traces for the learned process monitor.

The script runs each entry of the training matrix without a display.
It writes one Parquet file of feature rows and labels, and a summary file.

The monitor allows every proposal during a dataset run, so the rows record
the behaviour of the controller alone.

Usage:
    python scripts/generate_monitor_dataset.py configs/experiments/monitor-training.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from avalanche.config.run_identity import REPO_ROOT
from avalanche.monitors.dataset import generate_dataset

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "datasets" / "monitor-training.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_monitor_dataset")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--limit", type=int, default=None, help="run only the first entries"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written = generate_dataset(
        args.manifest, args.output, workers=args.workers, limit=args.limit
    )
    print(f"Wrote the labelled rows to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
