"""Run the required shortcut audit before monitor training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import InformationProfile
from avalanche.experiments.acceptance import load_shortcut_justifications
from avalanche.monitors.dataset import validate_generated_dataset
from avalanche.monitors.features import feature_names_for
from avalanche.monitors.shortcut_audit import run_shortcut_audit

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "audit" / "monitor-training"
DEFAULT_JUSTIFICATIONS = (
    REPO_ROOT / "configs" / "experiments" / "shortcut-justifications.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the shortcut audit command arguments."""
    parser = argparse.ArgumentParser(prog="audit_monitor_dataset")
    parser.add_argument("rows", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--justifications",
        type=Path,
        default=DEFAULT_JUSTIFICATIONS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Audit one generated principal dataset."""
    args = build_parser().parse_args(argv)
    frame = pd.read_parquet(args.rows)
    checksums = validate_generated_dataset(
        args.rows,
        frame,
        InformationProfile.PRINCIPAL,
    )
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("the shortcut audit needs training and validation rows")
    justifications, reviewed = load_shortcut_justifications(args.justifications)
    report = run_shortcut_audit(
        train,
        validation,
        args.output,
        feature_names=feature_names_for(InformationProfile.PRINCIPAL),
        accepted_justifications=justifications,
        reviewed_perfect_separation=reviewed,
        dataset_checksums=checksums,
    )
    if not report["approved"]:
        raise ValueError("the shortcut audit did not pass")
    print(f"Wrote the approved shortcut audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
