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
from avalanche.observability import (
    MetricEvent,
    MetricsAggregator,
    ObservabilitySession,
    StageStatus,
)

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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the Textual observer",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train one locked monitor from the fixed dataset splits."""
    args = build_parser().parse_args(argv)
    profile = InformationProfile(args.information_profile)
    stage_id = f"monitor-{profile.value.replace('_', '-')}"
    aggregator = _training_aggregator(stage_id, profile, args.epochs)
    session = ObservabilitySession(
        aggregator=aggregator,
        enabled=False if args.no_progress else None,
        log_path=_log_path(args.output),
    )
    session.emitter.emit(
        MetricEvent.create(
            "run_config",
            stage_id,
            seed=args.seed,
            dataset=str(args.rows),
            model=str(args.output),
            information_profile=profile.value,
            epochs=args.epochs,
            training_configuration="batch=256, learning-rate=0.001",
        )
    )
    with session:
        try:
            session.emitter.emit(
                MetricEvent.create(
                    "stage_phase",
                    f"{stage_id}-perceptron",
                    phase="loading dataset",
                )
            )
            frame = pd.read_parquet(args.rows)
            checksums = validate_generated_dataset(args.rows, frame, profile)
            train = frame[frame["split"] == "train"].reset_index(drop=True)
            validation = frame[frame["split"] == "validation"].reset_index(drop=True)
            if train.empty or validation.empty:
                message = "the monitor dataset needs training and validation rows"
                raise ValueError(message)
            session.emitter.emit(
                MetricEvent.create(
                    "run_config",
                    stage_id,
                    training_rows=len(train),
                    validation_rows=len(validation),
                )
            )
            train_locked_monitor(
                train,
                validation,
                args.shortcut_report,
                args.output,
                config=TrainingConfig(
                    seed=args.seed,
                    epochs=args.epochs,
                    information_profile=profile.value,
                ),
                dataset_checksums=checksums,
                emitter=session.emitter,
                stage_id=stage_id,
            )
        except Exception as error:
            _fail_running_stages(session, stage_id, error)
            raise
    return 0


def _training_aggregator(
    stage_id: str,
    profile: InformationProfile,
    epochs: int,
) -> MetricsAggregator:
    """Register each known training and calibration stage."""
    aggregator = MetricsAggregator()
    label = profile.value.replace("_", " ").title()
    aggregator.register_stage(stage_id, label=f"{label} training run", weight=0.25)
    aggregator.register_stage(
        f"{stage_id}-perceptron",
        label="Perceptron training",
        total_epochs=epochs,
    )
    aggregator.register_stage(
        f"{stage_id}-perceptron-calibration",
        label="Perceptron calibration",
    )
    aggregator.register_stage(
        f"{stage_id}-gru",
        label="GRU fallback",
        status=StageStatus.NOT_EVALUATED,
        weight=0.25,
    )
    return aggregator


def _fail_running_stages(
    session: ObservabilitySession,
    base_stage: str,
    error: Exception,
) -> None:
    """Mark the run and each active stage as failed."""
    snapshot = session.aggregator.snapshot()
    base = snapshot.stage(base_stage)
    active_children = [
        stage
        for stage in snapshot.stages
        if stage.stage_id != base_stage and stage.status == StageStatus.RUNNING
    ]
    if base.status != StageStatus.FAILED:
        session.emitter.emit(
            MetricEvent.create(
                "stage_failed",
                base_stage,
                phase=base.phase,
                error_type=type(error).__name__,
                error=str(error),
                count_failure=not active_children,
            )
        )
    for stage in snapshot.stages:
        if stage.status != StageStatus.RUNNING:
            continue
        session.emitter.emit(
            MetricEvent.create(
                "stage_failed",
                stage.stage_id,
                phase=stage.phase,
                error_type=type(error).__name__,
                error=str(error),
            )
        )


def _log_path(output: Path) -> Path:
    """Return a log beside the immutable model directory."""
    return output.with_name(output.name + ".observability.jsonl")


if __name__ == "__main__":
    raise SystemExit(main())
