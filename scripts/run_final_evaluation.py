"""Run the final matrix with unseen seeds and verified monitor references."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from avalanche.config import ModelLockReference, load_yaml
from avalanche.config.run_identity import REPO_ROOT
from avalanche.experiments.final_evaluation import (
    load_evaluation_config,
    require_unseen_evaluation_seeds,
    run_evaluation_matrix,
    write_final_evaluation,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / "final-evaluation.yaml"
DEFAULT_TRAINING = REPO_ROOT / "configs" / "experiments" / "monitor-training.yaml"
DEFAULT_REFERENCES = (
    REPO_ROOT / "outputs" / "models" / "final-evaluation-references.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "final-evaluation-routing-v8"


def build_parser() -> argparse.ArgumentParser:
    """Build the final evaluation command arguments."""
    parser = argparse.ArgumentParser(prog="run_final_evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--model-references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run and write the complete unseen-seed evaluation."""
    args = build_parser().parse_args(argv)
    config = load_evaluation_config(args.config)
    training = load_yaml(args.training_manifest)
    require_unseen_evaluation_seeds(config, training)
    references = load_model_references(args.model_references)
    records = run_evaluation_matrix(config, references, args.output)
    write_final_evaluation(records, args.output, references)
    print(f"Wrote the final evaluation to {args.output}")
    return 0


def load_model_references(path: Path) -> dict[str, ModelLockReference]:
    """Load the three verified profile references for final evaluation."""
    value = load_yaml(path)
    if value.get("model_references_version") != 1:
        raise ValueError("the model reference manifest version is incompatible")
    raw = value.get("references")
    required = {"principal", "fallback_oracle", "true_state_oracle"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("the final evaluation needs three model references")
    return {
        name: ModelLockReference.model_validate(_mapping(raw[name], name))
        for name in sorted(required)
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require one reference mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"the {name} model reference must be one mapping")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
