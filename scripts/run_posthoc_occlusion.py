"""Run a nonformal post-hoc feature occlusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from avalanche.monitors.features import FeatureProfile, feature_names_for


def build_parser() -> argparse.ArgumentParser:
    """Build the post-hoc occlusion arguments."""
    parser = argparse.ArgumentParser(prog="run_posthoc_occlusion")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "profile", choices=[profile.value for profile in FeatureProfile]
    )
    return parser


def occlude(frame: pd.DataFrame, profile: FeatureProfile | str) -> pd.DataFrame:
    """Zero excluded columns for a nonformal sensitivity analysis."""
    selected = set(feature_names_for(FeatureProfile(profile)))
    result = frame.copy()
    master = feature_names_for(FeatureProfile.PRINCIPAL_FULL)
    missing = sorted(set(master) - set(result))
    if missing:
        raise ValueError(f"the occlusion rows miss {len(missing)} feature columns")
    for name in master:
        if name not in selected:
            result[name] = 0.0
    return result


def main(argv: list[str] | None = None) -> int:
    """Write one clearly marked nonformal occlusion dataset."""
    args = build_parser().parse_args(argv)
    profile = FeatureProfile(args.profile)
    frame = occlude(pd.read_parquet(args.dataset), profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "formal": False,
                "analysis": "posthoc_occlusion",
                "feature_profile": profile.value,
                "warning": "Do not treat this result as a trained feature comparison.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
