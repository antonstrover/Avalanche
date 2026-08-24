"""The avalanche command-line interface."""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from avalanche.config import ResolvedConfig, load_and_merge, make_run_dir
from avalanche.experiments import run_episode


def _resolve_config(paths: list[str]) -> ResolvedConfig:
    merged = load_and_merge(*(Path(p) for p in paths))
    return ResolvedConfig.model_validate(merged)


def validate_config(args: argparse.Namespace) -> int:
    try:
        _resolve_config(args.config)
    except ValidationError as error:
        print(error, file=sys.stderr)
        return 1
    print("OK")
    return 0


def simulate(args: argparse.Namespace) -> int:
    try:
        resolved = _resolve_config(args.config)
    except ValidationError as error:
        print(error, file=sys.stderr)
        return 1

    run_dir = make_run_dir(resolved)
    run_episode(resolved, run_dir)
    print(f"Wrote the episode to {run_dir}")
    return 0


def sweep(args: argparse.Namespace) -> int:
    print("sweep: not yet implemented", file=sys.stderr)
    return 1


def analyse(args: argparse.Namespace) -> int:
    print("analyse: not yet implemented", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avalanche")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-config", help="validate one or more composable config files"
    )
    validate_parser.add_argument("config", nargs="+")
    validate_parser.set_defaults(func=validate_config)

    simulate_parser = subparsers.add_parser(
        "simulate", help="run one resolved simulator episode"
    )
    simulate_parser.add_argument("config", nargs="+")
    simulate_parser.set_defaults(func=simulate)

    sweep_parser = subparsers.add_parser("sweep", help="run an experiment matrix")
    sweep_parser.set_defaults(func=sweep)

    analyse_parser = subparsers.add_parser(
        "analyse", help="analyse a run output directory"
    )
    analyse_parser.set_defaults(func=analyse)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
