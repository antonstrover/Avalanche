"""The avalanche command-line interface."""

import argparse
import json
import sys

from avalanche.config import (
    ConfigurationResolutionError,
    ConfigurationResolver,
    ResolvedConfig,
    make_run_dir,
)
from avalanche.experiments import run_episode


def _resolve_config(args: argparse.Namespace) -> ResolvedConfig:
    return ConfigurationResolver().resolve(
        mountain=args.mountain,
        scenario=args.scenario,
        controller=args.controller,
        monitor=args.monitor,
        override=args.override,
    )


def _report_config_error(error: ConfigurationResolutionError) -> int:
    """Print one configuration error and return the failure code."""
    print(error, file=sys.stderr)
    return 1


def validate_config(args: argparse.Namespace) -> int:
    try:
        resolved = _resolve_config(args)
    except ConfigurationResolutionError as error:
        return _report_config_error(error)
    _print_preflight(resolved)
    return 0


def simulate(args: argparse.Namespace) -> int:
    try:
        resolved = _resolve_config(args)
    except ConfigurationResolutionError as error:
        return _report_config_error(error)

    if args.preflight:
        _print_preflight(resolved)
        return 0

    run_dir = make_run_dir(resolved)
    run_episode(resolved, run_dir)
    print(f"Wrote the episode to {run_dir}")
    return 0


def _print_preflight(resolved: ResolvedConfig) -> None:
    """Print one stable configuration preflight record."""
    value = {
        "provenance": [
            record.model_dump(mode="json") for record in resolved.provenance
        ],
        "resolved_configuration_sha256": resolved.resolved_configuration_sha256,
        "scientific_configuration_sha256": resolved.scientific_configuration_sha256,
        "status": "OK",
    }
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


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
        "validate-config", help="validate four formal configuration components"
    )
    _add_component_arguments(validate_parser)
    validate_parser.set_defaults(func=validate_config)

    simulate_parser = subparsers.add_parser(
        "simulate", help="run one resolved simulator episode"
    )
    _add_component_arguments(simulate_parser)
    simulate_parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate and report without running an episode",
    )
    simulate_parser.set_defaults(func=simulate)

    sweep_parser = subparsers.add_parser("sweep", help="run an experiment matrix")
    sweep_parser.set_defaults(func=sweep)

    analyse_parser = subparsers.add_parser(
        "analyse", help="analyse a run output directory"
    )
    analyse_parser.set_defaults(func=analyse)

    return parser


def _add_component_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the required formal component selections."""
    parser.add_argument("--mountain", required=True, action=_SingleValueAction)
    parser.add_argument("--scenario", required=True, action=_SingleValueAction)
    parser.add_argument("--controller", required=True, action=_SingleValueAction)
    parser.add_argument("--monitor", required=True, action=_SingleValueAction)
    parser.add_argument("--override", action=_SingleValueAction)


class _SingleValueAction(argparse.Action):
    """Reject a repeated named component option."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"the option {option_string} must occur only once")
        setattr(namespace, self.dest, values)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
