"""Verify one complete research freeze certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from avalanche.config.run_identity import REPO_ROOT
from avalanche.experiments.sealed import validate_freeze_certificate

ANALYSIS_PATHS = (
    "scripts/run_final_evaluation.py",
    "src/avalanche/experiments/final_evaluation.py",
    "src/avalanche/metrics/__init__.py",
    "src/avalanche/metrics/online.py",
)


def analysis_manifest_bytes() -> bytes:
    """Return sorted analysis paths and exact source digests."""
    lines = [f"{path}  {_sha256(REPO_ROOT / path)}" for path in sorted(ANALYSIS_PATHS)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """Write the analysis manifest or verify one certificate."""
    parser = argparse.ArgumentParser(prog="verify_protocol_freeze")
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--write-analysis-manifest", action="store_true")
    args = parser.parse_args(argv)
    analysis = REPO_ROOT / "protocols/final/analysis-code-v1.txt"
    expected = analysis_manifest_bytes()
    if args.write_analysis_manifest:
        analysis.parent.mkdir(parents=True, exist_ok=True)
        analysis.write_bytes(expected)
    if not analysis.is_file() or analysis.read_bytes() != expected:
        raise ValueError("the analysis source manifest is stale")
    if args.certificate is None:
        return 0
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    validate_freeze_certificate(certificate)
    if certificate["analysis_code_sha256"] != hashlib.sha256(expected).hexdigest():
        raise ValueError("the freeze certificate changes the analysis code")
    return 0


def _sha256(path: Path) -> str:
    """Return one exact source digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
