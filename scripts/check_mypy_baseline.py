"""Reject mypy errors that are absent from the tracked baseline."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "mypy-baseline.json"
ERROR_PATTERN = re.compile(
    r"^(?P<path>.+?):\d+(?::\d+)?: error: "
    r"(?P<message>.+?)  \[(?P<code>[^]]+)]$"
)


@dataclass(frozen=True, order=True)
class Fingerprint:
    """Identify one mypy error without its source line."""

    path: str
    code: str
    message: str


ErrorCounts = Counter[Fingerprint]


def parse_mypy_output(output: str) -> ErrorCounts:
    """Parse each mypy error into a stable fingerprint."""
    errors: ErrorCounts = Counter()
    for line in output.splitlines():
        if ": error:" not in line:
            continue
        match = ERROR_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"cannot parse a mypy error: {line}")
        path = _relative_path(match.group("path"))
        fingerprint = Fingerprint(
            path=path,
            code=match.group("code"),
            message=match.group("message"),
        )
        errors[fingerprint] += 1
    return errors


def _relative_path(value: str) -> str:
    """Return one repository-relative path."""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"a mypy path leaves the repository: {value}") from error


def load_baseline(path: Path = BASELINE_PATH) -> ErrorCounts:
    """Load and validate the tracked mypy baseline."""
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("the mypy baseline is missing or damaged") from error
    if not isinstance(document, dict) or set(document) != {
        "baseline_version",
        "errors",
    }:
        raise ValueError("the mypy baseline has invalid fields")
    if document["baseline_version"] != 1 or not isinstance(document["errors"], list):
        raise ValueError("the mypy baseline has an invalid version")

    entries = document["errors"]
    errors: ErrorCounts = Counter()
    sort_keys: list[tuple[str, str, str]] = []
    for entry in entries:
        fingerprint, count = _load_entry(entry)
        if fingerprint in errors:
            raise ValueError("the mypy baseline contains a duplicate error")
        errors[fingerprint] = count
        sort_keys.append((fingerprint.path, fingerprint.code, fingerprint.message))
    if sort_keys != sorted(sort_keys):
        raise ValueError("the mypy baseline errors are not sorted")
    return errors


def _load_entry(entry: Any) -> tuple[Fingerprint, int]:
    """Validate one stored baseline entry."""
    if not isinstance(entry, dict) or set(entry) != {
        "path",
        "code",
        "message",
        "count",
    }:
        raise ValueError("the mypy baseline contains invalid error fields")
    values = (entry["path"], entry["code"], entry["message"])
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("the mypy baseline contains an invalid error")
    count = entry["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("the mypy baseline contains an invalid count")
    return Fingerprint(values[0], values[1], values[2]), count


def compare_errors(
    current: ErrorCounts,
    baseline: ErrorCounts,
) -> tuple[ErrorCounts, ErrorCounts]:
    """Return added errors and removed errors."""
    return current - baseline, baseline - current


def baseline_document(errors: ErrorCounts) -> dict[str, object]:
    """Return a sorted JSON document for known errors."""
    return {
        "baseline_version": 1,
        "errors": [
            {
                "path": error.path,
                "code": error.code,
                "message": error.message,
                "count": errors[error],
            }
            for error in sorted(errors)
        ],
    }


def _print_errors(label: str, errors: ErrorCounts) -> None:
    """Print one fingerprint group with stable ordering."""
    print(label)
    for error in sorted(errors):
        suffix = f" ({errors[error]} occurrences)" if errors[error] != 1 else ""
        print(f"- {error.path}: error: {error.message} [{error.code}]{suffix}")


def main() -> int:
    """Run mypy and reject each new fingerprint."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--show-error-codes",
            "--no-error-summary",
            "src/avalanche",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode not in {0, 1}:
        print(output, file=sys.stderr)
        print("mypy could not complete", file=sys.stderr)
        return 1
    try:
        current = parse_mypy_output(output)
        baseline = load_baseline()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    added, removed = compare_errors(current, baseline)
    if removed:
        _print_errors("Resolved mypy errors:", removed)
    if added:
        _print_errors("New mypy errors:", added)
        return 1
    print(f"Mypy baseline passed with {sum(current.values())} known errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
