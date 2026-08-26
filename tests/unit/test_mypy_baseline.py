"""Test the line-independent mypy baseline."""

import json
import runpy
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_mypy_baseline.py"
SCRIPT = runpy.run_path(str(SCRIPT_PATH))
Fingerprint = SCRIPT["Fingerprint"]
compare_errors = SCRIPT["compare_errors"]
load_baseline = SCRIPT["load_baseline"]
parse_mypy_output = SCRIPT["parse_mypy_output"]


def _output(line: int, message: str = "Bad value", code: str = "arg-type") -> str:
    return f"src/avalanche/example.py:{line}: error: {message}  [{code}]"


def test_an_exact_baseline_has_no_changes():
    errors = parse_mypy_output(_output(10))

    assert compare_errors(errors, errors) == ({}, {})


def test_a_new_error_is_added():
    baseline = parse_mypy_output(_output(10))
    current = parse_mypy_output("\n".join((_output(10), _output(20, "New value"))))

    added, removed = compare_errors(current, baseline)

    assert sum(added.values()) == 1
    assert not removed


def test_a_resolved_error_is_removed():
    baseline = parse_mypy_output("\n".join((_output(10), _output(20, "Old value"))))
    current = parse_mypy_output(_output(10))

    added, removed = compare_errors(current, baseline)

    assert not added
    assert sum(removed.values()) == 1


def test_a_moved_error_keeps_its_fingerprint():
    baseline = parse_mypy_output(_output(10))
    current = parse_mypy_output(_output(90))

    assert compare_errors(current, baseline) == ({}, {})


def test_duplicate_errors_keep_their_counts():
    baseline = parse_mypy_output(_output(10))
    current = parse_mypy_output("\n".join((_output(10), _output(20))))

    added, removed = compare_errors(current, baseline)

    assert next(iter(added.values())) == 1
    assert not removed


def test_two_error_codes_in_one_file_stay_separate():
    errors = parse_mypy_output("\n".join((_output(10), _output(10, code="index"))))

    assert {error.code for error in errors} == {"arg-type", "index"}


def test_an_unparseable_error_is_rejected():
    with pytest.raises(ValueError, match="cannot parse"):
        parse_mypy_output("src/avalanche/example.py:10: error: missing code")


def test_a_malformed_baseline_is_rejected(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"baseline_version": 1, "errors": [{}]}))

    with pytest.raises(ValueError, match="invalid error fields"):
        load_baseline(path)


def test_an_unsorted_baseline_is_rejected(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "baseline_version": 1,
                "errors": [
                    {"path": "z.py", "code": "a", "message": "A", "count": 1},
                    {"path": "a.py", "code": "a", "message": "A", "count": 1},
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="not sorted"):
        load_baseline(path)


def test_a_fingerprint_records_the_path_code_and_message():
    errors = parse_mypy_output(_output(10))

    assert next(iter(errors)) == Fingerprint(
        "src/avalanche/example.py", "arg-type", "Bad value"
    )
