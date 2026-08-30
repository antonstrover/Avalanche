"""Check the dataset command observer boundary."""

import os
import runpy
from collections.abc import Callable
from pathlib import Path
from threading import current_thread, main_thread
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_monitor_dataset.py"


def _load_command(
    monkeypatch: pytest.MonkeyPatch,
    generate: Callable[..., None],
) -> tuple[Callable[[list[str]], int], list[Any]]:
    """Load the command and replace its external boundary objects."""
    namespace = runpy.run_path(str(SCRIPT))
    command_globals = namespace["main"].__globals__
    sessions: list[Any] = []

    class Session:
        def __init__(self, **values: Any) -> None:
            self.values = values
            self.events: list[Any] = []
            self.emitter = SimpleNamespace(emit=self.events.append)
            self.process_emitter = object()
            self.aggregator = SimpleNamespace(
                snapshot=lambda: SimpleNamespace(stages=())
            )
            self.close_calls = 0
            self.drain_calls = 0
            self.exit_type: type[BaseException] | None = None
            sessions.append(self)

        def __enter__(self) -> Any:
            return self

        def __exit__(
            self,
            error_type: type[BaseException] | None,
            _error: BaseException | None,
            _traceback: Any,
        ) -> bool:
            self.exit_type = error_type
            self.close_calls += 1
            return False

        def drain_pending(self) -> int:
            self.drain_calls += 1
            return 0

    monkeypatch.setitem(command_globals, "generate_dataset", generate)
    monkeypatch.setitem(command_globals, "ObservabilitySession", Session)
    return namespace["main"], sessions


def _arguments(tmp_path: Path, *, no_progress: bool = False) -> list[str]:
    """Create temporary command paths and return their arguments."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("seeds: [20260829]\nfamilies: []\n")
    arguments = [str(manifest), "--output", str(tmp_path / "rows.parquet")]
    if no_progress:
        arguments.append("--no-progress")
    return arguments


@pytest.mark.parametrize(
    ("no_progress", "expected_enabled"),
    ((False, None), (True, False)),
)
def test_generation_stays_in_the_caller_process_and_main_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_progress: bool,
    expected_enabled: bool | None,
) -> None:
    calls: list[tuple[int, object, Path]] = []

    def generate(_manifest: Path, output: Path, **_values: Any) -> None:
        calls.append((os.getpid(), current_thread(), output))

    main, sessions = _load_command(monkeypatch, generate)

    assert current_thread() is main_thread()
    assert main(_arguments(tmp_path, no_progress=no_progress)) == 0

    assert calls == [(os.getpid(), main_thread(), tmp_path / "rows.parquet")]
    assert len(sessions) == 1
    assert sessions[0].values["enabled"] is expected_enabled
    assert sessions[0].values["multiprocessing"] is True
    assert sessions[0].values["log_path"].parent == tmp_path
    assert sessions[0].close_calls == 1
    assert sessions[0].exit_type is None


def test_generation_failure_propagates_after_the_session_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def generate(_manifest: Path, _output: Path, **_values: Any) -> None:
        raise RuntimeError("generation failed")

    main, sessions = _load_command(monkeypatch, generate)

    with pytest.raises(RuntimeError, match="generation failed"):
        main(_arguments(tmp_path))

    assert sessions[0].drain_calls == 1
    assert sessions[0].close_calls == 1
    assert sessions[0].exit_type is RuntimeError


def test_keyboard_interrupt_propagates_after_the_session_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def generate(_manifest: Path, _output: Path, **_values: Any) -> None:
        raise KeyboardInterrupt

    main, sessions = _load_command(monkeypatch, generate)

    with pytest.raises(KeyboardInterrupt):
        main(_arguments(tmp_path))

    assert sessions[0].drain_calls == 0
    assert sessions[0].close_calls == 1
    assert sessions[0].exit_type is KeyboardInterrupt
