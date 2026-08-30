"""Test process resource sampling with controlled process handles."""

from __future__ import annotations

from types import SimpleNamespace

import psutil
import pytest

from avalanche.observability import ProcessIdentity, ProcessTreeSampler, resources


class ValueSeries:
    """Return controlled values and retain the final value."""

    def __init__(self, *values: object) -> None:
        self.values = list(values)

    def take(self) -> object:
        """Return the next controlled value."""
        value = self.values.pop(0) if len(self.values) > 1 else self.values[0]
        if isinstance(value, BaseException):
            raise value
        return value


class FakeProcess:
    """Provide the psutil process operations used by the sampler."""

    def __init__(
        self,
        pid: int,
        create_time: float,
        *cpu_values: object,
        name: str | None = None,
        rss_bytes: int = 100,
        memory_percent: float = 1.0,
    ) -> None:
        self.pid = pid
        self.created_at = create_time
        self.cpu_values = ValueSeries(*cpu_values)
        self.process_name = name or f"process-{pid}"
        self.rss = rss_bytes
        self.memory = memory_percent
        self.running = True
        self.current_children: list[FakeProcess] = []
        self.cpu_calls = 0

    def children(self, *, recursive: bool) -> list[FakeProcess]:
        """Return the current controlled descendants."""
        assert recursive
        return list(self.current_children)

    def create_time(self) -> float:
        """Return the controlled creation time."""
        return self.created_at

    def is_running(self) -> bool:
        """Return the controlled running state."""
        return self.running

    def cpu_percent(self, *, interval: None) -> float:
        """Return the next controlled process CPU value."""
        assert interval is None
        self.cpu_calls += 1
        return float(self.cpu_values.take())

    def name(self) -> str:
        """Return the controlled process name."""
        return self.process_name

    def memory_info(self) -> SimpleNamespace:
        """Return the controlled resident memory value."""
        return SimpleNamespace(rss=self.rss)

    def memory_percent(self) -> float:
        """Return the controlled process memory percentage."""
        return self.memory


def install_psutil(
    monkeypatch: pytest.MonkeyPatch,
    root: FakeProcess,
    *processes: FakeProcess,
    system_cpu: tuple[object, ...] = (0.0, 50.0),
    logical_cpus: int | None = 10,
) -> dict[int, FakeProcess]:
    """Install one controlled psutil process table."""
    registry = {process.pid: process for process in (root, *processes)}
    system_values = ValueSeries(*system_cpu)

    def process_for(pid: int) -> FakeProcess:
        try:
            return registry[pid]
        except KeyError as error:
            raise psutil.NoSuchProcess(pid) from error

    def system_cpu_percent(*, interval: None) -> float:
        assert interval is None
        return float(system_values.take())

    monkeypatch.setattr(resources.psutil, "Process", process_for)
    monkeypatch.setattr(resources.psutil, "cpu_percent", system_cpu_percent)
    monkeypatch.setattr(
        resources.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=62.5),
    )
    monkeypatch.setattr(
        resources.psutil,
        "cpu_count",
        lambda *, logical: logical_cpus if logical else None,
    )
    return registry


def test_sampler_retains_handles_and_reports_raw_cpu_units(monkeypatch):
    root = FakeProcess(10, 100.0, 0.0, 35.0, rss_bytes=100, memory_percent=1.0)
    child = FakeProcess(
        20,
        200.0,
        0.0,
        350.0,
        rss_bytes=300,
        memory_percent=3.0,
    )
    replacement_handle = FakeProcess(20, 200.0, 999.0)
    root.current_children = [child]
    install_psutil(monkeypatch, root, child, system_cpu=(0.0, 100.0))
    sampler = ProcessTreeSampler(root.pid, minimum_interval=0.0)
    root.current_children = [replacement_handle]

    sample = sampler.sample(force=True)

    assert sample.tree_cpu_percent == pytest.approx(385.0)
    assert sample.raw_cpu_percent == pytest.approx(385.0)
    assert sample.tree_cpu_cores == pytest.approx(3.85)
    assert sample.equivalent_cores == pytest.approx(3.85)
    assert sample.logical_cpu_count == 10
    assert sample.tree_cpu_capacity_percent == pytest.approx(38.5)
    assert sample.machine_capacity_percent == pytest.approx(38.5)
    assert sample.system_cpu_percent == pytest.approx(100.0)
    assert sample.tree_rss_bytes == 400
    assert sample.tree_memory_percent == pytest.approx(4.0)
    assert [item.pid for item in sample.processes] == [20, 10]
    assert child.cpu_calls == 2
    assert replacement_handle.cpu_calls == 0


def test_new_process_is_primed_before_its_cpu_is_reported(monkeypatch):
    root = FakeProcess(10, 100.0, 0.0, 25.0, 30.0)
    child = FakeProcess(20, 200.0, 0.0, 175.0)
    install_psutil(monkeypatch, root, child)
    sampler = ProcessTreeSampler(root.pid, minimum_interval=0.0)
    root.current_children = [child]

    first = sampler.sample(force=True)
    second = sampler.sample(force=True)

    child_first = next(item for item in first.processes if item.pid == child.pid)
    assert child_first.cpu_percent is None
    assert first.tree_cpu_percent is None
    assert first.tree_cpu_cores is None
    assert first.tree_cpu_capacity_percent is None
    assert child.cpu_calls == 2
    assert second.tree_cpu_percent == pytest.approx(205.0)
    assert [item.pid for item in second.processes] == [20, 10]


def test_pid_reuse_replaces_the_cached_process_identity(monkeypatch):
    root = FakeProcess(10, 100.0, 0.0, 10.0, 20.0, 30.0, 40.0)
    old_child = FakeProcess(20, 200.0, 0.0, 80.0, 999.0)
    new_child = FakeProcess(20, 300.0, 0.0, 240.0, 250.0)
    root.current_children = [old_child]
    install_psutil(monkeypatch, root, old_child, new_child)
    sampler = ProcessTreeSampler(root.pid, minimum_interval=0.0)

    initial = sampler.sample(force=True)
    root.current_children = [new_child]
    reused = sampler.sample(force=True)
    stable = sampler.sample(force=True)
    root.current_children = []
    exited = sampler.sample(force=True)

    assert initial.processes[0].identity == ProcessIdentity(20, 200.0)
    reused_child = next(item for item in reused.processes if item.pid == 20)
    assert reused_child.identity == ProcessIdentity(20, 300.0)
    assert reused_child.cpu_percent is None
    assert stable.tree_cpu_percent == pytest.approx(270.0)
    assert old_child.cpu_calls == 2
    assert new_child.cpu_calls == 2
    assert {item.pid for item in exited.processes} == {10}
    assert all(identity.pid != 20 for identity in sampler._process_cache)


def test_identity_exclusion_removes_cpu_and_memory_totals(monkeypatch):
    root = FakeProcess(
        10,
        100.0,
        0.0,
        40.0,
        50.0,
        60.0,
        rss_bytes=100,
        memory_percent=1.0,
    )
    observer = FakeProcess(
        20,
        200.0,
        0.0,
        300.0,
        rss_bytes=900,
        memory_percent=9.0,
    )
    reused_pid = FakeProcess(20, 300.0, 0.0, 250.0)
    root.current_children = [observer]
    install_psutil(monkeypatch, root, observer, reused_pid)
    sampler = ProcessTreeSampler(root.pid, minimum_interval=0.0)

    identity = sampler.exclude_process(observer.pid, observer.created_at)
    sample = sampler.sample(force=True)
    root.current_children = [reused_pid]
    primed = sampler.sample(force=True)
    reused = sampler.sample(force=True)

    assert identity == ProcessIdentity(observer.pid, observer.created_at)
    assert [item.pid for item in sample.processes] == [root.pid]
    assert sample.tree_cpu_percent == pytest.approx(40.0)
    assert sample.tree_rss_bytes == root.rss
    assert sample.tree_memory_percent == pytest.approx(root.memory)
    assert observer.cpu_calls == 1
    primed_child = next(item for item in primed.processes if item.pid == observer.pid)
    assert primed_child.identity == ProcessIdentity(observer.pid, 300.0)
    assert primed_child.cpu_percent is None
    assert reused.tree_cpu_percent == pytest.approx(310.0)


def test_an_unresolved_exclusion_uses_a_pid_fallback(monkeypatch):
    root = FakeProcess(10, 100.0, 0.0, 40.0)
    observer = FakeProcess(20, 200.0, 0.0, 300.0)
    registry = install_psutil(monkeypatch, root)
    sampler = ProcessTreeSampler(root.pid, minimum_interval=0.0)

    assert sampler.exclude_process(observer.pid) is None
    registry[observer.pid] = observer
    root.current_children = [observer]
    sample = sampler.sample(force=True)

    assert [item.pid for item in sample.processes] == [root.pid]
    assert observer.cpu_calls == 0


def test_unavailable_cpu_is_not_reported_as_zero(monkeypatch):
    root = FakeProcess(
        10,
        100.0,
        0.0,
        psutil.AccessDenied(10),
        0.0,
        50.0,
    )
    install_psutil(
        monkeypatch,
        root,
        system_cpu=(0.0, OSError("system sample unavailable")),
    )
    sampler = ProcessTreeSampler(root.pid, minimum_interval=0.0)

    unavailable = sampler.sample(force=True)
    reprime = sampler.sample(force=True)
    available = sampler.sample(force=True)

    assert unavailable.processes[0].cpu_percent is None
    assert unavailable.tree_cpu_percent is None
    assert unavailable.system_cpu_percent is None
    assert reprime.tree_cpu_percent is None
    assert available.tree_cpu_percent == pytest.approx(50.0)
