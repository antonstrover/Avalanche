"""Sample resources for one process tree."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite
from os import getpid
from threading import RLock
from time import monotonic, time
from typing import TypeVar

import psutil  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True, order=True)
class ProcessIdentity:
    """Identify one process instance without a PID reuse ambiguity."""

    pid: int
    create_time: float


@dataclass(frozen=True, slots=True)
class ProcessResource:
    """Hold resource values for one process."""

    pid: int
    create_time: float
    name: str
    cpu_percent: float | None
    rss_bytes: int | None
    memory_percent: float | None

    @property
    def identity(self) -> ProcessIdentity:
        """Return the stable identity for this process instance."""
        return ProcessIdentity(self.pid, self.create_time)


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """Hold one process-tree resource sample."""

    timestamp: float
    system_cpu_percent: float | None
    system_memory_percent: float | None
    tree_cpu_percent: float | None
    tree_memory_percent: float | None
    tree_rss_bytes: int | None
    process_count: int
    processes: tuple[ProcessResource, ...]
    logical_cpu_count: int | None = None
    tree_cpu_cores: float | None = None
    tree_cpu_capacity_percent: float | None = None
    gpu_percent: float | None = None
    gpu_memory_bytes: int | None = None

    @property
    def raw_cpu_percent(self) -> float | None:
        """Return the unnormalised workload CPU percentage."""
        return self.tree_cpu_percent

    @property
    def equivalent_cores(self) -> float | None:
        """Return the equivalent count of fully used CPU cores."""
        return self.tree_cpu_cores

    @property
    def machine_capacity_percent(self) -> float | None:
        """Return the workload share of the total machine capacity."""
        return self.tree_cpu_capacity_percent


class ProcessTreeSampler:
    """Sample a root process and its current children."""

    def __init__(
        self,
        pid: int | None = None,
        *,
        minimum_interval: float = 0.5,
        excluded_processes: Iterable[ProcessIdentity] = (),
        excluded_pids: Iterable[int] = (),
    ) -> None:
        if minimum_interval < 0.0:
            raise ValueError("the resource sample interval must be nonnegative")
        self.pid = getpid() if pid is None else pid
        self.minimum_interval = minimum_interval
        self._root = psutil.Process(self.pid)
        self._process_cache: dict[ProcessIdentity, psutil.Process] = {}
        self._cpu_ready: set[ProcessIdentity] = set()
        self._excluded_processes = set(excluded_processes)
        self._excluded_pids = {int(value) for value in excluded_pids}
        self._last_sample_at: float | None = None
        self._last_sample: ResourceSample | None = None
        self._lock = RLock()
        self._prime_cpu_counters()

    def exclude_process(
        self,
        pid: int,
        create_time: float | None = None,
    ) -> ProcessIdentity | None:
        """Exclude one process identity from each workload total."""
        process_pid = int(pid)
        resolved_time = create_time
        if resolved_time is None:
            try:
                resolved_time = float(psutil.Process(process_pid).create_time())
            except OSError, psutil.AccessDenied, psutil.NoSuchProcess:
                resolved_time = None
        with self._lock:
            if resolved_time is None or not isfinite(float(resolved_time)):
                self._excluded_pids.add(process_pid)
                self._evict_pid(process_pid)
                identity = None
            else:
                identity = ProcessIdentity(process_pid, float(resolved_time))
                self._excluded_processes.add(identity)
                self._excluded_pids.discard(process_pid)
                self._evict(identity)
            self._clear_sample_cache()
            return identity

    def sample(self, *, force: bool = False) -> ResourceSample:
        """Return one cached or current resource sample."""
        sampled_at = monotonic()
        with self._lock:
            if (
                not force
                and self._last_sample is not None
                and self._last_sample_at is not None
                and sampled_at - self._last_sample_at < self.minimum_interval
            ):
                return self._last_sample
            processes, newly_primed = self._refresh_processes()
            details = tuple(
                self._process_resource(
                    identity,
                    process,
                    report_cpu=identity not in newly_primed,
                )
                for identity, process in processes
            )
            details = tuple(sorted(details, key=_process_sort_key))
            raw_cpu = _sum_optional_float(item.cpu_percent for item in details)
            memory_percent = _sum_optional_float(
                item.memory_percent for item in details
            )
            rss_bytes = _sum_optional_int(item.rss_bytes for item in details)
            logical_cpu_count = _logical_cpu_count()
            equivalent_cores = raw_cpu / 100.0 if raw_cpu is not None else None
            capacity = (
                raw_cpu / logical_cpu_count
                if raw_cpu is not None and logical_cpu_count is not None
                else None
            )
            sample = ResourceSample(
                timestamp=time(),
                system_cpu_percent=_system_cpu_percent(),
                system_memory_percent=_system_memory_percent(),
                tree_cpu_percent=raw_cpu,
                tree_memory_percent=memory_percent,
                tree_rss_bytes=rss_bytes,
                process_count=len(details),
                processes=details,
                logical_cpu_count=logical_cpu_count,
                tree_cpu_cores=equivalent_cores,
                tree_cpu_capacity_percent=capacity,
            )
            self._last_sample_at = sampled_at
            self._last_sample = sample
            return sample

    def _prime_cpu_counters(self) -> None:
        with self._lock:
            self._refresh_processes()
            try:
                psutil.cpu_percent(interval=None)
            except OSError, psutil.Error:
                pass

    def _refresh_processes(
        self,
    ) -> tuple[
        tuple[tuple[ProcessIdentity, psutil.Process], ...],
        frozenset[ProcessIdentity],
    ]:
        try:
            children = self._root.children(recursive=True)
        except OSError, psutil.AccessDenied, psutil.NoSuchProcess:
            children = []
        seen: set[ProcessIdentity] = set()
        newly_primed: set[ProcessIdentity] = set()
        for process in (self._root, *children):
            identity = self._process_identity(process)
            if identity is None or self._is_excluded(identity):
                continue
            try:
                if not process.is_running():
                    continue
            except OSError, psutil.AccessDenied, psutil.NoSuchProcess:
                continue
            seen.add(identity)
            self._evict_reused_pid(identity)
            cached = self._process_cache.get(identity)
            if cached is None:
                self._process_cache[identity] = process
                cached = process
            if identity not in self._cpu_ready:
                newly_primed.add(identity)
                if self._prime_process_cpu(identity, cached):
                    self._cpu_ready.add(identity)
        for identity in tuple(self._process_cache):
            if identity not in seen or self._is_excluded(identity):
                self._evict(identity)
        available = tuple(
            (identity, self._process_cache[identity])
            for identity in sorted(self._process_cache)
        )
        return available, frozenset(newly_primed)

    @staticmethod
    def _process_identity(process: psutil.Process) -> ProcessIdentity | None:
        try:
            create_time = float(process.create_time())
            if not isfinite(create_time):
                return None
            return ProcessIdentity(int(process.pid), create_time)
        except OSError, psutil.AccessDenied, psutil.NoSuchProcess:
            return None

    def _prime_process_cpu(
        self,
        identity: ProcessIdentity,
        process: psutil.Process,
    ) -> bool:
        try:
            process.cpu_percent(interval=None)
            return True
        except psutil.NoSuchProcess:
            self._evict(identity)
            return False
        except OSError, psutil.AccessDenied:
            return False

    def _process_resource(
        self,
        identity: ProcessIdentity,
        process: psutil.Process,
        *,
        report_cpu: bool,
    ) -> ProcessResource:
        name = self._read_process_value(identity, process.name)
        cpu_percent: float | None = None
        if report_cpu and identity in self._cpu_ready:
            cpu_value = self._read_process_value(
                identity,
                lambda: float(process.cpu_percent(interval=None)),
            )
            if cpu_value is None or not isfinite(cpu_value) or cpu_value < 0.0:
                self._cpu_ready.discard(identity)
            else:
                cpu_percent = cpu_value
        rss_value = self._read_process_value(
            identity,
            lambda: int(process.memory_info().rss),
        )
        memory_value = self._read_process_value(
            identity,
            lambda: float(process.memory_percent()),
        )
        rss_bytes = rss_value if rss_value is not None and rss_value >= 0 else None
        memory_percent = (
            memory_value
            if memory_value is not None
            and isfinite(memory_value)
            and memory_value >= 0.0
            else None
        )
        return ProcessResource(
            pid=identity.pid,
            create_time=identity.create_time,
            name=name or "unavailable",
            cpu_percent=cpu_percent,
            rss_bytes=rss_bytes,
            memory_percent=memory_percent,
        )

    def _read_process_value(
        self,
        identity: ProcessIdentity,
        getter: Callable[[], _Value],
    ) -> _Value | None:
        try:
            return getter()
        except psutil.NoSuchProcess:
            self._evict(identity)
            return None
        except OSError, psutil.AccessDenied:
            return None

    def _is_excluded(self, identity: ProcessIdentity) -> bool:
        return (
            identity in self._excluded_processes or identity.pid in self._excluded_pids
        )

    def _evict_reused_pid(self, current: ProcessIdentity) -> None:
        for identity in tuple(self._process_cache):
            if identity.pid == current.pid and identity != current:
                self._evict(identity)

    def _evict_pid(self, pid: int) -> None:
        for identity in tuple(self._process_cache):
            if identity.pid == pid:
                self._evict(identity)

    def _evict(self, identity: ProcessIdentity) -> None:
        self._process_cache.pop(identity, None)
        self._cpu_ready.discard(identity)

    def _clear_sample_cache(self) -> None:
        self._last_sample_at = None
        self._last_sample = None


_Value = TypeVar("_Value")


def _process_sort_key(item: ProcessResource) -> tuple[bool, float, int, float]:
    cpu_percent = item.cpu_percent
    return (
        cpu_percent is None,
        -(cpu_percent if cpu_percent is not None else 0.0),
        item.pid,
        item.create_time,
    )


def _sum_optional_float(values: Iterable[float | None]) -> float | None:
    collected = tuple(values)
    if any(value is None for value in collected):
        return None
    return sum(value for value in collected if value is not None)


def _sum_optional_int(values: Iterable[int | None]) -> int | None:
    collected = tuple(values)
    if any(value is None for value in collected):
        return None
    return sum(value for value in collected if value is not None)


def _system_cpu_percent() -> float | None:
    try:
        value = float(psutil.cpu_percent(interval=None))
        return value if isfinite(value) and value >= 0.0 else None
    except OSError, psutil.Error:
        return None


def _system_memory_percent() -> float | None:
    try:
        value = float(psutil.virtual_memory().percent)
        return value if isfinite(value) and value >= 0.0 else None
    except OSError, psutil.Error:
        return None


def _logical_cpu_count() -> int | None:
    try:
        value = psutil.cpu_count(logical=True)
    except OSError, psutil.Error:
        return None
    return int(value) if value is not None and value > 0 else None
