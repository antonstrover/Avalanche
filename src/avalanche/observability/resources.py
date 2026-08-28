"""Sample resources for one process tree."""

from __future__ import annotations

from dataclasses import dataclass
from os import getpid
from time import monotonic, time

import psutil  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class ProcessResource:
    """Hold resource values for one process."""

    pid: int
    name: str
    cpu_percent: float
    rss_bytes: int
    memory_percent: float


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """Hold one process-tree resource sample."""

    timestamp: float
    system_cpu_percent: float
    system_memory_percent: float
    tree_cpu_percent: float
    tree_memory_percent: float
    tree_rss_bytes: int
    process_count: int
    processes: tuple[ProcessResource, ...]
    gpu_percent: float | None = None
    gpu_memory_bytes: int | None = None


class ProcessTreeSampler:
    """Sample a root process and its current children."""

    def __init__(
        self, pid: int | None = None, *, minimum_interval: float = 0.5
    ) -> None:
        if minimum_interval < 0.0:
            raise ValueError("the resource sample interval must be nonnegative")
        self.pid = getpid() if pid is None else pid
        self.minimum_interval = minimum_interval
        self._root = psutil.Process(self.pid)
        self._last_sample_at: float | None = None
        self._last_sample: ResourceSample | None = None
        self._prime_cpu_counters()

    def sample(self, *, force: bool = False) -> ResourceSample:
        """Return one cached or current resource sample."""
        sampled_at = monotonic()
        if (
            not force
            and self._last_sample is not None
            and self._last_sample_at is not None
            and sampled_at - self._last_sample_at < self.minimum_interval
        ):
            return self._last_sample
        processes = self._processes()
        details = tuple(self._process_resource(process) for process in processes)
        sample = ResourceSample(
            timestamp=time(),
            system_cpu_percent=_system_cpu_percent(),
            system_memory_percent=_system_memory_percent(),
            tree_cpu_percent=sum(item.cpu_percent for item in details),
            tree_memory_percent=sum(item.memory_percent for item in details),
            tree_rss_bytes=sum(item.rss_bytes for item in details),
            process_count=len(details),
            processes=details,
        )
        self._last_sample_at = sampled_at
        self._last_sample = sample
        return sample

    def _prime_cpu_counters(self) -> None:
        for process in self._processes():
            try:
                process.cpu_percent(interval=None)
            except OSError, psutil.AccessDenied, psutil.NoSuchProcess:
                continue
        try:
            psutil.cpu_percent(interval=None)
        except OSError:
            pass

    def _processes(self) -> tuple[psutil.Process, ...]:
        try:
            children = self._root.children(recursive=True)
        except OSError, psutil.AccessDenied, psutil.NoSuchProcess:
            children = []
        available: dict[int, psutil.Process] = {}
        for process in (self._root, *children):
            try:
                if process.is_running():
                    available[process.pid] = process
            except OSError, psutil.AccessDenied, psutil.NoSuchProcess:
                continue
        return tuple(available[pid] for pid in sorted(available))

    @staticmethod
    def _process_resource(process: psutil.Process) -> ProcessResource:
        try:
            with process.oneshot():
                return ProcessResource(
                    pid=process.pid,
                    name=process.name(),
                    cpu_percent=float(process.cpu_percent(interval=None)),
                    rss_bytes=int(process.memory_info().rss),
                    memory_percent=float(process.memory_percent()),
                )
        except OSError, psutil.AccessDenied, psutil.NoSuchProcess:
            return ProcessResource(
                pid=process.pid,
                name="unavailable",
                cpu_percent=0.0,
                rss_bytes=0,
                memory_percent=0.0,
            )


def _system_cpu_percent() -> float:
    try:
        return float(psutil.cpu_percent(interval=None))
    except OSError:
        return 0.0


def _system_memory_percent() -> float:
    try:
        return float(psutil.virtual_memory().percent)
    except OSError:
        return 0.0
