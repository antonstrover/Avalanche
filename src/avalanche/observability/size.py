"""Estimate a final Parquet size from encoded rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock


@dataclass(frozen=True, slots=True)
class ParquetSizeSnapshot:
    """Hold one output-size estimate."""

    expected_rows: int | None
    written_rows: int
    buffered_rows: int
    written_bytes: int
    row_groups: int
    average_bytes_per_row: float | None
    estimated_buffered_bytes: int | None
    estimated_final_bytes: int | None
    ready: bool
    provisional: bool
    final: bool

    @property
    def observed_rows(self) -> int:
        """Return written and buffered rows."""
        return self.written_rows + self.buffered_rows

    @property
    def state(self) -> str:
        """Return the estimate state for a reporter."""
        if self.final:
            return "final"
        if self.ready:
            return "ready"
        if self.provisional:
            return "provisional"
        return "waiting"


class ParquetSizeEstimator:
    """Track encoded bytes without scanning the output repeatedly."""

    def __init__(
        self,
        expected_rows: int | None = None,
        *,
        minimum_written_rows: int = 1_000,
        minimum_row_groups: int = 2,
    ) -> None:
        if expected_rows is not None and expected_rows < 0:
            raise ValueError("the expected row count must be nonnegative")
        if minimum_written_rows < 1:
            raise ValueError("the minimum written row count must be positive")
        if minimum_row_groups < 1:
            raise ValueError("the minimum row-group count must be positive")
        self.expected_rows = expected_rows
        self.minimum_written_rows = minimum_written_rows
        self.minimum_row_groups = minimum_row_groups
        self._written_rows = 0
        self._buffered_rows = 0
        self._written_bytes = 0
        self._row_groups = 0
        self._final = False
        self._lock = RLock()

    def update(
        self,
        *,
        written_rows: int,
        written_bytes: int,
        buffered_rows: int = 0,
        row_groups: int | None = None,
        final: bool = False,
    ) -> ParquetSizeSnapshot:
        """Set cumulative writer values and return an estimate."""
        values = (written_rows, written_bytes, buffered_rows)
        if any(value < 0 for value in values):
            raise ValueError("a Parquet size value must be nonnegative")
        if row_groups is not None and row_groups < 0:
            raise ValueError("a Parquet row-group count must be nonnegative")
        with self._lock:
            if written_rows < self._written_rows:
                raise ValueError("the written row count must not decrease")
            if written_bytes < self._written_bytes:
                raise ValueError("the written byte count must not decrease")
            if row_groups is not None and row_groups < self._row_groups:
                raise ValueError("the Parquet row-group count must not decrease")
            self._written_rows = written_rows
            self._written_bytes = written_bytes
            self._buffered_rows = buffered_rows
            if row_groups is not None:
                self._row_groups = row_groups
            self._final = self._final or final
            return self._snapshot_unlocked()

    def add_row_group(
        self,
        rows: int,
        encoded_bytes: int,
        *,
        buffered_rows: int = 0,
        final: bool = False,
    ) -> ParquetSizeSnapshot:
        """Add one completed row group."""
        if rows < 0 or encoded_bytes < 0 or buffered_rows < 0:
            raise ValueError("a Parquet row-group value must be nonnegative")
        with self._lock:
            self._written_rows += rows
            self._written_bytes += encoded_bytes
            self._buffered_rows = buffered_rows
            self._row_groups += 1
            self._final = self._final or final
            return self._snapshot_unlocked()

    def observe_file(
        self,
        path: Path,
        *,
        written_rows: int,
        buffered_rows: int = 0,
        row_groups: int | None = None,
        final: bool = False,
    ) -> ParquetSizeSnapshot:
        """Read one file size and update cumulative writer values."""
        return self.update(
            written_rows=written_rows,
            written_bytes=path.stat().st_size,
            buffered_rows=buffered_rows,
            row_groups=row_groups,
            final=final,
        )

    def snapshot(self) -> ParquetSizeSnapshot:
        """Return the current estimate."""
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> ParquetSizeSnapshot:
        average = (
            self._written_bytes / self._written_rows if self._written_rows > 0 else None
        )
        estimated_buffered = (
            round(average * self._buffered_rows) if average is not None else None
        )
        expected_rows = self.expected_rows
        estimate = (
            round(average * expected_rows)
            if average is not None and expected_rows is not None
            else None
        )
        if self._final:
            estimate = self._written_bytes
        ready = self._final or (
            self._written_rows >= self.minimum_written_rows
            and self._row_groups >= self.minimum_row_groups
        )
        return ParquetSizeSnapshot(
            expected_rows=expected_rows,
            written_rows=self._written_rows,
            buffered_rows=self._buffered_rows,
            written_bytes=self._written_bytes,
            row_groups=self._row_groups,
            average_bytes_per_row=average,
            estimated_buffered_bytes=estimated_buffered,
            estimated_final_bytes=estimate,
            ready=ready,
            provisional=estimate is not None and not ready,
            final=self._final,
        )
