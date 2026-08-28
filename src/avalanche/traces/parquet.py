"""Write ordered Parquet rows with bounded buffering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROW_GROUP_ROWS = 65_536


@dataclass(frozen=True)
class ParquetWriteProgress:
    """Describe the encoded and buffered output."""

    encoded_bytes: int
    written_rows: int
    buffered_rows: int
    row_groups: int
    final: bool = False


class BufferedParquetWriter:
    """Write one ordered table through complete row groups."""

    def __init__(
        self,
        path: Path,
        *,
        row_group_rows: int = DEFAULT_ROW_GROUP_ROWS,
        on_progress: Callable[[ParquetWriteProgress], None] | None = None,
    ) -> None:
        if row_group_rows < 1:
            raise ValueError("a Parquet row group must contain at least one row")
        self.path = Path(path)
        self.row_group_rows = row_group_rows
        self.on_progress = on_progress
        self._frames: list[pd.DataFrame] = []
        self._buffered_rows = 0
        self._written_rows = 0
        self._row_groups = 0
        self._sink: pa.NativeFile | None = None
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self._partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        self._closed = False

    def write(self, frame: pd.DataFrame) -> None:
        """Buffer one ordered frame and write each complete row group."""
        if self._closed:
            raise ValueError("the Parquet writer is closed")
        if frame.empty:
            return
        self._frames.append(frame)
        self._buffered_rows += len(frame)
        while self._buffered_rows >= self.row_group_rows:
            self._flush_rows(self.row_group_rows)
        self._report()

    def close(self) -> Path:
        """Flush the remaining rows and publish the complete file."""
        if self._closed:
            return self.path
        try:
            if self._buffered_rows:
                self._flush_rows(self._buffered_rows)
            if self._writer is None or self._sink is None:
                raise ValueError("the Parquet writer received no rows")
            self._writer.close()
            self._sink.flush()
            self._sink.close()
            self._writer = None
            self._sink = None
            self._partial_path.replace(self.path)
            self._closed = True
            self._report(final=True)
            return self.path
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        """Close the temporary output after a failed write."""
        writer = self._writer
        sink = self._sink
        self._writer = None
        self._sink = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if sink is not None:
            try:
                sink.close()
            except Exception:
                pass
        try:
            self._partial_path.unlink(missing_ok=True)
        except OSError:
            pass
        self._closed = True

    def __enter__(self) -> BufferedParquetWriter:
        return self

    def __exit__(self, error_type, error, traceback) -> Literal[False]:
        if error_type is None:
            self.close()
        else:
            self.abort()
        return False

    def _flush_rows(self, count: int) -> None:
        """Encode the next ordered rows as one row group."""
        combined = pd.concat(self._frames, ignore_index=True)
        selected = combined.iloc[:count].reset_index(drop=True)
        remainder = combined.iloc[count:].reset_index(drop=True)
        self._frames = [] if remainder.empty else [remainder]
        self._buffered_rows -= count
        table = pa.Table.from_pandas(selected, preserve_index=False)
        self._ensure_writer(table.schema)
        assert self._writer is not None
        self._writer.write_table(table, row_group_size=count)
        assert self._sink is not None
        self._sink.flush()
        self._written_rows += count
        self._row_groups += 1
        self._report()

    def _ensure_writer(self, schema: pa.Schema) -> None:
        """Open the temporary output for the first row group."""
        if self._writer is not None:
            assert self._schema is not None
            if not self._schema.equals(schema, check_metadata=True):
                raise ValueError("each Parquet row group must use one schema")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sink = pa.OSFile(str(self._partial_path), "wb")
        self._writer = pq.ParquetWriter(self._sink, schema, compression="snappy")
        self._schema = schema

    def _report(self, *, final: bool = False) -> None:
        """Report the current encoded and buffered counts."""
        if self.on_progress is None:
            return
        if final:
            encoded_bytes = self.path.stat().st_size
        elif self._sink is None:
            encoded_bytes = 0
        else:
            encoded_bytes = self._sink.tell()
        self.on_progress(
            ParquetWriteProgress(
                encoded_bytes=encoded_bytes,
                written_rows=self._written_rows,
                buffered_rows=self._buffered_rows,
                row_groups=self._row_groups,
                final=final,
            )
        )
