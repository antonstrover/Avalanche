"""Check the ordered monitor dataset writer."""

import pandas as pd
import pyarrow.parquet as pq
import pytest

from avalanche.traces import BufferedParquetWriter, ParquetWriteProgress


def test_the_writer_flushes_complete_groups_and_keeps_order(tmp_path):
    output = tmp_path / "rows.parquet"
    progress: list[ParquetWriteProgress] = []
    first = pd.DataFrame({"value": range(75), "label": ["a"] * 75})
    second = pd.DataFrame({"value": range(75, 210), "label": ["b"] * 135})

    with BufferedParquetWriter(
        output,
        row_group_rows=100,
        on_progress=progress.append,
    ) as writer:
        writer.write(first)
        assert progress[-1].written_rows == 0
        assert progress[-1].buffered_rows == 75
        writer.write(second)

    parquet = pq.ParquetFile(output)
    result = pd.read_parquet(output)
    assert parquet.metadata.num_row_groups == 3
    assert result["value"].tolist() == list(range(210))
    assert progress[-1] == ParquetWriteProgress(
        encoded_bytes=output.stat().st_size,
        written_rows=210,
        buffered_rows=0,
        row_groups=3,
        final=True,
    )


def test_the_writer_removes_a_partial_file_after_failure(tmp_path):
    output = tmp_path / "rows.parquet"

    try:
        with BufferedParquetWriter(output, row_group_rows=2) as writer:
            writer.write(pd.DataFrame({"value": [1, 2]}))
            raise RuntimeError("worker failed")
    except RuntimeError:
        pass

    assert not output.exists()
    assert not output.with_suffix(".parquet.partial").exists()


def test_abort_cleanup_does_not_replace_the_active_error(tmp_path):
    class BrokenClose:
        def close(self):
            raise OSError("close stopped")

    writer = BufferedParquetWriter(tmp_path / "rows.parquet")
    writer._writer = BrokenClose()
    writer._sink = BrokenClose()

    with pytest.raises(RuntimeError, match="worker stopped"):
        try:
            raise RuntimeError("worker stopped")
        except RuntimeError:
            writer.abort()
            raise
