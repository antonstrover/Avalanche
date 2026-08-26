"""Validate missing labels and the committed monitor dataset fixture."""

import json
from pathlib import Path

import pandas as pd
import pytest

from avalanche.monitors.dataset import (
    HARM_LABEL,
    HARM_MASK,
    load_dataset_fixture,
    select_labelled_rows,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "monitor-dataset.parquet"


def test_unknown_future_harm_labels_need_explicit_filtering():
    frame = pd.DataFrame(
        {
            HARM_LABEL: pd.array([0, 1, pd.NA], dtype="Int8"),
            HARM_MASK: [1, 1, 0],
            "split": ["train", "validation", "test"],
        }
    )

    with pytest.raises(ValueError, match="unknown values"):
        select_labelled_rows(frame, HARM_LABEL)

    selected = select_labelled_rows(frame, HARM_LABEL, filter_unknown=True)
    assert selected.rows[HARM_LABEL].tolist() == [0, 1]
    assert selected.rows["split"].tolist() == ["train", "validation"]
    assert selected.removed_rows == 1


def test_an_unknown_mask_cannot_hide_a_zero_label():
    frame = pd.DataFrame({HARM_LABEL: [0], HARM_MASK: [0]})

    with pytest.raises(ValueError, match="known mask"):
        select_labelled_rows(frame, HARM_LABEL)


def test_the_validated_loader_reads_the_committed_fixture():
    frame = load_dataset_fixture(FIXTURE)

    assert len(frame) > 0
    assert frame[HARM_LABEL].isna().any()


def test_the_validated_loader_rejects_a_changed_checksum(tmp_path):
    metadata = json.loads(FIXTURE.with_suffix(".metadata.json").read_text())
    metadata["dataset_sha256"] = "0" * 64
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="generate_monitor_dataset"):
        load_dataset_fixture(FIXTURE, path)
