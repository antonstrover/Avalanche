"""Validate missing labels and the committed monitor dataset fixture."""

import json
from pathlib import Path

import pandas as pd
import pytest

from avalanche.monitors.dataset import (
    STRANDING_LABEL,
    STRANDING_MASK,
    load_dataset_fixture,
    load_nonformal_legacy_dataset_v4_fixture,
    select_labelled_rows,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "monitor-dataset.parquet"


def test_unknown_future_stranding_labels_need_explicit_filtering():
    frame = pd.DataFrame(
        {
            STRANDING_LABEL: pd.array([0, 1, pd.NA], dtype="Int8"),
            STRANDING_MASK: [1, 1, 0],
            "split": ["train", "validation", "test"],
        }
    )

    with pytest.raises(ValueError, match="unknown values"):
        select_labelled_rows(frame, STRANDING_LABEL)

    selected = select_labelled_rows(frame, STRANDING_LABEL, filter_unknown=True)
    assert selected.rows[STRANDING_LABEL].tolist() == [0, 1]
    assert selected.rows["split"].tolist() == ["train", "validation"]
    assert selected.removed_rows == 1


def test_an_unknown_mask_cannot_hide_a_zero_label():
    frame = pd.DataFrame({STRANDING_LABEL: [0], STRANDING_MASK: [0]})

    with pytest.raises(ValueError, match="known mask"):
        select_labelled_rows(frame, STRANDING_LABEL)


def test_the_nonformal_legacy_loader_reads_the_committed_fixture():
    fixture = load_nonformal_legacy_dataset_v4_fixture(FIXTURE)
    metadata = json.loads(FIXTURE.with_suffix(".metadata.json").read_text())

    assert len(fixture.rows) > 0
    assert fixture.rows["harm_in_horizon"].isna().any()
    assert fixture.feature_names == tuple(metadata["feature_names"])
    assert fixture.feature_version == metadata["feature_version"]


def test_the_nonformal_legacy_loader_rejects_a_changed_checksum(tmp_path):
    metadata = json.loads(FIXTURE.with_suffix(".metadata.json").read_text())
    metadata["dataset_sha256"] = "0" * 64
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="historical monitor fixture"):
        load_nonformal_legacy_dataset_v4_fixture(FIXTURE, path)


def test_the_formal_loader_rejects_the_version_four_fixture():
    with pytest.raises(ValueError, match="current monitor fixture"):
        load_dataset_fixture(FIXTURE)
