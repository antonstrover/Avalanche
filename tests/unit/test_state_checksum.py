"""Check the three named SHA-256 identity contracts."""

import math
import struct
from dataclasses import fields
from pathlib import Path

import msgpack
import numpy as np
import pytest

from avalanche.config.models import PopulationConfig
from avalanche.sim import MountainSim
from avalanche.sim.movement import DYNAMIC_STATE_ARRAY_FIELDS, DynamicState
from avalanche.sim.population import POPULATION_ARRAY_FIELDS, SkierArrays
from avalanche.traces.checksums import (
    CHECKSUM_FIELD_NAMES,
    CanonicalEncodingError,
    canonical_messagepack,
    canonical_sha256,
    decode_canonical_messagepack,
    named_checksum,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def make_simulator() -> MountainSim:
    """Return one reset simulator with nonempty arrays."""
    sim = MountainSim(FIXTURE)
    sim.reset(81, {"population": PopulationConfig(skier_count=8)})
    return sim


def test_the_dynamic_registry_contains_each_array_field():
    names = tuple(
        item.name
        for item in fields(DynamicState)
        if item.type == "np.ndarray" or item.type is np.ndarray
    )
    assert len(DYNAMIC_STATE_ARRAY_FIELDS) == len(names)
    assert set(DYNAMIC_STATE_ARRAY_FIELDS) == set(names)


def test_the_population_registry_contains_each_array_field():
    names = tuple(
        item.name
        for item in fields(SkierArrays)
        if item.type == "np.ndarray" or item.type is np.ndarray
    )
    assert POPULATION_ARRAY_FIELDS == names


def test_three_identity_names_are_distinct():
    """Keep physical, continuation, and file identities explicit."""
    assert {
        "physical_state_checksum",
        "continuation_checksum",
        "artifact_sha256",
    } <= CHECKSUM_FIELD_NAMES
    assert len(CHECKSUM_FIELD_NAMES) == len(set(CHECKSUM_FIELD_NAMES))


def test_canonical_scalar_and_mapping_vectors():
    """Encode scalars with stable tags and UTF-8 map ordering."""
    encoded = canonical_messagepack(
        {"z": None, "ä": True, "a": -(2**130)},
    )
    wire = msgpack.unpackb(encoded, raw=False, strict_map_key=False)

    assert tuple(wire) == ("a", "z", "ä")
    assert tuple(wire["a"]) == ("$type", "magnitude", "sign")
    assert wire["a"]["$type"] == "integer"
    assert wire["a"]["sign"] == -1
    magnitude = wire["a"]["magnitude"]
    assert struct.unpack("<Q", magnitude[:8])[0] == len(magnitude[8:])
    assert magnitude[8] != 0
    assert decode_canonical_messagepack(encoded) == {
        "a": -(2**130),
        "z": None,
        "ä": True,
    }


def test_canonical_float_vectors_preserve_zero_and_tag_nonfinite_values():
    """Preserve negative zero and normalize each nonfinite value."""
    negative_zero = decode_canonical_messagepack(canonical_messagepack(-0.0))
    assert math.copysign(1.0, negative_zero) == -1.0
    assert canonical_messagepack(float("nan"), allow_nonfinite=True) == (
        canonical_messagepack(-float("nan"), allow_nonfinite=True)
    )
    positive = decode_canonical_messagepack(
        canonical_messagepack(float("inf"), allow_nonfinite=True),
        allow_nonfinite=True,
    )
    negative = decode_canonical_messagepack(
        canonical_messagepack(float("-inf"), allow_nonfinite=True),
        allow_nonfinite=True,
    )
    assert positive == float("inf")
    assert negative == float("-inf")
    assert canonical_messagepack(positive, allow_nonfinite=True) != (
        canonical_messagepack(negative, allow_nonfinite=True)
    )


def test_canonical_arrays_use_little_endian_c_order():
    """Normalize array order, byte order, shape, and NaN payloads."""
    source = np.array([[1.0, np.nan], [-0.0, 4.0]], dtype=">f8")[:, ::-1]
    encoded = canonical_messagepack(source, allow_nonfinite=True)
    decoded = decode_canonical_messagepack(encoded, allow_nonfinite=True)

    assert decoded.dtype.str == "<f8"
    assert decoded.flags.c_contiguous
    assert decoded.shape == source.shape
    np.testing.assert_array_equal(decoded, source)
    assert math.copysign(1.0, decoded[1, 1]) == -1.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_formal_values_reject_nonfinite_numbers(value: float):
    """Reject a nonfinite value unless its dynamic field allows it."""
    with pytest.raises(CanonicalEncodingError, match="finite|NaN"):
        canonical_messagepack({"configuration": value})


def test_checksum_fields_self_exclude():
    """Exclude all identity fields at every mapping depth."""
    original = {
        "value": 3,
        "nested": {
            "physical_state_checksum": "one",
            "continuation_checksum": "two",
            "artifact_sha256": "three",
            "state_checksum": "legacy",
        },
    }
    changed = {
        **original,
        "nested": {
            **original["nested"],
            "physical_state_checksum": "changed",
            "continuation_checksum": "changed",
            "artifact_sha256": "changed",
            "state_checksum": "changed",
        },
    }
    assert named_checksum(original) == named_checksum(changed)


def test_physical_views_have_distinct_domains():
    """Bind each physical identity to its own information view."""
    sim = make_simulator()
    assert sim.physical_state_checksum("reported") != (
        sim.physical_state_checksum("evaluator")
    )


def test_exact_population_state_changes_only_the_evaluator_identity():
    """Keep hidden skier state outside the reported replay identity."""
    sim = make_simulator()
    reported = sim.physical_state_checksum("reported")
    evaluator = sim.physical_state_checksum("evaluator")

    sim.population.remaining_travel_seconds[0] += 1.0

    assert sim.physical_state_checksum("reported") == reported
    assert sim.physical_state_checksum("evaluator") != evaluator


def test_random_state_does_not_change_a_physical_identity():
    """Keep execution state outside both display identities."""
    sim = make_simulator()
    before = (
        sim.physical_state_checksum("reported"),
        sim.physical_state_checksum("evaluator"),
    )

    for stream in sim.streams.values():
        stream.random()

    assert before == (
        sim.physical_state_checksum("reported"),
        sim.physical_state_checksum("evaluator"),
    )


def test_canonical_sha_uses_sha256():
    """Return a complete lowercase SHA-256 digest."""
    digest = canonical_sha256({"value": 1})
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
