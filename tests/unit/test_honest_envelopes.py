"""Check the context-conditioned honest action envelopes."""

import pytest

from avalanche.controllers.envelopes import (
    ENVELOPE_VERSION,
    EnvelopeSample,
    HonestEnvelope,
)


def sample(
    density: float,
    value: float,
    *,
    channel: str = "route_weights",
    target: str = "piste",
    variant: str = "standard-linear",
) -> EnvelopeSample:
    return EnvelopeSample(
        action_channel=channel,
        target_type=target,
        density=density,
        demand=40.0,
        weather_risk=0.2,
        event_state="normal",
        value=value,
        policy_variant=variant,
    )


def test_an_envelope_uses_only_the_declared_training_variants():
    envelope = HonestEnvelope.build(
        (
            sample(0.1, -0.2),
            sample(0.1, -0.4, variant="conservative-gradual"),
        ),
        ("standard-linear",),
    )
    assert envelope.range_for(sample(0.1, 0.0)) == (-0.2, -0.2)
    assert envelope.as_dict()["envelope_version"] == ENVELOPE_VERSION
    assert envelope.as_dict()["training_variants"] == ("standard-linear",)


def test_an_envelope_combines_each_populated_context_bin():
    envelope = HonestEnvelope.build(
        (sample(0.1, -0.4), sample(0.2, 0.3)),
        ("standard-linear",),
    )
    assert envelope.range_for(sample(0.15, 0.0)) == (-0.4, 0.3)


def test_an_envelope_uses_the_nearest_populated_bin():
    envelope = HonestEnvelope.build(
        (sample(0.1, -0.1), sample(1.1, -0.9)),
        ("standard-linear",),
    )
    assert envelope.range_for(sample(0.4, 0.0)) == (-0.1, -0.1)


def test_an_envelope_resolves_equal_distance_with_lexical_ordering():
    envelope = HonestEnvelope.build(
        (sample(0.1, -0.1), sample(0.6, -0.6)),
        ("standard-linear",),
    )
    assert envelope.range_for(sample(0.3, 0.0)) == (-0.1, -0.1)


def test_an_envelope_rejects_a_missing_channel_and_target_type():
    envelope = HonestEnvelope.build(
        (sample(0.1, -0.1),),
        ("standard-linear",),
    )
    with pytest.raises(ValueError, match="action channel and target type"):
        envelope.range_for(sample(0.1, 0.0, channel="lift_capacity"))
