from pathlib import Path

import pytest
from pydantic import ValidationError

from avalanche.control import DecisionType, MonitorDecision, freeze_action
from avalanche.env import neutral_action
from avalanche.sim import load_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


@pytest.mark.parametrize("decision", list(DecisionType))
def test_each_decision_type_round_trips(decision: DecisionType):
    replacement = None
    if decision is DecisionType.REPLACE:
        replacement = freeze_action(neutral_action(load_topology(FIXTURE)))
    value = MonitorDecision(
        risk_score=0.5,
        decision=decision,
        reason_codes=("TEST_REASON",),
        replacement_action=replacement,
        latency_seconds=0.01,
        predicted_result=(("risk", 0.5), ("limit", 0.75)),
    )

    restored = MonitorDecision.model_validate_json(value.model_dump_json())

    assert restored == value
    assert restored.decision is decision


@pytest.mark.parametrize("risk", [-0.1, 1.1])
def test_the_decision_rejects_an_invalid_risk(risk: float):
    with pytest.raises(ValidationError):
        MonitorDecision(risk_score=risk, decision=DecisionType.ALLOW)


def test_the_decision_rejects_a_negative_latency():
    with pytest.raises(ValidationError):
        MonitorDecision(
            risk_score=0.0,
            decision=DecisionType.ALLOW,
            latency_seconds=-0.1,
        )


@pytest.mark.parametrize(
    "prediction",
    [
        (("", 0.0),),
        (("risk", 0.0), ("risk", 1.0)),
        (("risk", float("nan")),),
        (("risk", float("inf")),),
    ],
)
def test_the_decision_rejects_an_invalid_prediction(prediction):
    with pytest.raises(ValidationError):
        MonitorDecision(
            risk_score=0.0,
            decision=DecisionType.ALLOW,
            predicted_result=prediction,
        )


def test_replace_requires_an_action():
    with pytest.raises(ValidationError, match="must contain"):
        MonitorDecision(risk_score=1.0, decision=DecisionType.REPLACE)


def test_allow_rejects_a_replacement():
    replacement = freeze_action(neutral_action(load_topology(FIXTURE)))
    with pytest.raises(ValidationError, match="only a replace"):
        MonitorDecision(
            risk_score=0.0,
            decision=DecisionType.ALLOW,
            replacement_action=replacement,
        )
