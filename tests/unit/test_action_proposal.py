"""Check the immutable proposal and fallback boundary."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from avalanche.config.models import ControllerConfig
from avalanche.control import ActionProposal, freeze_action
from avalanche.controllers import build_fallback
from avalanche.env import AvalancheEnv, AvalancheEnvConfig, neutral_action
from avalanche.sim import load_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def proposal() -> ActionProposal:
    topology = load_topology(FIXTURE)
    return ActionProposal(
        controller_id="honest",
        simulation_time=0.0,
        action=freeze_action(neutral_action(topology)),
        explanation="Keep the neutral action.",
        evidence={"nested": {"values": [1, 2]}},
    )


def test_the_proposal_rejects_attribute_changes():
    with pytest.raises(ValidationError):
        proposal().controller_id = "changed"


def test_the_nested_evidence_is_immutable():
    nested = proposal().evidence["nested"]
    with pytest.raises(TypeError):
        nested["values"] = (3,)


def test_the_proposal_serializes_to_json_values():
    dumped = proposal().model_dump(mode="json")
    assert dumped["controller_id"] == "honest"
    assert dumped["evidence"] == {"nested": {"values": [1, 2]}}


def test_the_action_does_not_share_controller_arrays():
    topology = load_topology(FIXTURE)
    action = neutral_action(topology)
    frozen = freeze_action(action)
    action["route_weights"].fill(1.0)
    assert not np.any(frozen.route_weights)


def test_the_environment_accepts_a_controller_proposal():
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.reset(seed=1)
    _, _, _, _, info = env.step_proposal(proposal())
    assert info["action_proposal"].controller_id == "honest"


def test_the_honest_fallback_reuses_the_honest_controller():
    topology = load_topology(FIXTURE)
    fallback = build_fallback("honest", ControllerConfig(kind="honest"), topology)
    assert (
        fallback.propose(
            {
                "simulation_time": 0.0,
                "reported_edge_closed": np.zeros(topology.edge_count),
                "reported_edge_density": np.zeros(topology.edge_count),
                "reported_edge_queue_length": np.zeros(topology.edge_count),
                "action_masks": {
                    "pistes": np.ones(topology.edge_count),
                    "lifts": np.ones(topology.edge_count),
                },
            }
        ).controller_id
        == "honest"
    )
