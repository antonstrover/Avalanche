"""The learned monitor features must keep one fixed shape.

The plan gives the feature blocks in section 9.4.
Every feature is an aggregate, so a larger mountain must not make a longer vector.
"""

from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import ControllerConfig, PopulationConfig
from avalanche.control import build_monitor_observation
from avalanche.controllers.factory import build_controller, build_fallback
from avalanche.env import AvalancheEnv, AvalancheEnvConfig
from avalanche.monitors.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    HISTORY_LENGTH,
    FeatureExtractor,
)
from avalanche.monitors.rules import RuleMonitor

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "mountain"
MOUNTAINS = ("small-resort.yaml", "medium-resort.yaml")
SEED = 20260825

POPULATION = PopulationConfig(
    skier_count=200,
    arrival_window_seconds=600.0,
    ability_weights=(0.3, 0.5, 0.2),
    compliance_mean=0.7,
    compliance_spread=0.2,
)


def make_context(mountain: str, history_entries: int = 0):
    """Return one real observation, one proposal, and one extractor."""
    env = AvalancheEnv(
        CONFIGS / mountain,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=60.0,
            episode_duration_seconds=600.0,
        ),
        simulator_options={"population": POPULATION},
    )
    env.reset(seed=SEED)
    controller_config = ControllerConfig(kind="honest")
    controller = build_controller(controller_config, env.topology)
    controller.reset(SEED)
    proposal = controller.propose(env.controller_observation())
    observation = build_monitor_observation(env.controller_observation(), env.sim)
    extractor = FeatureExtractor(
        build_fallback("honest", controller_config, env.topology),
        RuleMonitor(env.topology),
    )
    extractor.reset(SEED)
    history = tuple(
        {
            "proposal": proposal.model_dump(mode="json"),
            "decision": {"risk_score": 0.5, "decision": "ALLOW"},
        }
        for _ in range(history_entries)
    )
    return observation, proposal, extractor, history


@pytest.mark.parametrize("mountain", MOUNTAINS)
def test_the_feature_vector_shape_stays_fixed_across_mountains(mountain):
    observation, proposal, extractor, history = make_context(mountain)
    values = extractor.vector(observation, proposal, history)

    assert values.shape == (FEATURE_COUNT,)
    assert values.dtype == np.float32
    assert np.all(np.isfinite(values))


def test_the_names_match_the_values():
    assert len(FEATURE_NAMES) == FEATURE_COUNT
    assert len(set(FEATURE_NAMES)) == FEATURE_COUNT


def test_two_mountains_of_a_different_size_give_the_same_shape():
    small_topology_edges, medium_topology_edges = (
        len(make_context(mountain)[0]["reported_edge_density"])
        for mountain in MOUNTAINS
    )
    assert small_topology_edges != medium_topology_edges

    vectors = [
        extractor.vector(observation, proposal, history)
        for observation, proposal, extractor, history in (
            make_context(mountain) for mountain in MOUNTAINS
        )
    ]
    assert vectors[0].shape == vectors[1].shape


@pytest.mark.parametrize("entries", [0, 1, HISTORY_LENGTH, HISTORY_LENGTH + 4])
def test_a_short_history_pads_and_a_long_history_truncates(entries):
    observation, proposal, extractor, history = make_context(
        "small-resort.yaml", history_entries=entries
    )
    values = extractor.vector(observation, proposal, history)

    present = values[-HISTORY_LENGTH:]
    assert values.shape == (FEATURE_COUNT,)
    assert np.count_nonzero(present) == min(entries, HISTORY_LENGTH)


def test_an_honest_proposal_matches_the_reference_fallback():
    observation, proposal, extractor, history = make_context("small-resort.yaml")
    values = extractor.vector(observation, proposal, history)
    identical = values[FEATURE_NAMES.index("difference_identical_to_fallback")]

    assert identical == 1.0
