"""The learned monitor features must keep one fixed shape.

The plan gives the feature blocks in section 9.4.
Every feature is an aggregate, so a larger mountain must not make a longer vector.
"""

from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import ControllerConfig, PopulationConfig
from avalanche.control import (
    InformationProfile,
    build_monitor_observation,
    build_monitor_proposal,
    sanitize_trace_window,
)
from avalanche.controllers.factory import build_controller, build_fallback
from avalanche.env import AvalancheEnv, AvalancheEnvConfig
from avalanche.monitors.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_VERSION,
    HISTORY_LENGTH,
    FeatureExtractor,
    feature_names_for,
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


def make_context(
    mountain: str,
    history_entries: int = 0,
    profile: InformationProfile = InformationProfile.PRINCIPAL,
    feature_blocks: tuple[str, ...] | None = None,
):
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
    complete = controller.propose(env.controller_observation())
    proposal = build_monitor_proposal(complete)
    observation = build_monitor_observation(
        env.controller_observation(), env.sim, profile
    )
    extractor = FeatureExtractor(
        build_fallback("honest", controller_config, env.topology),
        RuleMonitor(env.topology),
        profile,
        feature_blocks,
    )
    extractor.reset(SEED)
    history = tuple(
        {
            "proposal": complete.model_dump(mode="json"),
            "decision": {"risk_score": 0.5, "decision": "ALLOW"},
        }
        for _ in range(history_entries)
    )
    return observation, proposal, extractor, sanitize_trace_window(history)


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


def test_the_principal_profile_excludes_prohibited_features():
    prohibited = ("identical", "remaining_time", "true_", "harm", "simulation_time")

    assert FEATURE_VERSION == 2
    assert all(part not in name for name in FEATURE_NAMES for part in prohibited)


def test_the_fallback_oracle_uses_distances_without_exact_equality():
    observation, proposal, extractor, history = make_context(
        "small-resort.yaml", profile=InformationProfile.ORACLE_FALLBACK
    )
    values = extractor.vector(observation, proposal, history)
    names = feature_names_for(InformationProfile.ORACLE_FALLBACK)
    distance_names = [name for name in names if name.startswith("oracle_fallback")]

    assert distance_names
    assert all(values[names.index(name)] == 0.0 for name in distance_names)
    assert all("identical" not in name for name in names)


@pytest.mark.parametrize("profile", list(InformationProfile))
def test_each_information_profile_has_one_fixed_schema(profile):
    observation, proposal, extractor, history = make_context(
        "small-resort.yaml", profile=profile
    )
    values = extractor.vector(observation, proposal, history)

    assert values.shape == (len(feature_names_for(profile)),)
    assert extractor.feature_names == feature_names_for(profile)


def test_an_ablation_zeros_each_excluded_block_during_extraction():
    observation, proposal, complete, history = make_context("small-resort.yaml")
    _, _, ablated, _ = make_context("small-resort.yaml", feature_blocks=("action",))
    complete_values = complete.vector(observation, proposal, history)
    ablated_values = ablated.vector(observation, proposal, history)
    action = np.asarray([name.startswith("action_") for name in FEATURE_NAMES])
    assert np.array_equal(ablated_values[action], complete_values[action])
    assert np.count_nonzero(ablated_values[~action]) == 0


def test_the_principal_profile_contains_each_context_measure():
    dimensions = ("capacity", "evacuation", "utility", "fairness", "consistency")
    assert all(
        any(name.startswith(f"context_{dimension}") for name in FEATURE_NAMES)
        for dimension in dimensions
    )
