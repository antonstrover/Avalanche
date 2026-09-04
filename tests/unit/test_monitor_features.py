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
    thaw_action,
)
from avalanche.controllers.factory import build_controller, build_fallback
from avalanche.env import AvalancheEnv, AvalancheEnvConfig
from avalanche.monitors.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_REGISTRIES,
    FEATURE_VERSION,
    HISTORY_LENGTH,
    MASTER_FEATURE_REGISTRY,
    PROFILE_CATEGORIES,
    FeatureExtractor,
    FeatureProfile,
    feature_names_for,
    feature_registry_for,
)
from avalanche.monitors.rules import RuleMonitor

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "mountain"
MOUNTAINS = ("small-resort.yaml", "medium-resort.yaml")
SEED = 20260825
FEATURE_PROFILES = (
    InformationProfile.PRINCIPAL,
    InformationProfile.ORACLE_FALLBACK,
    InformationProfile.ORACLE_TRUE_STATE,
)

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
    feature_profile: FeatureProfile = FeatureProfile.PRINCIPAL_FULL,
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
    observation = build_monitor_observation(env.sim, complete, profile)
    reference_fallback = (
        build_fallback("honest", controller_config, env.topology)
        if profile is InformationProfile.ORACLE_FALLBACK
        else None
    )
    extractor = FeatureExtractor(
        reference_fallback,
        RuleMonitor(env.topology),
        profile,
        feature_blocks,
        feature_profile,
    )
    extractor.reset(SEED)
    history = tuple(
        {"executed_action": thaw_action(complete.action)}
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
        len(make_context(mountain)[0].operational_evidence.value("edge_density"))
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

    assert FEATURE_VERSION == 3
    assert all(part not in name for name in FEATURE_NAMES for part in prohibited)


def test_the_fallback_oracle_uses_distances_without_exact_equality():
    observation, proposal, extractor, history = make_context(
        "small-resort.yaml", profile=InformationProfile.ORACLE_FALLBACK
    )
    values = extractor.vector(observation, proposal, history)
    names = feature_names_for(InformationProfile.ORACLE_FALLBACK)
    distance_names = [name for name in names if name.startswith("fallback_oracle")]

    assert distance_names
    assert all(values[names.index(name)] == 0.0 for name in distance_names)
    assert all("identical" not in name for name in names)


@pytest.mark.parametrize("profile", FEATURE_PROFILES)
def test_each_information_profile_has_one_fixed_schema(profile):
    observation, proposal, extractor, history = make_context(
        "small-resort.yaml", profile=profile
    )
    values = extractor.vector(observation, proposal, history)

    assert values.shape == (len(feature_names_for(profile)),)
    assert extractor.feature_names == feature_names_for(profile)


def test_the_true_state_oracle_separates_harm_from_density():
    names = feature_names_for(InformationProfile.ORACLE_TRUE_STATE)

    assert "true_state_oracle_unique_stranded_skiers" in names
    assert "true_state_oracle_dangerous_density_active_fraction" in names
    assert all("harm_count" not in name for name in names)


def test_each_projection_returns_only_its_declared_columns():
    observation, proposal, complete, history = make_context("small-resort.yaml")
    _, _, proposal_only, _ = make_context(
        "small-resort.yaml",
        feature_profile=FeatureProfile.PROPOSAL_ONLY,
    )
    complete_values = complete.vector(observation, proposal, history)
    proposal_values = proposal_only.vector(observation, proposal, history)
    indexes = [FEATURE_NAMES.index(name) for name in proposal_only.feature_names]
    assert np.array_equal(proposal_values, complete_values[indexes])
    assert proposal_values.shape == (len(proposal_only.feature_names),)


def test_every_feature_has_one_complete_provenance_contract():
    for feature in MASTER_FEATURE_REGISTRY.features:
        assert feature.category
        assert feature.source_fields
        assert feature.transformation
        assert feature.units
        assert feature.provenance_constraints
        assert feature.timestamp_rule
        assert feature.missingness_rule
        assert feature.allowed_profiles
        assert feature.source_categories
        assert feature.interaction == (len(feature.source_categories) > 1)


def test_each_profile_uses_only_its_allowed_source_categories():
    assert set(FEATURE_REGISTRIES) == set(FeatureProfile)
    for profile, registry in FEATURE_REGISTRIES.items():
        allowed = {category.value for category in PROFILE_CATEGORIES[profile]}
        assert all(
            set(feature.source_categories) <= allowed for feature in registry.features
        )
        assert registry.master_feature_registry_sha256 == MASTER_FEATURE_REGISTRY.sha256
        assert registry.sha256 == feature_registry_for(profile).sha256


def test_no_history_removes_each_executed_history_interaction():
    registry = feature_registry_for(FeatureProfile.NO_HISTORY)
    assert all(
        "executed-history" not in feature.source_categories
        for feature in registry.features
    )


def test_the_principal_profile_contains_each_context_measure():
    dimensions = ("capacity", "evacuation", "utility", "fairness", "consistency")
    assert all(
        any(name.startswith(f"context_{dimension}") for name in FEATURE_NAMES)
        for dimension in dimensions
    )
