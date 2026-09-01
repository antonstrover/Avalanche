"""Check mountain-specific evacuation target configuration."""

import pytest

from avalanche.config import ConfigurationResolutionError, ConfigurationResolver
from tests.configuration import resolve_test_configuration


def test_each_mountain_selects_its_own_environment_context():
    resolver = ConfigurationResolver()
    medium = resolver.resolve(
        "configs/mountain/default.yaml",
        "configs/scenarios/default.yaml",
        "configs/controllers/honest.yaml",
        "configs/monitors/none.yaml",
    )
    small = resolver.resolve(
        "configs/mountain/small.yaml",
        "configs/scenarios/default.yaml",
        "configs/controllers/small-resort/honest.yaml",
        "configs/monitors/none.yaml",
    )

    assert (
        medium.scenario.environment_context.for_mountain("val-tarin").mountain
        == "val-tarin"
    )
    assert (
        small.scenario.environment_context.for_mountain("small-resort").mountain
        == "small-resort"
    )


def test_an_unknown_active_target_is_rejected(tmp_path):
    context = {
        "evacuation_targets": [
            {
                "mountain": "small-resort",
                "evacuation_target_edges": [
                    {
                        "edge": "missing->edge",
                        "abilities": ["beginner"],
                    }
                ],
            }
        ]
    }

    with pytest.raises(ConfigurationResolutionError, match="unknown edge"):
        resolve_test_configuration(
            tmp_path,
            mountain="configs/mountain/small.yaml",
            scenario="configs/scenarios/default.yaml",
            controller="configs/controllers/small-resort/honest.yaml",
            monitor="configs/monitors/none.yaml",
            changes={"scenario": {"scenario": {"environment_context": context}}},
        )


def test_an_ability_unsafe_active_target_is_rejected(tmp_path):
    context = {
        "evacuation_targets": [
            {
                "mountain": "small-resort",
                "evacuation_target_edges": [
                    {
                        "edge": "lift2_top->ridge_junction",
                        "abilities": ["beginner"],
                    }
                ],
            }
        ]
    }

    with pytest.raises(ConfigurationResolutionError, match="unsafe for the beginner"):
        resolve_test_configuration(
            tmp_path,
            mountain="configs/mountain/small.yaml",
            scenario="configs/scenarios/default.yaml",
            controller="configs/controllers/small-resort/honest.yaml",
            monitor="configs/monitors/none.yaml",
            changes={"scenario": {"scenario": {"environment_context": context}}},
        )


def test_a_missing_mountain_context_is_rejected(tmp_path):
    context = {
        "evacuation_targets": [
            {
                "mountain": "val-tarin",
                "evacuation_target_edges": [
                    {
                        "edge": "combe_lower->crete_east",
                        "abilities": ["intermediate", "advanced"],
                    }
                ],
            }
        ]
    }

    with pytest.raises(ConfigurationResolutionError, match="no environment context"):
        resolve_test_configuration(
            tmp_path,
            mountain="configs/mountain/small.yaml",
            scenario="configs/scenarios/default.yaml",
            controller="configs/controllers/small-resort/honest.yaml",
            monitor="configs/monitors/none.yaml",
            changes={"scenario": {"scenario": {"environment_context": context}}},
        )
