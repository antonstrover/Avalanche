"""Check evaluator-truth evacuation capacity."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import (
    EnvironmentContextConfig,
    EvacuationTargetEdgeConfig,
    MountainEnvironmentContextConfig,
)
from avalanche.sim import MountainSim
from avalanche.sim.evacuation import (
    ResolvedEnvironmentContext,
    current_safe_evacuation_capacity,
    resolve_environment_context,
)
from avalanche.sim.movement import new_dynamic_state
from avalanche.sim.topology import load_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def edge_index(topology, source: str, destination: str) -> int:
    """Return one edge index from its endpoint names."""
    matches = np.flatnonzero(
        (topology.edge_source == topology.node_index[source])
        & (topology.edge_destination == topology.node_index[destination])
    )
    assert matches.size == 1
    return int(matches[0])


def target_config() -> MountainEnvironmentContextConfig:
    """Return one piste target and one lift target."""
    all_abilities = ("beginner", "intermediate", "advanced")
    return MountainEnvironmentContextConfig(
        mountain="small-resort",
        evacuation_target_edges=(
            EvacuationTargetEdgeConfig(
                edge="valley_junction->base_exit",
                abilities=all_abilities,
            ),
            EvacuationTargetEdgeConfig(
                edge="lift1_base->lift1_top",
                abilities=all_abilities,
            ),
        ),
    )


def test_capacity_uses_true_piste_time_and_lift_throughput():
    topology = load_topology(FIXTURE)
    state = new_dynamic_state(topology)
    piste = edge_index(topology, "valley_junction", "base_exit")
    lift = edge_index(topology, "lift1_base", "lift1_top")
    state.speed_factor[piste] = 0.5
    state.lift_capacity_factor[lift] = 0.25
    context = resolve_environment_context(topology, state, target_config())

    piste_capacity = float(topology.edge_safe_capacity[piste]) / (
        float(topology.edge_nominal_travel_time[piste]) / 0.5
    )
    lift_capacity = float(topology.edge_lift_throughput[lift]) / 3600.0 * 0.25
    expected = float(np.sum((piste_capacity, lift_capacity), dtype=np.float64))

    assert current_safe_evacuation_capacity(topology, state, context) == expected
    assert context.baseline_safe_evacuation_capacity_skiers_per_second == expected


@pytest.mark.parametrize(
    "closure_field",
    ("closed", "weather_closed", "failure_closed", "lift_stopped"),
)
def test_each_physical_closure_sets_a_target_to_zero(closure_field: str):
    topology = load_topology(FIXTURE)
    state = new_dynamic_state(topology)
    lift = edge_index(topology, "lift1_base", "lift1_top")
    context = ResolvedEnvironmentContext((lift,), ((0, 1, 2),), 1.0)
    getattr(state, closure_field)[lift] = True

    assert current_safe_evacuation_capacity(topology, state, context) == 0.0


def test_an_ability_unsafe_target_has_zero_capacity():
    topology = load_topology(FIXTURE)
    state = new_dynamic_state(topology)
    black_piste = edge_index(topology, "lift2_top", "ridge_junction")
    context = ResolvedEnvironmentContext((black_piste,), ((0,),), 1.0)

    assert current_safe_evacuation_capacity(topology, state, context) == 0.0


def test_the_initial_baseline_never_changes():
    topology = load_topology(FIXTURE)
    state = new_dynamic_state(topology)
    context = resolve_environment_context(topology, state, target_config())
    baseline = context.baseline_safe_evacuation_capacity_skiers_per_second
    state.speed_factor.fill(0.2)
    state.lift_capacity_factor.fill(0.1)

    assert current_safe_evacuation_capacity(topology, state, context) < baseline
    assert context.baseline_safe_evacuation_capacity_skiers_per_second == baseline


def test_the_baseline_uses_an_initial_physical_failure():
    target = MountainEnvironmentContextConfig(
        mountain="small-resort",
        evacuation_target_edges=(
            EvacuationTargetEdgeConfig(
                edge="lift1_base->lift1_top",
                abilities=("beginner", "intermediate", "advanced"),
            ),
        ),
    )
    sim = MountainSim(FIXTURE)
    sim.reset(
        23,
        {
            "environment_context": EnvironmentContextConfig(
                evacuation_targets=(target,)
            ),
            "failures": {
                "schedule": [
                    {
                        "kind": "lift_stoppage",
                        "target": "lift1_base->lift1_top",
                        "start_time_seconds": 0.0,
                        "duration_seconds": 30.0,
                        "controller_visible": True,
                    }
                ]
            },
        },
    )

    assert (
        sim.environment_context.baseline_safe_evacuation_capacity_skiers_per_second
        == 0.0
    )
