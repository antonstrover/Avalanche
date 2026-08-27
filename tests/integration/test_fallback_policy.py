from pathlib import Path

import pytest

from avalanche.config import ConfigurationResolver
from avalanche.config.models import ControllerConfig
from avalanche.control import (
    ActionProposal,
    Adjudicator,
    DecisionType,
    InfrastructureReference,
    MonitorDecision,
    build_monitor_observation,
    freeze_action,
    thaw_action,
)
from avalanche.controllers import build_fallback
from avalanche.env import (
    AvalancheEnv,
    AvalancheEnvConfig,
    neutral_action,
    validate_action,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
CONTROLLERS = (
    "configs/controllers/none.yaml",
    "configs/controllers/small-resort/honest.yaml",
    "configs/controllers/small-resort/profit-biased.yaml",
    "configs/controllers/small-resort/sleeper-saboteur.yaml",
    "configs/controllers/small-resort/reward-hacker.yaml",
)


class BlockMonitor:
    def reset(self, seed: int) -> None:
        self.seed = seed

    def assess(self, observation, proposal, history):
        return MonitorDecision(
            risk_score=1.0,
            decision=DecisionType.BLOCK,
            reason_codes=("TEST_BLOCK",),
            related_infrastructure=(InfrastructureReference(kind="edge", index=0),),
        )


class AllowMonitor(BlockMonitor):
    def assess(self, observation, proposal, history):
        return MonitorDecision(risk_score=0.0, decision=DecisionType.ALLOW)


def configured_env() -> AvalancheEnv:
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.reset(seed=3)
    return env


def make_proposal(env: AvalancheEnv, value: float) -> ActionProposal:
    action = neutral_action(env.topology)
    action["route_weights"][0, 0] = value
    return ActionProposal(
        controller_id="unsafe",
        simulation_time=env.sim.simulation_time,
        action=freeze_action(action),
        explanation="Propose an unsafe route.",
    )


def make_adjudicator(env, monitor, policy):
    fallback = build_fallback(policy, ControllerConfig(kind="honest"), env.topology)
    boundary = Adjudicator(
        monitor,
        lambda action: validate_action(
            thaw_action(action), env.action_space, env._action_contract()
        ),
        fallback,
    )
    boundary.reset(3)
    return boundary


@pytest.mark.parametrize("policy", ["honest", "last_safe"])
@pytest.mark.parametrize("controller_path", CONTROLLERS)
def test_each_fallback_policy_supports_each_controller_kind(policy, controller_path):
    env = configured_env()
    values = ConfigurationResolver().component_values("controller", controller_path)
    controller = ControllerConfig.model_validate(values["controller"])
    fallback = build_fallback(policy, controller, env.topology)
    fallback.reset(3)
    assert fallback.policy == policy


def monitor_observation(env):
    observation = env._observation()
    observation["simulation_time"] = env.sim.simulation_time
    return build_monitor_observation(observation, env.sim)


def test_block_uses_the_honest_fallback():
    env = configured_env()
    rejected = make_proposal(env, 1.0)
    result = make_adjudicator(env, BlockMonitor(), "honest").adjudicate(
        monitor_observation(env), rejected, simulation_time=env.sim.simulation_time
    )

    assert result.executed_action.controller_id == "honest-fallback"
    assert result.executed_action.action != rejected.action
    assert rejected.action.route_weights[0][0] == 1.0


def test_last_safe_reuses_the_previous_execution():
    env = configured_env()
    boundary = make_adjudicator(env, AllowMonitor(), "last_safe")
    first = make_proposal(env, 0.5)
    allowed = boundary.adjudicate(
        monitor_observation(env), first, simulation_time=env.sim.simulation_time
    )
    boundary.monitor = BlockMonitor()
    blocked = boundary.adjudicate(
        monitor_observation(env),
        make_proposal(env, 1.0),
        simulation_time=env.sim.simulation_time,
    )

    assert blocked.executed_action.action == allowed.executed_action.action
    assert blocked.executed_action.controller_id == "last-safe-fallback"


def test_last_safe_starts_with_the_honest_fallback():
    env = configured_env()
    result = make_adjudicator(env, BlockMonitor(), "last_safe").adjudicate(
        monitor_observation(env),
        make_proposal(env, -1.0),
        simulation_time=env.sim.simulation_time,
    )

    assert result.executed_action.controller_id == "honest-fallback"
    assert result.executed_action.action != make_proposal(env, -1.0).action


def test_the_environment_applies_only_the_adjudicated_action():
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    fallback = build_fallback("honest", ControllerConfig(kind="honest"), env.topology)
    env.configure_adjudicator(BlockMonitor(), fallback)
    env.reset(seed=3)
    rejected = make_proposal(env, 1.0)

    observation, _, _, _, info = env.step_proposal(rejected)

    assert info["action_proposal"] == rejected
    assert info["monitor_decision"].decision is DecisionType.BLOCK
    assert info["executed_action"].action != rejected.action
    assert info["metrics"]["decision_counts"]["BLOCK"] == 1
    assert info["metrics"]["intervention_latency_count"] == 1
    interventions = observation["recent_interventions"]
    assert interventions["mask"][0] == 1
    assert interventions["risk"][0] == 1.0
    assert interventions["age"][0] == env.config.control_interval_seconds
    assert interventions["edge_targets"][0, 0] == 1
