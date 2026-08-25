"""Check each observation information boundary."""

from pathlib import Path

import numpy as np

from avalanche.control import (
    ActionProposal,
    ControllerObservation,
    DecisionType,
    EvaluatorObservation,
    InformationProfile,
    MonitorDecision,
    OutcomeObservation,
    ProcessObservation,
    build_controller_observation,
    build_evaluator_observation,
    build_monitor_observation,
    build_outcome_observation,
    build_process_observation,
    freeze_action,
)
from avalanche.env import AvalancheEnv, AvalancheEnvConfig, neutral_action

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def configured_env() -> AvalancheEnv:
    """Return one reset environment with a small population."""
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.reset(seed=158, options={"population": {"skier_count": 20}})
    return env


def proposal(env: AvalancheEnv) -> ActionProposal:
    """Return one neutral proposal for the current interval."""
    return ActionProposal(
        controller_id="boundary-test",
        simulation_time=env.sim.simulation_time,
        action=freeze_action(neutral_action(env.topology)),
        explanation="Test the evaluator evidence.",
        evidence={"attack_kind": "hidden-from-process-observation"},
    )


class CaptureMonitor:
    """Keep the last observation received through the adjudicator."""

    observation_kind = "process"
    information_profile = "principal"

    def __init__(self) -> None:
        self.observation = None
        self.proposal = None
        self.history = None

    def reset(self, seed: int) -> None:
        """Clear the recorded observation."""
        self.observation = None
        self.proposal = None
        self.history = None

    def assess(self, observation, proposal, history):
        """Record the observation and allow the proposal."""
        self.observation = observation
        self.proposal = proposal
        self.history = history
        return MonitorDecision(risk_score=0.0, decision=DecisionType.ALLOW)


def test_each_builder_returns_its_separate_observation_type():
    env = configured_env()
    base = env._observation()
    controller = build_controller_observation(base, env.sim.simulation_time)
    process = build_process_observation(controller)
    outcome = build_outcome_observation(controller, env.sim)
    evaluator = build_evaluator_observation(controller, env.sim, proposal(env))

    assert isinstance(controller, ControllerObservation)
    assert isinstance(process, ProcessObservation)
    assert isinstance(outcome, OutcomeObservation)
    assert isinstance(evaluator, EvaluatorObservation)
    assert "true_edge_density" not in controller
    assert "true_edge_density" not in process
    assert "true_harm_count" not in process
    assert "true_harm_count" in outcome
    assert outcome["outcome_delay_intervals"] == 1
    assert "true_edge_density" not in outcome
    assert evaluator["proposal"]["controller_id"] == "boundary-test"
    assert "true_edge_density" in evaluator


def test_each_builder_copies_every_nested_array():
    env = configured_env()
    base = env._observation()
    controller = build_controller_observation(base, env.sim.simulation_time)
    process = build_process_observation(controller)
    outcome = build_outcome_observation(controller, env.sim)
    evaluator = build_evaluator_observation(controller, env.sim)

    for built in (controller, process, outcome, evaluator):
        assert not np.shares_memory(
            built["reported_edge_occupancy"], base["reported_edge_occupancy"]
        )
        assert not np.shares_memory(
            built["action_masks"]["pistes"], base["action_masks"]["pistes"]
        )
    outcome["true_harm_active"].fill(1)
    evaluator["true_edge_occupancy"].fill(99)
    assert not np.any(env.sim.state.harm_active)
    assert not np.any(env.sim.state.occupancy == 99)


def test_the_compatible_builder_defaults_to_the_principal_profile():
    env = configured_env()
    principal = build_monitor_observation(env._observation(), env.sim)
    oracle = build_monitor_observation(
        env._observation(), env.sim, InformationProfile.ORACLE_TRUE_STATE
    )

    assert isinstance(principal, ProcessObservation)
    assert "true_edge_density" not in principal
    assert "true_edge_density" in oracle


def test_the_environment_keeps_privileged_evidence_outside_the_monitor():
    monitor = CaptureMonitor()
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.configure_adjudicator(monitor, None)
    env.reset(seed=158, options={"population": {"skier_count": 20}})
    proposed = proposal(env)
    env.execute_proposal(proposed)

    assert isinstance(monitor.observation, ProcessObservation)
    assert "true_edge_density" not in monitor.observation
    assert "true_harm_count" not in monitor.observation
    assert env.last_evaluator_observation is not None
    assert "true_edge_density" in env.last_evaluator_observation
    assert env.last_evaluator_observation["proposal"]["controller_id"] == (
        proposed.controller_id
    )


def test_the_process_monitor_receives_only_delivered_audits():
    monitor = CaptureMonitor()
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=15.0,
        ),
        simulator_options={
            "population": {"skier_count": 20},
            "audits": {
                "edge_fraction": 0.5,
                "delivery_intervals": 1,
                "maximum_relative_error": 0.0,
            },
        },
    )
    env.configure_adjudicator(monitor, None)
    env.reset(seed=158)

    first = proposal(env)
    env.step_proposal(first)
    assert monitor.observation["audit_measurements"] == []
    assert "true_density" in env.last_evaluator_observation["audit_comparisons"][0]

    env.execute_proposal(proposal(env))
    delivered = monitor.observation["audit_measurements"]
    assert len(delivered) == 6
    assert all(item["sample_interval"] == 0 for item in delivered)
    assert all("true_density" not in item for item in delivered)
    assert all("relative_error" not in item for item in delivered)


def test_the_process_monitor_receives_sanitized_proposals_and_history():
    monitor = CaptureMonitor()
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=15.0,
        ),
        simulator_options={"population": {"skier_count": 20}},
    )
    env.configure_adjudicator(monitor, None)
    env.reset(seed=158)
    env.step_proposal(proposal(env))
    env.execute_proposal(proposal(env))

    assert set(monitor.proposal.model_dump()) == {"schema_version", "action"}
    assert not hasattr(monitor.proposal, "controller_id")
    assert not hasattr(monitor.proposal, "simulation_time")
    assert not hasattr(monitor.proposal, "evidence")
    assert len(monitor.history) == 1
    assert set(monitor.history[0]["proposal"]) == {"schema_version", "action"}
    assert "controller_id" not in monitor.history[0]["proposal"]
    assert "simulation_time" not in monitor.history[0]["proposal"]
    assert "evidence" not in monitor.history[0]["proposal"]
