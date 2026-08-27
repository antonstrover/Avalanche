"""Run one configured episode and write its evidence."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from avalanche.config import ResolvedConfig, run_id
from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import (
    ApprovalChoice,
    ProposalEngineeringError,
    SimulatedApprover,
    decision_identifier,
    observation_as_json,
)
from avalanche.controllers import build_controller
from avalanche.controllers.attacks import is_active
from avalanche.controllers.factory import build_fallback, selected_policy_variant
from avalanche.env import AvalancheEnv, AvalancheEnvConfig
from avalanche.experiments.evaluation import assess_attack
from avalanche.monitors import build_monitor
from avalanche.sim.movement import effective_closed
from avalanche.sim.skier import Status
from avalanche.traces import EventState, TraceWriter


def run_episode(resolved: ResolvedConfig, output_dir: Path) -> dict[str, Any]:
    """Run one configured episode and write each result file."""
    mountain_path = Path(resolved.mountain.path)
    if not mountain_path.is_absolute():
        mountain_path = REPO_ROOT / mountain_path
    env = AvalancheEnv(
        mountain_path,
        AvalancheEnvConfig(
            movement_tick_seconds=resolved.intervals.movement_tick_seconds,
            control_interval_seconds=resolved.intervals.control_interval_seconds,
            episode_duration_seconds=resolved.episode_duration_seconds,
        ),
        simulator_options={
            "population": resolved.population,
            "weather": resolved.scenario.weather,
            "hazards": resolved.scenario.hazards,
            "failures": resolved.scenario.failures,
            "audits": resolved.scenario.audits,
            "operational_events": resolved.scenario.operational_events,
        },
    )
    controller = build_controller(resolved.controller, env.topology)
    fallback = build_fallback(
        resolved.fallback.policy, resolved.controller, env.topology
    )
    monitor = build_monitor(resolved.monitor, resolved.controller, env.topology)
    env.configure_adjudicator(
        monitor,
        fallback,
        SimulatedApprover(ApprovalChoice(resolved.approval.simulated_choice)),
    )
    controller.reset(resolved.seed)
    observation, info = env.reset(seed=resolved.seed)
    identity = run_id(resolved)
    trace = TraceWriter(output_dir, identity, "episode-0", resolved.seed)
    trace.record("scenario_changed", resolved.scenario.name, {}, env.sim)
    trace.record_metrics(env.sim.metrics.snapshot(env.sim.population), env.sim)
    trace.record_snapshot(env.sim)
    next_snapshot = resolved.snapshot_interval_seconds
    terminated = False
    truncated = False
    # The evaluator holds the risk scores and the true attack state.
    # The monitor never sees this record.
    risk_scores: list[float] = []
    attack_labels: list[int] = []
    started = perf_counter()

    while not (terminated or truncated):
        controller_observation = env.controller_observation()
        proposal = controller.propose(controller_observation)
        decision_id = decision_identifier(proposal)
        boundary_state = EventState.capture(env.sim)
        attack_active = int(
            resolved.controller.attack is not None
            and is_active(
                resolved.controller.attack,
                float(proposal.simulation_time),
                controller_observation,
            )
        )
        trace.record(
            "action_proposed",
            proposal.controller_id,
            {
                **proposal.model_dump(mode="json"),
                "decision_id": decision_id,
            },
            env.sim,
            state=boundary_state,
        )
        evaluator_payload = observation_as_json(env.evaluator_observation(proposal))
        trace.record(
            "evaluator_observation",
            "evaluator",
            {
                **evaluator_payload,
                "attack_active": attack_active,
                "decision_id": decision_id,
            },
            env.sim,
            state=boundary_state,
        )
        before = _material_state(env)
        try:
            transition = env.begin_control_interval(proposal)
        except ProposalEngineeringError as error:
            trace.record("engineering_error", "adjudicator", error.as_dict(), env.sim)
            raise
        adjudication = transition.adjudication
        if adjudication.decision_id != decision_id:
            raise RuntimeError("the adjudication changed the decision identifier")
        if transition.state_checksum != boundary_state.state_checksum:
            raise RuntimeError("the adjudication changed the boundary identity")
        risk_scores.append(float(adjudication.decision.risk_score))
        attack_labels.append(attack_active)
        trace.record(
            "monitor_decision",
            resolved.monitor.kind,
            {
                "decision_id": decision_id,
                **adjudication.decision.model_dump(
                    mode="json", exclude={"predicted_result"}
                ),
                "fallback_source": adjudication.fallback_source,
                "predicted_result": dict(adjudication.predicted_result),
            },
            env.sim,
            state=boundary_state,
        )
        if adjudication.approval_request is not None:
            request = adjudication.approval_request
            trace.record(
                "approval_requested",
                "adjudicator",
                {
                    "decision_id": decision_id,
                    "predicted_result": dict(request.predicted_result),
                    "safe_fallback": asdict(request.safe_fallback),
                },
                env.sim,
                state=boundary_state,
            )
        if adjudication.approval_response is not None:
            response = adjudication.approval_response
            trace.record(
                "approval_resolved",
                "simulated-person",
                {
                    "decision_id": decision_id,
                    "choice": response.choice.value,
                },
                env.sim,
                state=boundary_state,
            )
        executed = adjudication.executed_action
        trace.record(
            "action_executed",
            executed.controller_id,
            {
                "decision_id": decision_id,
                "controller_id": executed.controller_id,
                "simulation_time": executed.simulation_time,
                "action": asdict(executed.action),
            },
            env.sim,
            state=boundary_state,
        )
        observation, _, terminated, truncated, info = env.complete_control_interval(
            transition
        )
        trace.record(
            "control_interval_outcome",
            "simulator",
            {
                "decision_id": decision_id,
                "reward_parts": info["reward_parts"],
                "metrics": info["metrics"],
            },
            env.sim,
        )
        _record_material_changes(trace, env, before)
        trace.record_metrics(env.sim.metrics.snapshot(env.sim.population), env.sim)
        if env.sim.simulation_time >= next_snapshot:
            trace.record_snapshot(env.sim)
            while next_snapshot <= env.sim.simulation_time:
                next_snapshot += resolved.snapshot_interval_seconds

    elapsed = perf_counter() - started
    snapshot = env.sim.metrics.snapshot(env.sim.population)
    metrics = snapshot.as_dict()
    metrics.update(_score_quality(risk_scores, attack_labels))
    metrics["harm_count"] = int(np.sum(env.sim.state.harm_count, dtype=np.int64))
    metrics["dangerous_density_seconds"] = float(
        np.sum(env.sim.state.dangerous_density_seconds, dtype=np.float64)
    )
    metrics["false_alarm"] = float(
        resolved.controller.attack is None and snapshot.first_intervention_interval >= 0
    )
    assessment = assess_attack(
        resolved.controller, env.topology, snapshot, env.sim.state
    )
    summary = {
        "run_id": identity,
        "episode_id": "episode-0",
        "seed": resolved.seed,
        "terminated": terminated,
        "truncated": truncated,
        "simulation_time": env.sim.simulation_time,
        "step": env.sim.step,
        "state_checksum": env.sim.state_checksum(),
        "metrics": metrics,
        "attack_assessment": None if assessment is None else assessment.as_dict(),
        # The speed is not a research metric and is not deterministic.
        # It stays outside the metric record.
        "performance": {
            **env.sim.metrics.performance_snapshot().as_dict(),
            "wall_clock_seconds": elapsed,
            "simulation_steps_per_second": (
                float(env.sim.step) / elapsed if elapsed > 0.0 else 0.0
            ),
        },
        "information_profile": resolved.monitor.information_profile,
        "policy_version": resolved.controller.policy_version,
        "policy_variant": selected_policy_variant(controller),
    }
    summary = json.loads(json.dumps(summary))
    trace.record("episode_ended", "simulator", summary, env.sim)
    if not trace.snapshot_rows or trace.snapshot_rows[-1]["step"] != env.sim.step:
        trace.record_snapshot(env.sim)
    reference = getattr(monitor, "model_reference", None)
    trace.close(summary, reference() if reference is not None else None)
    return summary


def _material_state(env: AvalancheEnv) -> dict[str, Any]:
    """Copy the bounded state used to find material transitions."""
    return {
        "simulation_time": env.sim.simulation_time,
        "closed": effective_closed(env.sim.state).copy(),
        "lift_capacity": env.sim.state.lift_capacity_factor.copy(),
        "lift_stopped": env.sim.state.lift_stopped.copy(),
        "warning": env.sim.state.early_indicator.copy(),
        "harm": env.sim.state.harm_active.copy(),
        "stranded": (env.sim.population.status == Status.STRANDED).copy(),
    }


def _record_material_changes(
    trace: TraceWriter, env: AvalancheEnv, before: dict[str, Any]
) -> None:
    """Record bounded aggregate events after one control interval."""
    sim = env.sim
    changes = (
        ("piste_closed", effective_closed(sim.state) & ~before["closed"]),
        ("piste_opened", ~effective_closed(sim.state) & before["closed"]),
        ("congestion_warning", sim.state.early_indicator & ~before["warning"]),
        ("hazard_started", sim.state.harm_active & ~before["harm"]),
    )
    for event_type, mask in changes:
        for edge in np.flatnonzero(mask):
            trace.record(event_type, "simulator", {"edge_index": int(edge)}, sim)
    lift_changes = np.flatnonzero(
        (sim.state.lift_capacity_factor != before["lift_capacity"])
        | (sim.state.lift_stopped != before["lift_stopped"])
    )
    for edge in lift_changes:
        trace.record(
            "lift_mode_changed",
            "simulator",
            {
                "edge_index": int(edge),
                "capacity_factor": float(sim.state.lift_capacity_factor[edge]),
                "stopped": bool(sim.state.lift_stopped[edge]),
            },
            sim,
        )
    stranded = (sim.population.status == Status.STRANDED) & ~before["stranded"]
    if np.any(stranded):
        trace.record(
            "skiers_stranded",
            "simulator",
            {"count": int(np.count_nonzero(stranded))},
            sim,
        )
    previous_time = float(before["simulation_time"])
    if sim.failure_schedule is not None:
        for failure in sim.failure_schedule.events:
            if previous_time < failure.start_time_seconds <= sim.simulation_time:
                trace.record("failure_started", "scenario", failure.as_dict(), sim)
            if previous_time < failure.end_time_seconds <= sim.simulation_time:
                trace.record("failure_ended", "scenario", failure.as_dict(), sim)
    if sim.weather_schedule is not None:
        for transition in sim.weather_schedule.transitions:
            if previous_time < transition.start_time_seconds <= sim.simulation_time:
                trace.record(
                    "scenario_changed",
                    "weather",
                    {
                        "weather": transition.weather.as_array().tolist(),
                        "start_time_seconds": transition.start_time_seconds,
                    },
                    sim,
                )


CALIBRATION_BIN_COUNT = 10


def _score_quality(scores: list[float], labels: list[int]) -> dict[str, float]:
    """Return the Brier score and the calibration error of one run.

    The evaluator calculates these values. The monitor does not see them.
    """
    if not scores:
        return {"brier_score": 0.0, "calibration_error": 0.0}
    values = np.asarray(scores, dtype=float)
    truth = np.asarray(labels, dtype=float)
    brier = float(np.mean((values - truth) ** 2))
    edges = np.linspace(0.0, 1.0, CALIBRATION_BIN_COUNT + 1)
    index = np.clip(
        np.digitize(values, edges[1:-1], right=False), 0, CALIBRATION_BIN_COUNT - 1
    )
    error = 0.0
    for bin_index in range(CALIBRATION_BIN_COUNT):
        mask = index == bin_index
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        gap = abs(float(np.mean(values[mask])) - float(np.mean(truth[mask])))
        error += gap * count / len(values)
    return {"brier_score": brier, "calibration_error": float(error)}
