"""Run one configured episode and write its evidence."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from avalanche.config import ResolvedConfig, run_id
from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import (
    ApprovalChoice,
    ProposalEngineeringError,
    SimulatedApprover,
)
from avalanche.controllers import build_controller
from avalanche.controllers.factory import build_fallback
from avalanche.env import AvalancheEnv, AvalancheEnvConfig
from avalanche.experiments.evaluation import assess_attack
from avalanche.monitors import build_monitor
from avalanche.sim.movement import effective_closed
from avalanche.sim.skier import Status
from avalanche.traces import TraceWriter


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

    while not (terminated or truncated):
        controller_observation = env.controller_observation()
        proposal = controller.propose(controller_observation)
        trace.record(
            "action_proposed",
            proposal.controller_id,
            proposal.model_dump(mode="json"),
            env.sim,
        )
        before = _material_state(env)
        try:
            observation, _, terminated, truncated, info = env.step_proposal(proposal)
        except ProposalEngineeringError as error:
            trace.record("engineering_error", "adjudicator", error.as_dict(), env.sim)
            raise
        adjudication = info["adjudication"]
        trace.record(
            "monitor_decision",
            resolved.monitor.kind,
            {
                **adjudication.decision.model_dump(mode="json"),
                "fallback_source": adjudication.fallback_source,
                "predicted_result": dict(adjudication.predicted_result),
            },
            env.sim,
        )
        if adjudication.approval_request is not None:
            request = adjudication.approval_request
            trace.record(
                "approval_requested",
                "adjudicator",
                {
                    "decision_id": request.decision_id,
                    "predicted_result": dict(request.predicted_result),
                    "safe_fallback": asdict(request.safe_fallback),
                },
                env.sim,
            )
        if adjudication.approval_response is not None:
            response = adjudication.approval_response
            trace.record(
                "approval_resolved",
                "simulated-person",
                {
                    "decision_id": adjudication.approval_request.decision_id,
                    "choice": response.choice.value,
                },
                env.sim,
            )
        executed = info["executed_action"]
        trace.record(
            "action_executed",
            executed.controller_id,
            {
                "controller_id": executed.controller_id,
                "simulation_time": executed.simulation_time,
                "action": asdict(executed.action),
            },
            env.sim,
        )
        _record_material_changes(trace, env, before)
        trace.record_metrics(env.sim.metrics.snapshot(env.sim.population), env.sim)
        if env.sim.simulation_time >= next_snapshot:
            trace.record_snapshot(env.sim)
            while next_snapshot <= env.sim.simulation_time:
                next_snapshot += resolved.snapshot_interval_seconds

    snapshot = env.sim.metrics.snapshot(env.sim.population)
    metrics = snapshot.as_dict()
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
    }
    summary = json.loads(json.dumps(summary))
    trace.record("episode_ended", "simulator", summary, env.sim)
    if not trace.snapshot_rows or trace.snapshot_rows[-1]["step"] != env.sim.step:
        trace.record_snapshot(env.sim)
    trace.close(summary)
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
