"""Run isolated live simulator sessions."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import queue
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

from avalanche.config import load_yaml
from avalanche.config.models import ControllerConfig, MonitorConfig, PopulationConfig
from avalanche.control import ActionProposal, AdjudicationResult
from avalanche.controllers import build_controller, build_fallback
from avalanche.env import (
    AvalancheEnv,
    AvalancheEnvConfig,
)
from avalanche.monitors import build_monitor
from avalanche.sim.engine import MountainSim
from avalanche.sim.movement import effective_closed
from avalanche.sim.skier import LocationKind

STREAM_VERSION = 3
SIMULATION_SPEED = 20.0
FRAME_INTERVAL_MS = 250
MAX_SKIERS = 10_000
TIMELINE_LIMIT = 64
MOUNTAIN_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "mountain" / "medium-resort.yaml"
)
DEMO_FAILURE_TARGET = "praz_plaza->plan_bois"
CONTROLLER_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "controllers" / "honest.yaml"
)
CONTROL_INTERVAL_SECONDS = 60.0
EPISODE_DURATION_SECONDS = 28_800.0


def topology_version(path: Path = MOUNTAIN_PATH) -> str:
    """Return a stable identity for the streamed topology."""
    return hashlib.blake2b(path.read_bytes(), digest_size=8).hexdigest()


def pack_frame(
    sim: MountainSim,
    session_id: str,
    sequence: int,
    topology: str,
    message_type: str = "frame",
    proposal: ActionProposal | None = None,
    adjudication: AdjudicationResult | None = None,
) -> bytes:
    """Pack one complete display state."""
    population = sim.population
    payload = {
        "skier_count": len(population),
        "location_kind": population.location_kind.astype(np.int8, copy=False).tobytes(),
        "location_index": population.location_index.astype("<i4", copy=False).tobytes(),
        "progress": population.progress.astype("<f4", copy=False).tobytes(),
        "display": display_state(sim, proposal, adjudication),
    }
    envelope = {
        "version": STREAM_VERSION,
        "type": message_type,
        "session_id": session_id,
        "sequence": sequence,
        "simulation_time": sim.simulation_time,
        "topology_version": topology,
        "state_checksum": sim.state_checksum(),
        "payload": payload,
    }
    return msgpack.packb(envelope, use_bin_type=True)


def _failure_id(kind: str, target: int, start: float) -> str:
    """Return the stable identity of one scheduled failure."""
    return f"failure:{kind}:{target}:{start:g}"


def display_state(
    sim: MountainSim,
    proposal: ActionProposal | None = None,
    adjudication: AdjudicationResult | None = None,
) -> dict[str, object]:
    """Return the bounded display state for one live frame."""
    assert sim.failure_schedule is not None
    assert sim.weather_schedule is not None
    assert sim.topology is not None

    failures = [
        {
            "event_id": _failure_id(
                event.kind.value, event.target, event.start_time_seconds
            ),
            **event.as_dict(),
            "end_time_seconds": event.end_time_seconds,
            "severity": ("medium" if event.kind.value == "late_telemetry" else "high"),
        }
        for event in sim.active_failures
    ]
    hazards = []
    for edge in np.flatnonzero(sim.state.early_indicator | sim.state.harm_active):
        harm = bool(sim.state.harm_active[edge])
        event_type = "true_harm" if harm else "early_indicator"
        count = sim.state.harm_count[edge] if harm else sim.state.indicator_count[edge]
        hazards.append(
            {
                "event_id": f"{event_type}:{int(edge)}:{int(count)}",
                "event_type": event_type,
                "edge_index": int(edge),
                "severity": "high" if harm else "medium",
                "hazard_score": float(sim.state.hazard_score[edge]),
            }
        )

    closed = effective_closed(sim.state)
    closures = [
        {
            "edge_index": int(edge),
            "weather": bool(sim.state.weather_closed[edge]),
            "failure": bool(sim.state.failure_closed[edge]),
            "operational": bool(sim.state.closed[edge]),
        }
        for edge in np.flatnonzero(closed)
    ]
    return {
        "weather": {
            "wind": sim.weather.wind,
            "visibility": sim.weather.visibility,
            "snowfall": sim.weather.snowfall,
            "temperature": sim.weather.temperature,
        },
        "failures": failures,
        "hazards": hazards,
        "closures": closures,
        "timeline": _timeline(sim)[-TIMELINE_LIMIT:],
        "decision": _decision_state(proposal, adjudication),
    }


def _decision_state(
    proposal: ActionProposal | None, adjudication: AdjudicationResult | None
) -> dict[str, object] | None:
    """Return the latest proposal and execution result."""
    if proposal is None or adjudication is None:
        return None
    executed = adjudication.executed_action
    return {
        "proposal": proposal.model_dump(mode="json"),
        "executed_action": {
            "controller_id": executed.controller_id,
            "simulation_time": executed.simulation_time,
            "action": asdict(executed.action),
        },
        "monitor_decision": None,
        "fallback_source": adjudication.fallback_source,
        "predicted_result": dict(adjudication.predicted_result),
    }


def _control_step(
    env: AvalancheEnv, controller: Any
) -> tuple[ActionProposal, AdjudicationResult]:
    """Propose and execute one live control action."""
    observation = env.controller_observation()
    proposal = controller.propose(observation)
    return proposal, env.execute_proposal(proposal)


def _timeline(sim: MountainSim) -> list[dict[str, object]]:
    """Build the recoverable event window at the current time."""
    assert sim.failure_schedule is not None
    assert sim.weather_schedule is not None
    events: list[dict[str, object]] = []
    for failure in sim.failure_schedule.events:
        identity = _failure_id(
            failure.kind.value, failure.target, failure.start_time_seconds
        )
        if failure.start_time_seconds <= sim.simulation_time:
            events.append(
                {
                    "event_id": f"{identity}:start",
                    "event_type": "failure_started",
                    "target": failure.target_id,
                    "edge_index": failure.target,
                    "start_time_seconds": failure.start_time_seconds,
                    "end_time_seconds": failure.end_time_seconds,
                    "severity": (
                        "medium" if failure.kind.value == "late_telemetry" else "high"
                    ),
                    "label": failure.kind.value.replace("_", " "),
                }
            )
        if failure.end_time_seconds <= sim.simulation_time:
            events.append(
                {
                    "event_id": f"{identity}:end",
                    "event_type": "failure_ended",
                    "target": failure.target_id,
                    "edge_index": failure.target,
                    "start_time_seconds": failure.end_time_seconds,
                    "end_time_seconds": failure.end_time_seconds,
                    "severity": "low",
                    "label": f"{failure.kind.value.replace('_', ' ')} ended",
                }
            )
    for hazard in sim.hazard_events:
        events.append(
            {
                "event_id": hazard.event_id,
                "event_type": hazard.event_type,
                "target": f"edge {hazard.edge_index}",
                "edge_index": hazard.edge_index,
                "start_time_seconds": hazard.start_time_seconds,
                "end_time_seconds": None,
                "severity": ("high" if hazard.event_type == "true_harm" else "medium"),
                "label": hazard.event_type.replace("_", " "),
            }
        )
    for index, transition in enumerate(sim.weather_schedule.transitions):
        if transition.start_time_seconds <= sim.simulation_time:
            events.append(
                {
                    "event_id": f"weather:{index}:{transition.start_time_seconds:g}",
                    "event_type": "weather_changed",
                    "target": "resort",
                    "edge_index": None,
                    "start_time_seconds": transition.start_time_seconds,
                    "end_time_seconds": None,
                    "severity": "low",
                    "label": "weather changed",
                }
            )
    return sorted(
        events,
        key=lambda event: (float(event["start_time_seconds"]), str(event["event_id"])),
    )


def _put_latest(output: Any, value: bytes) -> None:
    """Put a value without blocking the simulator."""
    try:
        output.put_nowait(value)
    except queue.Full:
        try:
            output.get_nowait()
        except queue.Empty:
            pass
        output.put_nowait(value)


def run_session(
    session_id: str,
    seed: int,
    skier_count: int,
    topology: str,
    output: Any,
    stop: Any,
    demo_failure: bool = False,
) -> None:
    """Run one simulator inside a child process."""
    try:
        options: dict[str, object] = {
            "population": PopulationConfig(skier_count=skier_count),
            "weather": {
                "initial": {
                    "wind": 5.0,
                    "visibility": 5_000.0,
                    "snowfall": 0.4,
                    "temperature": 1.0,
                },
                "schedule": [
                    {
                        "start_time_seconds": 30.0,
                        "wind": 12.0,
                        "visibility": 700.0,
                        "snowfall": 4.0,
                        "temperature": -3.0,
                    }
                ],
            },
        }
        if demo_failure:
            options["failures"] = {
                "schedule": [
                    {
                        "kind": "lift_stoppage",
                        "target": DEMO_FAILURE_TARGET,
                        "start_time_seconds": 5.0,
                        "duration_seconds": 60.0,
                        "controller_visible": True,
                    }
                ]
            }
        env = AvalancheEnv(
            MOUNTAIN_PATH,
            AvalancheEnvConfig(
                movement_tick_seconds=5.0,
                control_interval_seconds=CONTROL_INTERVAL_SECONDS,
                episode_duration_seconds=EPISODE_DURATION_SECONDS,
            ),
            simulator_options=options,
        )
        sim = env.sim
        controller_values = load_yaml(CONTROLLER_PATH)["controller"]
        controller_config = ControllerConfig.model_validate(controller_values)
        controller = build_controller(controller_config, env.topology)
        fallback = build_fallback("honest", controller_config, env.topology)
        monitor = build_monitor(
            MonitorConfig(kind="none"), controller_config, env.topology
        )
        env.configure_adjudicator(monitor, fallback)
        env.reset(seed=seed)
        controller.reset(seed)
        proposal, adjudication = _control_step(env, controller)
        sequence = 0
        _put_latest(
            output,
            pack_frame(
                sim,
                session_id,
                sequence,
                topology,
                "snapshot",
                proposal,
                adjudication,
            ),
        )
        interval = FRAME_INTERVAL_MS / 1000.0
        while not stop.wait(interval):
            sim.tick()
            if sim.simulation_time % CONTROL_INTERVAL_SECONDS == 0.0:
                proposal, adjudication = _control_step(env, controller)
            sequence += 1
            _put_latest(
                output,
                pack_frame(
                    sim,
                    session_id,
                    sequence,
                    topology,
                    proposal=proposal,
                    adjudication=adjudication,
                ),
            )
            if np.all(sim.population.location_kind == LocationKind.FINISHED):
                sequence += 1
                complete = msgpack.packb(
                    {
                        "version": STREAM_VERSION,
                        "type": "complete",
                        "session_id": session_id,
                        "sequence": sequence,
                        "simulation_time": sim.simulation_time,
                        "topology_version": topology,
                        "state_checksum": sim.state_checksum(),
                    },
                    use_bin_type=True,
                )
                _put_latest(output, complete)
                return
    except Exception as error:  # pragma: no cover - tested through the manager state
        _put_latest(
            output,
            msgpack.packb(
                {
                    "version": STREAM_VERSION,
                    "type": "error",
                    "session_id": session_id,
                    "sequence": 0,
                    "simulation_time": 0.0,
                    "topology_version": topology,
                    "state_checksum": "",
                    "message": str(error),
                },
                use_bin_type=True,
            ),
        )


@dataclass
class LiveSession:
    """Hold the transport state for one worker process."""

    session_id: str
    seed: int
    skier_count: int
    topology_version: str
    process: Any
    output: Any
    stop_event: Any
    demo_failure: bool = False
    status: str = "starting"
    latest: bytes | None = None
    latest_sequence: int = -1
    lock: threading.Lock = field(default_factory=threading.Lock)
    pump: threading.Thread | None = None

    def response(self) -> dict[str, object]:
        """Return the public session state."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "skier_count": self.skier_count,
            "simulation_speed": SIMULATION_SPEED,
            "frame_interval_ms": FRAME_INTERVAL_MS,
            "topology_version": self.topology_version,
            "demo_failure": self.demo_failure,
        }


class SessionManager:
    """Own the live worker processes."""

    def __init__(self) -> None:
        self.context = mp.get_context("spawn")
        self.sessions: dict[str, LiveSession] = {}
        self.lock = threading.Lock()

    def create(
        self, seed: int, skier_count: int, demo_failure: bool = False
    ) -> LiveSession:
        """Create and start one live session."""
        session_id = str(uuid.uuid4())
        version = topology_version()
        output = self.context.Queue(maxsize=4)
        stop_event = self.context.Event()
        process = self.context.Process(
            target=run_session,
            args=(
                session_id,
                seed,
                skier_count,
                version,
                output,
                stop_event,
                demo_failure,
            ),
            daemon=True,
        )
        session = LiveSession(
            session_id=session_id,
            seed=seed,
            skier_count=skier_count,
            topology_version=version,
            process=process,
            output=output,
            stop_event=stop_event,
            demo_failure=demo_failure,
        )
        with self.lock:
            self.sessions[session_id] = session
        process.start()
        session.pump = threading.Thread(target=self._pump, args=(session,), daemon=True)
        session.pump.start()
        return session

    def _pump(self, session: LiveSession) -> None:
        """Copy worker messages into the shared latest frame."""
        while session.process.is_alive() or not session.output.empty():
            try:
                packed = session.output.get(timeout=0.1)
            except queue.Empty:
                continue
            envelope = msgpack.unpackb(packed, raw=False)
            with session.lock:
                session.latest = packed
                session.latest_sequence = int(envelope["sequence"])
                kind = envelope["type"]
                if kind in {"snapshot", "frame"}:
                    session.status = "running"
                elif kind == "complete":
                    session.status = "complete"
                elif kind == "error":
                    session.status = "failed"
        stopped_early = session.status in {"starting", "running"}
        if stopped_early and not session.stop_event.is_set():
            session.status = "failed"

    def get(self, session_id: str) -> LiveSession | None:
        """Return one session when it exists."""
        with self.lock:
            return self.sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Stop and remove one session."""
        with self.lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        session.stop_event.set()
        session.process.join(timeout=2.0)
        if session.process.is_alive():
            session.process.terminate()
            session.process.join(timeout=1.0)
        session.output.close()
        return True

    def close(self) -> None:
        """Stop every live session."""
        with self.lock:
            session_ids = list(self.sessions)
        for session_id in session_ids:
            self.delete(session_id)


manager = SessionManager()


def snapshot_message(packed: bytes) -> bytes:
    """Change the latest frame into a recovery snapshot."""
    envelope = msgpack.unpackb(packed, raw=False)
    if envelope["type"] in {"snapshot", "frame"}:
        envelope["type"] = "snapshot"
    return msgpack.packb(envelope, use_bin_type=True)
