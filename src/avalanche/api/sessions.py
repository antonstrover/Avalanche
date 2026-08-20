"""Run isolated live simulator sessions."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

from avalanche.config.models import PopulationConfig
from avalanche.sim.engine import MountainSim
from avalanche.sim.skier import LocationKind

STREAM_VERSION = 1
SIMULATION_SPEED = 20.0
FRAME_INTERVAL_MS = 250
MAX_SKIERS = 10_000
MOUNTAIN_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "mountain" / "small-resort.yaml"
)


def topology_version(path: Path = MOUNTAIN_PATH) -> str:
    """Return a stable identity for the streamed topology."""
    return hashlib.blake2b(path.read_bytes(), digest_size=8).hexdigest()


def pack_frame(
    sim: MountainSim,
    session_id: str,
    sequence: int,
    topology: str,
    message_type: str = "frame",
) -> bytes:
    """Pack one complete display state."""
    population = sim.population
    payload = {
        "skier_count": len(population),
        "location_kind": population.location_kind.astype(np.int8, copy=False).tobytes(),
        "location_index": population.location_index.astype("<i4", copy=False).tobytes(),
        "progress": population.progress.astype("<f4", copy=False).tobytes(),
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
) -> None:
    """Run one simulator inside a child process."""
    try:
        sim = MountainSim(MOUNTAIN_PATH)
        sim.reset(
            seed,
            options={
                "population": PopulationConfig(skier_count=skier_count),
            },
        )
        sequence = 0
        _put_latest(output, pack_frame(sim, session_id, sequence, topology, "snapshot"))
        interval = FRAME_INTERVAL_MS / 1000.0
        while not stop.wait(interval):
            sim.tick()
            sequence += 1
            _put_latest(output, pack_frame(sim, session_id, sequence, topology))
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
        }


class SessionManager:
    """Own the live worker processes."""

    def __init__(self) -> None:
        self.context = mp.get_context("spawn")
        self.sessions: dict[str, LiveSession] = {}
        self.lock = threading.Lock()

    def create(self, seed: int, skier_count: int) -> LiveSession:
        """Create and start one live session."""
        session_id = str(uuid.uuid4())
        version = topology_version()
        output = self.context.Queue(maxsize=4)
        stop_event = self.context.Event()
        process = self.context.Process(
            target=run_session,
            args=(session_id, seed, skier_count, version, output, stop_event),
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
