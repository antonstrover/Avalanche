"""Write material events and periodic replay snapshots."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from avalanche.metrics import METRICS_VERSION, MetricSnapshot
from avalanche.sim.engine import MountainSim
from avalanche.traces.snapshots import (
    EVALUATOR_REPLAY_FILENAME,
    REPORTED_REPLAY_FILENAME,
    encode_physical_replay_snapshot,
)

EVENT_SCHEMA_VERSION = 5
SUMMARY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class EventState:
    """Identify the simulator state for one trace event."""

    simulation_time: float
    step: int
    state_checksum: str

    @classmethod
    def capture(cls, sim: MountainSim) -> EventState:
        """Capture the current simulator state identity."""
        return cls(sim.simulation_time, sim.step, sim.physical_state_checksum())


@dataclass(frozen=True)
class EventRecord:
    """Hold one versioned material event."""

    schema_version: int
    run_id: str
    episode_id: str
    seed: int
    simulation_time: float
    step: int
    event_type: str
    actor_id: str
    payload: dict[str, Any]
    state_checksum: str

    def as_dict(self) -> dict[str, Any]:
        """Return the complete event envelope."""
        return asdict(self)


class TraceWriter:
    """Buffer and write one episode trace."""

    def __init__(
        self, output_dir: Path, run_id: str, episode_id: str, seed: int
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.episode_id = episode_id
        self.seed = seed
        self.events: list[EventRecord] = []
        self.metric_rows: list[dict[str, Any]] = []
        self.reported_snapshot_rows: list[dict[str, Any]] = []
        self.evaluator_snapshot_rows: list[dict[str, Any]] = []
        self.continuation_artifacts: list[dict[str, str]] = []

    @property
    def snapshot_rows(self) -> list[dict[str, Any]]:
        """Return the evaluator rows for the former local interface."""
        return self.evaluator_snapshot_rows

    def record(
        self,
        event_type: str,
        actor_id: str,
        payload: dict[str, Any],
        sim: MountainSim,
        *,
        state: EventState | None = None,
    ) -> None:
        """Buffer one material event with the current state identity."""
        identity = state or EventState.capture(sim)
        self.events.append(
            EventRecord(
                schema_version=EVENT_SCHEMA_VERSION,
                run_id=self.run_id,
                episode_id=self.episode_id,
                seed=self.seed,
                simulation_time=identity.simulation_time,
                step=identity.step,
                event_type=event_type,
                actor_id=actor_id,
                payload=payload,
                state_checksum=identity.state_checksum,
            )
        )

    def record_metrics(self, metrics: MetricSnapshot, sim: MountainSim) -> None:
        """Buffer one wide metric sample."""
        if metrics.metrics_version != METRICS_VERSION:
            raise ValueError(
                f"a metric sample must use metrics version {METRICS_VERSION}"
            )
        self.metric_rows.append(
            {
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "seed": self.seed,
                "simulation_time": sim.simulation_time,
                "step": sim.step,
                **metrics.as_dict(),
            }
        )

    def record_snapshot(self, sim: MountainSim) -> None:
        """Buffer separate reported and evaluator replay views."""
        self.reported_snapshot_rows.append(
            encode_physical_replay_snapshot(
                sim,
                view_kind="reported",
                run_id=self.run_id,
                episode_id=self.episode_id,
            )
        )
        self.evaluator_snapshot_rows.append(
            encode_physical_replay_snapshot(
                sim,
                view_kind="evaluator",
                run_id=self.run_id,
                episode_id=self.episode_id,
            )
        )

    def record_continuation_artifact(self, record: dict[str, str]) -> None:
        """Add one externally checksummed continuation artifact."""
        self.continuation_artifacts.append(dict(record))

    def snapshot_state(self) -> dict[str, Any]:
        """Return every buffered trace and writer position."""
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "events": tuple(event.as_dict() for event in self.events),
            "metric_rows": tuple(self.metric_rows),
            "reported_snapshot_rows": tuple(self.reported_snapshot_rows),
            "evaluator_snapshot_rows": tuple(self.evaluator_snapshot_rows),
            "continuation_artifacts": tuple(self.continuation_artifacts),
            "output_append_positions": {
                "events": len(self.events),
                "metrics": len(self.metric_rows),
                "reported_replay": len(self.reported_snapshot_rows),
                "evaluator_replay": len(self.evaluator_snapshot_rows),
            },
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore trace buffers before appending resumed output."""
        identity = (state["run_id"], state["episode_id"], int(state["seed"]))
        if identity != (self.run_id, self.episode_id, self.seed):
            raise ValueError("the trace writer identity is incompatible")
        self.events = [EventRecord(**item) for item in state["events"]]
        self.metric_rows = [dict(item) for item in state["metric_rows"]]
        self.reported_snapshot_rows = [
            dict(item) for item in state["reported_snapshot_rows"]
        ]
        self.evaluator_snapshot_rows = [
            dict(item) for item in state["evaluator_snapshot_rows"]
        ]
        self.continuation_artifacts = [
            dict(item) for item in state["continuation_artifacts"]
        ]
        positions = state["output_append_positions"]
        actual = {
            "events": len(self.events),
            "metrics": len(self.metric_rows),
            "reported_replay": len(self.reported_snapshot_rows),
            "evaluator_replay": len(self.evaluator_snapshot_rows),
        }
        if positions != actual:
            raise ValueError("the trace writer positions are inconsistent")

    def close(
        self, summary: dict[str, Any], model_reference: dict[str, Any] | None = None
    ) -> None:
        """Write each buffered artifact."""
        summary_metrics = summary.get("metrics")
        if summary.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                f"a run summary must use summary version {SUMMARY_SCHEMA_VERSION}"
            )
        if not isinstance(summary_metrics, dict) or (
            summary_metrics.get("metrics_version") != METRICS_VERSION
        ):
            raise ValueError(
                f"a run summary must use metrics version {METRICS_VERSION}"
            )
        if "harm_count" in summary_metrics:
            raise ValueError("a current run summary must not contain harm_count")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.output_dir / "events.jsonl"
        with events_path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(
                    json.dumps(event.as_dict(), sort_keys=True, separators=(",", ":"))
                )
                handle.write("\n")
        pq.write_table(
            pa.Table.from_pylist(self.metric_rows), self.output_dir / "metrics.parquet"
        )
        reported_path = self.output_dir / REPORTED_REPLAY_FILENAME
        evaluator_path = self.output_dir / EVALUATOR_REPLAY_FILENAME
        pq.write_table(pa.Table.from_pylist(self.reported_snapshot_rows), reported_path)
        pq.write_table(pa.Table.from_pylist(self.evaluator_snapshot_rows), evaluator_path)
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        model_reference = model_reference or {
            "model_kind": None,
            "model_path": None,
            "model_revision": None,
        }
        (self.output_dir / "model-reference.json").write_text(
            json.dumps(model_reference, indent=2, sort_keys=True), encoding="utf-8"
        )
        replay_artifacts = [
            _artifact_record("physical_replay_reported", reported_path),
            _artifact_record("physical_replay_evaluator", evaluator_path),
        ]
        manifest = {
            "artifact_manifest_version": 1,
            "artifacts": [*replay_artifacts, *self.continuation_artifacts],
        }
        (self.output_dir / "artifact-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _artifact_record(artifact_type: str, path: Path) -> dict[str, str]:
    """Return one exact persisted file identity."""
    return {
        "artifact_type": artifact_type,
        "path": path.name,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
