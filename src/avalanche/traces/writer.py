"""Write material events and periodic replay snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from avalanche.metrics import METRICS_VERSION, MetricSnapshot
from avalanche.sim.engine import MountainSim
from avalanche.traces.snapshots import encode_snapshot

EVENT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class EventState:
    """Identify the simulator state for one trace event."""

    simulation_time: float
    step: int
    state_checksum: str

    @classmethod
    def capture(cls, sim: MountainSim) -> EventState:
        """Capture the current simulator state identity."""
        return cls(sim.simulation_time, sim.step, sim.state_checksum())


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
        self.snapshot_rows: list[dict[str, Any]] = []

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
        """Buffer one typed replay snapshot."""
        self.snapshot_rows.append(
            encode_snapshot(
                sim,
                run_id=self.run_id,
                episode_id=self.episode_id,
                seed=self.seed,
            )
        )

    def close(
        self, summary: dict[str, Any], model_reference: dict[str, Any] | None = None
    ) -> None:
        """Write each buffered artifact."""
        summary_metrics = summary.get("metrics")
        if not isinstance(summary_metrics, dict) or (
            summary_metrics.get("metrics_version") != METRICS_VERSION
        ):
            raise ValueError(
                f"a run summary must use metrics version {METRICS_VERSION}"
            )
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
        pq.write_table(
            pa.Table.from_pylist(self.snapshot_rows),
            self.output_dir / "snapshots.parquet",
        )
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
