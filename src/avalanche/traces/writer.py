"""Write material events and periodic replay snapshots."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from avalanche.metrics import MetricSnapshot
from avalanche.sim.engine import MountainSim

EVENT_SCHEMA_VERSION = 1


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
    ) -> None:
        """Buffer one material event with the current state identity."""
        self.events.append(
            EventRecord(
                schema_version=EVENT_SCHEMA_VERSION,
                run_id=self.run_id,
                episode_id=self.episode_id,
                seed=self.seed,
                simulation_time=sim.simulation_time,
                step=sim.step,
                event_type=event_type,
                actor_id=actor_id,
                payload=payload,
                state_checksum=sim.state_checksum(),
            )
        )

    def record_metrics(self, metrics: MetricSnapshot, sim: MountainSim) -> None:
        """Buffer one wide metric sample."""
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
        population = sim.population
        state = sim.state
        self.snapshot_rows.append(
            {
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "seed": self.seed,
                "simulation_time": sim.simulation_time,
                "step": sim.step,
                "state_checksum": sim.state_checksum(),
                "skier_count": len(population),
                "edge_count": int(state.occupancy.size),
                "location_kind_i8": _bytes(population.location_kind, "i1"),
                "location_index_i32": _bytes(population.location_index, "<i4"),
                "progress_f32": _bytes(population.progress, "<f4"),
                "status_i8": _bytes(population.status, "i1"),
                "edge_closed_i8": _bytes(state.closed, "i1"),
                "edge_occupancy_i32": _bytes(state.occupancy, "<i4"),
                "edge_queue_i32": _bytes(state.queue_length, "<i4"),
            }
        )

    def close(self, summary: dict[str, Any]) -> None:
        """Write each buffered artifact."""
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


def _bytes(values: np.ndarray, dtype: str) -> bytes:
    """Return contiguous bytes with one declared portable dtype."""
    return np.asarray(values, dtype=np.dtype(dtype)).tobytes()
