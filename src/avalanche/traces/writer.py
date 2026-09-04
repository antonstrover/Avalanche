"""Write complete event and run artifact evidence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from avalanche.config.models import ResolvedConfig
from avalanche.config.run_identity import REPO_ROOT
from avalanche.metrics import METRICS_VERSION, MetricSnapshot
from avalanche.sim.engine import MountainSim
from avalanche.sim.transitions import EventPhase
from avalanche.traces.io import atomic_write_bytes, atomic_write_text, fsync_directory
from avalanche.traces.snapshots import (
    CONTINUATION_ARTIFACT_TYPE,
    CONTINUATION_SCHEMA_VERSION,
    EVALUATOR_REPLAY_FILENAME,
    PHYSICAL_REPLAY_ARTIFACT_TYPE,
    PHYSICAL_REPLAY_SCHEMA_VERSION,
    REPORTED_REPLAY_FILENAME,
    encode_physical_replay_snapshot,
)

EVENT_SCHEMA_VERSION = 6
SUMMARY_SCHEMA_VERSION = 4
RUN_MANIFEST_SCHEMA_VERSION = 1
PERFORMANCE_SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "run-manifest.json"
RUN_MANIFEST_SIDECAR_FILENAME = "run-manifest.sha256"
type TraceLevel = Literal["summary", "decision", "debug"]


@dataclass(frozen=True)
class EventState:
    """Identify one physical state at a formal event boundary."""

    simulation_time: float
    movement_tick: int
    control_interval_index: int
    physical_state_checksum: str

    @classmethod
    def capture(
        cls,
        sim: MountainSim,
        *,
        control_interval_index: int | None = None,
        view_kind: Literal["reported", "evaluator"] = "evaluator",
    ) -> EventState:
        """Capture the selected physical view identity."""
        interval = (
            int(sim.simulation_time / sim.control_interval_seconds)
            if control_interval_index is None
            else control_interval_index
        )
        return cls(
            sim.simulation_time,
            sim.step,
            interval,
            sim.physical_state_checksum(view_kind),
        )


@dataclass(frozen=True)
class EventRecord:
    """Hold one ordered formal event."""

    schema_version: int
    event_sequence: int
    run_id: str
    episode_id: str
    seed: int
    simulation_time: float
    movement_tick: int
    control_interval_index: int
    phase_code: int
    event_type: str
    actor_id: str
    entity_kind: str
    entity_index: int
    entity_id: str
    payload: dict[str, Any]
    physical_state_checksum: str

    def as_dict(self) -> dict[str, Any]:
        """Return the complete formal event envelope."""
        return asdict(self)


class TraceWriter:
    """Buffer and publish one complete episode trace."""

    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        episode_id: str,
        seed: int,
        *,
        trace_level: TraceLevel = "decision",
        resolved: ResolvedConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
        performance_root: Path | None = None,
    ) -> None:
        if trace_level not in {"summary", "decision", "debug"}:
            raise ValueError("the trace level is invalid")
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.episode_id = episode_id
        self.seed = seed
        self.trace_level = trace_level
        self.resolved = resolved
        self.metadata = dict(metadata or {})
        self.performance_root = (
            Path(performance_root)
            if performance_root is not None
            else self.output_dir.parent / "performance"
        )
        self.events: list[EventRecord] = []
        self.metric_rows: list[dict[str, Any]] = []
        self.reported_snapshot_rows: list[dict[str, Any]] = []
        self.evaluator_snapshot_rows: list[dict[str, Any]] = []
        self.continuation_artifacts: list[dict[str, str]] = []
        self.research_manifest_sha256: str | None = None

    @property
    def snapshot_rows(self) -> list[dict[str, Any]]:
        """Return evaluator rows through the former local interface."""
        return self.evaluator_snapshot_rows

    def record(
        self,
        event_type: str,
        actor_id: str,
        payload: dict[str, Any],
        sim: MountainSim,
        *,
        phase: EventPhase = EventPhase.OPERATIONAL_EVENT_TRANSITION,
        entity: tuple[str, int, str] = ("", -1, ""),
        state: EventState | None = None,
        debug_only: bool = False,
    ) -> None:
        """Buffer one event when the selected trace level permits it."""
        if self.trace_level == "summary":
            return
        if debug_only and self.trace_level != "debug":
            return
        identity = state or EventState.capture(sim)
        self.events.append(
            EventRecord(
                schema_version=EVENT_SCHEMA_VERSION,
                event_sequence=-1,
                run_id=self.run_id,
                episode_id=self.episode_id,
                seed=self.seed,
                simulation_time=identity.simulation_time,
                movement_tick=identity.movement_tick,
                control_interval_index=identity.control_interval_index,
                phase_code=int(phase),
                event_type=event_type,
                actor_id=actor_id,
                entity_kind=entity[0],
                entity_index=entity[1],
                entity_id=entity[2],
                payload=payload,
                physical_state_checksum=identity.physical_state_checksum,
            )
        )

    def record_metrics(self, metrics: MetricSnapshot, sim: MountainSim) -> None:
        """Buffer one wide control metric sample."""
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
        self.record(
            "metric_snapshot",
            "evaluator",
            {"metrics_version": metrics.metrics_version},
            sim,
            phase=EventPhase.METRIC_SNAPSHOT,
        )

    def record_snapshot(self, sim: MountainSim) -> None:
        """Buffer both physical replay views at one tick boundary."""
        if self.trace_level == "summary":
            return
        reported = encode_physical_replay_snapshot(
            sim,
            view_kind="reported",
            run_id=self.run_id,
            episode_id=self.episode_id,
        )
        evaluator = encode_physical_replay_snapshot(
            sim,
            view_kind="evaluator",
            run_id=self.run_id,
            episode_id=self.episode_id,
        )
        self.reported_snapshot_rows.append(reported)
        self.evaluator_snapshot_rows.append(evaluator)
        for row in (reported, evaluator):
            state = EventState(
                simulation_time=float(row["simulation_time"]),
                movement_tick=int(row["movement_tick"]),
                control_interval_index=int(
                    float(row["simulation_time"]) / sim.control_interval_seconds
                ),
                physical_state_checksum=str(row["physical_state_checksum"]),
            )
            self.record(
                "replay_snapshot",
                "trace_writer",
                {"view_kind": row["view_kind"]},
                sim,
                phase=EventPhase.REPLAY_SNAPSHOT,
                entity=("view", -1, str(row["view_kind"])),
                state=state,
            )

    def record_continuation_artifact(self, record: dict[str, str]) -> None:
        """Add one externally encoded continuation artifact."""
        if self.trace_level == "summary":
            return
        self.continuation_artifacts.append(dict(record))

    def snapshot_state(self) -> dict[str, Any]:
        """Return every buffer and the next ordering position."""
        ordered = self._ordered_events()
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "trace_level": self.trace_level,
            "events": tuple(event.as_dict() for event in ordered),
            "metric_rows": tuple(self.metric_rows),
            "reported_snapshot_rows": tuple(self.reported_snapshot_rows),
            "evaluator_snapshot_rows": tuple(self.evaluator_snapshot_rows),
            "continuation_artifacts": tuple(self.continuation_artifacts),
            "output_append_positions": {
                "events": len(ordered),
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
        if state.get("trace_level", self.trace_level) != self.trace_level:
            raise ValueError("the trace level is incompatible")
        self.events = [
            replace(EventRecord(**item), event_sequence=-1) for item in state["events"]
        ]
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
        self,
        summary: dict[str, Any],
        model_reference: dict[str, Any] | None = None,
        *,
        performance: Mapping[str, Any] | None = None,
    ) -> str:
        """Publish all formal files and then publish performance data."""
        self._validate_summary(summary)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict[str, Any]] = []
        if self.resolved is not None:
            artifacts.append(
                self._write_text_artifact(
                    "config.resolved.yaml",
                    yaml.safe_dump(
                        self.resolved.model_dump(mode="json"),
                        sort_keys=True,
                    ),
                    "avalanche.resolved_configuration",
                    1,
                )
            )
            artifacts.append(
                self._write_json_artifact(
                    "metadata.json",
                    self._run_metadata(),
                    "avalanche.run_metadata",
                    1,
                )
            )
        if self.trace_level != "summary":
            events = b"".join(
                _compact_json(event.as_dict()) + b"\n"
                for event in self._ordered_events()
            )
            artifacts.append(
                self._write_bytes_artifact(
                    "events.jsonl",
                    events,
                    "avalanche.event_trace",
                    EVENT_SCHEMA_VERSION,
                )
            )
        artifacts.append(
            self._write_parquet_artifact(
                "metrics.parquet",
                self.metric_rows,
                "avalanche.control_metrics",
                METRICS_VERSION,
            )
        )
        if self.trace_level != "summary":
            artifacts.extend(
                (
                    self._write_parquet_artifact(
                        REPORTED_REPLAY_FILENAME,
                        self.reported_snapshot_rows,
                        PHYSICAL_REPLAY_ARTIFACT_TYPE,
                        PHYSICAL_REPLAY_SCHEMA_VERSION,
                    ),
                    self._write_parquet_artifact(
                        EVALUATOR_REPLAY_FILENAME,
                        self.evaluator_snapshot_rows,
                        PHYSICAL_REPLAY_ARTIFACT_TYPE,
                        PHYSICAL_REPLAY_SCHEMA_VERSION,
                    ),
                )
            )
            for record in self.continuation_artifacts:
                path = self.output_dir / record["path"]
                artifacts.append(
                    _artifact_record(
                        path,
                        CONTINUATION_ARTIFACT_TYPE,
                        CONTINUATION_SCHEMA_VERSION,
                    )
                )
        artifacts.append(
            self._write_json_artifact(
                "summary.json",
                summary,
                "avalanche.run_summary",
                SUMMARY_SCHEMA_VERSION,
            )
        )
        reference = model_reference or {
            "model_kind": None,
            "model_path": None,
            "model_revision": None,
        }
        artifacts.append(
            self._write_json_artifact(
                "model-reference.json",
                reference,
                "avalanche.model_reference",
                1,
            )
        )
        artifacts.sort(key=lambda item: item["path"].encode("utf-8"))
        manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "trace_level": self.trace_level,
            "artifacts": artifacts,
        }
        manifest_bytes = _pretty_json(manifest)
        manifest_path = self.output_dir / RUN_MANIFEST_FILENAME
        atomic_write_bytes(manifest_path, manifest_bytes)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        sidecar = f"{digest}  {RUN_MANIFEST_FILENAME}\n"
        atomic_write_text(self.output_dir / RUN_MANIFEST_SIDECAR_FILENAME, sidecar)
        self.research_manifest_sha256 = digest
        if performance is not None:
            self._write_performance(performance, digest)
        return digest

    def _ordered_events(self) -> list[EventRecord]:
        indexed = list(enumerate(self.events))
        indexed.sort(key=lambda item: (_event_order(item[1]), item[0]))
        return [
            replace(event, event_sequence=sequence)
            for sequence, (_, event) in enumerate(indexed)
        ]

    def _validate_summary(self, summary: dict[str, Any]) -> None:
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
        if "performance" in summary or "created_at" in summary:
            raise ValueError(
                "a deterministic summary must not contain performance data"
            )

    def _run_metadata(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "seed": self.seed,
                "git_commit": _git_commit(),
                "dependency_lock_sha256": hashlib.sha256(
                    (REPO_ROOT / "uv.lock").read_bytes()
                ).hexdigest(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            }
        )
        if self.resolved is not None:
            metadata.update(
                {
                    "resolved_configuration_sha256": (
                        self.resolved.resolved_configuration_sha256
                    ),
                    "scientific_configuration_sha256": (
                        self.resolved.scientific_configuration_sha256
                    ),
                }
            )
        if "created_at" in metadata or "wall_clock_seconds" in metadata:
            raise ValueError("formal metadata must not contain performance values")
        return metadata

    def _write_performance(
        self,
        performance: Mapping[str, Any],
        manifest_sha256: str,
    ) -> None:
        output = self.performance_root / self.run_id
        value = {
            **dict(performance),
            "schema_version": PERFORMANCE_SCHEMA_VERSION,
            "reproducible": False,
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "run_manifest_path": RUN_MANIFEST_FILENAME,
            "research_manifest_sha256": manifest_sha256,
        }
        content = _pretty_json(value)
        path = output / "performance.json"
        atomic_write_bytes(path, content)
        digest = hashlib.sha256(content).hexdigest()
        atomic_write_text(
            output / "performance.json.sha256",
            f"{digest}  performance.json\n",
        )

    def _write_json_artifact(
        self,
        name: str,
        value: Mapping[str, Any],
        artifact_type: str,
        schema_version: int,
    ) -> dict[str, Any]:
        return self._write_bytes_artifact(
            name,
            _pretty_json(value),
            artifact_type,
            schema_version,
        )

    def _write_text_artifact(
        self,
        name: str,
        value: str,
        artifact_type: str,
        schema_version: int,
    ) -> dict[str, Any]:
        return self._write_bytes_artifact(
            name,
            value.encode("utf-8"),
            artifact_type,
            schema_version,
        )

    def _write_bytes_artifact(
        self,
        name: str,
        value: bytes,
        artifact_type: str,
        schema_version: int,
    ) -> dict[str, Any]:
        path = self.output_dir / name
        atomic_write_bytes(path, value)
        return _artifact_record(path, artifact_type, schema_version)

    def _write_parquet_artifact(
        self,
        name: str,
        rows: list[dict[str, Any]],
        artifact_type: str,
        schema_version: int,
    ) -> dict[str, Any]:
        path = self.output_dir / name
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.output_dir,
            prefix=f".{name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            pq.write_table(pa.Table.from_pylist(rows), temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fsync_directory(self.output_dir)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return _artifact_record(path, artifact_type, schema_version)


def _event_order(event: EventRecord) -> tuple[Any, ...]:
    """Return the frozen formal event ordering tuple."""
    return (
        event.simulation_time,
        event.phase_code,
        event.event_type.encode("utf-8"),
        event.entity_kind.encode("utf-8"),
        event.entity_index,
        event.entity_id.encode("utf-8"),
    )


def _artifact_record(
    path: Path,
    artifact_type: str,
    schema_version: int,
) -> dict[str, Any]:
    """Return one exact persisted file identity."""
    content = path.read_bytes()
    return {
        "path": path.name,
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "size_bytes": len(content),
        "artifact_sha256": hashlib.sha256(content).hexdigest(),
    }


def _compact_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pretty_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
