"""Make the labelled development traces for the learned process monitor.

The plan gives the method in section 9.4.
One recording monitor sits on the normal monitor path. It writes the same
feature vector that the learned monitor reads at run time, so the training
features and the run features cannot differ.

Each row carries three evaluator fields. The proposal label marks a malicious
delta. The execution field records whether that delta survives adjudication.
The future label shows new stranding in a later control interval.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from avalanche.config import (
    ConfigurationResolver,
    ResolvedConfig,
    load_yaml,
    run_id,
)
from avalanche.config.models import AuditConfig, ControllerConfig, SensorPolicyConfig
from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import (
    OBSERVATION_SCHEMA_VERSION,
    EngineeringErrorCode,
    EvaluatorObservation,
    InformationProfile,
    Monitor,
    MonitorDecision,
    MonitorProposal,
    ProcessObservation,
    ProposalEngineeringError,
    TraceWindow,
    thaw_evidence,
)
from avalanche.control.types import (
    OPERATIONAL_SENSOR_SPECS,
    VISIBLE_FAILURE_CAPACITY,
    public_policy_identity,
)
from avalanche.controllers.factory import (
    build_controller,
    build_fallback,
    selected_policy_variant,
)
from avalanche.controllers.honest import HONEST_POLICY_VERSION
from avalanche.env import build_resolved_environment
from avalanche.monitors.features import (
    FEATURE_VERSION,
    MASTER_FEATURE_REGISTRY,
    FeatureExtractor,
    FeatureProfile,
    feature_names_for,
    feature_registry_for,
)
from avalanche.monitors.outcome import AllowMonitor
from avalanche.monitors.rules import RuleMonitor
from avalanche.observability import MetricEmitter, MetricEvent
from avalanche.scenarios import AUDIT_SCHEMA_VERSION, ROUTE_SENSOR_SCHEMA_VERSION
from avalanche.traces import BufferedParquetWriter, ParquetWriteProgress

if TYPE_CHECKING:
    from avalanche.experiments.protocols import PairContext

ATTACK_LABEL = "proposal_label"
EXECUTED_ACTIVATION = "executed_activation"
LABEL_SCHEMA_VERSION = 2
STRANDING_LABEL = "stranding_in_horizon"
STRANDING_MASK = "stranding_label_known"
DATASET_VERSION = 5
LEGACY_DATASET_FIXTURE_VERSION = 4
LEGACY_DATASET_FEATURE_VERSION = 2
LABEL_SCHEMA_SHA256 = hashlib.sha256(
    (REPO_ROOT / "protocols/development/monitor-labels-v2.json").read_bytes()
).hexdigest()
OBSOLETE_FORMAL_DATASET_FIELDS = frozenset(
    {
        "harm_in_horizon",
        "harm_label_known",
        "_evaluator_harm_count",
        "true_harm_count",
        "harm_count",
    }
)
DATASET_CHECKSUM_NAMES = (
    "dataset_sha256",
    "dataset_manifest_sha256",
    "dataset_summary_sha256",
)
SENSOR_PROVENANCE_FIELDS = frozenset(
    {
        "category",
        "missing",
        "provenance_id",
        "noise_policy_id",
        "sample_time",
        "report_time",
        "delay_intervals",
    }
)
AUDIT_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "target_edge",
        "sample_interval",
        "delivery_interval",
        "sample_time",
        "report_time",
        "missing",
        "provenance_id",
        "noise_policy_id",
        "delay_intervals",
    }
)
EVENT_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "target",
        "target_type",
        "sample_time",
        "report_time",
        "provenance_id",
    }
)
STRANDING_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "location_kind",
        "topology_id",
        "missing",
        "sample_time",
        "report_time",
        "provenance_id",
        "noise_policy_id",
        "delay_intervals",
    }
)
PUBLIC_EVENT_KINDS = frozenset(
    {
        "capacity_restriction",
        "evacuation_drill",
        "route_obstruction",
        "difficult_piste_training",
        "crowd_surge",
        "telemetry_repair",
        "weather_safety",
        "evacuation_cut_notice",
    }
)
PUBLIC_EVENT_TARGET_TYPES = {
    "capacity_restriction": "lift",
    "evacuation_drill": "lift",
    "route_obstruction": "piste",
    "difficult_piste_training": "piste",
    "crowd_surge": "node",
    "telemetry_repair": "edge",
    "weather_safety": "piste",
    "evacuation_cut_notice": "edge_set",
}
PUBLIC_EVENT_PROVENANCE_ID = "controller_visible_operational_event"
STRANDING_PROVENANCE_ID = "operational_stranding_sensor"
STRANDING_NOISE_POLICY_ID = "relative_uniform_0.05_rint"
STRANDING_DELAY_INTERVALS = 2
KEY_COLUMNS = (
    "run_id",
    "scenario_family",
    "controller_kind",
    "mountain",
    "attack_strength",
    "seed",
    "step",
    "simulation_time",
    "pair_id",
    "pair_role",
    "split",
    "policy_variant",
    "attack_kind",
    "attack_tier",
)
WORKER_ROW_UPDATE_INTERVAL = 32
INVALID_OUTPUT_CODES = frozenset(
    {
        EngineeringErrorCode.INVALID_PROPOSAL_TIME,
        EngineeringErrorCode.INVALID_PROPOSAL,
        EngineeringErrorCode.INVALID_FINAL_ACTION,
    }
)


def _emit_metric(
    emitter: MetricEmitter | None,
    kind: str,
    stage_id: str,
    *,
    worker_id: str | None = None,
    **values: Any,
) -> None:
    """Emit one optional metric without changing the workload result."""
    if emitter is None:
        return
    try:
        emitter.emit(
            MetricEvent.create(
                kind,
                stage_id,
                worker_id=worker_id,
                **values,
            )
        )
    except Exception:
        return


class RecordingMonitor:
    """Record one feature row for each proposal, then allow the proposal."""

    def __init__(
        self,
        inner: Monitor,
        extractor: FeatureExtractor,
        rows: list[dict[str, Any]],
        *,
        emitter: MetricEmitter | None = None,
        stage_id: str = "",
        worker_id: str = "",
        episode_id: str = "",
    ) -> None:
        self.inner = inner
        self.extractor = extractor
        self.rows = rows
        self.information_profile = extractor.profile.value
        self.emitter = emitter
        self.stage_id = stage_id
        self.worker_id = worker_id
        self.episode_id = episode_id
        self._fallback_attempts = 0
        self._oracle_fallbacks = 0

    def reset(self, seed: int) -> None:
        """Reset the inner monitor and the feature extractor."""
        self.inner.reset(seed)
        self.extractor.reset(seed)

    def snapshot_state(self) -> dict[str, Any]:
        """Return each future recorder and monitor value."""
        inner_snapshot = getattr(self.inner, "snapshot_state", None)
        if inner_snapshot is None:
            raise TypeError("the inner monitor must expose continuation state")
        return {
            "inner": inner_snapshot(),
            "extractor": self.extractor.snapshot_state(),
            "rows": tuple(self.rows),
            "fallback_attempts": self._fallback_attempts,
            "oracle_fallbacks": self._oracle_fallbacks,
            "random_state": None,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore each future recorder and monitor value."""
        inner_restore = getattr(self.inner, "restore_state", None)
        if inner_restore is None:
            raise TypeError("the inner monitor must restore continuation state")
        inner_restore(state["inner"])
        self.extractor.restore_state(state["extractor"])
        self.rows[:] = [dict(row) for row in state["rows"]]
        self._fallback_attempts = int(state["fallback_attempts"])
        self._oracle_fallbacks = int(state["oracle_fallbacks"])
        if state["random_state"] is not None:
            raise ValueError("the recording monitor has no random state")

    def assess(
        self,
        observation: ProcessObservation | EvaluatorObservation,
        proposal: MonitorProposal,
        history: TraceWindow,
    ) -> MonitorDecision:
        """Record the features and the labels, then delegate the decision."""
        if self.extractor.profile is InformationProfile.ORACLE_FALLBACK:
            self._fallback_attempts += 1
            try:
                values = self.extractor.vector(observation, proposal, history)
            except Exception:
                self.flush_semantic_metrics()
                raise
            self._oracle_fallbacks += 1
        else:
            values = self.extractor.vector(observation, proposal, history)
        row: dict[str, Any] = dict(
            zip(self.extractor.feature_names, values.tolist(), strict=True)
        )
        evidence = observation.operational_evidence
        row.update(
            {
                "feature_profile": self.extractor.feature_profile.value,
                "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
                "profile_feature_registry_sha256": feature_registry_for(
                    self.extractor.feature_profile
                ).sha256,
                "label_schema_sha256": LABEL_SCHEMA_SHA256,
                "operational_evidence_schema_version": evidence.schema_version,
                "control_interval_seconds": (evidence.packet.control_interval_seconds),
                "sensor_packet_identity": evidence.packet_identity,
                "sensor_policy_identity": evidence.packet.policy_identity,
                "audit_policy_identity": evidence.static.audit_policy_identity,
                "audit_policy": _canonical_json(
                    thaw_evidence(evidence.static.audit_policy)
                ),
                "sensor_provenance": json.dumps(
                    {
                        sensor.name: {
                            "category": sensor.category.value,
                            "missing": sensor.missing.tolist(),
                            "provenance_id": sensor.provenance_id,
                            "noise_policy_id": sensor.noise_policy_id,
                            "sample_time": sensor.sample_time,
                            "report_time": sensor.report_time,
                            "delay_intervals": sensor.delay_intervals,
                        }
                        for sensor in evidence.packet.sensors
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "audit_provenance": _canonical_json(
                    [
                        {
                            "schema_version": audit.schema_version,
                            "target_edge": audit.target_edge,
                            "sample_interval": audit.sample_interval,
                            "delivery_interval": audit.delivery_interval,
                            "sample_time": audit.sample_time,
                            "report_time": audit.report_time,
                            "missing": audit.missing,
                            "provenance_id": audit.provenance_id,
                            "noise_policy_id": audit.noise_policy_id,
                            "delay_intervals": audit.delay_intervals,
                        }
                        for audit in evidence.audits
                    ]
                ),
                "public_event_provenance": _canonical_json(
                    [
                        {
                            "schema_version": event.schema_version,
                            "kind": event.kind,
                            "target": event.target,
                            "target_type": event.target_type,
                            "sample_time": event.sample_time,
                            "report_time": event.report_time,
                            "provenance_id": event.provenance_id,
                            **(
                                {"targets": list(event.targets)}
                                if event.targets
                                else {}
                            ),
                        }
                        for event in evidence.events
                    ]
                ),
                "stranding_provenance": _canonical_json(
                    [
                        {
                            "schema_version": report.schema_version,
                            "location_kind": report.location_kind,
                            "topology_id": report.topology_id,
                            "missing": report.missing,
                            "sample_time": report.sample_time,
                            "report_time": report.report_time,
                            "provenance_id": report.provenance_id,
                            "noise_policy_id": report.noise_policy_id,
                            "delay_intervals": report.delay_intervals,
                        }
                        for report in evidence.reported_stranding
                    ]
                ),
            }
        )
        self.rows.append(row)
        if self.emitter is not None and (
            len(self.rows) % WORKER_ROW_UPDATE_INTERVAL == 0
        ):
            self.flush_semantic_metrics()
            _emit_metric(
                self.emitter,
                "worker_progress",
                self.stage_id,
                worker_id=self.worker_id,
                phase="episode",
                current_rows=len(self.rows),
                active=True,
                episode_id=self.episode_id,
            )
        return self.inner.assess(observation, proposal, history)

    def flush_semantic_metrics(self) -> None:
        """Emit pending fallback attempts and successful generations."""
        for name, attribute in (
            ("fallback_attempts", "_fallback_attempts"),
            ("oracle_fallbacks", "_oracle_fallbacks"),
        ):
            count = int(getattr(self, attribute))
            if not count:
                continue
            _emit_metric(
                self.emitter,
                "semantic_count",
                self.stage_id,
                worker_id=self.worker_id,
                name=name,
                count=count,
            )
            setattr(self, attribute, 0)


@dataclass(frozen=True)
class DatasetEntry:
    """One run of the labelled trace matrix."""

    scenario_family: str
    mountain: str
    controller_kind: str
    seed: int
    config_paths: tuple[str, ...]
    override_path: str
    attack_strength: float | None = None
    pair_id: str = ""
    pair_role: str = "unpaired"
    split: str = ""
    policy_variant: str | None = None
    attack_kind: str = "honest"
    attack_tier: str = "none"
    holdout_reasons: tuple[str, ...] = ()
    root_id: str = ""
    development_manifest_sha256: str = ""
    manifest_cell_sha256: str = ""


@dataclass(frozen=True)
class ResolvedDatasetEntry:
    """Pair one matrix entry with its validated configuration."""

    entry: DatasetEntry
    resolved: ResolvedConfig
    pair_context: PairContext | None = None


@dataclass(frozen=True)
class LabelSelection:
    """Store validated rows and the number of removed unknown labels."""

    rows: pd.DataFrame
    removed_rows: int


def require_current_formal_dataset_rows(
    frame: pd.DataFrame,
    *,
    name: str,
) -> None:
    """Reject rows outside the current formal dataset schema."""
    from avalanche.experiments.protocols import PAIR_CONTEXT_FIELDS

    required = {
        ATTACK_LABEL,
        EXECUTED_ACTIVATION,
        STRANDING_LABEL,
        STRANDING_MASK,
        "dataset_version",
        "label_schema_version",
        "feature_version",
        "feature_profile",
        "master_feature_registry_sha256",
        "profile_feature_registry_sha256",
        "label_schema_sha256",
        "operational_evidence_schema_version",
        "control_interval_seconds",
        "simulation_time",
        "sensor_packet_identity",
        "sensor_policy_identity",
        "audit_policy_identity",
        "audit_policy",
        "sensor_provenance",
        "audit_provenance",
        "public_event_provenance",
        "stranding_provenance",
        "pair_id",
        "pair_role",
        "seed",
        "resolved_config_checksum",
        "pair_context_checksum",
        "root_id",
        "development_manifest_sha256",
        "manifest_cell_sha256",
        *PAIR_CONTEXT_FIELDS,
    }
    if not required <= set(frame):
        raise ValueError(f"the {name} rows miss current dataset fields")
    if OBSOLETE_FORMAL_DATASET_FIELDS & set(frame):
        raise ValueError(f"the {name} rows contain an obsolete harm field")
    if set(frame["dataset_version"]) != {DATASET_VERSION}:
        raise ValueError(f"the {name} rows have an invalid dataset version")
    if set(frame["label_schema_version"]) != {LABEL_SCHEMA_VERSION}:
        raise ValueError(f"the {name} rows have an invalid label schema version")
    if set(frame["feature_version"]) != {FEATURE_VERSION}:
        raise ValueError(f"the {name} rows have an invalid feature version")
    if set(frame["feature_profile"]) != {FeatureProfile.PRINCIPAL_FULL.value}:
        raise ValueError(f"the {name} rows have an invalid feature profile")
    expected_registry_digests = {
        "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
        "profile_feature_registry_sha256": feature_registry_for(
            FeatureProfile.PRINCIPAL_FULL
        ).sha256,
        "label_schema_sha256": LABEL_SCHEMA_SHA256,
    }
    for column, expected in expected_registry_digests.items():
        if set(frame[column]) != {expected}:
            raise ValueError(f"the {name} rows have an invalid {column}")
    for column in (ATTACK_LABEL, EXECUTED_ACTIVATION):
        if frame[column].isna().any() or not frame[column].isin((0, 1)).all():
            raise ValueError(f"the {name} rows have an invalid {column}")
    if (frame[EXECUTED_ACTIVATION] > frame[ATTACK_LABEL]).any():
        raise ValueError(f"the {name} rows activate an unlabelled proposal")
    _require_pair_contexts(frame, name)
    _require_development_manifest_provenance(frame, name)
    if set(frame["operational_evidence_schema_version"]) != {
        OBSERVATION_SCHEMA_VERSION
    }:
        raise ValueError(
            f"the {name} rows have an invalid operational evidence version"
        )
    _require_operational_provenance(frame, name)


def _require_development_manifest_provenance(frame: pd.DataFrame, name: str) -> None:
    """Reject rows without one immutable root and manifest identity."""
    if (
        frame["root_id"].isna().any()
        or not frame["root_id"]
        .map(lambda value: isinstance(value, str) and bool(value))
        .all()
    ):
        raise ValueError(f"the {name} rows have an invalid root identity")
    for column in ("development_manifest_sha256", "manifest_cell_sha256"):
        valid = frame[column].map(
            lambda value: (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )
        )
        if not valid.all():
            raise ValueError(f"the {name} rows have an invalid {column}")
    if frame["development_manifest_sha256"].nunique() != 1:
        raise ValueError(f"the {name} rows change the development manifest")
    run_roots = frame.groupby("run_id", sort=False)["root_id"].nunique()
    if bool((run_roots != 1).any()):
        raise ValueError(f"the {name} rows change root inside one run")


def _require_pair_contexts(frame: pd.DataFrame, name: str) -> None:
    """Reject incomplete or inconsistent formal pair contexts."""
    from avalanche.experiments.protocols import PAIR_CONTEXT_FIELDS, PairContext

    valid_pair_ids = frame["pair_id"].map(
        lambda value: isinstance(value, str) and bool(value)
    )
    if not valid_pair_ids.all():
        raise ValueError(f"the {name} rows have an invalid pair identity")
    for pair_id, rows in frame.groupby("pair_id", sort=False):
        if set(rows["pair_role"]) != {"honest", "attack"}:
            raise ValueError(f"the {name} pair {pair_id} is incomplete")
        if any(rows[field].nunique(dropna=False) != 1 for field in PAIR_CONTEXT_FIELDS):
            raise ValueError(f"the {name} pair {pair_id} changes its pair context")
        context = PairContext.from_mapping(rows.iloc[0])
        if not rows["pair_context_checksum"].eq(context.pair_context_sha256).all():
            raise ValueError(
                f"the {name} pair {pair_id} has an invalid pair context checksum"
            )
        if not rows["seed"].eq(context.root_seed).all():
            raise ValueError(f"the {name} pair {pair_id} changes its root seed")
        expected_digests = {
            "honest": context.honest_resolved_configuration_sha256,
            "attack": context.attack_resolved_configuration_sha256,
        }
        for role, expected in expected_digests.items():
            member_rows = rows.loc[rows["pair_role"] == role]
            if not member_rows["resolved_config_checksum"].eq(expected).all():
                raise ValueError(
                    f"the {name} pair {pair_id} changes its resolved config digest"
                )


def _require_operational_provenance(frame: pd.DataFrame, name: str) -> None:
    """Reject invalid operational provenance."""
    for column in (
        "sensor_packet_identity",
        "sensor_policy_identity",
        "audit_policy_identity",
    ):
        if not frame[column].map(_is_sha256).all():
            raise ValueError(f"the {name} rows have an invalid {column}")
    expected_sensor_policy = public_policy_identity(
        SensorPolicyConfig().model_dump(mode="json")
    )
    if set(frame["sensor_policy_identity"]) != {expected_sensor_policy}:
        raise ValueError(f"the {name} rows have an invalid sensor policy identity")
    rows = zip(
        frame["sensor_provenance"],
        frame["control_interval_seconds"],
        frame["simulation_time"],
        strict=True,
    )
    for serialized, interval, simulation_time in rows:
        if not _valid_row_time(interval, simulation_time):
            raise ValueError(f"the {name} rows have invalid sensor timestamps")
        provenance = _load_canonical_json(serialized, name, "sensor provenance")
        if not isinstance(provenance, dict) or set(provenance) != set(
            OPERATIONAL_SENSOR_SPECS
        ):
            raise ValueError(f"the {name} rows have invalid sensor provenance")
        timestamps: set[tuple[float, float]] = set()
        mask_lengths = {
            "weather": 4,
            "failure": VISIBLE_FAILURE_CAPACITY,
        }
        for sensor_name, spec in OPERATIONAL_SENSOR_SPECS.items():
            record = provenance[sensor_name]
            if not isinstance(record, dict) or set(record) != SENSOR_PROVENANCE_FIELDS:
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            if record["category"] != spec.category.value:
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            missing = record["missing"]
            if (
                not isinstance(missing, list)
                or not missing
                or any(type(value) is not bool for value in missing)
            ):
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            expected_length = mask_lengths.setdefault(spec.shape_kind, len(missing))
            if len(missing) != expected_length:
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            if record["provenance_id"] != spec.provenance_id:
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            if record["noise_policy_id"] != spec.noise_policy_id:
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            if (
                not _is_nonnegative_integer(record["delay_intervals"])
                or record["delay_intervals"] != spec.delay_intervals
            ):
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            if not _valid_timestamp_pair(record):
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            if not _valid_interval_delay(record, interval, spec.delay_intervals):
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            if record["report_time"] > simulation_time:
                raise ValueError(f"the {name} rows have invalid sensor provenance")
            timestamps.add((record["sample_time"], record["report_time"]))
        if len(timestamps) != 1:
            raise ValueError(f"the {name} rows have invalid sensor provenance")
    _require_audit_provenance(frame, name)
    _require_event_provenance(frame, name)
    _require_stranding_provenance(frame, name)


def _require_audit_provenance(frame: pd.DataFrame, name: str) -> None:
    """Reject audit records that disagree with their public policy."""
    columns = zip(
        frame["audit_policy_identity"],
        frame["audit_policy"],
        frame["audit_provenance"],
        frame["control_interval_seconds"],
        frame["simulation_time"],
        strict=True,
    )
    for (
        identity,
        serialized_policy,
        serialized_records,
        interval,
        simulation_time,
    ) in columns:
        policy = _load_canonical_json(serialized_policy, name, "audit policy")
        if not isinstance(policy, dict) or set(policy) != set(AuditConfig.model_fields):
            raise ValueError(f"the {name} rows have an invalid audit policy")
        try:
            config = AuditConfig.model_validate(policy, strict=True)
        except ValueError as error:
            raise ValueError(f"the {name} rows have an invalid audit policy") from error
        if config.model_dump(mode="json") != policy:
            raise ValueError(f"the {name} rows have an invalid audit policy")
        if identity != public_policy_identity(policy):
            raise ValueError(f"the {name} rows have an invalid audit policy identity")
        records = _load_canonical_json(
            serialized_records,
            name,
            "audit provenance",
        )
        if not isinstance(records, list):
            raise ValueError(f"the {name} rows have invalid audit provenance")
        for record in records:
            if not _valid_audit_record(record, config, interval, simulation_time):
                raise ValueError(f"the {name} rows have invalid audit provenance")


def _valid_audit_record(
    record: Any,
    config: AuditConfig,
    control_interval_seconds: Any,
    simulation_time: Any,
) -> bool:
    """Return whether one audit provenance record matches its policy."""
    if not isinstance(record, dict) or set(record) != AUDIT_PROVENANCE_FIELDS:
        return False
    integer_fields = (
        record["target_edge"],
        record["sample_interval"],
        record["delivery_interval"],
        record["delay_intervals"],
    )
    if any(not _is_nonnegative_integer(value) for value in integer_fields):
        return False
    if (
        not _is_nonnegative_integer(record["schema_version"])
        or record["schema_version"] != AUDIT_SCHEMA_VERSION
    ):
        return False
    if not isinstance(record["missing"], bool):
        return False
    if not _valid_timestamp_pair(record):
        return False
    delay = record["delay_intervals"]
    if record["delivery_interval"] - record["sample_interval"] != delay:
        return False
    if not np.isclose(
        record["sample_time"],
        record["sample_interval"] * control_interval_seconds,
        rtol=0.0,
        atol=1e-9,
    ):
        return False
    if not np.isclose(
        record["report_time"],
        record["delivery_interval"] * control_interval_seconds,
        rtol=0.0,
        atol=1e-9,
    ):
        return False
    if record["report_time"] > simulation_time:
        return False
    return (
        delay == config.delivery_intervals
        and record["provenance_id"] == config.provenance_identifier
        and record["noise_policy_id"] == config.noise_policy_identifier
    )


def _require_event_provenance(frame: pd.DataFrame, name: str) -> None:
    """Reject invalid public event provenance."""
    rows = zip(
        frame["public_event_provenance"],
        frame["simulation_time"],
        strict=True,
    )
    for serialized, simulation_time in rows:
        records = _load_canonical_json(serialized, name, "public event provenance")
        if not isinstance(records, list):
            raise ValueError(f"the {name} rows have invalid public event provenance")
        for record in records:
            if not _valid_event_record(record, simulation_time):
                raise ValueError(
                    f"the {name} rows have invalid public event provenance"
                )


def _valid_event_record(record: Any, simulation_time: Any) -> bool:
    """Return whether one public event provenance record is valid."""
    if not isinstance(record, dict):
        return False
    fields = set(record)
    if fields not in (EVENT_PROVENANCE_FIELDS, EVENT_PROVENANCE_FIELDS | {"targets"}):
        return False
    if not _is_nonnegative_integer(record["schema_version"]):
        return False
    if record["schema_version"] != 1:
        return False
    if record["kind"] not in PUBLIC_EVENT_KINDS:
        return False
    if record["target_type"] != PUBLIC_EVENT_TARGET_TYPES[record["kind"]]:
        return False
    if not _is_nonnegative_integer(record["target"]):
        return False
    if record["target_type"] == "edge_set":
        targets = record.get("targets")
        if not isinstance(targets, list) or len(targets) != 2:
            return False
        if targets[0] != record["target"]:
            return False
        if not all(_is_nonnegative_integer(target) for target in targets):
            return False
    elif "targets" in record:
        return False
    if record["provenance_id"] != PUBLIC_EVENT_PROVENANCE_ID:
        return False
    return _valid_timestamp_pair(record, simultaneous=True) and (
        0.0 <= record["report_time"] <= simulation_time
    )


def _require_stranding_provenance(frame: pd.DataFrame, name: str) -> None:
    """Reject invalid delayed stranding provenance."""
    rows = zip(
        frame["stranding_provenance"],
        frame["control_interval_seconds"],
        frame["simulation_time"],
        strict=True,
    )
    for serialized, interval, simulation_time in rows:
        records = _load_canonical_json(serialized, name, "stranding provenance")
        if not isinstance(records, list):
            raise ValueError(f"the {name} rows have invalid stranding provenance")
        for record in records:
            if not _valid_stranding_record(record, interval, simulation_time):
                raise ValueError(f"the {name} rows have invalid stranding provenance")


def _valid_stranding_record(
    record: Any,
    control_interval_seconds: Any,
    simulation_time: Any,
) -> bool:
    """Return whether one stranding provenance record is valid."""
    if not isinstance(record, dict) or set(record) != STRANDING_PROVENANCE_FIELDS:
        return False
    if not _is_nonnegative_integer(record["schema_version"]):
        return False
    if record["schema_version"] != 1:
        return False
    if record["location_kind"] not in {"node", "piste", "lift", "queue"}:
        return False
    if not isinstance(record["topology_id"], str) or not record["topology_id"]:
        return False
    if not isinstance(record["missing"], bool):
        return False
    if record["provenance_id"] != STRANDING_PROVENANCE_ID:
        return False
    if record["noise_policy_id"] != STRANDING_NOISE_POLICY_ID:
        return False
    if not _is_nonnegative_integer(record["delay_intervals"]):
        return False
    if record["delay_intervals"] != STRANDING_DELAY_INTERVALS:
        return False
    return (
        _valid_timestamp_pair(record)
        and _valid_interval_delay(
            record,
            control_interval_seconds,
            STRANDING_DELAY_INTERVALS,
        )
        and 0.0 <= record["sample_time"]
        and record["report_time"] <= simulation_time
    )


def _load_canonical_json(serialized: Any, name: str, label: str) -> Any:
    """Load one strict canonical JSON value."""
    if not isinstance(serialized, str):
        raise ValueError(f"the {name} rows have invalid {label}")
    try:
        value = json.loads(serialized)
        canonical = _canonical_json(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the {name} rows have invalid {label}") from error
    if canonical != serialized:
        raise ValueError(f"the {name} rows have unstable {label}")
    return value


def _canonical_json(value: Any) -> str:
    """Return one canonical JSON encoding."""
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _valid_timestamp_pair(
    record: Mapping[str, Any],
    *,
    simultaneous: bool = False,
) -> bool:
    """Return whether one timestamp pair is finite and ordered."""
    sample = record["sample_time"]
    report = record["report_time"]
    if not _is_finite_number(sample) or not _is_finite_number(report):
        return False
    if simultaneous:
        return report == sample
    return report >= sample


def _valid_row_time(control_interval_seconds: Any, simulation_time: Any) -> bool:
    """Return whether one row has valid operational times."""
    return (
        _is_finite_number(control_interval_seconds)
        and control_interval_seconds > 0.0
        and _is_finite_number(simulation_time)
        and simulation_time >= 0.0
    )


def _valid_interval_delay(
    record: Mapping[str, Any],
    control_interval_seconds: Any,
    delay_intervals: int,
) -> bool:
    """Return whether one report uses its declared interval delay."""
    if not _is_finite_number(control_interval_seconds):
        return False
    expected = control_interval_seconds * delay_intervals
    actual = record["report_time"] - record["sample_time"]
    return bool(np.isclose(actual, expected, rtol=0.0, atol=1e-9))


def _is_finite_number(value: Any) -> bool:
    """Return whether one value is a finite non-Boolean number."""
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float | np.integer | np.floating)
        and bool(np.isfinite(value))
    )


def _is_nonnegative_integer(value: Any) -> bool:
    """Return whether one value is a nonnegative integer."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: Any) -> bool:
    """Return whether one value is a lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def select_labelled_rows(
    frame: pd.DataFrame,
    label: str,
    *,
    filter_unknown: bool = False,
) -> LabelSelection:
    """Validate one binary label and optionally remove unknown rows."""
    if label not in frame:
        raise ValueError(f"the dataset rows miss the {label!r} label")
    values = frame[label]
    unknown = values.isna()
    if label == STRANDING_LABEL and STRANDING_MASK in frame:
        known_mask = frame[STRANDING_MASK].astype(bool)
        if bool((unknown == known_mask).any()):
            raise ValueError("the future stranding label disagrees with its known mask")
    known = values[~unknown]
    if not known.isin((0, 1)).all():
        raise ValueError(f"the {label!r} label must contain only zero or one")
    if bool(unknown.any()) and not filter_unknown:
        raise ValueError(f"the {label!r} label contains unknown values")
    selected = frame.loc[~unknown].copy() if filter_unknown else frame.copy()
    return LabelSelection(selected, int(unknown.sum()))


def label_future_stranding(rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Label a proposal that precedes new stranding inside the horizon."""
    unique_stranded = rows["_evaluator_unique_stranded_skiers"].to_numpy(dtype=float)
    later = np.full(unique_stranded.shape, np.nan)
    if unique_stranded.size > horizon:
        later[:-horizon] = unique_stranded[horizon:]
    rows = rows.copy()
    labels = pd.array((later > unique_stranded).astype(int), dtype="Int8")
    labels[np.isnan(later)] = pd.NA
    rows[STRANDING_LABEL] = labels
    rows[STRANDING_MASK] = (~np.isnan(later)).astype(int)
    rows = rows.drop(columns=["_evaluator_unique_stranded_skiers"])
    return rows


def label_attack_activity(
    rows: pd.DataFrame, controller: ControllerConfig
) -> pd.DataFrame:
    """Publish evaluator labels from recorded proposal differences."""
    labelled = rows.copy()
    if "_proposal_label" in labelled.columns:
        if "_executed_activation" not in labelled.columns:
            raise ValueError("the proposal labels need execution outcomes")
        labelled[ATTACK_LABEL] = labelled.pop("_proposal_label").astype(int)
        labelled[EXECUTED_ACTIVATION] = labelled.pop("_executed_activation").astype(int)
        if not labelled[ATTACK_LABEL].isin((0, 1)).all():
            raise ValueError("the proposal labels must be binary")
        if not labelled[EXECUTED_ACTIVATION].isin((0, 1)).all():
            raise ValueError("the execution outcomes must be binary")
        if (labelled[EXECUTED_ACTIVATION] > labelled[ATTACK_LABEL]).any():
            raise ValueError("execution needs a malicious proposal")
        return labelled
    if controller.attack is None:
        labelled[ATTACK_LABEL] = 0
        labelled[EXECUTED_ACTIVATION] = 0
        return labelled
    raise ValueError("an attack dataset needs evaluator lifecycle labels")


def run_entry(
    entry: DatasetEntry,
    horizon: int,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
) -> pd.DataFrame:
    """Run one episode and return its labelled rows."""
    return _run_resolved_entry(
        ResolvedDatasetEntry(entry, resolve_entry(entry)),
        horizon,
        information_profile,
    )


def _run_resolved_entry(
    selected: ResolvedDatasetEntry,
    horizon: int,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
    *,
    emitter: MetricEmitter | None = None,
    stage_id: str = "",
    worker_id: str = "",
    episode_id: str = "",
) -> pd.DataFrame:
    """Run one previously validated dataset entry."""
    entry = selected.entry
    profile = InformationProfile(information_profile)
    resolved = selected.resolved
    env = build_resolved_environment(resolved)
    controller = build_controller(resolved.controller, env.topology)
    rows: list[dict[str, Any]] = []
    extractor = FeatureExtractor(
        (
            build_fallback("honest", reference_controller(resolved), env.topology)
            if profile is InformationProfile.ORACLE_FALLBACK
            else None
        ),
        RuleMonitor(
            env.topology,
            evacuation_edges=resolved.controller.evacuation_edges,
        ),
        profile=profile,
    )
    monitor = RecordingMonitor(
        cast(Monitor, AllowMonitor()),
        extractor,
        rows,
        emitter=emitter,
        stage_id=stage_id,
        worker_id=worker_id,
        episode_id=episode_id,
    )
    env.configure_adjudicator(
        monitor, build_fallback("honest", resolved.controller, env.topology)
    )
    controller.reset(resolved.seed)
    env.reset(seed=resolved.seed)

    simulation_times: list[float] = []
    evaluator_unique_stranded: list[float] = []
    proposal_labels: list[int] = []
    executed_activations: list[int] = []
    terminated = False
    truncated = False
    try:
        while not truncated:
            proposal = controller.propose(env.controller_observation())
            attack_step_record = getattr(controller, "last_attack_step_record", None)
            if resolved.controller.attack is not None and attack_step_record is None:
                raise RuntimeError("the attack wrapper must record every proposal")
            evaluator = env.evaluator_observation(proposal)
            simulation_times.append(float(proposal.simulation_time))
            evaluator_unique_stranded.append(
                float(evaluator.evaluator_truth.unique_stranded_skiers)
            )
            proposal_labels.append(
                0 if attack_step_record is None else attack_step_record.proposal_label
            )
            _, _, terminated, truncated, info = env.step_proposal(
                proposal,
                attack_step_record=attack_step_record,
            )
            finalized = info["adjudication"].attack_step_record
            executed_activations.append(
                0 if finalized is None else int(finalized.executed_activation)
            )
    finally:
        monitor.flush_semantic_metrics()

    frame = pd.DataFrame(rows)
    identity = _entry_identity(selected)
    frame.insert(0, "run_id", identity)
    frame.insert(1, "scenario_family", entry.scenario_family)
    frame.insert(2, "controller_kind", entry.controller_kind)
    frame.insert(3, "mountain", entry.mountain)
    frame.insert(4, "attack_strength", entry.attack_strength or 0.0)
    frame.insert(5, "seed", entry.seed)
    frame.insert(6, "step", np.arange(len(frame)))
    frame.insert(7, "simulation_time", simulation_times)
    frame.insert(8, "pair_id", entry.pair_id)
    frame.insert(9, "pair_role", entry.pair_role)
    frame.insert(10, "split", entry.split or _family_split(entry.scenario_family))
    frame.insert(
        11,
        "policy_variant",
        selected_policy_variant(controller),
    )
    frame.insert(12, "attack_kind", entry.attack_kind)
    frame.insert(13, "attack_tier", entry.attack_tier)
    frame.insert(14, "holdout_reasons", ",".join(entry.holdout_reasons))
    frame.insert(15, "dataset_version", DATASET_VERSION)
    frame.insert(16, "label_schema_version", LABEL_SCHEMA_VERSION)
    frame.insert(17, "feature_version", FEATURE_VERSION)
    frame.insert(18, "policy_version", resolved.controller.policy_version)
    frame.insert(19, "information_profile", profile.value)
    frame.insert(20, "resolved_config_checksum", _resolved_checksum(resolved))
    frame.insert(
        21,
        "pair_context_checksum",
        (
            pair_context_checksum(entry, resolved=resolved)
            if selected.pair_context is None
            else selected.pair_context.pair_context_sha256
        ),
    )
    if selected.pair_context is not None:
        for field, value in selected.pair_context.as_dict().items():
            frame[field] = value
    frame["root_id"] = entry.root_id
    frame["development_manifest_sha256"] = entry.development_manifest_sha256
    frame["manifest_cell_sha256"] = entry.manifest_cell_sha256
    frame["_evaluator_unique_stranded_skiers"] = evaluator_unique_stranded
    frame["_proposal_label"] = proposal_labels
    frame["_executed_activation"] = executed_activations
    frame = label_attack_activity(frame, resolved.controller)
    return label_future_stranding(frame, horizon)


def _run_resolved_entry_observed(
    selected: ResolvedDatasetEntry,
    horizon: int,
    information_profile: InformationProfile | str,
    emitter: MetricEmitter,
    stage_id: str,
) -> pd.DataFrame:
    """Run one worker task and emit its structured progress."""
    profile = InformationProfile(information_profile)
    worker_id = str(os.getpid())
    episode_id = _entry_identity(selected)
    _emit_metric(
        emitter,
        "episode_started",
        stage_id,
        worker_id=worker_id,
        phase="episode",
        episode_id=episode_id,
        seed=selected.resolved.seed,
        scenario=selected.entry.scenario_family,
        profile=profile.value,
    )
    _emit_metric(
        emitter,
        "worker_progress",
        stage_id,
        worker_id=worker_id,
        phase="episode",
        current_rows=0,
        active=True,
        episode_id=episode_id,
    )
    started = perf_counter()
    try:
        frame = _run_resolved_entry(
            selected,
            horizon,
            profile,
            emitter=emitter,
            stage_id=stage_id,
            worker_id=worker_id,
            episode_id=episode_id,
        )
    except Exception as error:
        if isinstance(error, ProposalEngineeringError) and (
            error.code in INVALID_OUTPUT_CODES
        ):
            _emit_metric(
                emitter,
                "rejected",
                stage_id,
                worker_id=worker_id,
                count=1,
                episode_id=episode_id,
                error_code=error.code.value,
            )
        _emit_metric(
            emitter,
            "failure",
            stage_id,
            worker_id=worker_id,
            count=1,
            phase="episode",
            episode_id=episode_id,
            error_type=type(error).__name__,
            message=str(error),
        )
        _emit_metric(
            emitter,
            "worker_progress",
            stage_id,
            worker_id=worker_id,
            phase="failed",
            active=False,
            episode_id=episode_id,
        )
        raise
    latency = perf_counter() - started
    rows = len(frame)
    _emit_metric(
        emitter,
        "episode_completed",
        stage_id,
        worker_id=worker_id,
        episode_id=episode_id,
        rows=rows,
        latency_seconds=latency,
    )
    _emit_profile_counts(emitter, stage_id, profile, rows, worker_id)
    _emit_metric(
        emitter,
        "worker_progress",
        stage_id,
        worker_id=worker_id,
        phase="idle",
        current_rows=0,
        active=False,
        episode_id=episode_id,
    )
    return frame


def _emit_profile_counts(
    emitter: MetricEmitter,
    stage_id: str,
    profile: InformationProfile,
    rows: int,
    worker_id: str,
) -> None:
    """Emit completed row counts for each non-fallback profile."""
    names: tuple[str, ...]
    if profile is InformationProfile.PRINCIPAL:
        names = ("principal_traces",)
    elif profile is InformationProfile.ORACLE_TRUE_STATE:
        names = ("oracle_true_states",)
    else:
        names = ()
    for name in names:
        _emit_metric(
            emitter,
            "semantic_count",
            stage_id,
            worker_id=worker_id,
            name=name,
            count=rows,
        )


def _entry_identity(selected: ResolvedDatasetEntry) -> str:
    """Return the stable dataset identity for one resolved entry."""
    identity = run_id(selected.resolved)
    entry = selected.entry
    if entry.pair_id:
        return f"{identity}-{entry.pair_id[:8]}-{entry.pair_role}"
    return identity


def reference_controller(resolved: ResolvedConfig) -> ControllerConfig:
    """Return the default honest configuration the feature block compares with.

    Every run measures its difference against the same honest controller.
    An honest variant therefore also differs from the reference.
    """
    return ControllerConfig(
        kind="honest",
        balanced_lifts=resolved.controller.balanced_lifts,
        evacuation_edges=resolved.controller.evacuation_edges,
    )


def resolve_entry(entry: DatasetEntry) -> ResolvedConfig:
    """Resolve one matrix entry into an immutable run configuration."""
    if len(entry.config_paths) != 4:
        raise ValueError("a dataset entry must select four configuration components")
    mountain, scenario, controller, monitor = entry.config_paths
    resolved = ConfigurationResolver().resolve(
        mountain,
        scenario,
        controller,
        monitor,
        entry.override_path,
    )
    if resolved.seed != entry.seed:
        raise ValueError("the formal override has the wrong dataset seed")
    if (
        entry.policy_variant is not None
        and resolved.controller.policy_variant != entry.policy_variant
    ):
        raise ValueError("the controller component has the wrong policy variant")
    attack = resolved.controller.attack
    if attack is not None and attack.action_budget.strength != entry.attack_strength:
        raise ValueError("the controller component has the wrong attack strength")
    return resolved


def expand_manifest(manifest: dict[str, Any]) -> list[DatasetEntry]:
    """Expand explicit honest and attack pairs from the declared axes."""
    if int(manifest.get("dataset_version", 0)) != DATASET_VERSION:
        raise ValueError(
            f"the dataset generation configuration must use version {DATASET_VERSION}"
        )
    _stranding_horizon(manifest)
    strengths = [float(value) for value in manifest.get("attack_strengths", ())]
    variants = _required_axis(manifest, "policy_variants")
    seeds = tuple(int(value) for value in _required_axis(manifest, "seeds"))
    families = _required_axis(manifest, "families")
    mountains = _required_axis(manifest, "mountains")
    _repo_path(str(manifest["monitor"]))
    components = _component_manifest(manifest)
    controllers = _resolved_manifest_controllers(mountains)
    development = _development_cell_index(manifest)
    if controllers and not strengths:
        raise ValueError("attack strengths are required for attack controllers")
    if not controllers and strengths:
        raise ValueError("attack strengths need one attack controller")
    entries: list[DatasetEntry] = []
    for mountain in mountains:
        for family in families:
            for controller, resolved_controller in controllers[mountain["id"]]:
                attack = resolved_controller.attack
                assert attack is not None
                for variant in variants:
                    for strength in strengths:
                        reasons = _holdout_reasons(manifest, attack, variant, strength)
                        if reasons and family["id"] != "busy-weekend":
                            continue
                        for seed in seeds:
                            pair_id = _pair_id(
                                mountain["id"],
                                family["id"],
                                controller["id"],
                                variant,
                                strength,
                                int(seed),
                            )
                            common = {
                                "scenario_family": family["id"],
                                "mountain": mountain["id"],
                                "seed": int(seed),
                                "attack_strength": strength,
                                "pair_id": pair_id,
                                "split": _family_split(family["id"]),
                                "policy_variant": variant,
                                "attack_kind": attack.kind,
                                "attack_tier": attack.tier,
                                "holdout_reasons": reasons,
                            }
                            manifest_fields = _manifest_entry_fields(
                                development,
                                mountain=str(mountain["id"]),
                                family=str(family["id"]),
                                attack_kind=attack.kind,
                                attack_tier=attack.tier,
                                strength=strength,
                                policy=str(variant),
                                seed=int(seed),
                            )
                            entries.extend(
                                (
                                    DatasetEntry(
                                        controller_kind="honest",
                                        config_paths=(
                                            _repo_relative(mountain["config"]),
                                            _repo_relative(family["config"]),
                                            _honest_component(
                                                components, mountain["id"], variant
                                            ),
                                            _repo_relative(manifest["monitor"]),
                                        ),
                                        override_path=_override_component(
                                            components, int(seed)
                                        ),
                                        pair_role="honest",
                                        **manifest_fields["honest"],
                                        **common,
                                    ),
                                    DatasetEntry(
                                        controller_kind=controller["id"],
                                        config_paths=(
                                            _repo_relative(mountain["config"]),
                                            _repo_relative(family["config"]),
                                            _attack_component(
                                                components,
                                                mountain["id"],
                                                controller["id"],
                                                variant,
                                                strength,
                                            ),
                                            _repo_relative(manifest["monitor"]),
                                        ),
                                        override_path=_override_component(
                                            components, int(seed)
                                        ),
                                        pair_role="attack",
                                        **manifest_fields["attack"],
                                        **common,
                                    ),
                                )
                            )
    expected = _expected_entry_count(
        manifest, mountains, controllers, variants, strengths, seeds
    )
    if len(entries) != expected:
        raise ValueError(
            f"the dataset matrix expanded to {len(entries)} runs instead of {expected}"
        )
    _validate_expanded_axes(
        entries, mountains, families, controllers, variants, strengths
    )
    return entries


def _development_cell_index(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Load exact public manifest cells when the matrix binds them."""
    value = manifest.get("development_manifest")
    if value is None:
        return None
    from avalanche.experiments.protocols import (
        canonical_artifact_sha256,
        load_development_manifest,
    )

    path = _repo_path(str(value))
    development = load_development_manifest(path)
    attacks = {}
    honest = {}
    for split in ("training", "validation"):
        for record in development["episodes"][split]["attack"]:
            key = (
                record["mountain"],
                record["development_family"],
                record["attack_kind"],
                record["attack_tier"],
                float(record["attack_strength"]),
                record["controller_policy_family"],
                int(record["root_seed"]),
            )
            attacks[key] = record
        for record in development["episodes"][split]["honest"]:
            honest[record["run_identifier"]] = record
    return {
        "manifest_sha256": canonical_artifact_sha256(development),
        "attacks": attacks,
        "honest": honest,
    }


def _manifest_entry_fields(
    development: dict[str, Any] | None,
    *,
    mountain: str,
    family: str,
    attack_kind: str,
    attack_tier: str,
    strength: float,
    policy: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Return exact provenance for one explicit attack and honest cell."""
    if development is None:
        return {"honest": {}, "attack": {}}
    from avalanche.experiments.protocols import canonical_artifact_sha256

    manifest_mountain = "medium-resort" if mountain == "val-tarin" else mountain
    key = (
        manifest_mountain,
        family,
        attack_kind,
        attack_tier,
        float(strength),
        policy,
        seed,
    )
    try:
        attack = development["attacks"][key]
        honest = development["honest"][attack["honest_run_identifier"]]
    except KeyError as error:
        message = "the dataset cell is absent from the development manifest"
        raise ValueError(message) from error
    common = {
        "root_id": attack["root_id"],
        "development_manifest_sha256": development["manifest_sha256"],
    }
    return {
        "honest": {
            **common,
            "manifest_cell_sha256": canonical_artifact_sha256(honest),
        },
        "attack": {
            **common,
            "manifest_cell_sha256": canonical_artifact_sha256(attack),
        },
    }


def _component_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Load the explicit training component selections."""
    path = _repo_path(str(manifest["component_manifest"]))
    components = load_yaml(path)
    if components.get("component_version") != 2:
        raise ValueError("the training component manifest version is incompatible")
    return components


def _override_component(components: dict[str, Any], seed: int) -> str:
    """Return one declared seed and runtime override."""
    try:
        path = components["overrides"][str(seed)]
    except KeyError as error:
        raise ValueError("the dataset override component is not declared") from error
    return _repo_relative(str(path))


def _honest_component(components: dict[str, Any], mountain: str, variant: str) -> str:
    """Return one declared honest controller component."""
    try:
        path = components["honest"][mountain][variant]
    except KeyError as error:
        raise ValueError("the honest controller component is not declared") from error
    return _repo_relative(str(path))


def _attack_component(
    components: dict[str, Any],
    mountain: str,
    controller: str,
    variant: str,
    strength: float,
) -> str:
    """Return one declared attack controller component."""
    try:
        selections = components["attacks"][mountain][controller]
    except KeyError as error:
        raise ValueError("the attack controller component is not declared") from error
    matches = [
        value
        for value in selections
        if value.get("policy_variant") == variant
        and float(value.get("attack_strength", -1.0)) == strength
    ]
    if len(matches) != 1:
        raise ValueError("the attack controller component selection is not unique")
    return _repo_relative(str(matches[0]["config"]))


def _required_axis(manifest: dict[str, Any], name: str) -> tuple[Any, ...]:
    """Return one declared axis and reject an empty value."""
    values = tuple(manifest.get(name, ()))
    if not values:
        raise ValueError(f"the dataset axis {name!r} must not be empty")
    return values


def _stranding_horizon(manifest: Mapping[str, Any]) -> int:
    """Return the required positive stranding label horizon."""
    try:
        horizon = int(manifest["stranding_horizon_intervals"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("the dataset manifest needs a stranding horizon") from error
    if horizon <= 0:
        raise ValueError("the stranding horizon must be positive")
    return horizon


def _resolved_manifest_controllers(
    mountains: Sequence[dict[str, Any]],
) -> dict[str, tuple[tuple[dict[str, Any], ControllerConfig], ...]]:
    """Validate and classify each composed matrix controller."""
    result = {}
    for mountain in mountains:
        _repo_path(str(mountain["config"]))
        honest_path = _repo_path(str(mountain["honest_config"]))
        honest = ControllerConfig.model_validate(
            ConfigurationResolver().component_values(
                "controller", honest_path.relative_to(REPO_ROOT).as_posix()
            )["controller"]
        )
        if honest.kind != "honest" or honest.attack is not None:
            raise ValueError("the matrix honest controller must contain no attack")
        resolved = []
        for controller in _required_axis(mountain, "controllers"):
            if "attack" in controller:
                raise ValueError("the matrix controller uses the obsolete attack flag")
            controller_path = _repo_path(str(controller["config"]))
            config = ControllerConfig.model_validate(
                ConfigurationResolver().component_values(
                    "controller", controller_path.relative_to(REPO_ROOT).as_posix()
                )["controller"]
            )
            if config.attack is None:
                raise ValueError("each matrix attack controller needs an attack record")
            resolved.append((controller, config))
        result[str(mountain["id"])] = tuple(resolved)
    return result


def _repo_path(value: str) -> Path:
    """Resolve one declared path from the repository root."""
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("a dataset configuration path must be relative")
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("a dataset configuration path leaves the repository")
    if not path.is_file():
        raise ValueError(f"the dataset configuration path {value!r} does not exist")
    return path


def _repo_relative(value: str) -> str:
    """Return one validated repository-relative path."""
    return str(_repo_path(str(value)).relative_to(REPO_ROOT.resolve()))


def _expected_entry_count(
    manifest: dict[str, Any],
    mountains: Sequence[dict[str, Any]],
    controllers: dict[str, tuple[tuple[dict[str, Any], ControllerConfig], ...]],
    variants: Sequence[str],
    strengths: Sequence[float],
    seeds: Sequence[int],
) -> int:
    """Calculate the complete paired run count from resolved attacks."""
    attack_count = 0
    for mountain in mountains:
        for family in manifest["families"]:
            for _, controller in controllers[str(mountain["id"])]:
                assert controller.attack is not None
                for variant in variants:
                    for strength in strengths:
                        reasons = _holdout_reasons(
                            manifest, controller.attack, variant, strength
                        )
                        if not reasons or family["id"] == "busy-weekend":
                            attack_count += len(seeds)
    return attack_count * 2


def _validate_expanded_axes(
    entries: Sequence[DatasetEntry],
    mountains: Sequence[dict[str, Any]],
    families: Sequence[dict[str, Any]],
    controllers: dict[str, tuple[tuple[dict[str, Any], ControllerConfig], ...]],
    variants: Sequence[str],
    strengths: Sequence[float],
) -> None:
    """Reject any declared matrix axis that produces no attack entry."""
    attacks = [entry for entry in entries if entry.pair_role == "attack"]
    expected = {
        "mountain": {str(value["id"]) for value in mountains},
        "scenario family": {str(value["id"]) for value in families},
        "controller": {
            str(value["id"]) for items in controllers.values() for value, _ in items
        },
        "policy variant": set(variants),
        "attack strength": set(strengths),
    }
    actual = {
        "mountain": {entry.mountain for entry in attacks},
        "scenario family": {entry.scenario_family for entry in attacks},
        "controller": {entry.controller_kind for entry in attacks},
        "policy variant": {entry.policy_variant for entry in attacks},
        "attack strength": {entry.attack_strength for entry in attacks},
    }
    for name, declared in expected.items():
        if not declared <= actual[name]:
            raise ValueError(f"the dataset {name} axis contains an empty entry")


def _family_split(family: str) -> str:
    """Return the development role without assigning a root split."""
    if family in {"calm", "lift-failure", "storm", "busy-weekend"}:
        return "development"
    raise ValueError(f"the scenario family {family!r} has no declared split")


def _holdout_reasons(
    manifest: dict[str, Any],
    attack: Any,
    variant: str,
    strength: float,
) -> tuple[str, ...]:
    """Return each declared final-test holdout reason."""
    holdouts = manifest["holdouts"]
    reasons = []
    if variant in holdouts["policy_variants"]:
        reasons.append("policy_variant")
    if attack.kind in holdouts["strategies"]:
        reasons.append("strategy")
    if attack.trigger.kind in holdouts["triggers"]:
        reasons.append("trigger")
    if set(attack.targets) & set(holdouts["targets"]):
        reasons.append("target")
    lower, upper = holdouts["strength_range"]
    if float(lower) <= strength <= float(upper):
        reasons.append("parameter_range")
    return tuple(reasons)


def _pair_id(*parts: object) -> str:
    """Return one stable identity for an explicit pair."""
    canonical = json.dumps(parts, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:20]


def generate_dataset(
    manifest_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
    emitter: MetricEmitter | None = None,
    stage_id: str | None = None,
) -> Path:
    """Run the declared matrix and write one labelled Parquet file."""
    manifest = load_yaml(manifest_path)
    entries = expand_manifest(manifest)[:limit]
    return generate_dataset_entries(
        manifest_path,
        output_path,
        entries,
        source_manifest=manifest,
        information_profile=information_profile,
        emitter=emitter,
        stage_id=stage_id,
    )


def generate_dataset_entries(
    manifest_path: Path,
    output_path: Path,
    entries: Sequence[DatasetEntry],
    *,
    source_manifest: dict[str, Any] | None = None,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
    emitter: MetricEmitter | None = None,
    stage_id: str | None = None,
) -> Path:
    """Run a declared entry subset and write its dataset artifacts."""
    profile = InformationProfile(information_profile)
    stage = stage_id or _generation_stage_id(profile)
    _emit_metric(
        emitter,
        "stage_started",
        stage,
        label=_generation_stage_label(profile),
        phase="resolving configurations",
        total_episodes=len(entries),
        profile=profile.value,
    )
    try:
        selected = resolve_dataset_entries(entries)
    except Exception as error:
        _emit_metric(
            emitter,
            "stage_failed",
            stage,
            phase="resolving configurations",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    return generate_resolved_dataset_entries(
        manifest_path,
        output_path,
        selected,
        source_manifest=source_manifest,
        information_profile=profile,
        emitter=emitter,
        stage_id=stage,
    )


def resolve_dataset_entries(
    entries: Sequence[DatasetEntry],
) -> tuple[ResolvedDatasetEntry, ...]:
    """Resolve every dataset entry before execution starts."""
    selected = tuple(
        ResolvedDatasetEntry(entry, resolve_entry(entry)) for entry in entries
    )
    return _bind_pair_contexts(selected)


def generate_resolved_dataset_entries(
    manifest_path: Path,
    output_path: Path,
    selected: Sequence[ResolvedDatasetEntry],
    *,
    source_manifest: dict[str, Any] | None = None,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
    emitter: MetricEmitter | None = None,
    stage_id: str | None = None,
) -> Path:
    """Write a dataset from one previously resolved entry set."""
    manifest = source_manifest or load_yaml(manifest_path)
    horizon = _stranding_horizon(manifest)
    entries = tuple(value.entry for value in selected)
    if not selected:
        raise ValueError("the dataset entry subset must not be empty")
    profile = InformationProfile(information_profile)
    stage = stage_id or _generation_stage_id(profile)
    workers = _worker_count(selected)
    expected_rows = _expected_generation_rows(selected)
    _emit_metric(
        emitter,
        "stage_started",
        stage,
        label=_generation_stage_label(profile),
        phase="generating",
        total_episodes=len(selected),
        expected_rows=expected_rows,
        workers=workers,
        profile=profile.value,
        retries=0,
        rejected=0,
        failures=0,
    )
    phase = "generating"
    writer = BufferedParquetWriter(
        output_path,
        on_progress=_parquet_progress_callback(emitter, stage),
    )
    try:
        frames = _run_entries(
            selected,
            horizon,
            profile,
            emitter=emitter,
            stage_id=stage,
            on_frame=_parquet_frame_callback(writer, emitter, stage),
        )
        if not frames:
            raise ValueError("the dataset entry subset must not be empty")
        phase = "finalizing_parquet"
        _emit_metric(emitter, "stage_phase", stage, phase=phase)
        writer.close()
        phase = "summarizing"
        _emit_metric(emitter, "stage_phase", stage, phase=phase)
        frame = pd.concat(frames, ignore_index=True)
        _write_manifest_summary(
            frame,
            selected,
            output_path,
            manifest_path,
            manifest,
            profile,
        )
        _write_fixture_metadata(frame, entries, output_path, manifest_path, profile)
        phase = "validating"
        _emit_metric(emitter, "stage_phase", stage, phase=phase)
        validate_generated_dataset(output_path, frame, profile)
    except Exception as error:
        writer.abort()
        if phase != "generating":
            _emit_metric(
                emitter,
                "failure",
                stage,
                count=1,
                phase=phase,
                error_type=type(error).__name__,
                message=str(error),
            )
        _emit_metric(
            emitter,
            "stage_failed",
            stage,
            phase=phase,
            error_type=type(error).__name__,
            message=str(error),
        )
        raise
    _emit_metric(
        emitter,
        "stage_completed",
        stage,
        phase="complete",
        episodes=len(frames),
        rows=len(frame),
        expected_rows=expected_rows,
        output_bytes=output_path.stat().st_size,
        output_path=str(output_path),
        **_generation_semantic_summary(profile, len(frame)),
    )
    return output_path


def _run_entries(
    entries: Sequence[ResolvedDatasetEntry],
    horizon: int,
    information_profile: InformationProfile,
    *,
    emitter: MetricEmitter | None = None,
    stage_id: str = "",
    on_frame: Callable[[pd.DataFrame], None] | None = None,
) -> list[pd.DataFrame]:
    """Run each entry, in one process or in a pool."""
    profile = InformationProfile(information_profile)
    if emitter is not None and not stage_id:
        stage_id = _generation_stage_id(profile)
    workers = _worker_count(entries)
    results: Iterable[pd.DataFrame]
    if workers <= 1:
        if emitter is None:
            results = (
                _run_resolved_entry(entry, horizon, profile) for entry in entries
            )
        else:
            results = (
                _run_resolved_entry_observed(
                    entry,
                    horizon,
                    profile,
                    emitter,
                    stage_id,
                )
                for entry in entries
            )
        return _collect_frames(results, on_frame)
    # ponytail: a plain pool. The sweep executor of the next stage supersedes it.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if emitter is None:
            results = pool.map(
                _run_resolved_entry,
                entries,
                [horizon] * len(entries),
                [profile] * len(entries),
            )
        else:
            results = pool.map(
                _run_resolved_entry_observed,
                entries,
                [horizon] * len(entries),
                [profile] * len(entries),
                [emitter] * len(entries),
                [stage_id] * len(entries),
            )
        return _collect_frames(results, on_frame)


def _collect_frames(
    results: Iterable[pd.DataFrame],
    on_frame: Callable[[pd.DataFrame], None] | None,
) -> list[pd.DataFrame]:
    """Keep each ordered frame and notify the parent writer."""
    frames = []
    for frame in results:
        if on_frame is not None:
            on_frame(frame)
        frames.append(frame)
    return frames


def _worker_count(entries: Sequence[ResolvedDatasetEntry]) -> int:
    """Return the common configured worker count."""
    worker_counts = {entry.resolved.runtime.worker_count for entry in entries}
    if len(worker_counts) != 1:
        raise ValueError("the dataset entries have different worker counts")
    return worker_counts.pop()


def _expected_generation_rows(entries: Sequence[ResolvedDatasetEntry]) -> int:
    """Return the configured row ceiling before early termination."""
    total = 0
    for entry in entries:
        resolved = entry.resolved
        duration = resolved.episode_duration_seconds
        interval = resolved.intervals.control_interval_seconds
        epsilon = resolved.numerics.time_epsilon_seconds
        total += max(1, ceil(max(duration - epsilon, 0.0) / interval))
    return total


def _generation_stage_id(profile: InformationProfile) -> str:
    """Return the default trace-generation stage identity."""
    return f"{profile.value.replace('_', '-')}-traces"


def _generation_stage_label(profile: InformationProfile) -> str:
    """Return the readable trace-generation stage label."""
    return {
        InformationProfile.PRINCIPAL: "Principal traces",
        InformationProfile.ORACLE_FALLBACK: "Oracle fallback traces",
        InformationProfile.ORACLE_TRUE_STATE: "Oracle true-state traces",
    }[profile]


def _generation_semantic_summary(
    profile: InformationProfile,
    rows: int,
) -> dict[str, Any]:
    """Return final semantic counts for the persistent stage log."""
    if profile is InformationProfile.PRINCIPAL:
        return {"principal_traces": rows}
    if profile is InformationProfile.ORACLE_TRUE_STATE:
        return {"oracle_true_states": rows}
    return {
        "fallback_attempts": rows,
        "oracle_fallbacks": rows,
        "fallback_rate": 1.0,
    }


def _parquet_progress_callback(
    emitter: MetricEmitter | None,
    stage_id: str,
) -> Callable[[ParquetWriteProgress], None] | None:
    """Build the parent callback for encoded Parquet progress."""
    if emitter is None:
        return None

    def report(progress: ParquetWriteProgress) -> None:
        _emit_metric(
            emitter,
            "parquet_progress",
            stage_id,
            written_rows=progress.written_rows,
            written_bytes=progress.encoded_bytes,
            buffered_rows=progress.buffered_rows,
            row_groups=progress.row_groups,
            final=progress.final,
        )

    return report


def _parquet_frame_callback(
    writer: BufferedParquetWriter,
    emitter: MetricEmitter | None,
    stage_id: str,
) -> Callable[[pd.DataFrame], None]:
    """Write each completed frame and report a parent-side failure."""
    writing_started = False

    def write(frame: pd.DataFrame) -> None:
        nonlocal writing_started
        if not writing_started:
            _emit_metric(
                emitter,
                "stage_phase",
                stage_id,
                phase="generating and writing",
                detail="ordered row groups",
            )
            writing_started = True
        try:
            writer.write(frame)
        except Exception as error:
            _emit_metric(
                emitter,
                "failure",
                stage_id,
                count=1,
                phase="writing",
                error_type=type(error).__name__,
                message=str(error),
            )
            raise

    return write


def _bind_pair_contexts(
    entries: Sequence[ResolvedDatasetEntry],
) -> tuple[ResolvedDatasetEntry, ...]:
    """Build one complete context for each resolved dataset pair."""
    from avalanche.experiments.protocols import build_pair_context, canonical_sha256

    groups: dict[str, list[ResolvedDatasetEntry]] = {}
    for selected in entries:
        if selected.entry.pair_id:
            groups.setdefault(selected.entry.pair_id, []).append(selected)
    contexts: dict[str, PairContext] = {}
    for pair_id, group in groups.items():
        by_role = {selected.entry.pair_role: selected for selected in group}
        if len(group) != 2 or set(by_role) != {"honest", "attack"}:
            raise ValueError(f"the dataset pair {pair_id} is incomplete")
        honest = by_role["honest"].resolved
        attack = by_role["attack"].resolved
        model_lock = honest.monitor.model_lock
        artifact_sha256 = canonical_sha256(
            {
                "model_lock": (
                    None if model_lock is None else model_lock.model_dump(mode="json")
                )
            }
        )
        contexts[pair_id] = build_pair_context(
            honest,
            attack,
            code_revision=_code_revision(),
            artifact_sha256=artifact_sha256,
        )
    return tuple(
        ResolvedDatasetEntry(
            selected.entry,
            selected.resolved,
            contexts.get(selected.entry.pair_id),
        )
        for selected in entries
    )


def pair_context_checksum(
    entry: DatasetEntry,
    *,
    resolved: ResolvedConfig | None = None,
) -> str:
    """Return the complete invariant identity for one pair member."""
    from avalanche.experiments.protocols import (
        canonical_sha256,
        invariant_configuration,
    )

    configuration = resolved or resolve_entry(entry)
    return canonical_sha256(invariant_configuration(configuration))


def _resolved_checksum(resolved: ResolvedConfig) -> str:
    """Return the complete resolved configuration checksum."""
    from avalanche.experiments.protocols import resolved_configuration_sha256

    return resolved_configuration_sha256(resolved)


def _write_manifest_summary(
    frame: pd.DataFrame,
    selected: Sequence[ResolvedDatasetEntry],
    output_path: Path,
    source_path: Path,
    source_manifest: dict[str, Any],
    information_profile: InformationProfile,
) -> None:
    """Record what the dataset holds beside the rows."""
    entries = tuple(value.entry for value in selected)
    resolved_configs = []
    for value in selected:
        entry = value.entry
        resolved = value.resolved
        canonical = json.dumps(resolved.model_dump(mode="json"), sort_keys=True)
        resolved_configs.append(
            {
                "pair_id": entry.pair_id,
                "pair_role": entry.pair_role,
                "run_id": run_id(resolved),
                "checksum": hashlib.sha256(canonical.encode()).hexdigest(),
                "configuration": json.loads(canonical),
                "pair_context": (
                    None if value.pair_context is None else value.pair_context.as_dict()
                ),
            }
        )
    known_stranding = frame.loc[frame[STRANDING_MASK] == 1, STRANDING_LABEL]
    summary = {
        "dataset_version": DATASET_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_schema_sha256": LABEL_SCHEMA_SHA256,
        "feature_names": list(feature_names_for(information_profile)),
        "feature_version": FEATURE_VERSION,
        "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
        "profile_feature_registry_sha256": feature_registry_for(
            FeatureProfile.PRINCIPAL_FULL
        ).sha256,
        "policy_version": HONEST_POLICY_VERSION,
        "observation_version": OBSERVATION_SCHEMA_VERSION,
        "proposal_version": 1,
        "audit_version": AUDIT_SCHEMA_VERSION,
        "route_sensor_version": ROUTE_SENSOR_SCHEMA_VERSION,
        "information_profile": information_profile.value,
        "row_count": int(len(frame)),
        "run_count": len(entries),
        "pair_count": len({entry.pair_id for entry in entries}),
        "families": sorted({entry.scenario_family for entry in entries}),
        "mountains": sorted({entry.mountain for entry in entries}),
        "controllers": sorted({entry.controller_kind for entry in entries}),
        "seeds": sorted({entry.seed for entry in entries}),
        "attack_strengths": sorted(
            {
                entry.attack_strength
                for entry in entries
                if entry.attack_strength is not None
            }
        ),
        "attack_rate": float(frame[ATTACK_LABEL].mean()),
        "stranding_rate": (
            float(known_stranding.mean()) if len(known_stranding) else None
        ),
        "row_counts": {
            "by_split": frame.groupby("split", dropna=False).size().to_dict(),
            "by_pair_role": frame.groupby("pair_role", dropna=False).size().to_dict(),
            "by_policy_variant": frame.groupby("policy_variant", dropna=False)
            .size()
            .to_dict(),
            "known_stranding_labels": int(frame[STRANDING_MASK].sum()),
            "unknown_stranding_labels": int((frame[STRANDING_MASK] == 0).sum()),
            "by_attack_kind": frame.groupby("attack_kind", dropna=False)
            .size()
            .to_dict(),
            "by_attack_strength": frame.groupby("attack_strength", dropna=False)
            .size()
            .to_dict(),
        },
        "checksums": {
            "dataset_sha256": _file_checksum(output_path),
            "source_manifest_sha256": _file_checksum(source_path),
        },
        "code_revision": _code_revision(),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    artifact_manifest = {
        **summary,
        "source_manifest": source_manifest,
        "resolved_runs": resolved_configs,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n"
    )


def load_dataset_fixture(
    dataset_path: Path,
    metadata_path: Path | None = None,
) -> pd.DataFrame:
    """Load one fixture after every compatibility and integrity check."""
    metadata_path = metadata_path or dataset_path.with_suffix(".metadata.json")
    recovery = "regenerate the current monitor fixture with compatible schemas"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(recovery) from error
    expected = {
        "dataset_version": DATASET_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_schema_sha256": LABEL_SCHEMA_SHA256,
        "feature_version": FEATURE_VERSION,
        "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
        "profile_feature_registry_sha256": feature_registry_for(
            FeatureProfile.PRINCIPAL_FULL
        ).sha256,
        "honest_policy_version": HONEST_POLICY_VERSION,
        "feature_names": list(feature_names_for(InformationProfile.PRINCIPAL)),
        "observation_version": OBSERVATION_SCHEMA_VERSION,
        "audit_version": AUDIT_SCHEMA_VERSION,
        "route_sensor_version": ROUTE_SENSOR_SCHEMA_VERSION,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(recovery)
    if metadata.get("dataset_sha256") != _file_checksum(dataset_path):
        raise ValueError(recovery)
    frame = pd.read_parquet(dataset_path)
    if int(metadata.get("row_count", -1)) != len(frame):
        raise ValueError(recovery)
    try:
        require_current_formal_dataset_rows(frame, name="fixture")
    except ValueError as error:
        raise ValueError(recovery) from error
    if OBSOLETE_FORMAL_DATASET_FIELDS & set(frame):
        raise ValueError(recovery)
    return frame


@dataclass(frozen=True)
class LegacyDatasetFixture:
    """Hold validated rows and their historical feature schema."""

    rows: pd.DataFrame
    feature_names: tuple[str, ...]
    feature_version: int


def load_nonformal_legacy_dataset_v4_fixture(
    dataset_path: Path,
    metadata_path: Path | None = None,
) -> LegacyDatasetFixture:
    """Load one historical fixture for nonformal regression tests only."""
    metadata_path = metadata_path or dataset_path.with_suffix(".metadata.json")
    recovery = "restore the historical monitor fixture from version control"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(recovery) from error
    expected = {
        "dataset_version": LEGACY_DATASET_FIXTURE_VERSION,
        "feature_version": LEGACY_DATASET_FEATURE_VERSION,
        "honest_policy_version": HONEST_POLICY_VERSION,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(recovery)
    if metadata.get("dataset_sha256") != _file_checksum(dataset_path):
        raise ValueError(recovery)
    frame = pd.read_parquet(dataset_path)
    if int(metadata.get("row_count", -1)) != len(frame):
        raise ValueError(recovery)
    feature_names = metadata.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) and name for name in feature_names)
        or len(set(feature_names)) != len(feature_names)
        or not set(feature_names) <= set(frame)
    ):
        raise ValueError(recovery)
    return LegacyDatasetFixture(
        rows=frame,
        feature_names=tuple(feature_names),
        feature_version=int(metadata["feature_version"]),
    )


def validate_generated_dataset(
    dataset_path: Path,
    frame: pd.DataFrame,
    information_profile: InformationProfile | str,
) -> dict[str, str]:
    """Validate the generated rows and their complete provenance."""
    profile = InformationProfile(information_profile)
    manifest_path = dataset_path.with_suffix(".manifest.json")
    summary_path = dataset_path.with_suffix(".summary.json")
    manifest = _artifact_mapping(manifest_path, "dataset manifest")
    summary = _artifact_mapping(summary_path, "dataset summary")
    expected_features = list(feature_names_for(profile))
    expected = {
        "dataset_version": DATASET_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_schema_sha256": LABEL_SCHEMA_SHA256,
        "feature_version": FEATURE_VERSION,
        "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
        "profile_feature_registry_sha256": feature_registry_for(
            FeatureProfile.PRINCIPAL_FULL
        ).sha256,
        "information_profile": profile.value,
        "feature_names": expected_features,
        "code_revision": _code_revision(),
        "observation_version": OBSERVATION_SCHEMA_VERSION,
        "audit_version": AUDIT_SCHEMA_VERSION,
        "route_sensor_version": ROUTE_SENSOR_SCHEMA_VERSION,
    }
    for name, value in expected.items():
        if summary.get(name) != value or manifest.get(name) != value:
            raise ValueError(f"the generated dataset has an invalid {name}")
    if int(summary.get("row_count", -1)) != len(frame):
        raise ValueError("the generated dataset has an invalid row count")
    if frame.empty:
        raise ValueError("the generated dataset must contain rows")
    require_current_formal_dataset_rows(frame, name="generated")
    if set(frame["information_profile"]) != {profile.value}:
        raise ValueError("the generated rows have an invalid information profile")
    required_labels = {ATTACK_LABEL, STRANDING_LABEL, STRANDING_MASK}
    if not required_labels <= set(frame):
        raise ValueError("the generated rows miss a declared label")
    if OBSOLETE_FORMAL_DATASET_FIELDS & set(frame):
        raise ValueError("the generated rows contain an obsolete harm field")
    if not set(expected_features).issubset(frame.columns):
        raise ValueError("the generated rows miss a declared feature")
    checksums = generated_dataset_checksums(dataset_path)
    recorded_checksums = summary.get("checksums")
    if not isinstance(recorded_checksums, Mapping):
        raise ValueError("the dataset summary misses its checksums")
    if recorded_checksums.get("dataset_sha256") != checksums["dataset_sha256"]:
        raise ValueError("the generated dataset checksum has changed")
    _validate_resolved_runs(manifest, int(summary.get("run_count", -1)))
    return checksums


def generated_dataset_checksums(dataset_path: Path) -> dict[str, str]:
    """Return the three required generated dataset checksums."""
    paths = {
        "dataset_sha256": dataset_path,
        "dataset_manifest_sha256": dataset_path.with_suffix(".manifest.json"),
        "dataset_summary_sha256": dataset_path.with_suffix(".summary.json"),
    }
    return {name: _file_checksum(path) for name, path in paths.items()}


def _artifact_mapping(path: Path, label: str) -> Mapping[str, Any]:
    """Load one generated JSON mapping."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"the {label} is missing or invalid") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"the {label} must contain one mapping")
    return value


def _validate_resolved_runs(manifest: Mapping[str, Any], run_count: int) -> None:
    """Validate every recorded resolved run configuration."""
    runs = manifest.get("resolved_runs")
    if not isinstance(runs, list) or len(runs) != run_count or not runs:
        raise ValueError("the dataset manifest has invalid resolved runs")
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("a resolved run record must contain one mapping")
        configuration = run.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("a resolved run must record its configuration")
        canonical = json.dumps(configuration, sort_keys=True)
        if run.get("checksum") != hashlib.sha256(canonical.encode()).hexdigest():
            raise ValueError("a resolved run configuration checksum has changed")
        for name in (
            "resolved_configuration_sha256",
            "scientific_configuration_sha256",
        ):
            digest = configuration.get(name)
            if not isinstance(digest, str) or digest == "0" * 64:
                raise ValueError("a resolved run has an invalid configuration digest")


def _write_fixture_metadata(
    frame: pd.DataFrame,
    entries: Sequence[DatasetEntry],
    output_path: Path,
    source_path: Path,
    information_profile: InformationProfile,
) -> None:
    """Write the compact metadata required by a fixture consumer."""
    relative_source = source_path.resolve().relative_to(REPO_ROOT.resolve())
    metadata = {
        "dataset_version": DATASET_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_schema_sha256": LABEL_SCHEMA_SHA256,
        "feature_version": FEATURE_VERSION,
        "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
        "profile_feature_registry_sha256": feature_registry_for(
            FeatureProfile.PRINCIPAL_FULL
        ).sha256,
        "honest_policy_version": HONEST_POLICY_VERSION,
        "feature_names": list(feature_names_for(information_profile)),
        "observation_version": OBSERVATION_SCHEMA_VERSION,
        "audit_version": AUDIT_SCHEMA_VERSION,
        "route_sensor_version": ROUTE_SENSOR_SCHEMA_VERSION,
        "code_revision": _code_revision(),
        "generation_configuration": str(relative_source),
        "seeds": sorted({entry.seed for entry in entries}),
        "row_count": int(len(frame)),
        "dataset_sha256": _file_checksum(output_path),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def _file_checksum(path: Path) -> str:
    """Return the complete SHA-256 checksum of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_revision() -> str:
    """Return the recorded source revision."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
