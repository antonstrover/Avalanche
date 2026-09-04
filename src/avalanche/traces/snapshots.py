"""Define separate display replay and executable continuation snapshots."""

from __future__ import annotations

import hashlib
import hmac
import platform
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np

from avalanche.config.models import ResolvedConfig
from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import ApprovalChoice, SimulatedApprover, StatefulComponent
from avalanche.controllers.attacks import AttackLifecycle
from avalanche.controllers.factory import build_controller, build_fallback
from avalanche.env.adapter import AvalancheEnv
from avalanche.env.factory import build_resolved_environment
from avalanche.monitors.factory import build_monitor
from avalanche.sim.engine import MountainSim
from avalanche.traces.checksums import (
    CanonicalEncodingError,
    canonical_messagepack,
    canonical_sha256,
    decode_canonical_messagepack,
    named_checksum,
)
from avalanche.traces.continuation_state import (
    capture_simulator_state,
    restore_simulator_state,
)

PHYSICAL_REPLAY_SCHEMA_VERSION = 1
CONTINUATION_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = PHYSICAL_REPLAY_SCHEMA_VERSION
PHYSICAL_REPLAY_ARTIFACT_TYPE = "avalanche.physical_replay_snapshot"
CONTINUATION_ARTIFACT_TYPE = "avalanche.continuation_snapshot"
REPORTED_REPLAY_FILENAME = "physical-replay-reported.parquet"
EVALUATOR_REPLAY_FILENAME = "physical-replay-evaluator.parquet"
CONTINUATION_EXTENSION = ".avalanche-continuation.msgpack"

_PHYSICAL_KEYS = {
    "artifact_type",
    "schema_version",
    "view_kind",
    "run_id",
    "episode_id",
    "simulation_time",
    "movement_tick",
    "topology_artifact_name",
    "topology_artifact_sha256",
    "state_messagepack",
    "physical_state_checksum",
}
_CONTINUATION_KEYS = {
    "artifact_type",
    "schema_version",
    "compatibility",
    "references",
    "simulator",
    "environment",
    "controller",
    "monitor",
    "fallback",
    "approval",
    "adjudicator",
    "feature_extractor",
    "attack_lifecycle",
    "trace",
    "runtime",
    "continuation_checksum",
}
_PROTOCOL_PATHS = (
    "src/avalanche/traces/checksums.py",
    "src/avalanche/traces/snapshots.py",
    "src/avalanche/traces/continuation_state.py",
    "src/avalanche/control/protocols.py",
)


class SnapshotSchemaError(ValueError):
    """Report invalid or unsupported snapshot data."""


def encode_physical_replay_snapshot(
    sim: MountainSim,
    *,
    view_kind: Literal["reported", "evaluator"],
    run_id: str,
    episode_id: str,
) -> dict[str, Any]:
    """Return one display-only physical replay row."""
    replay = sim.physical_replay_state(view_kind)
    topology = replay["topology_artifact_reference"]
    return {
        "artifact_type": PHYSICAL_REPLAY_ARTIFACT_TYPE,
        "schema_version": PHYSICAL_REPLAY_SCHEMA_VERSION,
        "view_kind": view_kind,
        "run_id": run_id,
        "episode_id": episode_id,
        "simulation_time": replay["simulation_time"],
        "movement_tick": replay["movement_tick"],
        "topology_artifact_name": topology["name"],
        "topology_artifact_sha256": topology["sha256"],
        "state_messagepack": canonical_messagepack(
            replay["state"],
            allow_nonfinite=True,
        ),
        "physical_state_checksum": named_checksum(
            replay,
            allow_nonfinite=True,
        ),
    }


def load_physical_replay_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    """Validate and return one non-executable replay view."""
    value = _mapping(row, "physical replay")
    _require_keys(value, _PHYSICAL_KEYS, "physical replay")
    if value["artifact_type"] != PHYSICAL_REPLAY_ARTIFACT_TYPE:
        raise SnapshotSchemaError("the artifact is not a physical replay")
    if value["schema_version"] != PHYSICAL_REPLAY_SCHEMA_VERSION:
        raise SnapshotSchemaError("the physical replay schema is unsupported")
    view_kind = value["view_kind"]
    if view_kind not in {"reported", "evaluator"}:
        raise SnapshotSchemaError("the physical replay view is invalid")
    try:
        state = decode_canonical_messagepack(
            value["state_messagepack"],
            allow_nonfinite=True,
        )
    except CanonicalEncodingError as error:
        raise SnapshotSchemaError("the physical replay state is invalid") from error
    replay = {
        "view_kind": view_kind,
        "simulation_time": value["simulation_time"],
        "movement_tick": value["movement_tick"],
        "topology_artifact_reference": {
            "name": value["topology_artifact_name"],
            "sha256": value["topology_artifact_sha256"],
        },
        "state": state,
    }
    expected = named_checksum(replay, allow_nonfinite=True)
    if not hmac.compare_digest(str(value["physical_state_checksum"]), expected):
        raise SnapshotSchemaError("the physical state checksum does not match")
    return {**replay, "executable": False}


def encode_snapshot(
    sim: MountainSim,
    *,
    run_id: str,
    episode_id: str,
    seed: int,
) -> dict[str, Any]:
    """Return the evaluator replay through the former local entry point."""
    del seed
    return encode_physical_replay_snapshot(
        sim,
        view_kind="evaluator",
        run_id=run_id,
        episode_id=episode_id,
    )


def restore_snapshot(sim: MountainSim, row: dict[str, Any]) -> None:
    """Reject execution restoration from every display replay."""
    del sim
    load_physical_replay_snapshot(row)
    raise SnapshotSchemaError("the physical replay is display-only")


def encode_continuation_snapshot(
    env: AvalancheEnv,
    controller: StatefulComponent,
    resolved: ResolvedConfig,
    *,
    attack_lifecycle: AttackLifecycle,
    trace_state: dict[str, Any],
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    """Return one complete executable continuation snapshot."""
    _validate_formal_configuration(resolved)
    monitor = _stateful(env.adjudicator.monitor, "monitor")
    fallback = _optional_stateful(env.adjudicator.fallback, "fallback")
    approval = _stateful(env.adjudicator.approval, "approval")
    feature = _optional_stateful(getattr(monitor, "extractor", None), "feature")
    snapshot = {
        "artifact_type": CONTINUATION_ARTIFACT_TYPE,
        "schema_version": CONTINUATION_SCHEMA_VERSION,
        "compatibility": _compatibility_state(env.sim),
        "references": _reference_state(resolved, env.sim),
        "simulator": capture_simulator_state(env.sim),
        "environment": env.snapshot_state(),
        "controller": _component_state(controller),
        "monitor": _component_state(monitor),
        "fallback": _component_state(fallback),
        "approval": _component_state(approval),
        "adjudicator": _component_state(env.adjudicator),
        "feature_extractor": _component_state(feature),
        "attack_lifecycle": attack_lifecycle.snapshot_state(),
        "trace": trace_state,
        "runtime": runtime_state,
    }
    snapshot["continuation_checksum"] = named_checksum(
        snapshot,
        allow_nonfinite=True,
    )
    _validate_continuation(
        snapshot,
        resolved,
        current_compatibility=snapshot["compatibility"],
    )
    return snapshot


def write_continuation_snapshot(
    path: Path,
    snapshot: dict[str, Any],
) -> dict[str, str]:
    """Write canonical bytes and return their enclosing manifest record."""
    target = Path(path)
    if not target.name.endswith(CONTINUATION_EXTENSION):
        raise SnapshotSchemaError("the continuation extension is invalid")
    content = canonical_messagepack(snapshot, allow_nonfinite=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {
        "artifact_type": CONTINUATION_ARTIFACT_TYPE,
        "path": target.name,
        "artifact_sha256": hashlib.sha256(content).hexdigest(),
    }


def load_continuation_snapshot(
    path: Path,
    *,
    expected_artifact_sha256: str,
    resolved: ResolvedConfig,
) -> dict[str, Any]:
    """Verify bytes before parsing one continuation snapshot."""
    target = Path(path)
    content = target.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(actual, expected_artifact_sha256):
        raise SnapshotSchemaError("the continuation artifact SHA-256 does not match")
    try:
        value = decode_canonical_messagepack(content, allow_nonfinite=True)
    except CanonicalEncodingError as error:
        raise SnapshotSchemaError("the continuation MessagePack is invalid") from error
    snapshot = _mapping(value, "continuation")
    _validate_continuation(snapshot, resolved)
    if not target.name.endswith(CONTINUATION_EXTENSION):
        raise SnapshotSchemaError("the continuation extension is invalid")
    return snapshot


def restore_continuation_snapshot(
    snapshot: dict[str, Any],
    *,
    resolved: ResolvedConfig,
) -> dict[str, Any]:
    """Construct and restore compatible executable components."""
    _validate_continuation(snapshot, resolved)
    env = build_resolved_environment(resolved)
    controller = build_controller(resolved.controller, env.topology)
    fallback = build_fallback(
        resolved.fallback.policy,
        resolved.controller,
        env.topology,
    )
    monitor = build_monitor(resolved.monitor, resolved.controller, env.topology)
    approval = SimulatedApprover(ApprovalChoice(resolved.approval.simulated_choice))
    env.configure_adjudicator(
        monitor,
        fallback,
        approval,
        resolved.approval.timeout_seconds,
    )
    controller.reset(resolved.seed)
    env.reset(seed=resolved.seed)
    restore_simulator_state(env.sim, snapshot["simulator"])
    _restore_component(controller, snapshot["controller"], "controller")
    _restore_component(monitor, snapshot["monitor"], "monitor")
    _restore_component(fallback, snapshot["fallback"], "fallback")
    _restore_component(approval, snapshot["approval"], "approval")
    _restore_component(env.adjudicator, snapshot["adjudicator"], "adjudicator")
    feature = getattr(monitor, "extractor", None)
    _restore_component(feature, snapshot["feature_extractor"], "feature")
    env.restore_state(snapshot["environment"])
    lifecycle = AttackLifecycle()
    lifecycle.restore_state(snapshot["attack_lifecycle"])
    restored_state = capture_simulator_state(env.sim)
    if canonical_sha256(restored_state, allow_nonfinite=True) != canonical_sha256(
        snapshot["simulator"],
        allow_nonfinite=True,
    ):
        raise SnapshotSchemaError("the restored simulator state does not match")
    env.sim.physical_state_checksum("reported")
    env.sim.physical_state_checksum("evaluator")
    return {
        "environment": env,
        "controller": controller,
        "attack_lifecycle": lifecycle,
        "trace": snapshot["trace"],
        "runtime": snapshot["runtime"],
    }


def _validate_continuation(
    snapshot: dict[str, Any],
    resolved: ResolvedConfig,
    *,
    current_compatibility: dict[str, Any] | None = None,
) -> None:
    _require_keys(snapshot, _CONTINUATION_KEYS, "continuation")
    if snapshot["artifact_type"] != CONTINUATION_ARTIFACT_TYPE:
        raise SnapshotSchemaError("the artifact is not a continuation snapshot")
    if snapshot["schema_version"] != CONTINUATION_SCHEMA_VERSION:
        raise SnapshotSchemaError("the continuation schema is unsupported")
    expected = named_checksum(snapshot, allow_nonfinite=True)
    actual = snapshot["continuation_checksum"]
    if not isinstance(actual, str) or not hmac.compare_digest(actual, expected):
        raise SnapshotSchemaError("the continuation checksum does not match")
    compatibility = _mapping(snapshot["compatibility"], "compatibility")
    current = current_compatibility or _compatibility_state()
    for name, expected_value in current.items():
        if compatibility.get(name) != expected_value:
            raise SnapshotSchemaError(f"the {name} compatibility identity differs")
    references = _mapping(snapshot["references"], "references")
    if references != _reference_state(resolved):
        raise SnapshotSchemaError("the continuation references do not match")
    _validate_component_identities(snapshot, resolved)


def _validate_component_identities(
    snapshot: dict[str, Any],
    resolved: ResolvedConfig,
) -> None:
    env = build_resolved_environment(resolved)
    controller = build_controller(resolved.controller, env.topology)
    fallback = build_fallback(
        resolved.fallback.policy, resolved.controller, env.topology
    )
    monitor = build_monitor(resolved.monitor, resolved.controller, env.topology)
    approval = SimulatedApprover(ApprovalChoice(resolved.approval.simulated_choice))
    expected = {
        "controller": controller,
        "monitor": monitor,
        "fallback": fallback,
        "approval": approval,
        "adjudicator": env.adjudicator,
        "feature_extractor": getattr(monitor, "extractor", None),
    }
    for name, component in expected.items():
        section = _mapping(snapshot[name], name)
        expected_type = None if component is None else _type_identity(component)
        if section.get("component_type") != expected_type:
            raise SnapshotSchemaError(f"the {name} component type differs")


def _component_state(component: StatefulComponent | None) -> dict[str, Any]:
    if component is None:
        return {"component_type": None, "state": None}
    value = _stateful(component, "component")
    return {
        "component_type": _type_identity(value),
        "state": value.snapshot_state(),
    }


def _restore_component(component: Any, section: Any, label: str) -> None:
    value = _mapping(section, label)
    if component is None:
        if value != {"component_type": None, "state": None}:
            raise SnapshotSchemaError(f"the {label} component is incompatible")
        return
    if value.get("component_type") != _type_identity(component):
        raise SnapshotSchemaError(f"the {label} component type differs")
    _stateful(component, label).restore_state(value.get("state"))


def _stateful(value: Any, label: str) -> StatefulComponent:
    if not isinstance(value, StatefulComponent):
        raise TypeError(f"the {label} must expose continuation state")
    return value


def _optional_stateful(value: Any, label: str) -> StatefulComponent | None:
    return None if value is None else _stateful(value, label)


def _type_identity(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _compatibility_state(sim: MountainSim | None = None) -> dict[str, Any]:
    bit_generator = (
        np.random.default_rng().bit_generator
        if sim is None or not sim.streams
        else next(iter(sim.streams.values())).bit_generator
    )
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "bit_generator_class": _type_identity(bit_generator),
        "code_revision": _code_revision(),
        "protocol_digests": {
            name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
            for name in _PROTOCOL_PATHS
        },
    }


def _reference_state(
    resolved: ResolvedConfig,
    sim: MountainSim | None = None,
) -> dict[str, Any]:
    topology_sha256 = (
        _topology_sha256(resolved) if sim is None else _sim_topology_sha256(sim)
    )
    return {
        "resolved_configuration_sha256": resolved.resolved_configuration_sha256,
        "scientific_configuration_sha256": resolved.scientific_configuration_sha256,
        "topology_artifact": {
            "path": resolved.mountain.path,
            "sha256": topology_sha256,
        },
        "source_artifacts": tuple(
            {
                "path": item.source_path,
                "sha256": item.source_sha256,
            }
            for item in resolved.provenance
            if item.source_path is not None and item.source_sha256 is not None
        ),
        "model_lock": (
            None
            if resolved.monitor.model_lock is None
            else resolved.monitor.model_lock.model_dump(mode="python")
        ),
    }


def _topology_sha256(resolved: ResolvedConfig) -> str:
    path = Path(resolved.mountain.path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sim_topology_sha256(sim: MountainSim) -> str:
    if sim.topology is None:
        raise SnapshotSchemaError("reset the simulator before continuation work")
    return sim.topology.mountain_sha256


def _code_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _validate_formal_configuration(resolved: ResolvedConfig) -> None:
    try:
        canonical_messagepack(resolved.model_dump(mode="python"))
    except CanonicalEncodingError as error:
        raise SnapshotSchemaError("the resolved configuration is not finite") from error


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SnapshotSchemaError(f"the {label} must be a string-keyed mapping")
    return value


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        raise SnapshotSchemaError(f"the {label} field {missing[0]!r} is missing")
    if extra:
        raise SnapshotSchemaError(f"the {label} field {extra[0]!r} is unknown")
