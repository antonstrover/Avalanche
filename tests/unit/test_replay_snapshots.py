"""Check separate physical replay and continuation snapshots."""

import copy
import hashlib
from pathlib import Path

import pytest

from avalanche.config import ResolvedConfig
from avalanche.control import ApprovalChoice, SimulatedApprover
from avalanche.controllers import build_controller
from avalanche.controllers.attacks import AttackLifecycle
from avalanche.controllers.factory import build_fallback
from avalanche.env import AvalancheEnv, build_resolved_environment
from avalanche.monitors import build_monitor
from avalanche.traces import (
    CONTINUATION_ARTIFACT_TYPE,
    CONTINUATION_SCHEMA_VERSION,
    PHYSICAL_REPLAY_ARTIFACT_TYPE,
    PHYSICAL_REPLAY_SCHEMA_VERSION,
    SnapshotSchemaError,
    encode_continuation_snapshot,
    encode_physical_replay_snapshot,
    load_continuation_snapshot,
    load_physical_replay_snapshot,
    restore_continuation_snapshot,
    restore_snapshot,
    write_continuation_snapshot,
)
from avalanche.traces.checksums import canonical_messagepack, named_checksum
from tests.configuration import resolve_test_configuration


def resolved_config(root: Path) -> ResolvedConfig:
    """Return one short formal episode configuration."""
    return resolve_test_configuration(
        root,
        mountain="configs/mountain/small.yaml",
        scenario="configs/scenarios/default.yaml",
        controller="configs/controllers/small-resort/honest.yaml",
        monitor="configs/monitors/none.yaml",
        changes={
            "scenario": {
                "intervals": {
                    "movement_tick_seconds": 5.0,
                    "control_interval_seconds": 5.0,
                },
                "snapshot_interval_seconds": 5.0,
            }
        },
        override={
            "population": {"skier_count": 8},
            "episode_duration_seconds": 15.0,
        },
    )


def running_components(
    resolved: ResolvedConfig,
) -> tuple[AvalancheEnv, object, AttackLifecycle]:
    """Return one reset environment and its executable components."""
    env = build_resolved_environment(resolved)
    controller = build_controller(resolved.controller, env.topology)
    fallback = build_fallback(
        resolved.fallback.policy,
        resolved.controller,
        env.topology,
    )
    monitor = build_monitor(resolved.monitor, resolved.controller, env.topology)
    env.configure_adjudicator(
        monitor,
        fallback,
        SimulatedApprover(ApprovalChoice(resolved.approval.simulated_choice)),
        resolved.approval.timeout_seconds,
    )
    controller.reset(resolved.seed)
    env.reset(seed=resolved.seed)
    return env, controller, AttackLifecycle()


def continuation(root: Path) -> tuple[ResolvedConfig, dict]:
    """Return one snapshot after a complete control interval."""
    resolved = resolved_config(root)
    env, controller, lifecycle = running_components(resolved)
    proposal = controller.propose(env.controller_observation())
    env.step_proposal(proposal)
    return resolved, encode_continuation_snapshot(
        env,
        controller,
        resolved,
        attack_lifecycle=lifecycle,
        trace_state={"sequence": 1, "snapshot_cadence": 5.0},
        runtime_state={"next_snapshot_time": 10.0},
    )


def test_reported_snapshot_has_no_exact_status(tmp_path):
    """Keep per-skier truth out of the reported replay view."""
    resolved = resolved_config(tmp_path / "config")
    env, _, _ = running_components(resolved)
    row = encode_physical_replay_snapshot(
        env.sim,
        view_kind="reported",
        run_id="run-one",
        episode_id="episode-0",
    )
    replay = load_physical_replay_snapshot(row)

    assert row["artifact_type"] == PHYSICAL_REPLAY_ARTIFACT_TYPE
    assert row["schema_version"] == PHYSICAL_REPLAY_SCHEMA_VERSION
    assert row["view_kind"] == "reported"
    assert "population" not in replay["state"]
    assert "status" not in replay["state"]
    assert "physical_state_checksum" in row
    assert "state_checksum" not in row
    assert not replay["executable"]


def test_evaluator_snapshot_has_exact_physical_fields(tmp_path):
    """Include the exact per-skier display fields only for evaluators."""
    resolved = resolved_config(tmp_path / "config")
    env, _, _ = running_components(resolved)
    row = encode_physical_replay_snapshot(
        env.sim,
        view_kind="evaluator",
        run_id="run-one",
        episode_id="episode-0",
    )
    replay = load_physical_replay_snapshot(row)

    assert set(replay["state"]["population"]) == {
        "location_kind",
        "location_index",
        "required_travel_seconds",
        "remaining_travel_seconds",
        "status",
    }


def test_physical_replay_cannot_resume(tmp_path):
    """Reject execution restoration from both replay views."""
    resolved = resolved_config(tmp_path / "config")
    env, _, _ = running_components(resolved)
    before = env.sim.physical_state_checksum()

    for view_kind in ("reported", "evaluator"):
        row = encode_physical_replay_snapshot(
            env.sim,
            view_kind=view_kind,
            run_id="run-one",
            episode_id="episode-0",
        )
        with pytest.raises(SnapshotSchemaError, match="display-only"):
            restore_snapshot(env.sim, row)

    assert env.sim.physical_state_checksum() == before


def test_continuation_restores_every_simulator_array(tmp_path):
    """Restore the exact simulator state into new components."""
    resolved, snapshot = continuation(tmp_path / "config")
    restored = restore_continuation_snapshot(snapshot, resolved=resolved)
    env = restored["environment"]

    assert env.sim.physical_state_checksum("reported") == named_checksum(
        env.sim.physical_replay_state("reported"),
        allow_nonfinite=True,
    )
    assert env.sim.physical_state_checksum("evaluator") == named_checksum(
        env.sim.physical_replay_state("evaluator"),
        allow_nonfinite=True,
    )
    assert snapshot["artifact_type"] == CONTINUATION_ARTIFACT_TYPE
    assert snapshot["schema_version"] == CONTINUATION_SCHEMA_VERSION
    assert "continuation_checksum" in snapshot
    assert "physical_state_checksum" not in snapshot
    assert "artifact_sha256" not in snapshot


@pytest.mark.parametrize(
    "section",
    [
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
        "references",
        "compatibility",
    ],
)
def test_each_continuation_section_is_tamper_evident(tmp_path, section):
    """Reject an independent change to every continuation section."""
    resolved, snapshot = continuation(tmp_path / section)
    tampered = copy.deepcopy(snapshot)
    tampered[section]["tampered"] = True

    with pytest.raises(SnapshotSchemaError, match="checksum"):
        restore_continuation_snapshot(tampered, resolved=resolved)


@pytest.mark.parametrize(
    "path",
    [
        ("simulator", "random_streams", "audit"),
        ("monitor", "state"),
        ("simulator", "route_sensor"),
        ("environment", "control_history"),
    ],
)
def test_future_state_tampering_fails(tmp_path, path):
    """Reject tampering in a known future-influencing state owner."""
    resolved, snapshot = continuation(tmp_path / path[-1])
    tampered = copy.deepcopy(snapshot)
    target = tampered
    for name in path[:-1]:
        target = target[name]
    name = path[-1]
    target[name] = {"original": target[name], "tampered": True}

    with pytest.raises(SnapshotSchemaError, match="checksum"):
        restore_continuation_snapshot(tampered, resolved=resolved)


@pytest.mark.parametrize(
    "field",
    [
        "python_version",
        "numpy_version",
        "bit_generator_class",
        "code_revision",
        "protocol_digests",
    ],
)
def test_runtime_compatibility_rejection_table(tmp_path, field):
    """Reject every recorded runtime or code identity mismatch."""
    resolved, snapshot = continuation(tmp_path / field)
    tampered = copy.deepcopy(snapshot)
    if field == "protocol_digests":
        tampered["compatibility"][field]["test"] = "0" * 64
    else:
        tampered["compatibility"][field] = "incompatible"
    tampered["continuation_checksum"] = named_checksum(
        tampered,
        allow_nonfinite=True,
    )

    with pytest.raises(SnapshotSchemaError, match="compatibility"):
        restore_continuation_snapshot(tampered, resolved=resolved)


def test_file_sha_fails_before_parse(tmp_path):
    """Check the expected file identity before MessagePack parsing."""
    resolved = resolved_config(tmp_path / "config")
    path = tmp_path / "bad.avalanche-continuation.msgpack"
    path.write_bytes(b"not canonical MessagePack")

    with pytest.raises(SnapshotSchemaError, match="artifact SHA-256"):
        load_continuation_snapshot(
            path,
            expected_artifact_sha256="0" * 64,
            resolved=resolved,
        )


def test_continuation_file_uses_three_identity_boundaries(tmp_path):
    """Keep the logical and exact file identities in separate records."""
    resolved, snapshot = continuation(tmp_path / "config")
    path = tmp_path / "state.avalanche-continuation.msgpack"
    manifest = write_continuation_snapshot(path, snapshot)
    loaded = load_continuation_snapshot(
        path,
        expected_artifact_sha256=manifest["artifact_sha256"],
        resolved=resolved,
    )

    assert loaded["continuation_checksum"] == snapshot["continuation_checksum"]
    assert manifest["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "artifact_sha256" not in loaded


def test_cross_type_load_is_rejected(tmp_path):
    """Reject each snapshot type at the other loader boundary."""
    resolved, snapshot = continuation(tmp_path / "config")
    env, _, _ = running_components(resolved)
    replay = encode_physical_replay_snapshot(
        env.sim,
        view_kind="evaluator",
        run_id="run-one",
        episode_id="episode-0",
    )
    with pytest.raises(SnapshotSchemaError):
        load_physical_replay_snapshot(snapshot)

    path = tmp_path / "replay.avalanche-continuation.msgpack"
    content = canonical_messagepack(replay, allow_nonfinite=True)
    path.write_bytes(content)
    with pytest.raises(SnapshotSchemaError):
        load_continuation_snapshot(
            path,
            expected_artifact_sha256=hashlib.sha256(content).hexdigest(),
            resolved=resolved,
        )


def test_a_wrong_continuation_extension_is_rejected_after_parse(tmp_path):
    """Reject a valid continuation carried by the wrong artifact extension."""
    resolved, snapshot = continuation(tmp_path / "config")
    content = canonical_messagepack(snapshot, allow_nonfinite=True)
    path = tmp_path / "state.msgpack"
    path.write_bytes(content)

    with pytest.raises(SnapshotSchemaError, match="extension"):
        load_continuation_snapshot(
            path,
            expected_artifact_sha256=hashlib.sha256(content).hexdigest(),
            resolved=resolved,
        )
