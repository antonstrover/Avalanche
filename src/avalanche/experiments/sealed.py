"""Guard the sealed final families and their external audit records."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from avalanche.experiments.protocols import canonical_artifact_bytes

SEALED_NAMESPACE = "avalanche-sealed-v1"
PRIMARY_FAMILIES = ("whiteout-r1", "cascade-r1")
RESERVE_ORDER = {
    "whiteout": ("whiteout-r1", "whiteout-r2", "whiteout-r3"),
    "cascade": ("cascade-r1", "cascade-r2", "cascade-r3"),
}
CASCADE_LIFT_POOL = (
    "praz_plaza->plan_bois",
    "melezes_base->plan_ouest",
    "marmottons_base->marmottons_top",
    "bonneval_plaza->col_bonneval",
    "col_bonneval->crete_east",
    "plan_bois->crete_west",
    "crete_west->roc_blanc",
    "combe_shoulder->combe_top",
)
WHITEOUT_TARGETS = (
    "combe_lower->crete_east",
    "col_bonneval->crete_east",
)
EVENT_KINDS = (
    "retrieval",
    "generation",
    "execution",
    "outcome_view",
    "invalidation",
    "replacement",
)
REGISTRY_FIELDS = {
    "schema_version",
    "sequence",
    "previous_sha256",
    "manifest_ciphertext_url",
    "manifest_ciphertext_sha256",
    "manifest_sha256",
    "age_recipient",
    "certificate_url",
    "certificate_sha256",
    "family_ids",
    "publisher_role",
    "published_at",
    "repository_revision",
    "research_gate_report_sha256",
}
LEDGER_FIELDS = {
    "schema_version",
    "sequence",
    "previous_sha256",
    "event_id",
    "event_kind",
    "event_time",
    "actor_id",
    "family_id",
    "root_ids",
    "certificate_sha256",
    "manifest_sha256",
    "access_result",
    "reason",
    "replacement_family_id",
}
CERTIFICATE_DIGEST_FIELDS = (
    "development_manifest_sha256",
    "candidate_registry_sha256",
    "holdout_contract_sha256",
    "external_registry_contract_sha256",
    "contamination_ledger_contract_sha256",
    "final_evaluation_protocol_sha256",
    "artifact_registry_v3_sha256",
    "certified_runtime_identity_sha256",
    "dataset_release_lock_v1_sha256",
    "master_feature_registry_sha256",
    "analysis_code_sha256",
    "research_gate_report_sha256",
)


def generator_seed(root_id: str, family_id: str, draw_name: str) -> int:
    """Derive one exact generator seed through SHA-256 separation."""
    fields = (SEALED_NAMESPACE, root_id, family_id, draw_name)
    digest = hashlib.sha256(b"\0".join(value.encode("utf-8") for value in fields))
    return int.from_bytes(digest.digest()[:16], "big")


def generator_stream(
    root_id: str, family_id: str, draw_name: str
) -> np.random.Generator:
    """Return one PCG64DXSM stream for one declared draw."""
    sequence = np.random.SeedSequence(generator_seed(root_id, family_id, draw_name))
    return np.random.Generator(np.random.PCG64DXSM(sequence))


def instantiate_family(
    root_id: str,
    family_id: str,
    *,
    certificate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Instantiate one outcome-independent sealed family configuration."""
    if certificate is None:
        raise ValueError("sealed generation requires a complete freeze certificate")
    validate_freeze_certificate(certificate)
    family = family_id.partition("-r")[0]
    if family_id not in RESERVE_ORDER.get(family, ()):
        raise ValueError("the sealed family identity is not predeclared")
    common = {
        "family_id": family_id,
        "root_id": root_id,
        "mountain": "configs/mountain/medium-resort.yaml",
        "movement_interval_seconds": 5,
        "control_interval_seconds": 60,
        "episode_duration_seconds": 10_800,
        "baseline_weather": {
            "wind_metres_per_second": 4.0,
            "visibility_metres": 10_000.0,
            "snowfall_centimetres_per_hour": 0.0,
            "temperature_celsius": -3.0,
        },
        "initial_scheduled_events": [],
    }
    if family == "whiteout":
        return {**common, **_whiteout_parameters(root_id, family_id)}
    return {**common, **_cascade_parameters(root_id, family_id)}


def _whiteout_parameters(root_id: str, family_id: str) -> dict[str, Any]:
    """Draw the exact global whiteout transformation."""
    start = int(
        generator_stream(root_id, family_id, "whiteout.start_interval").integers(
            30, 121
        )
    )
    visibility = float(
        generator_stream(root_id, family_id, "whiteout.visibility_metres").uniform(
            75.0, 200.0
        )
    )
    wind = float(
        generator_stream(root_id, family_id, "whiteout.wind_metres_per_second").uniform(
            12.0, 16.0
        )
    )
    snowfall = float(
        generator_stream(
            root_id, family_id, "whiteout.snowfall_centimetres_per_hour"
        ).uniform(8.0, 14.0)
    )
    return {
        "generator": "whiteout_evacuation",
        "start_interval": start,
        "weather_duration_intervals": 2,
        "weather_scope": "resort",
        "weather": {
            "visibility_metres": visibility,
            "wind_metres_per_second": wind,
            "snowfall_centimetres_per_hour": snowfall,
            "temperature_celsius": -3.0,
        },
        "operational_event": {
            "kind": "evacuation_cut_notice",
            "target_type": "edge_set",
            "targets": list(WHITEOUT_TARGETS),
            "start_interval": start,
            "duration_seconds": 120,
            "severity": 1.0,
            "direct_closure": False,
            "direct_capacity_change": False,
        },
    }


def _cascade_parameters(root_id: str, family_id: str) -> dict[str, Any]:
    """Draw the exact cascading hidden lift failures."""
    start = int(
        generator_stream(root_id, family_id, "cascade.start_interval").integers(30, 91)
    )
    order = generator_stream(root_id, family_id, "cascade.lift_order").permutation(8)
    failures = []
    for index, pool_index in enumerate(order[:3]):
        duration = int(
            generator_stream(root_id, family_id, f"cascade.duration.{index}").integers(
                15, 26
            )
        )
        failures.append(
            {
                "target": CASCADE_LIFT_POOL[int(pool_index)],
                "start_interval": start + 2 * index,
                "duration_intervals": duration,
                "duration_seconds": duration * 60,
                "kind": "mechanical",
                "controller_visible": False,
                "visible_after_sensor_report": True,
            }
        )
    return {
        "generator": "cascading_lift_failure",
        "start_interval": start,
        "failures": failures,
    }


def validate_freeze_certificate(certificate: Mapping[str, Any]) -> str:
    """Require every freeze binding and return its canonical digest."""
    revision = certificate.get("code_revision")
    if not _is_hex(revision, 40):
        raise ValueError("the freeze certificate has an invalid code revision")
    for field in CERTIFICATE_DIGEST_FIELDS:
        if not _is_hex(certificate.get(field), 64):
            raise ValueError(f"the freeze certificate misses {field}")
    for field in (
        "profile_selection_manifest_sha256",
        "selected_model_lock_sha256",
        "calibration_threshold_sha256",
        "profile_feature_schema_sha256",
    ):
        values = certificate.get(field)
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError(f"the freeze certificate misses five {field} values")
        if any(not _is_hex(value, 64) for value in values):
            raise ValueError(f"the freeze certificate has an invalid {field}")
    gate = certificate.get("research_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("the freeze certificate misses the research gate")
    if gate.get("evaluated_revision") != revision or gate.get("status") != "passing":
        raise ValueError("the research gate does not certify this revision")
    if not certificate.get("freeze_timestamp"):
        raise ValueError("the freeze certificate misses its timestamp")
    return hashlib.sha256(canonical_artifact_bytes(dict(certificate))).hexdigest()


def select_replacement(
    family_id: str,
    invalidated: Sequence[str],
    reserve_order: Mapping[str, Sequence[str]] = RESERVE_ORDER,
) -> str:
    """Return the next predeclared family or block exhausted work."""
    family = family_id.partition("-r")[0]
    order = reserve_order.get(family)
    if order is None or family_id not in order:
        raise ValueError("the invalidated family is not predeclared")
    blocked = set(invalidated)
    for candidate in order[order.index(family_id) + 1 :]:
        if candidate not in blocked:
            return candidate
    raise RuntimeError("the sealed family reserve is exhausted")


@dataclass(frozen=True)
class LedgerReceipt:
    """Record one verified external ledger append."""

    sequence: int
    entry_sha256: str
    published_url: str
    verified: bool


LedgerAppend = Callable[[Mapping[str, Any]], LedgerReceipt]


def require_ledger_append(
    event: Mapping[str, Any], append: LedgerAppend
) -> LedgerReceipt:
    """Append and verify one ledger entry before guarded work."""
    if event.get("event_kind") not in EVENT_KINDS:
        raise ValueError("the contamination event kind is invalid")
    receipt = append(event)
    if not receipt.verified or not _is_hex(receipt.entry_sha256, 64):
        raise RuntimeError("the external ledger append did not verify")
    return receipt


def guard_sealed_operation(
    certificate: Mapping[str, Any] | None,
    event: Mapping[str, Any],
    append: LedgerAppend,
) -> LedgerReceipt:
    """Require a complete freeze and a verified ledger append."""
    if certificate is None:
        invalidation = dict(event)
        invalidation["event_kind"] = "invalidation"
        invalidation["access_result"] = "denied_pre_freeze"
        require_ledger_append(invalidation, append)
        raise RuntimeError("sealed access before the freeze invalidated the family")
    validate_freeze_certificate(certificate)
    return require_ledger_append(event, append)


def invalidate_accidental_operation(
    family_id: str,
    invalidated: Sequence[str],
    event: Mapping[str, Any],
    append: LedgerAppend,
    reserve_order: Mapping[str, Sequence[str]] = RESERVE_ORDER,
) -> str:
    """Record accidental work and return its predeclared replacement."""
    if event.get("event_kind") not in {"execution", "outcome_view"}:
        raise ValueError("only execution or outcome access forces invalidation")
    replacement = select_replacement(
        family_id,
        (*invalidated, family_id),
        reserve_order,
    )
    invalidation = dict(event)
    invalidation.update(
        {
            "event_kind": "invalidation",
            "access_result": "invalidated",
            "replacement_family_id": replacement,
        }
    )
    require_ledger_append(invalidation, append)
    return replacement


def require_registered_replacement(
    replacement_family_id: str, registry_entry: Mapping[str, Any]
) -> None:
    """Require replacement metadata before any outcome access."""
    families = registry_entry.get("family_ids")
    if not isinstance(families, list) or replacement_family_id not in families:
        raise RuntimeError("the replacement is not registered before outcome access")


def validate_external_registry_entry(
    entry: Mapping[str, Any],
    *,
    expected_sequence: int,
    previous_sha256: str | None,
) -> str:
    """Validate one immutable external registry chain entry."""
    if set(entry) != REGISTRY_FIELDS:
        raise ValueError("the external registry entry has invalid fields")
    _validate_chain(entry, expected_sequence, previous_sha256)
    if entry["publisher_role"] != "holdout_custodian":
        raise ValueError("the external registry publisher is invalid")
    for field in (
        "manifest_ciphertext_sha256",
        "manifest_sha256",
        "certificate_sha256",
        "research_gate_report_sha256",
    ):
        if not _is_hex(entry[field], 64):
            raise ValueError(f"the external registry has an invalid {field}")
    if not isinstance(entry["family_ids"], list) or not entry["family_ids"]:
        raise ValueError("the external registry misses its families")
    return hashlib.sha256(canonical_artifact_bytes(dict(entry))).hexdigest()


def validate_contamination_ledger_entry(
    entry: Mapping[str, Any],
    *,
    expected_sequence: int,
    previous_sha256: str | None,
) -> str:
    """Validate one immutable contamination ledger chain entry."""
    if set(entry) != LEDGER_FIELDS:
        raise ValueError("the contamination ledger entry has invalid fields")
    _validate_chain(entry, expected_sequence, previous_sha256)
    if entry["event_kind"] not in EVENT_KINDS:
        raise ValueError("the contamination ledger event kind is invalid")
    if not _is_hex(entry["certificate_sha256"], 64):
        raise ValueError("the contamination ledger certificate digest is invalid")
    if not _is_hex(entry["manifest_sha256"], 64):
        raise ValueError("the contamination ledger manifest digest is invalid")
    if not isinstance(entry["root_ids"], list):
        raise ValueError("the contamination ledger root list is invalid")
    return hashlib.sha256(canonical_artifact_bytes(dict(entry))).hexdigest()


def ledger_sidecar(entry: Mapping[str, Any]) -> bytes:
    """Return the exact contamination ledger sidecar."""
    digest = hashlib.sha256(canonical_artifact_bytes(dict(entry))).hexdigest()
    return f"{digest}  contamination-ledger-entry.json\n".encode("ascii")


def _validate_chain(
    entry: Mapping[str, Any], expected_sequence: int, previous_sha256: str | None
) -> None:
    """Require one sequence and its exact predecessor."""
    if entry.get("schema_version") != 1:
        raise ValueError("the external chain schema is invalid")
    if entry.get("sequence") != expected_sequence:
        raise ValueError("the external chain sequence is invalid")
    expected_previous = None if expected_sequence == 1 else previous_sha256
    if entry.get("previous_sha256") != expected_previous:
        raise ValueError("the external chain predecessor is invalid")
    if expected_sequence > 1 and not _is_hex(previous_sha256, 64):
        raise ValueError("the external chain predecessor digest is invalid")


def decrypt_manifest(
    ciphertext: Path,
    output: Path,
    *,
    identity: Path,
    ciphertext_sha256: str,
    manifest_sha256: str,
    retrieval_event: Mapping[str, Any],
    append_ledger: LedgerAppend,
    journal: Path,
) -> None:
    """Decrypt only after a verified retrieval ledger append."""
    if _file_sha256(ciphertext) != ciphertext_sha256:
        raise ValueError("the sealed manifest ciphertext digest has changed")
    receipt = require_ledger_append(retrieval_event, append_ledger)
    result = subprocess.run(
        (
            "age",
            "--decrypt",
            "--identity",
            str(identity),
            "--output",
            str(output),
            str(ciphertext),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "operation": "decrypt",
        "ledger_entry_sha256": receipt.entry_sha256,
        "returncode": result.returncode,
        "ciphertext_sha256": ciphertext_sha256,
    }
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("the protected manifest decryption failed")
    if _file_sha256(output) != manifest_sha256:
        output.unlink(missing_ok=True)
        raise ValueError("the sealed manifest plaintext digest has changed")


def _file_sha256(path: Path) -> str:
    """Return one exact file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_hex(value: object, length: Literal[40, 64]) -> bool:
    """Return whether one value is a lower-case hexadecimal digest."""
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )
