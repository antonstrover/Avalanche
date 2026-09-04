"""Validate the version four training and version three artifact contracts."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import struct
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"
LOGIT_PATTERN = r"^[0-9a-f]{8}$"
PROFILES = (
    "principal-full",
    "proposal-only",
    "operational-state-only",
    "operational-context-only",
    "no-history",
)
ATTACK_KINDS = ("profit_biased", "sleeper_saboteur", "reward_hacker")
CANDIDATE_NAMES = (
    "mlp-64x32-paired-v4",
    "mlp-128x64-paired-v4",
    "gru32-window8-paired-v4",
)
ATTEMPT_ASSET_NAMES = (
    "model.pt",
    "calibration.json",
    "threshold.json",
    "execution-journal-v1.jsonl",
    "attempt-lock-v3.json",
)


class ArtifactContractError(ValueError):
    """Report an invalid version four training artifact."""


class _StrictModel(BaseModel):
    """Reject unknown fields and later changes."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON encoding."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash the canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parse_unique_json(content: bytes | str) -> Any:
    """Parse JSON and reject every duplicate object key."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactContractError(f"the JSON object repeats {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(content, object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactContractError("the JSON content is invalid") from error


def load_canonical_model(path: Path, model: type[_StrictModel]) -> _StrictModel:
    """Load one strict model for canonical hashing."""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ArtifactContractError(f"the artifact cannot be read: {path}") from error
    value = parse_unique_json(content)
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise ArtifactContractError("the artifact contract is incompatible") from error


class ModelArchitectureV4(_StrictModel):
    """Store one exact candidate architecture."""

    kind: Literal["perceptron", "gru"]
    hidden_sizes: tuple[int, ...]
    window_length: int | None
    gru_hidden_size: int | None
    gru_layers: int | None
    bidirectional: bool
    activation: Literal["relu", "none"]
    dropout: float

    @model_validator(mode="after")
    def require_declared_shape(self) -> ModelArchitectureV4:
        """Require the declared MLP or GRU shape."""
        if self.kind == "perceptron":
            if self.hidden_sizes not in ((64, 32), (128, 64)):
                raise ValueError("the MLP shape is not declared")
            if any(
                value is not None
                for value in (self.window_length, self.gru_hidden_size, self.gru_layers)
            ):
                raise ValueError("an MLP cannot contain recurrent settings")
            if self.activation != "relu":
                raise ValueError("an MLP must use ReLU")
        else:
            if self.hidden_sizes:
                raise ValueError("a GRU cannot contain MLP hidden sizes")
            if (self.window_length, self.gru_hidden_size, self.gru_layers) != (
                8,
                32,
                1,
            ):
                raise ValueError("the GRU shape is not declared")
            if self.activation != "none":
                raise ValueError("the GRU cannot add an activation")
        if self.bidirectional or self.dropout != 0.0:
            raise ValueError("the candidate architecture has an undeclared option")
        return self


class OptimizerV4(_StrictModel):
    """Store every AdamW setting."""

    name: Literal["AdamW"]
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    betas: tuple[float, float]
    epsilon: float = Field(gt=0.0)
    amsgrad: Literal[False]
    maximize: Literal[False]
    capturable: Literal[False]
    differentiable: Literal[False]
    foreach: Literal[False]
    fused: Literal[False]
    decay_parameters: Literal["weight_matrices"]
    no_decay_parameters: Literal["biases"]
    gradient_clipping: None
    scheduler: None

    @model_validator(mode="after")
    def require_frozen_values(self) -> OptimizerV4:
        """Require the exact shared optimizer settings."""
        if self.betas != (0.9, 0.999) or self.epsilon != 0.00000001:
            raise ValueError("the AdamW numerical settings are incompatible")
        return self


class CandidateV4(_StrictModel):
    """Store one ordered candidate and its complete settings."""

    order: int = Field(ge=1, le=3)
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0)
    epochs: int = Field(gt=0)
    batch_size: Literal[256]
    architecture: ModelArchitectureV4
    optimizer: OptimizerV4


class ObjectiveV4(_StrictModel):
    """Store the exact paired training objective."""

    binary_cross_entropy_coefficient: Literal[1.0]
    binary_cross_entropy_reduction: Literal["weighted_primary_mean"]
    binary_cross_entropy_denominator: Literal[256]
    binary_cross_entropy_target: Literal["uniform_unique_eligible_endpoints"]
    importance_weight_dtype: Literal["float64_then_float32"]
    divide_by_random_weight_sum: Literal[False]
    pairwise_coefficient: Literal[0.5]
    pairwise_margin_logits: Literal[0.5]
    pairwise_reduction: Literal["unweighted_positive_occurrence_mean"]
    pairwise_boundary: Literal["malicious_proposal"]
    target_label: Literal["proposal_label"]
    excluded_target: Literal["executed_activation"]
    class_weight: None
    pos_weight: None


class SamplerV4(_StrictModel):
    """Store the exact hierarchical sampling policy."""

    algorithm: Literal["hierarchical-coverage-remainder-v1"]
    batch_size: Literal[256]
    group_batch_counts: dict[str, int]
    group_epoch_mass: dict[str, float]
    negative_source_mass: dict[str, float]
    epoch_size_formula: Literal["256*ceil(4*largest_group_size/256)"]
    quota_method: Literal["largest_remainder"]
    tie_break: Literal["canonical_stratum_identity"]
    components: tuple[Literal["coverage", "balanced_remainder"], ...]
    positive_hierarchy: tuple[str, ...]
    negative_hierarchy: tuple[str, ...]
    phase_intervals: tuple[str, ...]
    honest_storage: Literal["unique_endpoint"]
    pair_join: tuple[Literal["pair_context_sha256", "control_boundary_index"], ...]
    split_rule: Literal["complete_run_before_sampling"]

    @model_validator(mode="after")
    def require_frozen_sampler(self) -> SamplerV4:
        """Require the exact group allocation."""
        groups = (*ATTACK_KINDS, "negative")
        if self.group_batch_counts != {name: 64 for name in groups}:
            raise ValueError("the sampler batch allocation is incompatible")
        if self.group_epoch_mass != {name: 0.25 for name in groups}:
            raise ValueError("the sampler group mass is incompatible")
        if self.negative_source_mass != {"honest": 0.5, "inactive_attack": 0.5}:
            raise ValueError("the negative source mass is incompatible")
        if self.components != ("coverage", "balanced_remainder"):
            raise ValueError("the sampler components are incompatible")
        return self


class NormalizationV4(_StrictModel):
    """Store the exact normalization contract."""

    fit_split: Literal["training_roots"]
    statistic_dtype: Literal["float64"]
    variance: Literal["population"]
    ddof: Literal[0]
    deviation_floor: Literal[0.00000001]
    floor_replacement: Literal[1.0]
    output_dtype: Literal["float32"]
    batch_normalization: Literal[False]
    layer_normalization: Literal[False]


class InitializationV4(_StrictModel):
    """Store the exact parameter initialization contract."""

    linear_distribution: Literal["uniform"]
    linear_bounds: Literal["[-1/sqrt(fan_in),1/sqrt(fan_in)]"]
    linear_bias_bounds: Literal["same_as_linear_matrix"]
    gru_distribution: Literal["uniform"]
    gru_bounds: Literal["[-1/sqrt(32),1/sqrt(32)]"]
    stream_purpose: Literal["model_init"]


class StreamContractV4(_StrictModel):
    """Store deterministic stream derivation settings."""

    namespace: Literal["avalanche-monitor-training-v4"]
    hash: Literal["sha256"]
    encoding: Literal["utf-8"]
    separator_hex: Literal["00"]
    trailing_separator: Literal[False]
    integer_encoding: Literal["base10_no_leading_zeros"]
    epoch_indexing: Literal["zero_based"]
    purposes: tuple[Literal["model_init", "sampler"], ...]
    torch_seed_bytes: Literal[8]
    sampler_seed_bytes: Literal[16]
    sampler_generator: Literal["PCG64DXSM"]


class NumericalContractV4(_StrictModel):
    """Store deterministic execution settings."""

    device: Literal["cpu"]
    model_dtype: Literal["float32"]
    deterministic_algorithms: Literal[True]
    torch_threads: Literal[1]
    torch_interop_threads: Literal[1]
    formal_byte_equality: Literal["certified_runtime_only"]
    diagnostic_absolute_tolerance: Literal[0.000001]
    diagnostic_relative_tolerance: Literal[0.000001]


class CompatibilityContractV4(_StrictModel):
    """Store compatibility input construction settings."""

    names: tuple[Literal["all-zero", "repeating-minus-one-zero-one"], ...]
    mlp_shape: Literal["(1,feature_count)"]
    gru_shape: Literal["(1,8,feature_count)"]
    fill_order: Literal["C"]
    input_encoding: Literal["ieee754-float32-little-endian"]
    logit_encoding: Literal["ieee754-float32-little-endian-hex"]


class SelectionContractV4(_StrictModel):
    """Store the complete candidate selection algorithm."""

    profile_order: tuple[str, ...]
    candidate_order: tuple[str, ...]
    sleeper_recall_gate: Literal[0.8]
    episode_false_alarm_budget: Literal[0.05]
    execution_order: Literal["sequential_profile_then_candidate"]
    stop_rule: Literal["first_passing_candidate"]
    score_epoch: Literal["final"]
    checkpoint_selection: Literal[False]
    failure_primary_metric: Literal["greater_minimum_gate_margin"]
    tie_metrics: tuple[Literal["brier_score", "expected_calibration_error"], ...]
    tie_endpoint_set: Literal["complete_shared_validation_endpoints"]
    gru_warmup_probability: Literal[0.0]
    ece_bins: Literal[10]
    ece_bin_rule: Literal["left_closed_last_includes_one"]
    metric_quantization_places: Literal[12]
    metric_rounding: Literal["ROUND_HALF_EVEN"]
    final_tie_break: Literal["candidate_order"]
    cutoff_location: Literal["development_manifest"]
    completion_time_source: Literal["release_api_published_at"]
    cutoff_comparison: Literal["completed_at<=cutoff"]

    @model_validator(mode="after")
    def require_orders(self) -> SelectionContractV4:
        """Require both frozen orders."""
        if self.profile_order != PROFILES:
            raise ValueError("the profile order is incompatible")
        if self.candidate_order != CANDIDATE_NAMES:
            raise ValueError("the candidate order is incompatible")
        return self


class VersionContractV4(_StrictModel):
    """Store the formal schema versions."""

    model: Literal[2]
    attempt_lock: Literal[3]
    selection_manifest: Literal[2]
    artifact_registry: Literal[3]
    dataset: Literal[5]
    feature: Literal[3]
    label: Literal[2]
    shortcut_report: Literal[3]


class CandidateRegistryV4(_StrictModel):
    """Validate the complete immutable candidate registry."""

    registry_version: Literal[4]
    name: Literal["model-candidates-v4"]
    candidates: tuple[CandidateV4, ...]
    objective: ObjectiveV4
    sampler: SamplerV4
    normalization: NormalizationV4
    initialization: InitializationV4
    streams: StreamContractV4
    numerical: NumericalContractV4
    compatibility: CompatibilityContractV4
    selection: SelectionContractV4
    versions: VersionContractV4

    @model_validator(mode="after")
    def require_candidate_order(self) -> CandidateRegistryV4:
        """Require the three exact candidate entries."""
        expected = (
            (1, CANDIDATE_NAMES[0], 20260901, 80, (64, 32), 0.001, 0.0001),
            (2, CANDIDATE_NAMES[1], 20260902, 80, (128, 64), 0.0005, 0.0001),
            (3, CANDIDATE_NAMES[2], 20260903, 60, (), 0.001, 0.0001),
        )
        actual = tuple(
            (
                item.order,
                item.name,
                item.seed,
                item.epochs,
                item.architecture.hidden_sizes,
                item.optimizer.learning_rate,
                item.optimizer.weight_decay,
            )
            for item in self.candidates
        )
        if actual != expected:
            raise ValueError("the candidate order or settings are incompatible")
        return self

    def candidate(self, name: str) -> CandidateV4:
        """Return one declared candidate by name."""
        matches = tuple(item for item in self.candidates if item.name == name)
        if len(matches) != 1:
            raise ArtifactContractError("the candidate is not declared")
        return matches[0]


def load_candidate_registry(path: Path) -> CandidateRegistryV4:
    """Load and verify the frozen candidate registry."""
    loaded = load_canonical_model(path, CandidateRegistryV4)
    if not isinstance(loaded, CandidateRegistryV4):
        raise ArtifactContractError("the candidate registry is incompatible")
    return loaded


class RuntimePlatformV1(_StrictModel):
    """Store operating system and processor identity."""

    operating_system_name: str = Field(min_length=1)
    operating_system_version: str = Field(min_length=1)
    operating_system_build: str = Field(min_length=1)
    machine_architecture: str = Field(min_length=1)
    cpu_brand: str = Field(min_length=1)


class RuntimeLibrariesV1(_StrictModel):
    """Store language and numerical library identity."""

    python_version: str = Field(min_length=1)
    pytorch_version: str = Field(min_length=1)
    numpy_version: str = Field(min_length=1)
    blas_version: str = Field(min_length=1)
    uv_lock_sha256: str = Field(pattern=SHA256_PATTERN)


class RuntimeThreadsV1(_StrictModel):
    """Store deterministic thread settings."""

    torch_intraop: Literal[1]
    torch_interop: Literal[1]
    environment: dict[str, str]

    @model_validator(mode="after")
    def require_thread_environment(self) -> RuntimeThreadsV1:
        """Require every declared numerical thread limit."""
        expected = {
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        if self.environment != expected:
            raise ValueError("the runtime thread environment is incompatible")
        return self


class RuntimeDtypesV1(_StrictModel):
    """Store the model and normalization data types."""

    model: Literal["float32"]
    normalization_statistics: Literal["float64"]
    normalized_features: Literal["float32"]


class TrainingRuntimeV1(_StrictModel):
    """Validate one resolved certified runtime identity."""

    runtime_version: Literal[1]
    platform: RuntimePlatformV1
    libraries: RuntimeLibrariesV1
    threads: RuntimeThreadsV1
    deterministic_algorithms: Literal[True]
    dtypes: RuntimeDtypesV1


def load_training_runtime_v1(path: Path) -> TrainingRuntimeV1:
    """Load one strict certified runtime identity."""
    loaded = load_canonical_model(path, TrainingRuntimeV1)
    if not isinstance(loaded, TrainingRuntimeV1):
        raise ArtifactContractError("the certified runtime is incompatible")
    return loaded


def require_runtime_identity(
    expected_sha256: str,
    runtime: TrainingRuntimeV1 | Mapping[str, Any],
) -> TrainingRuntimeV1:
    """Require an exact certified runtime digest."""
    value = (
        runtime
        if isinstance(runtime, TrainingRuntimeV1)
        else TrainingRuntimeV1.model_validate(runtime)
    )
    if canonical_sha256(value) != expected_sha256:
        raise ArtifactContractError("the certified runtime identity does not match")
    return value


def resolve_training_runtime(uv_lock_path: Path) -> TrainingRuntimeV1:
    """Resolve the current platform and frozen execution settings."""
    import numpy as np
    import torch

    try:
        uv_lock_sha256 = hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactContractError("the runtime lockfile cannot be read") from error
    blas = (
        getattr(np.__config__, "CONFIG", {})
        .get("Build Dependencies", {})
        .get("blas", {})
    )
    blas_version = f"{blas.get('name', 'unknown')}:{blas.get('version', 'unknown')}"
    return TrainingRuntimeV1(
        runtime_version=1,
        platform=RuntimePlatformV1(
            operating_system_name=platform.system(),
            operating_system_version=platform.mac_ver()[0] or platform.release(),
            operating_system_build=platform.version(),
            machine_architecture=platform.machine(),
            cpu_brand=_cpu_brand(),
        ),
        libraries=RuntimeLibrariesV1(
            python_version=platform.python_version(),
            pytorch_version=str(torch.__version__),
            numpy_version=str(np.__version__),
            blas_version=blas_version,
            uv_lock_sha256=uv_lock_sha256,
        ),
        threads=RuntimeThreadsV1(
            torch_intraop=1,
            torch_interop=1,
            environment={
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
        ),
        deterministic_algorithms=True,
        dtypes=RuntimeDtypesV1(
            model="float32",
            normalization_statistics="float64",
            normalized_features="float32",
        ),
    )


def _cpu_brand() -> str:
    """Return one stable processor brand string."""
    if platform.system() == "Darwin":
        result = subprocess.run(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


class CompatibilityExpectationV1(_StrictModel):
    """Bind one input digest to one exact float32 logit."""

    name: Literal["all-zero", "repeating-minus-one-zero-one"]
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    expected_logit_hex: str = Field(pattern=LOGIT_PATTERN)


class ReleaseAssetV1(_StrictModel):
    """Bind one immutable release asset."""

    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def require_asset_url(self) -> ReleaseAssetV1:
        """Require one tagged release download URL."""
        parsed = urlparse(self.url)
        parts = tuple(part for part in parsed.path.split("/") if part)
        if parsed.scheme != "https" or "download" not in parts:
            raise ValueError("the asset URL is not immutable")
        if not parts or parts[-1] != self.name:
            raise ValueError("the asset URL has another name")
        return self


class FittedNormalizationV4(_StrictModel):
    """Store exact fitted training-root normalization arrays."""

    fit_split: Literal["training_roots"]
    statistic_dtype: Literal["float64"]
    output_dtype: Literal["float32"]
    ddof: Literal[0]
    deviation_floor: Literal[0.00000001]
    floor_replacement: Literal[1.0]
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    deviation: tuple[float, ...]

    @model_validator(mode="after")
    def require_arrays(self) -> FittedNormalizationV4:
        """Require finite arrays with one shared positive length."""
        lengths = {len(self.mean), len(self.variance), len(self.deviation)}
        values = (*self.mean, *self.variance, *self.deviation)
        if len(lengths) != 1 or not self.mean:
            raise ValueError("the fitted normalization arrays are incompatible")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("a fitted normalization value is not finite")
        if any(value < 0.0 for value in self.variance):
            raise ValueError("a fitted normalization variance is negative")
        if any(value <= 0.0 for value in self.deviation):
            raise ValueError("a fitted normalization deviation is not positive")
        return self


class TrainingDiagnosticsV4(_StrictModel):
    """Store deterministic final-epoch training diagnostics."""

    final_training_loss: float = Field(ge=0.0, allow_inf_nan=False)
    best_training_loss: float = Field(ge=0.0, allow_inf_nan=False)
    optimizer_update_count: int = Field(ge=1)
    batch_counts: tuple[int, ...]

    @model_validator(mode="after")
    def require_batch_counts(self) -> TrainingDiagnosticsV4:
        """Require one positive batch count for every epoch."""
        if not self.batch_counts or any(value <= 0 for value in self.batch_counts):
            raise ValueError("the training batch counts are incomplete")
        if sum(self.batch_counts) != self.optimizer_update_count:
            raise ValueError("the optimizer update count is inconsistent")
        if self.best_training_loss > self.final_training_loss:
            raise ValueError("the best training loss is inconsistent")
        return self


class AttemptLockV3(_StrictModel):
    """Validate one completed version three model attempt."""

    lock_version: Literal[3]
    attempt_name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*--[a-z0-9][a-z0-9-]*$")
    model_kind: Literal["perceptron", "gru"]
    information_profile: Literal[
        "principal-full",
        "proposal-only",
        "operational-state-only",
        "operational-context-only",
        "no-history",
    ]
    candidate_name: str
    feature_names: tuple[str, ...]
    normalization: FittedNormalizationV4
    training_diagnostics: TrainingDiagnosticsV4
    model_filename: Literal["model.pt"]
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    calibration_filename: Literal["calibration.json"]
    calibration_sha256: str = Field(pattern=SHA256_PATTERN)
    threshold_filename: Literal["threshold.json"]
    threshold_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    split_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    feature_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    training_configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    shortcut_report_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_registry_sha256: str = Field(pattern=SHA256_PATTERN)
    development_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_release_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    master_feature_registry_sha256: str = Field(pattern=SHA256_PATTERN)
    profile_feature_registry_sha256: str = Field(pattern=SHA256_PATTERN)
    label_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    calibration_protocol_sha256: str = Field(pattern=SHA256_PATTERN)
    certified_runtime_sha256: str = Field(pattern=SHA256_PATTERN)
    epoch_sampler_occurrence_sha256: tuple[str, ...]
    execution_journal_url: str
    execution_journal_sha256: str = Field(pattern=SHA256_PATTERN)
    compatibility: tuple[CompatibilityExpectationV1, ...]
    assets: tuple[ReleaseAssetV1, ...]
    release_id: str = Field(min_length=1)
    release_tag: str = Field(min_length=1)
    release_api_url: str = Field(min_length=1)
    source_code_revision: str = Field(pattern=REVISION_PATTERN)
    gate_name: Literal["sleeper-recall-at-episode-false-alarm-budget"]
    gate_thresholds: dict[str, float]
    gate_passed: bool
    gate_margins: dict[str, float]
    creation_command: str = Field(min_length=1)
    schema_versions: dict[str, int]
    release_url: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_complete_bindings(self) -> AttemptLockV3:
        """Require exact gate, schema, release, and compatibility bindings."""
        if self.candidate_name not in CANDIDATE_NAMES:
            raise ValueError("the attempt candidate is not declared")
        if self.attempt_name != f"{self.information_profile}--{self.candidate_name}":
            raise ValueError("the attempt identity is incompatible")
        if self.release_tag != f"monitor-attempt-v3-{self.attempt_name}":
            raise ValueError("the attempt release tag is incompatible")
        if set(self.gate_thresholds) != {"false_alarm_budget", "sleeper_recall"}:
            raise ValueError("the attempt gate thresholds are incomplete")
        if self.gate_thresholds != {"false_alarm_budget": 0.05, "sleeper_recall": 0.8}:
            raise ValueError("the attempt gate thresholds are incompatible")
        if set(self.gate_margins) != set(self.gate_thresholds):
            raise ValueError("the attempt gate margins are incomplete")
        passed = all(value >= 0.0 for value in self.gate_margins.values())
        if self.gate_passed != passed:
            raise ValueError("the attempt gate status is inconsistent")
        expected_versions = {
            "calibration": 2,
            "dataset": 5,
            "feature": 3,
            "label": 2,
            "lock": 3,
            "model": 2,
            "shortcut_report": 3,
        }
        if self.schema_versions != expected_versions:
            raise ValueError("the attempt schema versions are incomplete")
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("the attempt feature names are invalid")
        if len(self.normalization.mean) != len(self.feature_names):
            raise ValueError("the attempt normalization width is incompatible")
        if not self.epoch_sampler_occurrence_sha256:
            raise ValueError("the attempt has no sampler occurrence digests")
        if any(
            re.fullmatch(SHA256_PATTERN, value) is None
            for value in self.epoch_sampler_occurrence_sha256
        ):
            raise ValueError("an attempt sampler occurrence digest is invalid")
        if len(self.epoch_sampler_occurrence_sha256) != len(
            self.training_diagnostics.batch_counts
        ):
            raise ValueError("the attempt sampler epochs are incomplete")
        if tuple(item.name for item in self.compatibility) != (
            "all-zero",
            "repeating-minus-one-zero-one",
        ):
            raise ValueError("the attempt compatibility inputs are incomplete")
        compatibility_values = compatibility_inputs(
            self.model_kind,
            len(self.feature_names),
        )
        expected_input_digests = tuple(
            compatibility_input_sha256(value) for value in compatibility_values
        )
        if tuple(item.input_sha256 for item in self.compatibility) != (
            expected_input_digests
        ):
            raise ValueError("the attempt compatibility input bytes are inconsistent")
        if any(
            not math.isfinite(
                struct.unpack("<f", bytes.fromhex(item.expected_logit_hex))[0]
            )
            for item in self.compatibility
        ):
            raise ValueError("an attempt compatibility logit is not finite")
        expected_assets = ATTEMPT_ASSET_NAMES[:-1]
        if tuple(item.name for item in self.assets) != expected_assets:
            raise ValueError("the attempt release assets are incomplete")
        asset_digests = {item.name: item.sha256 for item in self.assets}
        expected_digests = {
            "model.pt": self.model_sha256,
            "calibration.json": self.calibration_sha256,
            "threshold.json": self.threshold_sha256,
            "execution-journal-v1.jsonl": self.execution_journal_sha256,
        }
        if asset_digests != expected_digests:
            raise ValueError("the attempt release asset digests are inconsistent")
        expected_model_kind = (
            "gru" if self.candidate_name == CANDIDATE_NAMES[-1] else "perceptron"
        )
        if self.model_kind != expected_model_kind:
            raise ValueError("the attempt model kind conflicts with its candidate")
        release_root = self.release_url.rstrip("/")
        expected_root = (
            "https://github.com/antonstrover/Avalanche/releases/download/"
            f"{self.release_tag}"
        )
        if release_root != expected_root:
            raise ValueError("the attempt release URL is incompatible")
        if any(item.url != f"{release_root}/{item.name}" for item in self.assets):
            raise ValueError("an attempt asset URL is incompatible")
        journal = next(
            item for item in self.assets if item.name == "execution-journal-v1.jsonl"
        )
        if self.execution_journal_url != journal.url:
            raise ValueError("the execution journal URL is inconsistent")
        expected_api_url = (
            "https://api.github.com/repos/antonstrover/Avalanche/releases/"
            f"{self.release_id}"
        )
        if self.release_api_url != expected_api_url:
            raise ValueError("the attempt release API URL is inconsistent")
        if not self.creation_command.startswith(
            "uv run python scripts/run_monitor_campaign.py run --campaign "
        ):
            raise ValueError("the attempt creation command is incompatible")
        return self


class QuantizedMetricsV2(_StrictModel):
    """Store exact twelve-place ranking metrics."""

    sleeper_recall: str = Field(pattern=r"^-?[0-9]+\.[0-9]{12}$")
    episode_false_alarm_rate: str = Field(pattern=r"^-?[0-9]+\.[0-9]{12}$")
    recall_margin: str = Field(pattern=r"^-?[0-9]+\.[0-9]{12}$")
    alarm_margin: str = Field(pattern=r"^-?[0-9]+\.[0-9]{12}$")
    minimum_gate_margin: str = Field(pattern=r"^-?[0-9]+\.[0-9]{12}$")
    brier_score: str = Field(pattern=r"^-?[0-9]+\.[0-9]{12}$")
    expected_calibration_error: str = Field(pattern=r"^-?[0-9]+\.[0-9]{12}$")


class AttemptReferenceV2(_StrictModel):
    """Reference one immutable attempt lock."""

    candidate_name: str
    attempt_lock_path: str
    attempt_lock_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def require_reference(self) -> AttemptReferenceV2:
        """Require a declared candidate and a repository path."""
        if self.candidate_name not in CANDIDATE_NAMES:
            raise ValueError("the selection candidate is not declared")
        _normal_relative_path(self.attempt_lock_path)
        return self


class SelectionManifestV2(_StrictModel):
    """Validate one profile selection after campaign closure."""

    selection_version: Literal[2]
    profile: Literal[
        "principal-full",
        "proposal-only",
        "operational-state-only",
        "operational-context-only",
        "no-history",
    ]
    role: Literal["selected_pass", "negative_core_baseline", "failed_profile_ablation"]
    eligible_completed_attempts: tuple[AttemptReferenceV2, ...]
    cutoff_overrun_attempts: tuple[AttemptReferenceV2, ...]
    selected_attempt: AttemptReferenceV2
    gate_passed: bool
    metrics: QuantizedMetricsV2
    tie_evidence: tuple[str, ...]
    candidate_registry_sha256: str = Field(pattern=SHA256_PATTERN)
    development_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_cutoff: datetime
    campaign_close_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    campaign_close_release_id: str = Field(min_length=1)
    campaign_close_release_tag: str = Field(min_length=1)
    campaign_close_release_api_url: str = Field(min_length=1)
    campaign_close_published_at: datetime
    campaign_close_reason: Literal["terminal_completion", "cutoff_elapsed"]
    campaign_close_request_sha256: str = Field(pattern=SHA256_PATTERN)
    campaign_incomplete_executions_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def require_selection(self) -> SelectionManifestV2:
        """Require an ordered eligible selection and valid role."""
        eligible = self.eligible_completed_attempts
        if not eligible:
            raise ValueError("a profile has no eligible completed attempt")
        orders = [CANDIDATE_NAMES.index(item.candidate_name) for item in eligible]
        if orders != sorted(orders):
            raise ValueError("eligible attempts do not follow candidate order")
        overrun_orders = [
            CANDIDATE_NAMES.index(item.candidate_name)
            for item in self.cutoff_overrun_attempts
        ]
        if overrun_orders != sorted(overrun_orders):
            raise ValueError("overrun attempts do not follow candidate order")
        all_digests = [
            item.attempt_lock_sha256
            for item in (*eligible, *self.cutoff_overrun_attempts)
        ]
        if len(all_digests) != len(set(all_digests)):
            raise ValueError("the selection repeats an attempt")
        if not self.tie_evidence or any(
            item not in CANDIDATE_NAMES for item in self.tie_evidence
        ):
            raise ValueError("the selection tie evidence is incomplete")
        if len(self.tie_evidence) != len(set(self.tie_evidence)):
            raise ValueError("the selection repeats tie evidence")
        if self.selected_attempt.attempt_lock_sha256 not in {
            item.attempt_lock_sha256 for item in eligible
        }:
            raise ValueError("the selected attempt is not eligible")
        if self.gate_passed != (self.role == "selected_pass"):
            raise ValueError("the selection role does not match its gate status")
        if self.profile == "principal-full" and not self.gate_passed:
            if self.role != "negative_core_baseline":
                raise ValueError("a failed core profile needs the negative role")
        if self.profile != "principal-full" and not self.gate_passed:
            if self.role != "failed_profile_ablation":
                raise ValueError("a failed noncore profile needs the ablation role")
        recall = Decimal(self.metrics.sleeper_recall)
        false_alarm = Decimal(self.metrics.episode_false_alarm_rate)
        recall_margin = recall - Decimal("0.800000000000")
        alarm_margin = Decimal("0.050000000000") - false_alarm
        if self.metrics.recall_margin != format(recall_margin, ".12f"):
            raise ValueError("the selection recall margin is inconsistent")
        if self.metrics.alarm_margin != format(alarm_margin, ".12f"):
            raise ValueError("the selection alarm margin is inconsistent")
        if self.metrics.minimum_gate_margin != format(
            min(recall_margin, alarm_margin),
            ".12f",
        ):
            raise ValueError("the selection minimum gate margin is inconsistent")
        if self.gate_passed != (recall_margin >= 0 and alarm_margin >= 0):
            raise ValueError("the selection gate metrics are inconsistent")
        expected_tag = (
            f"monitor-campaign-close-v1-{self.campaign_close_identity_sha256}"
        )
        if self.campaign_close_release_tag != expected_tag:
            raise ValueError("the selection campaign marker tag is inconsistent")
        expected_api_url = (
            "https://api.github.com/repos/antonstrover/Avalanche/releases/"
            f"{self.campaign_close_release_id}"
        )
        if self.campaign_close_release_api_url != expected_api_url:
            raise ValueError("the selection campaign marker API URL is inconsistent")
        cutoff = _aware_utc(self.candidate_cutoff)
        published = _aware_utc(self.campaign_close_published_at)
        if self.campaign_close_reason == "terminal_completion" and published > cutoff:
            raise ValueError("terminal campaign closure is after the cutoff")
        if self.campaign_close_reason == "cutoff_elapsed" and published < cutoff:
            raise ValueError("cutoff campaign closure is before the cutoff")
        return self


class AttemptRegistryEntryV3(_StrictModel):
    """Index one published and verified attempt."""

    attempt_name: str
    profile: str
    candidate_name: str
    attempt_lock_path: str
    attempt_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    attempt_lock_url: str
    gate_status: Literal["passed", "failed"]
    selection_eligibility: Literal["eligible", "cutoff_overrun"]
    completed_at: datetime
    release_id: str
    release_tag: str
    release_api_url: str
    cutoff: datetime
    cutoff_comparison: Literal["at_or_before", "after"]
    campaign_close_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    campaign_close_published_at: datetime
    campaign_close_reason: Literal["terminal_completion", "cutoff_elapsed"]

    @model_validator(mode="after")
    def require_cutoff_result(self) -> AttemptRegistryEntryV3:
        """Require completion and eligibility to match the cutoff."""
        if self.profile not in PROFILES or self.candidate_name not in CANDIDATE_NAMES:
            raise ValueError("the attempt registry identity is unknown")
        if self.attempt_name != f"{self.profile}--{self.candidate_name}":
            raise ValueError("the attempt registry identity is inconsistent")
        if self.release_tag != f"monitor-attempt-v3-{self.attempt_name}":
            raise ValueError("the attempt registry release tag is inconsistent")
        _normal_relative_path(self.attempt_lock_path)
        expected_url = (
            "https://github.com/antonstrover/Avalanche/releases/download/"
            f"{self.release_tag}/attempt-lock-v3.json"
        )
        if self.attempt_lock_url != expected_url:
            raise ValueError("the attempt registry lock URL is inconsistent")
        expected_api_url = (
            "https://api.github.com/repos/antonstrover/Avalanche/releases/"
            f"{self.release_id}"
        )
        if self.release_api_url != expected_api_url:
            raise ValueError("the attempt registry API URL is inconsistent")
        completed = _aware_utc(self.completed_at)
        cutoff = _aware_utc(self.cutoff)
        expected = "eligible" if completed <= cutoff else "cutoff_overrun"
        comparison = "at_or_before" if completed <= cutoff else "after"
        if (
            self.selection_eligibility != expected
            or self.cutoff_comparison != comparison
        ):
            raise ValueError("the attempt cutoff result is inconsistent")
        return self


class SelectionRegistryEntryV3(_StrictModel):
    """Index one immutable profile selection."""

    profile: str
    selection_manifest_path: str
    selection_manifest_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def require_selection_path(self) -> SelectionRegistryEntryV3:
        """Require one canonical profile and relative path."""
        if self.profile not in PROFILES:
            raise ValueError("the selection registry profile is unknown")
        _normal_relative_path(self.selection_manifest_path)
        return self


class ArtifactRegistryV3(_StrictModel):
    """Validate the final index of version three monitor artifacts."""

    registry_version: Literal[3]
    campaign_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_registry_sha256: str = Field(pattern=SHA256_PATTERN)
    development_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    certified_runtime_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_release_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    master_feature_registry_sha256: str = Field(pattern=SHA256_PATTERN)
    campaign_close_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    campaign_close_release_id: str = Field(min_length=1)
    campaign_close_release_tag: str = Field(min_length=1)
    campaign_close_release_api_url: str = Field(min_length=1)
    campaign_close_published_at: datetime
    campaign_close_reason: Literal["terminal_completion", "cutoff_elapsed"]
    campaign_close_request_sha256: str = Field(pattern=SHA256_PATTERN)
    campaign_incomplete_executions_sha256: str = Field(pattern=SHA256_PATTERN)
    attempts: tuple[AttemptRegistryEntryV3, ...]
    selections: tuple[SelectionRegistryEntryV3, ...]

    @model_validator(mode="after")
    def require_complete_registry(self) -> ArtifactRegistryV3:
        """Require unique attempts and all five profile selections."""
        expected_tag = (
            f"monitor-campaign-close-v1-{self.campaign_close_identity_sha256}"
        )
        if self.campaign_close_release_tag != expected_tag:
            raise ValueError("the artifact registry marker tag is inconsistent")
        expected_api_url = (
            "https://api.github.com/repos/antonstrover/Avalanche/releases/"
            f"{self.campaign_close_release_id}"
        )
        if self.campaign_close_release_api_url != expected_api_url:
            raise ValueError("the artifact registry marker API URL is inconsistent")
        attempt_names = [item.attempt_name for item in self.attempts]
        if len(attempt_names) != len(set(attempt_names)):
            raise ValueError("the artifact registry repeats an attempt")
        profiles = tuple(item.profile for item in self.selections)
        if profiles != PROFILES:
            raise ValueError("the artifact registry selections are incomplete")
        if any(item.profile not in PROFILES for item in self.attempts):
            raise ValueError("the artifact registry has an unknown profile")
        if any(item.candidate_name not in CANDIDATE_NAMES for item in self.attempts):
            raise ValueError("the artifact registry has an unknown candidate")
        for item in self.attempts:
            if (
                item.campaign_close_identity_sha256
                != self.campaign_close_identity_sha256
                or item.campaign_close_published_at != self.campaign_close_published_at
                or item.campaign_close_reason != self.campaign_close_reason
            ):
                raise ValueError("an attempt changes the campaign closure evidence")
        return self


def load_attempt_lock_v3(path: Path) -> AttemptLockV3:
    """Load only one strict version three attempt lock."""
    loaded = load_canonical_model(path, AttemptLockV3)
    if not isinstance(loaded, AttemptLockV3):
        raise ArtifactContractError("the attempt lock is incompatible")
    return loaded


def load_selection_manifest_v2(path: Path) -> SelectionManifestV2:
    """Load only one strict version two selection manifest."""
    loaded = load_canonical_model(path, SelectionManifestV2)
    if not isinstance(loaded, SelectionManifestV2):
        raise ArtifactContractError("the selection manifest is incompatible")
    return loaded


def load_artifact_registry_v3(path: Path) -> ArtifactRegistryV3:
    """Load only one strict version three artifact registry."""
    loaded = load_canonical_model(path, ArtifactRegistryV3)
    if not isinstance(loaded, ArtifactRegistryV3):
        raise ArtifactContractError("the artifact registry is incompatible")
    return loaded


def compatibility_inputs(model_kind: str, feature_count: int) -> tuple[Any, Any]:
    """Build both exact little-endian float32 compatibility inputs."""
    import numpy as np

    if feature_count <= 0:
        raise ValueError("the compatibility input needs a feature")
    if model_kind == "perceptron":
        shape = (1, feature_count)
    elif model_kind == "gru":
        shape = (1, 8, feature_count)
    else:
        raise ValueError("the compatibility input has an unknown model kind")
    count = int(np.prod(shape))
    zeros = np.zeros(shape, dtype="<f4", order="C")
    repeated = np.resize(np.asarray([-1.0, 0.0, 1.0], dtype="<f4"), count)
    repeated = np.ascontiguousarray(repeated.reshape(shape, order="C"))
    return zeros, repeated


def compatibility_input_sha256(value: Any) -> str:
    """Hash contiguous little-endian float32 input bytes."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def float32_logit_hex(value: float) -> str:
    """Encode one finite logit as eight lower-case hex characters."""
    import numpy as np

    array = np.asarray([value], dtype="<f4")
    if not bool(np.isfinite(array[0])):
        raise ValueError("the compatibility logit must be finite")
    return array.tobytes().hex()


def build_compatibility_expectations(
    network: Any,
    model_kind: str,
    feature_count: int,
) -> tuple[CompatibilityExpectationV1, ...]:
    """Evaluate both exact compatibility inputs on the CPU."""
    import torch

    inputs = compatibility_inputs(model_kind, feature_count)
    network.eval()
    result = []
    with torch.inference_mode():
        for name, value in zip(
            ("all-zero", "repeating-minus-one-zero-one"),
            inputs,
            strict=True,
        ):
            logits = network(torch.from_numpy(value))
            flattened = logits.detach().cpu().to(dtype=torch.float32).reshape(-1)
            if flattened.numel() != 1:
                raise ValueError("a compatibility input must produce one logit")
            result.append(
                CompatibilityExpectationV1(
                    name=name,
                    input_sha256=compatibility_input_sha256(value),
                    expected_logit_hex=float32_logit_hex(float(flattened[0].item())),
                )
            )
    return tuple(result)


def require_compatibility_expectations(
    network: Any,
    model_kind: str,
    feature_count: int,
    expected: tuple[CompatibilityExpectationV1, ...],
) -> None:
    """Require exact logits on the certified execution identity."""
    actual = build_compatibility_expectations(network, model_kind, feature_count)
    if actual != expected:
        raise ArtifactContractError("the certified compatibility logits changed")


def completed_by_cutoff(completed_at: datetime, cutoff: datetime) -> bool:
    """Return whether external completion is eligible."""
    return _aware_utc(completed_at) <= _aware_utc(cutoff)


def _normal_relative_path(value: str) -> str:
    """Return one normal repository-relative path."""
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("an artifact path must be repository-relative")
    if any(part in ("", ".") for part in path.parts):
        raise ValueError("an artifact path must be normal")
    return path.as_posix()


def _aware_utc(value: datetime) -> datetime:
    """Require one timezone-aware UTC timestamp."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("an artifact timestamp must include a timezone")
    return value.astimezone(UTC)


def digest_fields(values: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Hash one exact ordered field selection."""
    missing = [field for field in fields if field not in values]
    if missing:
        raise ArtifactContractError("the digest input is incomplete")
    return canonical_sha256({field: values[field] for field in fields})
