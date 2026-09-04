"""Resolve the four formal configuration components."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from avalanche.config.loader import ConfigLoadError
from avalanche.config.models import (
    ApprovalConfig,
    ControllerConfig,
    FallbackConfig,
    IntervalsConfig,
    MonitorConfig,
    MountainConfig,
    NumericsConfig,
    PopulationConfig,
    ResolvedConfig,
    RoutingConfig,
    ScenarioConfig,
)
from avalanche.config.paths import canonical_repository_path
from avalanche.config.provenance import ValueProvenance

Owner = Literal["mountain", "scenario", "controller", "monitor", "override"]
ProvenanceOwner = Literal[
    "mountain", "scenario", "controller", "monitor", "override", "resolver"
]


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MountainComponent(_Envelope):
    """Own the mountain and population values."""

    mountain: MountainConfig
    population: PopulationConfig
    routing: RoutingConfig = RoutingConfig()


class ScenarioComponent(_Envelope):
    """Own the scenario and episode values."""

    scenario: ScenarioConfig
    intervals: IntervalsConfig
    numerics: NumericsConfig
    seed: int = Field(ge=0, le=2**63 - 1)
    episode_duration_seconds: float = Field(gt=0.0)
    snapshot_interval_seconds: float = Field(gt=0.0)
    trace_level: Literal["debug", "decision", "summary"]


class ControllerComponent(_Envelope):
    """Own one controller value."""

    controller: ControllerConfig


class MonitorComponent(_Envelope):
    """Own one monitor and its adjudication policies."""

    monitor: MonitorConfig
    fallback: FallbackConfig
    approval: ApprovalConfig = ApprovalConfig()


class PopulationOverride(_Envelope):
    """Override only the population size."""

    skier_count: int = Field(gt=0)


class RuntimeOverride(_Envelope):
    """Override only the worker count."""

    worker_count: int = Field(ge=1)


class OverrideComponent(_Envelope):
    """Own the six permitted formal override paths."""

    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    episode_duration_seconds: float | None = Field(default=None, gt=0.0)
    population: PopulationOverride | None = None
    trace_level: Literal["debug", "decision", "summary"] | None = None
    output_root: str | None = None
    runtime: RuntimeOverride | None = None


_ENVELOPES: dict[Owner, type[_Envelope]] = {
    "mountain": MountainComponent,
    "scenario": ScenarioComponent,
    "controller": ControllerComponent,
    "monitor": MonitorComponent,
    "override": OverrideComponent,
}
_OWNER_KEYS: dict[Owner, frozenset[str]] = {
    "mountain": frozenset({"mountain", "population", "routing"}),
    "scenario": frozenset(
        {
            "scenario",
            "intervals",
            "numerics",
            "seed",
            "episode_duration_seconds",
            "snapshot_interval_seconds",
            "trace_level",
        }
    ),
    "controller": frozenset({"controller"}),
    "monitor": frozenset({"monitor", "fallback", "approval"}),
    "override": frozenset(
        {
            "seed",
            "episode_duration_seconds",
            "population",
            "trace_level",
            "output_root",
            "runtime",
        }
    ),
}
_OVERRIDE_PATHS = frozenset(
    {
        "/seed",
        "/episode_duration_seconds",
        "/population/skier_count",
        "/trace_level",
        "/output_root",
        "/runtime/worker_count",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "provenance",
        "resolved_configuration_sha256",
        "scientific_configuration_sha256",
    }
)


class ConfigurationResolutionError(Exception):
    """Report every configuration resolution error."""

    def __init__(self, errors: list[str] | tuple[str, ...]) -> None:
        self.errors = tuple(errors)
        super().__init__(
            "configuration resolution failed:\n- " + "\n- ".join(self.errors)
        )


@dataclass(frozen=True)
class _Location:
    owner: Owner
    source_path: str
    line: int
    column: int
    source_sha256: str


@dataclass(frozen=True)
class _Source:
    values: dict[str, Any]
    locations: dict[str, _Location]


@dataclass(frozen=True)
class _ParsedSource:
    """Hold one parsed component before include merging."""

    values: dict[str, Any]
    locations: dict[str, _Location]
    includes: tuple[str, ...]


def _pointer(parts: tuple[str, ...]) -> str:
    encoded = (part.replace("~", "~0").replace("/", "~1") for part in parts)
    return "/" + "/".join(encoded) if parts else ""


def _walk_locations(
    node: Node,
    parts: tuple[str, ...],
    location: _Location,
    result: dict[str, _Location],
    errors: list[str],
) -> None:
    result[_pointer(parts)] = _Location(
        owner=location.owner,
        source_path=location.source_path,
        line=node.start_mark.line + 1,
        column=node.start_mark.column + 1,
        source_sha256=location.source_sha256,
    )
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode):
                errors.append(f"{location.source_path}: a mapping key must be text")
                continue
            key = str(key_node.value)
            if key in seen:
                errors.append(
                    f"{location.source_path}:{key_node.start_mark.line + 1}:"
                    f"{key_node.start_mark.column + 1}: the key {key!r} is duplicated"
                )
            seen.add(key)
            _walk_locations(value_node, (*parts, key), location, result, errors)
    elif isinstance(node, SequenceNode):
        for index, value_node in enumerate(node.value):
            _walk_locations(value_node, (*parts, str(index)), location, result, errors)


def _merge(
    target: dict[str, Any],
    source: Mapping[str, Any],
    target_locations: dict[str, _Location],
    source_locations: Mapping[str, _Location],
    parts: tuple[str, ...] = (),
) -> None:
    for key, value in source.items():
        path = (*parts, str(key))
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _merge(existing, value, target_locations, source_locations, path)
            continue
        target[key] = value
        prefix = _pointer(path)
        for pointer in tuple(target_locations):
            if pointer == prefix or pointer.startswith(prefix + "/"):
                target_locations.pop(pointer)
        for pointer, location in source_locations.items():
            if pointer == prefix or pointer.startswith(prefix + "/"):
                target_locations[pointer] = location


def _logical_path(value: str | Path, description: str) -> PurePosixPath:
    text = value.as_posix() if isinstance(value, Path) else str(value)
    return PurePosixPath(canonical_repository_path(text, description))


def _leaf_pointers(value: Any, parts: tuple[str, ...] = ()) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if not value:
            return (_pointer(parts),)
        return tuple(
            pointer
            for key, nested in value.items()
            for pointer in _leaf_pointers(nested, (*parts, str(key)))
        )
    if isinstance(value, (list, tuple)):
        if not value:
            return (_pointer(parts),)
        return tuple(
            pointer
            for index, nested in enumerate(value)
            for pointer in _leaf_pointers(nested, (*parts, str(index)))
        )
    return (_pointer(parts),)


def _owner_for_pointer(pointer: str) -> ProvenanceOwner:
    head = pointer.split("/", 2)[1]
    if head in {"mountain", "population"}:
        return "mountain"
    if head in {
        "scenario",
        "intervals",
        "numerics",
        "seed",
        "episode_duration_seconds",
        "snapshot_interval_seconds",
        "trace_level",
    }:
        return "scenario"
    if head == "controller":
        return "controller"
    if head in {"monitor", "fallback", "approval"}:
        return "monitor"
    return "resolver"


def _edge_map(topology: Any) -> dict[str, int]:
    return {
        f"{topology.node_ids[int(topology.edge_source[index])]}->"
        f"{topology.node_ids[int(topology.edge_destination[index])]}": index
        for index in range(topology.edge_count)
    }


def _safe_routes(topology: Any) -> list[str]:
    from avalanche.sim.ability import ABILITY_NAMES
    from avalanche.sim.routes import build_route_table, required_destinations
    from avalanche.sim.topology import NODE_TYPE_NAMES

    entrance_code = NODE_TYPE_NAMES.index("entrance")
    entrances = tuple(
        index
        for index, node_type in enumerate(topology.node_type)
        if int(node_type) == entrance_code
    )
    destinations = required_destinations(topology)
    routes = build_route_table(topology)
    errors = []
    for ability_index, ability in enumerate(ABILITY_NAMES):
        for entrance in entrances:
            for destination in destinations:
                if isfinite(
                    float(routes.travel_time[ability_index, entrance, destination])
                ):
                    continue
                errors.append(
                    f"mountain /mountain/path: the {ability} ability has no safe "
                    f"route from the entrance {topology.node_ids[entrance]!r} to "
                    f"the required destination {topology.node_ids[destination]!r}"
                )
    return errors


class ConfigurationResolver:
    """Resolve and validate one formal component selection."""

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        artifact_root: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
        self.artifact_root = (artifact_root or self.repo_root).resolve()
        self._parsed_sources: dict[tuple[Owner, PurePosixPath, str], _ParsedSource] = {}
        self._source_digests: dict[tuple[Owner, PurePosixPath], str] = {}
        self._topologies: dict[tuple[Path, str], Any] = {}
        self._safe_route_errors: dict[tuple[Path, str], tuple[str, ...]] = {}
        self._topology_digests: dict[Path, str] = {}

    def _read(
        self,
        logical: PurePosixPath,
        owner: Owner,
        stack: tuple[PurePosixPath, ...] = (),
    ) -> _Source:
        if logical in stack:
            chain = " -> ".join(path.as_posix() for path in (*stack, logical))
            raise ConfigurationResolutionError([f"{owner}: include cycle: {chain}"])
        path = self._repository_file(logical, f"{owner} component")
        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            raise ConfigLoadError(
                path, "the configuration file does not exist"
            ) from error
        except OSError as error:
            raise ConfigLoadError(
                path, "the configuration file cannot be read"
            ) from error
        digest = hashlib.sha256(content).hexdigest()
        parsed = self._parsed_source(logical, owner, path, content, digest)
        values = deepcopy(parsed.values)
        locations = dict(parsed.locations)
        merged: dict[str, Any] = {}
        merged_locations: dict[str, _Location] = {}
        for include in parsed.includes:
            try:
                relative = _logical_path(include, "include")
            except ValueError as error:
                raise ConfigurationResolutionError(
                    [f"{owner} {logical.as_posix()}: {error}"]
                ) from error
            included_logical = PurePosixPath(logical.parent, relative)
            included = self._read(included_logical, owner, (*stack, logical))
            _merge(merged, included.values, merged_locations, included.locations)
        _merge(merged, values, merged_locations, locations)
        return _Source(merged, merged_locations)

    def _parsed_source(
        self,
        logical: PurePosixPath,
        owner: Owner,
        path: Path,
        content: bytes,
        digest: str,
    ) -> _ParsedSource:
        """Return one content-addressed parsed component."""
        source_key = (owner, logical)
        previous_digest = self._source_digests.get(source_key)
        if previous_digest != digest:
            if previous_digest is not None:
                self._parsed_sources.pop((*source_key, previous_digest), None)
            self._source_digests[source_key] = digest
        cache_key = (*source_key, digest)
        cached = self._parsed_sources.get(cache_key)
        if cached is not None:
            return cached
        try:
            text = content.decode("utf-8")
        except UnicodeError as error:
            raise ConfigLoadError(
                path, "the configuration file is not valid UTF-8"
            ) from error
        try:
            node = yaml.compose(text, Loader=yaml.SafeLoader)
            values = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ConfigLoadError(
                path, "the configuration file contains invalid YAML"
            ) from error
        if not isinstance(node, MappingNode) or not isinstance(values, dict):
            raise ConfigLoadError(path, "the configuration root must be a mapping")
        errors: list[str] = []
        base_location = _Location(owner, logical.as_posix(), 1, 1, digest)
        locations: dict[str, _Location] = {}
        _walk_locations(node, (), base_location, locations, errors)
        unknown = sorted(set(values) - _OWNER_KEYS[owner] - {"include"})
        if unknown:
            errors.extend(
                f"{owner} {logical.as_posix()}: the top-level key {key!r} is not owned"
                for key in unknown
            )
        declared_includes = values.pop("include", ())
        includes: tuple[str, ...]
        if isinstance(declared_includes, str):
            includes = (declared_includes,)
        elif isinstance(declared_includes, (list, tuple)) and all(
            isinstance(value, str) for value in declared_includes
        ):
            includes = tuple(declared_includes)
        else:
            errors.append(f"{owner} {logical.as_posix()}: include must name text paths")
            includes = ()
        if errors:
            raise ConfigurationResolutionError(errors)
        locations = {
            pointer: location
            for pointer, location in locations.items()
            if pointer != "/include" and not pointer.startswith("/include/")
        }
        parsed = _ParsedSource(values, locations, includes)
        self._parsed_sources[cache_key] = parsed
        return parsed

    def _argument(self, value: str | Path, owner: Owner) -> PurePosixPath:
        try:
            logical = _logical_path(value, f"{owner} component")
        except ValueError as error:
            raise ConfigurationResolutionError([f"{owner}: {error}"]) from error
        try:
            path = self._repository_file(logical, f"{owner} component")
        except ConfigurationResolutionError:
            raise
        if not path.is_file():
            raise ConfigurationResolutionError(
                [f"{owner}: the component file does not exist: {logical.as_posix()}"]
            )
        return logical

    def _repository_file(self, logical: PurePosixPath, description: str) -> Path:
        """Return one repository-contained source path."""
        path = self.repo_root.joinpath(*logical.parts)
        resolved = path.resolve()
        if not resolved.is_relative_to(self.repo_root):
            raise ConfigurationResolutionError(
                [f"the {description} path leaves the repository: {logical.as_posix()}"]
            )
        return path

    def _cached_topology(self, path: Path) -> tuple[Any, tuple[Path, str]]:
        """Return one topology keyed by its current file bytes."""
        from avalanche.sim.topology import load_topology

        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            raise ConfigLoadError(
                path, "the configuration file does not exist"
            ) from error
        except OSError as error:
            raise ConfigLoadError(
                path, "the configuration file cannot be read"
            ) from error
        digest = hashlib.sha256(content).hexdigest()
        resolved_path = path.resolve()
        previous_digest = self._topology_digests.get(resolved_path)
        if previous_digest != digest:
            if previous_digest is not None:
                previous_key = (resolved_path, previous_digest)
                self._topologies.pop(previous_key, None)
                self._safe_route_errors.pop(previous_key, None)
            self._topology_digests[resolved_path] = digest
        cache_key = (resolved_path, digest)
        topology = self._topologies.get(cache_key)
        if topology is None:
            topology = load_topology(path)
            self._topologies[cache_key] = topology
        return topology, cache_key

    def _cached_safe_routes(
        self, topology: Any, cache_key: tuple[Path, str]
    ) -> tuple[str, ...]:
        """Return one cached safe-route validation result."""
        errors = self._safe_route_errors.get(cache_key)
        if errors is None:
            errors = tuple(_safe_routes(topology))
            self._safe_route_errors[cache_key] = errors
        return errors

    def component_values(self, owner: Owner, value: str | Path) -> dict[str, Any]:
        """Return one typed component envelope for selection displays."""
        if owner == "override":
            raise ValueError("an override is not a selectable component")
        source = self._read(self._argument(value, owner), owner)
        try:
            _ENVELOPES[owner].model_validate(source.values)
        except ValidationError as error:
            raise ConfigurationResolutionError(
                [
                    f"{owner} {_pointer(tuple(str(value) for value in item['loc']))}: "
                    f"{item['msg']}"
                    for item in error.errors(include_url=False)
                ]
            ) from error
        return source.values

    def resolve_live(
        self,
        mountain: str | Path,
        scenario: str | Path,
        controller: str | Path,
        monitor: str | Path,
        *,
        seed: int | None = None,
        episode_duration_seconds: float | None = None,
        skier_count: int | None = None,
        trace_level: str | None = None,
    ) -> ResolvedConfig:
        """Apply the four scientific live overrides to a formal selection."""
        live_values = {
            pointer: value
            for pointer, value in (
                ("/seed", seed),
                ("/episode_duration_seconds", episode_duration_seconds),
                ("/population/skier_count", skier_count),
                ("/trace_level", trace_level),
            )
            if value is not None
        }
        return self._resolve(
            mountain,
            scenario,
            controller,
            monitor,
            live_values=live_values,
        )

    def resolve(
        self,
        mountain: str | Path,
        scenario: str | Path,
        controller: str | Path,
        monitor: str | Path,
        override: str | Path | None = None,
    ) -> ResolvedConfig:
        """Resolve all sources or return every validation error."""
        return self._resolve(mountain, scenario, controller, monitor, override)

    def _resolve(
        self,
        mountain: str | Path,
        scenario: str | Path,
        controller: str | Path,
        monitor: str | Path,
        override: str | Path | None = None,
        *,
        live_values: Mapping[str, Any] | None = None,
    ) -> ResolvedConfig:
        """Resolve one formal or live selection in one validation pass."""
        live_values = live_values or {}
        selections: list[tuple[Owner, str | Path]] = [
            ("mountain", mountain),
            ("scenario", scenario),
            ("controller", controller),
            ("monitor", monitor),
        ]
        if override is not None:
            selections.append(("override", override))
        sources: list[tuple[Owner, _Source]] = []
        errors: list[str] = []
        for owner, value in selections:
            try:
                source = self._read(self._argument(value, owner), owner)
                _ENVELOPES[owner].model_validate(source.values)
                sources.append((owner, source))
            except ValidationError as error:
                errors.extend(
                    f"{owner} {_pointer(tuple(str(value) for value in item['loc']))}: "
                    f"{item['msg']}"
                    for item in error.errors(include_url=False)
                )
            except (ConfigLoadError, ConfigurationResolutionError) as error:
                if isinstance(error, ConfigurationResolutionError):
                    errors.extend(error.errors)
                else:
                    errors.append(f"{owner}: {error}")
        if errors:
            raise ConfigurationResolutionError(errors)
        values: dict[str, Any] = {}
        locations: dict[str, _Location] = {}
        for owner, source in sources:
            if owner == "override":
                leaves = set(_leaf_pointers(source.values))
                forbidden = sorted(leaves - _OVERRIDE_PATHS)
                if forbidden:
                    errors.extend(
                        f"override {pointer}: the formal override path is forbidden"
                        for pointer in forbidden
                    )
                    continue
            duplicate = sorted(set(values) & set(source.values))
            if owner != "override" and duplicate:
                errors.extend(
                    f"{owner} /{key}: the value already has an owner"
                    for key in duplicate
                )
                continue
            _merge(values, source.values, locations, source.locations)
        values.setdefault("output_root", "outputs")
        values.setdefault("runtime", {"worker_count": 1})
        if errors:
            raise ConfigurationResolutionError(errors)
        for pointer, value in live_values.items():
            target = values
            parts = pointer.strip("/").split("/")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        try:
            resolved = ResolvedConfig.model_validate(values)
        except ValidationError as error:
            formatted = []
            for item in error.errors(include_url=False):
                pointer = _pointer(tuple(str(value) for value in item["loc"]))
                error_owner = (
                    "live"
                    if pointer in live_values
                    else (
                        "configuration" if not pointer else _owner_for_pointer(pointer)
                    )
                )
                formatted.append(f"{error_owner} {pointer}: {item['msg']}")
            raise ConfigurationResolutionError(formatted) from error
        errors.extend(self._validate_semantics(resolved))
        if errors:
            raise ConfigurationResolutionError(errors)
        logical = resolved.model_dump(mode="json", exclude=set(_IDENTITY_FIELDS))
        records = []
        override_pointers = {
            pointer
            for owner, source in sources
            if owner == "override"
            for pointer in source.locations
        }
        for pointer in sorted(_leaf_pointers(logical)):
            if pointer in live_values:
                records.append(
                    ValueProvenance(
                        pointer=pointer,
                        kind="derived",
                        owner="override",
                        formula_version="validated-live-request-v1",
                        input_paths=(f"/live_request{pointer}",),
                    )
                )
                continue
            location = locations.get(pointer)
            if location is None:
                records.append(
                    ValueProvenance(
                        pointer=pointer,
                        kind="schema_default",
                        owner=_owner_for_pointer(pointer),
                        schema_path=f"ResolvedConfig{pointer}",
                    )
                )
                continue
            record_owner: Owner = (
                "override" if pointer in override_pointers else location.owner
            )
            records.append(
                ValueProvenance(
                    pointer=pointer,
                    kind="explicit",
                    owner=record_owner,
                    source_path=location.source_path,
                    line=location.line,
                    column=location.column,
                    source_sha256=location.source_sha256,
                )
            )
        records.extend(
            (
                ValueProvenance(
                    pointer="/resolved_configuration_sha256",
                    kind="derived",
                    owner="resolver",
                    formula_version="canonical-json-sha256-v1",
                    input_paths=("/*",),
                ),
                ValueProvenance(
                    pointer="/scientific_configuration_sha256",
                    kind="derived",
                    owner="resolver",
                    formula_version="scientific-json-sha256-v1",
                    input_paths=("/*", "/output_root", "/runtime/worker_count"),
                ),
            )
        )
        return _with_identity(resolved, tuple(records))

    def _validate_semantics(self, resolved: ResolvedConfig) -> list[str]:
        from avalanche.monitors.training import (
            ArtifactError,
            verify_formal_model_reference,
        )

        errors: list[str] = []
        try:
            _logical_path(resolved.output_root, "output root")
        except ValueError as error:
            errors.append(f"resolver /output_root: {error}")
        else:
            output = self.repo_root.joinpath(*PurePosixPath(resolved.output_root).parts)
            if not output.resolve().is_relative_to(self.repo_root):
                errors.append("resolver /output_root: the path leaves the repository")
        errors.extend(self._validate_schedule_bounds(resolved))
        topology = None
        try:
            mountain_path = _logical_path(resolved.mountain.path, "mountain topology")
            topology_path = self._repository_file(mountain_path, "mountain topology")
            topology, topology_key = self._cached_topology(topology_path)
        except (ConfigLoadError, ConfigurationResolutionError, ValueError) as error:
            errors.append(f"mountain /mountain/path: {error}")
        if topology is not None:
            errors.extend(self._validate_topology_references(resolved, topology))
            errors.extend(self._cached_safe_routes(topology, topology_key))
        if resolved.monitor.kind == "learned":
            if resolved.monitor.model_lock is None:
                errors.append(
                    "monitor /monitor/model_lock: "
                    "a learned monitor needs a verified selection"
                )
            else:
                try:
                    verify_formal_model_reference(
                        resolved.monitor.model_lock,
                        repo_root=self.artifact_root,
                    )
                except ArtifactError as error:
                    errors.append(f"monitor /monitor/model_lock: {error}")
        return errors

    def _validate_topology_references(
        self, resolved: ResolvedConfig, topology: Any
    ) -> list[str]:
        """Validate every topology reference and wrapper precondition."""
        from avalanche.sim.ability import ABILITY_NAMES, ability_edge_mask
        from avalanche.sim.topology import EDGE_TYPE_NAMES

        errors: list[str] = []
        for field, declared, loaded in (
            ("node_count", resolved.mountain.node_count, topology.node_count),
            ("edge_count", resolved.mountain.edge_count, topology.edge_count),
        ):
            if declared != loaded:
                errors.append(
                    f"mountain /mountain/{field}: declares {declared} "
                    f"but loads {loaded}"
                )
        edges = _edge_map(topology)

        def require_edge(
            pointer: str,
            reference: str,
            *,
            lift: bool = False,
            controllable: bool = False,
        ) -> int | None:
            index = edges.get(reference)
            if index is None:
                errors.append(
                    f"{_owner_for_pointer(pointer)} {pointer}: "
                    f"unknown edge {reference!r}"
                )
            elif lift and EDGE_TYPE_NAMES[int(topology.edge_type[index])] != "lift":
                errors.append(
                    f"{_owner_for_pointer(pointer)} {pointer}: "
                    f"{reference!r} is not a lift"
                )
            if (
                index is not None
                and controllable
                and not bool(topology.edge_controllable[index])
            ):
                errors.append(
                    f"{_owner_for_pointer(pointer)} {pointer}: "
                    f"{reference!r} is not controllable"
                )
            return index

        environment_context = None
        try:
            environment_context = resolved.scenario.environment_context.for_mountain(
                resolved.mountain.name
            )
        except ValueError as error:
            errors.append(f"scenario /scenario/environment_context: {error}")
        if environment_context is not None:
            context_index = next(
                index
                for index, context in enumerate(
                    resolved.scenario.environment_context.evacuation_targets
                )
                if context.mountain == resolved.mountain.name
            )
            for target_index, target in enumerate(
                environment_context.evacuation_target_edges
            ):
                pointer = (
                    "/scenario/environment_context/evacuation_targets/"
                    f"{context_index}/evacuation_target_edges/{target_index}/edge"
                )
                edge_index = require_edge(pointer, target.edge)
                if edge_index is None:
                    continue
                for ability in target.abilities:
                    ability_index = ABILITY_NAMES.index(ability)
                    if not bool(ability_edge_mask(topology, ability_index)[edge_index]):
                        errors.append(
                            f"scenario {pointer}: {target.edge!r} is unsafe for "
                            f"the {ability} ability"
                        )

        controller = resolved.controller
        if controller.balanced_lifts is not None:
            for index, reference in enumerate(controller.balanced_lifts):
                require_edge(
                    f"/controller/balanced_lifts/{index}", reference, lift=True
                )
        for index, reference in enumerate(controller.evacuation_edges):
            require_edge(f"/controller/evacuation_edges/{index}", reference)
        if controller.attack is not None:
            attack = controller.attack
            target_indices = []
            for index, reference in enumerate(attack.targets):
                target_indices.append(
                    require_edge(
                        f"/controller/attack/targets/{index}",
                        reference,
                        controllable=True,
                    )
                )
            proxy_indices = []
            for index, reference in enumerate(controller.attack.journey_proxies):
                proxy_indices.append(
                    require_edge(
                        f"/controller/attack/journey_proxies/{index}",
                        reference,
                        controllable=True,
                    )
                )
            if attack.target_group not in {None, "standard", "premium"}:
                errors.append(
                    "controller /controller/attack/target_group: unknown group"
                )
            effective_indices = (
                proxy_indices
                if attack.kind == "profit_biased" and attack.tier == "stealth"
                else target_indices
            )[: attack.action_budget.maximum_targets]
            if attack.kind == "reward_hacker" and not any(
                index is not None
                and EDGE_TYPE_NAMES[int(topology.edge_type[index])] == "lift"
                for index in effective_indices
            ):
                errors.append(
                    "controller /controller/attack/targets: "
                    "a reward hacker needs one lift service target"
                )
            if attack.kind == "sleeper_saboteur":
                evacuation = set(controller.evacuation_edges)
                for index, (reference, edge_index) in enumerate(
                    zip(attack.targets, target_indices, strict=True)
                ):
                    pointer = f"/controller/attack/targets/{index}"
                    if (
                        edge_index is not None
                        and EDGE_TYPE_NAMES[int(topology.edge_type[edge_index])]
                        != "lift"
                    ):
                        errors.append(
                            f"controller {pointer}: {reference!r} is not a lift"
                        )
                    if reference not in evacuation:
                        errors.append(
                            f"controller {pointer}: {reference!r} is not an escape"
                        )
            if attack.kind == "profit_biased" and attack.tier == "overt":
                for index, edge_index in enumerate(target_indices):
                    if edge_index is None:
                        continue
                    source = int(topology.edge_source[edge_index])
                    if not bool(topology.node_controllable[source]):
                        errors.append(
                            f"controller /controller/attack/targets/{index}: "
                            "the target source node is not controllable"
                        )
        for index, reference in enumerate(resolved.monitor.evacuation_edges):
            require_edge(f"/monitor/evacuation_edges/{index}", reference)
        for index, event in enumerate(resolved.scenario.failures.schedule):
            pointer = f"/scenario/failures/schedule/{index}/target"
            if isinstance(event.target, int):
                if event.target < 0 or event.target >= topology.edge_count:
                    errors.append(
                        f"scenario {pointer}: edge index is outside the topology"
                    )
                target_index = event.target
            else:
                target_index = edges.get(event.target, -1)
                require_edge(pointer, event.target, lift=event.kind == "lift_stoppage")
            if (
                event.kind == "lift_stoppage"
                and 0 <= target_index < topology.edge_count
            ):
                if EDGE_TYPE_NAMES[int(topology.edge_type[target_index])] != "lift":
                    errors.append(
                        f"scenario {pointer}: a lift stoppage must target a lift"
                    )
        return errors

    def _validate_schedule_bounds(self, resolved: ResolvedConfig) -> list[str]:
        """Validate every fixed and sampled schedule against the episode."""
        errors: list[str] = []
        for index, event in enumerate(resolved.scenario.failures.schedule):
            if (
                event.start_time_seconds + event.duration_seconds
                > resolved.episode_duration_seconds
            ):
                errors.append(
                    f"scenario /scenario/failures/schedule/{index}/target: "
                    "the failure ends after the episode"
                )
        weather = resolved.scenario.weather
        for index, entry in enumerate(weather.schedule):
            if entry.start_time_seconds >= resolved.episode_duration_seconds:
                errors.append(
                    f"scenario /scenario/weather/schedule/{index}/start_time_seconds: "
                    "the weather change must precede the episode end"
                )
        if weather.sampling is not None:
            final_start = weather.sampling.interval_seconds * (
                weather.sampling.transition_count - 1
            )
            if final_start >= resolved.episode_duration_seconds:
                errors.append(
                    "scenario /scenario/weather/sampling: "
                    "the schedule exceeds the episode"
                )
        failure_sampling = resolved.scenario.failures.sampling
        if failure_sampling is not None and (
            failure_sampling.latest_start_seconds
            + failure_sampling.maximum_duration_seconds
            > resolved.episode_duration_seconds
        ):
            errors.append(
                "scenario /scenario/failures/sampling: the schedule exceeds the episode"
            )
        events = resolved.scenario.operational_events
        if events.enabled:
            for index, start in enumerate(events.matched_periods_seconds):
                if (
                    start
                    + events.maximum_offset_seconds
                    + events.maximum_duration_seconds
                    > resolved.episode_duration_seconds
                ):
                    errors.append(
                        f"scenario /scenario/operational_events/"
                        f"matched_periods_seconds/{index}: "
                        "the event can exceed the episode"
                    )
        if resolved.snapshot_interval_seconds > resolved.episode_duration_seconds:
            errors.append(
                "scenario /snapshot_interval_seconds: exceeds the episode duration"
            )
        return errors


def resolve_configuration(
    mountain: str | Path,
    scenario: str | Path,
    controller: str | Path,
    monitor: str | Path,
    override: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> ResolvedConfig:
    """Resolve one formal configuration with the default resolver."""
    return ConfigurationResolver(repo_root).resolve(
        mountain, scenario, controller, monitor, override
    )


def _with_identity(
    resolved: ResolvedConfig, provenance: tuple[ValueProvenance, ...]
) -> ResolvedConfig:
    """Add both canonical configuration identities."""
    logical = resolved.model_dump(mode="json", exclude=set(_IDENTITY_FIELDS))
    canonical = json.dumps(
        logical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    resolved_digest = hashlib.sha256(canonical).hexdigest()
    scientific = dict(logical)
    scientific.pop("output_root", None)
    scientific.pop("runtime", None)
    scientific_digest = hashlib.sha256(
        json.dumps(
            scientific, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return resolved.model_copy(
        update={
            "provenance": provenance,
            "resolved_configuration_sha256": resolved_digest,
            "scientific_configuration_sha256": scientific_digest,
        }
    )
