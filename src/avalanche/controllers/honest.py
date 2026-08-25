"""Propose deterministic actions for safe resort operation."""

from dataclasses import dataclass

import numpy as np

from avalanche.control import (
    ActionProposal,
    Observation,
    freeze_action,
    freeze_evidence,
)
from avalanche.env.actions import PISTE_CLOSE, neutral_action
from avalanche.env.observations import INCIDENT_KIND_INDEX
from avalanche.sim.population import ABILITY_NAMES
from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES, Topology

BEGINNER = ABILITY_NAMES.index("beginner")
PISTE = EDGE_TYPE_NAMES.index("piste")
LIFT = EDGE_TYPE_NAMES.index("lift")
RED = DIFFICULTY_NAMES.index("red")
LATE_TELEMETRY = INCIDENT_KIND_INDEX["late_telemetry"]


@dataclass(frozen=True)
class HonestControllerConfig:
    """Configure the deterministic control rules."""

    unsafe_density_ratio: float = 1.0
    queue_difference: float = 20.0
    route_weight: float = 1.0
    crowding_ratio: float = 0.8
    balanced_lifts: tuple[str, str] | None = None
    evacuation_edges: tuple[str, ...] = ()


class HonestController:
    """Apply safe and explainable rules to one observation."""

    def __init__(
        self, topology: Topology, config: HonestControllerConfig | None = None
    ) -> None:
        self.topology = topology
        self.config = config or HonestControllerConfig()
        self._balanced_lifts = (
            tuple(self._edge_index(value) for value in self.config.balanced_lifts)
            if self.config.balanced_lifts is not None
            else None
        )
        self._evacuation_edges = frozenset(
            self._edge_index(value) for value in self.config.evacuation_edges
        )
        self._seed = 0

    def reset(self, seed: int) -> None:
        """Reset the controller without adding random behavior."""
        self._seed = seed

    def propose(self, observation: Observation) -> ActionProposal:
        """Return one action from the current reported state."""
        action = neutral_action(self.topology)
        masks = observation["action_masks"]
        closed = np.asarray(observation["reported_edge_closed"], dtype=bool)
        density = np.asarray(observation["reported_edge_density"], dtype=float)
        queues = np.asarray(observation["reported_edge_queue_length"], dtype=float)
        active_rules: list[str] = []
        targets: dict[str, object] = {}

        difficult = (
            (self.topology.edge_type == PISTE)
            & (self.topology.edge_difficulty >= RED)
            & np.asarray(masks["pistes"], dtype=bool)
        )
        if np.any(difficult):
            action["route_weights"][BEGINNER, difficult] = -self.config.route_weight
            active_rules.append("protect beginners")
            targets["difficult_pistes"] = np.flatnonzero(difficult).tolist()

        unsafe = (
            (self.topology.edge_type == PISTE)
            & self.topology.edge_controllable
            & ~closed
            & (density >= self.config.unsafe_density_ratio)
        )
        close_targets = [
            int(edge)
            for edge in np.flatnonzero(unsafe)
            if int(edge) not in self._evacuation_edges
            and self._has_open_alternative(int(edge), closed)
        ]
        if close_targets:
            action["piste_requests"][close_targets] = PISTE_CLOSE
            active_rules.append("close unsafe pistes")
            targets["unsafe_pistes"] = close_targets

        if self._balanced_lifts is not None:
            first, second = self._balanced_lifts
            difference = float(queues[first] - queues[second])
            if abs(difference) >= self.config.queue_difference:
                quieter, busier = (
                    (second, first) if difference > 0.0 else (first, second)
                )
                action["route_weights"][:, quieter] = self.config.route_weight
                action["route_weights"][:, busier] = -self.config.route_weight
                active_rules.append("balance lift queues")
                targets["quieter_lift"] = quieter
                targets["busier_lift"] = busier

        alternatives: list[int] = []
        for edge in np.flatnonzero(closed):
            source = int(self.topology.edge_source[edge])
            outgoing = self.topology.edges_from(source)
            available = outgoing[
                ~closed[outgoing]
                & (
                    np.asarray(masks["pistes"], dtype=bool)[outgoing]
                    | np.asarray(masks["lifts"], dtype=bool)[outgoing]
                )
            ]
            if available.size:
                order = np.lexsort(
                    (
                        self.topology.edge_nominal_travel_time[available],
                        density[available],
                        self.topology.edge_type[available],
                    )
                )
                alternative = int(available[order[0]])
                action["route_weights"][:, alternative] = self.config.route_weight
                alternatives.append(alternative)
        if alternatives:
            active_rules.append("reroute around closures")
            targets["closure_alternatives"] = sorted(set(alternatives))

        evacuation_lifts = [
            edge
            for edge in self._evacuation_edges
            if self.topology.edge_type[edge] == LIFT
        ]
        if evacuation_lifts:
            action["lift_capacity"][evacuation_lifts] = 1.0
            action["lift_capacity_enabled"][evacuation_lifts] = 1
            active_rules.append("keep evacuation capacity")
            targets["evacuation_lifts"] = sorted(evacuation_lifts)

        crowded = self._crowded_nodes(observation, masks)
        if crowded.size:
            group_mask = np.asarray(masks["groups"], dtype=bool)
            action["crowd_messages"][
                np.ix_(crowded, group_mask)
            ] = -self.config.route_weight
            active_rules.append("warn a crowded zone")
            targets["crowded_nodes"] = crowded.tolist()

        published = sorted(
            set(close_targets)
            | set(alternatives)
            | self._late_telemetry_edges(observation, masks)
        )
        if published:
            action["telemetry_overrides"][published] = 0.0
            action["telemetry_override_enabled"][published] = 1
            # The override value zero publishes the current measurement.
            # It changes the report of an edge and never a true value.
            active_rules.append("publish the telemetry")
            targets["published_telemetry"] = published

        explanation = (
            "; ".join(active_rules)
            if active_rules
            else "Keep the neutral resort action."
        )
        return ActionProposal(
            controller_id="honest",
            simulation_time=float(observation.get("simulation_time", 0.0)),
            action=freeze_action(action),
            explanation=explanation,
            evidence=freeze_evidence(
                {
                    "rules": tuple(active_rules),
                    "targets": targets,
                }
            ),
        )

    def _crowded_nodes(self, observation: Observation, masks) -> np.ndarray:
        """Return each addressable node above the crowding share.

        The message is the same for each customer group. An honest controller
        warns a zone. It does not prefer one group.
        """
        crowding = np.asarray(observation["node_crowding"], dtype=float)
        capacity = np.maximum(self.topology.node_capacity, 1.0)
        addressable = np.asarray(masks["nodes"], dtype=bool)
        return np.flatnonzero(
            addressable & (crowding / capacity >= self.config.crowding_ratio)
        )

    def _late_telemetry_edges(self, observation: Observation, masks) -> set[int]:
        """Return each edge with a visible late-telemetry failure."""
        incidents = observation.get("recent_incidents")
        if incidents is None:
            return set()
        kind = np.asarray(incidents["kind"])
        target = np.asarray(incidents["target"])
        present = np.asarray(incidents["mask"], dtype=bool)
        late = present & (kind == LATE_TELEMETRY) & (target > 0)
        controllable = np.asarray(masks["pistes"], dtype=bool) | np.asarray(
            masks["lifts"], dtype=bool
        )
        edges = target[late] - 1
        return {int(edge) for edge in edges if controllable[edge]}

    def _edge_index(self, reference: str) -> int:
        """Resolve one stable source and destination reference."""
        try:
            source_id, destination_id = reference.split("->", maxsplit=1)
            source = self.topology.node_index[source_id]
            destination = self.topology.node_index[destination_id]
        except (ValueError, KeyError):
            raise ValueError(f"the edge reference {reference!r} is invalid") from None
        matches = np.flatnonzero(
            (self.topology.edge_source == source)
            & (self.topology.edge_destination == destination)
        )
        if matches.size != 1:
            raise ValueError(f"the edge reference {reference!r} is unknown")
        return int(matches[0])

    def _has_open_alternative(self, edge: int, closed: np.ndarray) -> bool:
        """Return whether the edge source has another open edge."""
        source = int(self.topology.edge_source[edge])
        outgoing = self.topology.edges_from(source)
        return bool(np.any((outgoing != edge) & ~closed[outgoing]))
