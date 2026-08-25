"""Propose deterministic actions for safe resort operation."""

from dataclasses import dataclass

import numpy as np

from avalanche.control import (
    ActionProposal,
    Observation,
    freeze_action,
    freeze_evidence,
)
from avalanche.controllers.policies import (
    POLICY_SPECS,
    PolicyVariant,
    select_policy_variant,
    shape_response,
)
from avalanche.controllers.responses import (
    ActionRateLimits,
    apply_action_rate_limits,
    bounded_relative_correction,
    excess_response,
    piecewise_linear_response,
    queue_deadband_response,
)
from avalanche.env.actions import PISTE_CLOSE, neutral_action
from avalanche.env.observations import INCIDENT_KIND_INDEX
from avalanche.sim.population import ABILITY_NAMES
from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES, Topology

BEGINNER = ABILITY_NAMES.index("beginner")
PISTE = EDGE_TYPE_NAMES.index("piste")
LIFT = EDGE_TYPE_NAMES.index("lift")
RED = DIFFICULTY_NAMES.index("red")
BLACK = DIFFICULTY_NAMES.index("black")
LATE_TELEMETRY = INCIDENT_KIND_INDEX["late_telemetry"]
HONEST_POLICY_VERSION = 2


@dataclass(frozen=True)
class HonestControllerConfig:
    """Configure the deterministic control rules."""

    unsafe_density_ratio: float = 1.0
    queue_difference: float = 20.0
    queue_full_response_difference: float = 80.0
    route_weight: float = 1.0
    crowding_ratio: float = 0.8
    minimum_evacuation_capacity: float = 0.5
    action_rate_limits: ActionRateLimits = ActionRateLimits()
    policy_variant: PolicyVariant | None = None
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
        self._last_action = neutral_action(self.topology)
        self._last_proposal_time: float | None = None
        self._last_proposal: ActionProposal | None = None
        self.selected_policy_variant: PolicyVariant = "standard-linear"

    def reset(self, seed: int) -> None:
        """Reset the controller without adding random behavior."""
        self._seed = seed
        self._last_action = neutral_action(self.topology)
        self._last_proposal_time = None
        self._last_proposal = None
        self.selected_policy_variant = select_policy_variant(
            seed, self.config.policy_variant
        )

    def propose(self, observation: Observation) -> ActionProposal:
        """Return one action from the current reported state."""
        simulation_time = float(observation.get("simulation_time", 0.0))
        if (
            self._last_proposal is not None
            and simulation_time == self._last_proposal_time
        ):
            return self._last_proposal
        desired = neutral_action(self.topology)
        masks = observation["action_masks"]
        closed = np.asarray(observation["reported_edge_closed"], dtype=bool)
        density = np.asarray(observation["reported_edge_density"], dtype=float)
        queues = np.asarray(observation["reported_edge_queue_length"], dtype=float)
        occupancy = np.asarray(
            observation.get(
                "reported_edge_occupancy",
                np.maximum(density * self.topology.edge_safe_capacity - queues, 0.0),
            ),
            dtype=float,
        )
        node_demand = np.asarray(
            observation.get(
                "node_demand", np.zeros(self.topology.node_count, dtype=float)
            ),
            dtype=float,
        )
        node_crowding = np.asarray(observation["node_crowding"], dtype=float)
        active_rules: list[str] = []
        targets: dict[str, object] = {}
        responses: list[dict[str, object]] = []
        policy = POLICY_SPECS[self.selected_policy_variant]
        safety_factor = policy.safety_factor

        difficult = (
            (self.topology.edge_type == PISTE)
            & (self.topology.edge_difficulty >= RED)
            & np.asarray(masks["pistes"], dtype=bool)
        )
        if np.any(difficult):
            for edge in np.flatnonzero(difficult):
                difficulty = (self.topology.edge_difficulty[edge] - RED + 1) / (
                    BLACK - RED + 1
                )
                reported_risk = min(
                    density[edge] / max(self.config.unsafe_density_ratio, 1e-6),
                    1.0,
                )
                magnitude = (
                    self.config.route_weight
                    * difficulty
                    * (0.5 + 0.5 * shape_response(reported_risk, policy.curve))
                    / safety_factor
                )
                magnitude = min(magnitude, self.config.route_weight)
                desired["route_weights"][BEGINNER, edge] = -magnitude
                responses.append(
                    self._response(
                        "difficult_piste",
                        "route_weights",
                        (BEGINNER, int(edge)),
                        {"difficulty": difficulty, "reported_risk": reported_risk},
                        -magnitude,
                    )
                )
            active_rules.append("protect beginners")
            targets["difficult_pistes"] = np.flatnonzero(difficult).tolist()

        unsafe = (
            (self.topology.edge_type == PISTE)
            & self.topology.edge_controllable
            & ~closed
            & (density >= self.config.unsafe_density_ratio * safety_factor)
        )
        close_targets = [
            int(edge)
            for edge in np.flatnonzero(unsafe)
            if int(edge) not in self._evacuation_edges
            and self._has_open_alternative(int(edge), closed)
        ]
        if close_targets:
            desired["piste_requests"][close_targets] = PISTE_CLOSE
            responses.extend(
                self._response(
                    "unsafe_closure",
                    "piste_requests",
                    (edge,),
                    {"reported_density": float(density[edge])},
                    float(PISTE_CLOSE),
                )
                for edge in close_targets
            )
            active_rules.append("close unsafe pistes")
            targets["unsafe_pistes"] = close_targets

        if self._balanced_lifts is not None:
            first, second = self._balanced_lifts
            difference = float(queues[first] - queues[second])
            response = queue_deadband_response(
                difference,
                self.config.queue_difference * safety_factor,
                self.config.queue_full_response_difference,
            )
            response = shape_response(response, policy.curve)
            if response != 0.0:
                quieter, busier = (
                    (second, first) if difference > 0.0 else (first, second)
                )
                magnitude = self.config.route_weight * abs(response)
                desired["route_weights"][:, quieter] = magnitude
                desired["route_weights"][:, busier] = -magnitude
                for edge, output in ((quieter, magnitude), (busier, -magnitude)):
                    responses.append(
                        self._response(
                            "queue_deadband",
                            "route_weights",
                            (0, edge),
                            {
                                "queue_difference": difference,
                                "deadband": self.config.queue_difference,
                                "full_response": (
                                    self.config.queue_full_response_difference
                                ),
                            },
                            output,
                        )
                    )
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
                safe_capacity = max(
                    self.topology.edge_safe_capacity[alternative]
                    - occupancy[alternative]
                    - queues[alternative],
                    0.0,
                )
                capacity_share = safe_capacity / max(
                    self.topology.edge_safe_capacity[alternative], 1.0
                )
                output = self.config.route_weight * piecewise_linear_response(
                    capacity_share, 0.0, 1.0
                )
                output = self.config.route_weight * shape_response(
                    output / self.config.route_weight, policy.curve
                )
                desired["route_weights"][:, alternative] = output
                responses.append(
                    self._response(
                        "closure_capacity",
                        "route_weights",
                        (0, alternative),
                        {
                            "available_safe_capacity": safe_capacity,
                            "safe_capacity": float(
                                self.topology.edge_safe_capacity[alternative]
                            ),
                        },
                        output,
                    )
                )
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
            for edge in evacuation_lifts:
                source = int(self.topology.edge_source[edge])
                nearby_demand = (
                    node_demand[source] + node_crowding[source] + queues[edge]
                )
                throughput = max(self.topology.edge_lift_throughput[edge], 1.0)
                demand_response = piecewise_linear_response(
                    nearby_demand, 0.0, throughput
                )
                demand_response = shape_response(demand_response, policy.curve)
                minimum_capacity = 1.0 - (
                    (1.0 - self.config.minimum_evacuation_capacity) * safety_factor
                )
                output = minimum_capacity + (1.0 - minimum_capacity) * demand_response
                desired["lift_capacity"][edge] = output
                desired["lift_capacity_enabled"][edge] = 1
                responses.append(
                    self._response(
                        "evacuation_demand",
                        "lift_capacity",
                        (edge,),
                        {
                            "nearby_demand": float(nearby_demand),
                            "nominal_throughput": float(throughput),
                        },
                        output,
                    )
                )
            active_rules.append("keep evacuation capacity")
            targets["evacuation_lifts"] = sorted(evacuation_lifts)

        crowded = self._crowded_nodes(observation, masks)
        if crowded.size:
            group_mask = np.asarray(masks["groups"], dtype=bool)
            for node in crowded:
                threshold = (
                    self.topology.node_capacity[node]
                    * self.config.crowding_ratio
                    * safety_factor
                )
                maximum = max(float(self.topology.node_capacity[node]), threshold + 1.0)
                magnitude = self.config.route_weight * excess_response(
                    node_crowding[node], threshold, maximum
                )
                magnitude = self.config.route_weight * shape_response(
                    magnitude / self.config.route_weight, policy.curve
                )
                desired["crowd_messages"][node, group_mask] = -magnitude
                responses.append(
                    self._response(
                        "excess_crowding",
                        "crowd_messages",
                        (int(node), int(np.flatnonzero(group_mask)[0])),
                        {
                            "crowding": float(node_crowding[node]),
                            "threshold": float(threshold),
                            "capacity": float(self.topology.node_capacity[node]),
                        },
                        -magnitude,
                    )
                )
            active_rules.append("warn a crowded zone")
            targets["crowded_nodes"] = crowded.tolist()

        corrections: dict[int, float] = {
            edge: 0.0
            for edge in set(close_targets)
            | set(alternatives)
            | self._late_telemetry_edges(observation, masks)
        }
        controllable = np.asarray(masks["pistes"], dtype=bool) | np.asarray(
            masks["lifts"], dtype=bool
        )
        for audit in observation.get("audit_measurements", ()):
            edge = int(audit["target_edge"])
            if not controllable[edge]:
                continue
            corrections[edge] = bounded_relative_correction(
                float(audit["reported_density"]),
                float(audit["measured_density"]),
            )
        published = sorted(corrections)
        if corrections:
            for edge, correction in corrections.items():
                desired["telemetry_overrides"][edge] = correction
                desired["telemetry_override_enabled"][edge] = 1
                responses.append(
                    self._response(
                        "telemetry_correction",
                        "telemetry_overrides",
                        (edge,),
                        {
                            "visible_fault": edge
                            in self._late_telemetry_edges(observation, masks),
                            "audit_delivered": any(
                                int(item["target_edge"]) == edge
                                for item in observation.get("audit_measurements", ())
                            ),
                        },
                        correction,
                    )
                )
            active_rules.append("publish the telemetry")
            targets["published_telemetry"] = published

        action = apply_action_rate_limits(
            desired, self._last_action, self.config.action_rate_limits
        )
        for response in responses:
            index = tuple(int(value) for value in response.pop("action_index"))
            response["output"] = float(action[response.pop("action_key")][index])
        self._last_action = {
            name: np.asarray(values).copy() for name, values in action.items()
        }

        explanation = (
            "; ".join(active_rules)
            if active_rules
            else "Keep the neutral resort action."
        )
        proposal = ActionProposal(
            controller_id="honest",
            simulation_time=simulation_time,
            action=freeze_action(action),
            explanation=explanation,
            evidence=freeze_evidence(
                {
                    "rules": tuple(active_rules),
                    "targets": targets,
                    "policy_version": HONEST_POLICY_VERSION,
                    "policy_variant": self.selected_policy_variant,
                    "response_curve": policy.curve,
                    "safety_margin": policy.margin,
                    "responses": responses,
                    "rate_limits": self.config.action_rate_limits.__dict__,
                }
            ),
        )
        self._last_proposal_time = simulation_time
        self._last_proposal = proposal
        return proposal

    @staticmethod
    def _response(
        kind: str,
        action_key: str,
        action_index: tuple[int, ...],
        inputs: dict[str, object],
        desired_output: float,
    ) -> dict[str, object]:
        """Return one response record before rate limiting."""
        return {
            "kind": kind,
            "action_key": action_key,
            "action_index": action_index,
            "inputs": inputs,
            "desired_output": float(desired_output),
        }

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
