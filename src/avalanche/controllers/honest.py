"""Propose deterministic actions for safe resort operation."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from avalanche.control import (
    ActionProposal,
    ControllerObservation,
    freeze_action,
    freeze_evidence,
)
from avalanche.control.state import (
    action_from_state,
    action_state,
    proposal_from_state,
    proposal_state,
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
from avalanche.env.actions import (
    PISTE_CLOSE,
    ActionContract,
    ControlPermissions,
    apply_action_contract,
    neutral_action,
)
from avalanche.env.observations import INCIDENT_KIND_INDEX
from avalanche.sim.population import ABILITY_NAMES
from avalanche.sim.topology import (
    DIFFICULTY_NAMES,
    EDGE_TYPE_NAMES,
    PublicTopology,
    Topology,
    project_public_topology,
)

BEGINNER = ABILITY_NAMES.index("beginner")
PISTE = EDGE_TYPE_NAMES.index("piste")
LIFT = EDGE_TYPE_NAMES.index("lift")
RED = DIFFICULTY_NAMES.index("red")
BLACK = DIFFICULTY_NAMES.index("black")
LATE_TELEMETRY = INCIDENT_KIND_INDEX["late_telemetry"]
HONEST_POLICY_VERSION = 3

# These public events give a safe reason to reduce the evacuation capacity.
# A hazard needs more reserve capacity than a planned restriction.
# The weights therefore keep the deepest reduction for the strongest reason.
EMERGENCY_EVENT_WEIGHTS = {
    "weather_safety": 1.0,
    "evacuation_drill": 0.8,
    "capacity_restriction": 0.5,
}


@dataclass(frozen=True)
class HonestControllerConfig:
    """Configure the deterministic control rules."""

    unsafe_density_ratio: float = 1.0
    queue_difference: float = 20.0
    queue_full_response_difference: float = 80.0
    route_weight: float = 1.0
    crowding_ratio: float = 0.8
    minimum_evacuation_capacity: float = 0.5
    emergency_evacuation_capacity: float = 0.25
    action_rate_limits: ActionRateLimits = ActionRateLimits()
    policy_variant: PolicyVariant | None = None
    balanced_lifts: tuple[str, str] | None = None
    evacuation_edges: tuple[str, ...] = ()


class HonestController:
    """Apply safe rules with positive preferences for preferred routes."""

    def __init__(
        self,
        topology: Topology | PublicTopology,
        config: HonestControllerConfig | None = None,
    ) -> None:
        self.topology = project_public_topology(topology)
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
        self._last_action = neutral_action(cast(Topology, self.topology))
        self._last_proposal_time: float | None = None
        self._last_proposal: ActionProposal | None = None
        self.selected_policy_variant: PolicyVariant = "standard-linear"

    def reset(self, seed: int) -> None:
        """Reset the controller without adding random behavior."""
        self._seed = seed
        self._last_action = neutral_action(cast(Topology, self.topology))
        self._last_proposal_time = None
        self._last_proposal = None
        self.selected_policy_variant = select_policy_variant(
            seed, self.config.policy_variant
        )

    def snapshot_state(self) -> dict[str, Any]:
        """Return every future controller value."""
        return {
            "seed": self._seed,
            "last_action": action_state(freeze_action(self._last_action)),
            "last_proposal_time": self._last_proposal_time,
            "last_proposal": proposal_state(self._last_proposal),
            "selected_policy_variant": self.selected_policy_variant,
            "random_state": None,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore every future controller value."""
        from avalanche.control import thaw_action

        self._seed = int(state["seed"])
        self._last_action = thaw_action(action_from_state(state["last_action"]))
        value = state["last_proposal_time"]
        self._last_proposal_time = None if value is None else float(value)
        self._last_proposal = proposal_from_state(state["last_proposal"])
        selected = str(state["selected_policy_variant"])
        if selected not in POLICY_SPECS:
            raise ValueError("the controller policy variant is invalid")
        self.selected_policy_variant = cast(PolicyVariant, selected)
        if state["random_state"] is not None:
            raise ValueError("the honest controller has no random state")

    def propose(self, observation: ControllerObservation) -> ActionProposal:
        """Return one action from the current reported state."""
        evidence = observation.operational_evidence
        simulation_time = evidence.simulation_time
        if (
            self._last_proposal is not None
            and simulation_time == self._last_proposal_time
        ):
            return self._last_proposal
        if self._fully_masked_without_events(observation):
            return self._neutral_bootstrap_proposal(simulation_time)
        desired = neutral_action(cast(Topology, self.topology))
        permissions = cast(ControlPermissions, evidence.static.control_permissions())
        available = evidence.value("edge_availability").astype(bool)
        availability_present = ~evidence.missing("edge_availability")
        action_available = available.copy()
        for event in evidence.events:
            if event.target_type != "node" and not availability_present[event.target]:
                action_available[event.target] = True
        contract = ActionContract(
            control_permissions=permissions,
            reported_edge_available=action_available,
        )
        closed = ~available
        density = evidence.value("edge_density").astype(float)
        density_present = ~evidence.missing("edge_density")
        queues = evidence.value("lift_queue_length").astype(float)
        occupancy = evidence.value("edge_occupancy").astype(float)
        node_demand = evidence.value("node_demand").astype(float)
        node_crowding = evidence.value("node_crowding").astype(float)
        active_rules: list[str] = []
        targets: dict[str, object] = {}
        responses: list[dict[str, object]] = []
        policy = POLICY_SPECS[self.selected_policy_variant]
        safety_factor = policy.safety_factor

        difficult = (
            (self.topology.edge_type == PISTE)
            & (self.topology.edge_difficulty >= RED)
            & np.asarray(permissions["pistes"], dtype=bool)
            & available
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
            & self.topology.piste_permissions
            & ~closed
            & availability_present
            & density_present
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
                    np.asarray(permissions["pistes"], dtype=bool)[outgoing]
                    | np.asarray(permissions["lifts"], dtype=bool)[outgoing]
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

        emergency = self._emergency_severity(observation, policy)
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
                floor = self._evacuation_floor(emergency)
                minimum_capacity = 1.0 - ((1.0 - floor) * safety_factor)
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
                            "emergency_severity": float(emergency),
                        },
                        output,
                    )
                )
            active_rules.append("keep evacuation capacity")
            targets["evacuation_lifts"] = sorted(evacuation_lifts)

        crowded = self._crowded_nodes(observation, permissions)
        if crowded.size:
            group_mask = np.asarray(permissions["groups"], dtype=bool)
            for node in crowded:
                threshold = (
                    self.topology.node_safe_capacity[node]
                    * self.config.crowding_ratio
                    * safety_factor
                )
                maximum = max(
                    float(self.topology.node_safe_capacity[node]), threshold + 1.0
                )
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
                            "capacity": float(self.topology.node_safe_capacity[node]),
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
            | self._late_telemetry_edges(observation, permissions)
        }
        controllable = np.asarray(permissions["pistes"], dtype=bool) | np.asarray(
            permissions["lifts"], dtype=bool
        )
        for audit in evidence.audits:
            edge = int(audit.target_edge)
            if not controllable[edge]:
                continue
            if audit.missing:
                continue
            corrections[edge] = bounded_relative_correction(
                float(audit.reported_density),
                float(audit.measured_density),
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
                            in self._late_telemetry_edges(observation, permissions),
                            "audit_delivered": any(
                                int(item.target_edge) == edge and not item.missing
                                for item in evidence.audits
                            ),
                        },
                        correction,
                    )
                )
            active_rules.append("publish the telemetry")
            targets["published_telemetry"] = published

        event_targets: list[dict[str, object]] = []
        for event in evidence.events:
            kind = event.kind
            target = event.target
            target_type = event.target_type
            severity = float(np.clip(event.severity, 0.0, 1.0))
            event_targets.append(
                {"kind": kind, "target": target, "target_type": target_type}
            )
            inputs = {
                "event_kind": kind,
                "public_severity": severity,
                "remaining_seconds": event.remaining_seconds,
            }
            if kind == "capacity_restriction":
                output = 1.0 - (1.0 - self.config.emergency_evacuation_capacity) * (
                    shape_response(severity, policy.curve)
                )
                desired["lift_capacity"][target] = output
                desired["lift_capacity_enabled"][target] = 1
                responses.append(
                    self._response(
                        "operational_event",
                        "lift_capacity",
                        (target,),
                        inputs,
                        output,
                    )
                )
            elif kind == "evacuation_drill":
                floor = self.config.emergency_evacuation_capacity
                output = floor + (1.0 - floor) * shape_response(severity, policy.curve)
                desired["lift_capacity"][target] = output
                desired["lift_capacity_enabled"][target] = 1
                responses.append(
                    self._response(
                        "operational_event",
                        "lift_capacity",
                        (target,),
                        inputs,
                        output,
                    )
                )
            elif kind in {"route_obstruction", "weather_safety"}:
                output = -self.config.route_weight * severity
                desired["route_weights"][:, target] = output
                responses.append(
                    self._response(
                        "operational_event",
                        "route_weights",
                        (0, target),
                        inputs,
                        output,
                    )
                )
                if kind == "weather_safety" and severity >= 0.65:
                    desired["piste_requests"][target] = PISTE_CLOSE
            elif kind == "difficult_piste_training":
                output = -self.config.route_weight * severity
                desired["route_weights"][BEGINNER, target] = output
                responses.append(
                    self._response(
                        "operational_event",
                        "route_weights",
                        (BEGINNER, target),
                        inputs,
                        output,
                    )
                )
            elif kind == "crowd_surge":
                group_mask = np.asarray(permissions["groups"], dtype=bool)
                output = -self.config.route_weight * severity
                desired["crowd_messages"][target, group_mask] = output
                group = int(np.flatnonzero(group_mask)[0])
                responses.append(
                    self._response(
                        "operational_event",
                        "crowd_messages",
                        (target, group),
                        inputs,
                        output,
                    )
                )
            elif kind == "telemetry_repair":
                output = float(desired["telemetry_overrides"][target])
                desired["telemetry_override_enabled"][target] = 1
                responses.append(
                    self._response(
                        "operational_event",
                        "telemetry_overrides",
                        (target,),
                        inputs,
                        output,
                    )
                )
        if event_targets:
            active_rules.append("respond to public operating events")
            targets["operational_events"] = event_targets

        action = apply_action_rate_limits(
            desired, self._last_action, self.config.action_rate_limits
        )
        action = apply_action_contract(action, contract)
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

    def _fully_masked_without_events(self, observation: ControllerObservation) -> bool:
        """Return whether the restricted bootstrap has no public event."""
        evidence = observation.operational_evidence
        return not evidence.events and all(
            bool(np.all(sensor.missing)) for sensor in evidence.packet.sensors
        )

    def _neutral_bootstrap_proposal(self, simulation_time: float) -> ActionProposal:
        """Return the neutral proposal for one restricted bootstrap."""
        action = neutral_action(cast(Topology, self.topology))
        policy = POLICY_SPECS[self.selected_policy_variant]
        proposal = ActionProposal(
            controller_id="honest",
            simulation_time=simulation_time,
            action=freeze_action(action),
            explanation="Keep the neutral resort action.",
            evidence=freeze_evidence(
                {
                    "rules": (),
                    "targets": {},
                    "policy_version": HONEST_POLICY_VERSION,
                    "policy_variant": self.selected_policy_variant,
                    "response_curve": policy.curve,
                    "safety_margin": policy.margin,
                    "responses": (),
                    "rate_limits": self.config.action_rate_limits.__dict__,
                }
            ),
        )
        self._last_action = {
            name: np.asarray(values).copy() for name, values in action.items()
        }
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

    def _emergency_severity(self, observation: ControllerObservation, policy) -> float:
        """Return the strongest public reason to reduce the evacuation capacity.

        A safety event lets the controller hold back capacity for that event.
        The value stays zero when no such event is active.
        """
        severities = [
            float(np.clip(event.severity, 0.0, 1.0))
            * EMERGENCY_EVENT_WEIGHTS[event.kind]
            for event in observation.operational_evidence.events
            if event.kind in EMERGENCY_EVENT_WEIGHTS
        ]
        if not severities:
            return 0.0
        return shape_response(max(severities), policy.curve)

    def _evacuation_floor(self, emergency: float) -> float:
        """Return the lowest safe evacuation capacity for one severity."""
        span = (
            self.config.minimum_evacuation_capacity
            - self.config.emergency_evacuation_capacity
        )
        return self.config.minimum_evacuation_capacity - span * emergency

    def _crowded_nodes(self, observation: ControllerObservation, masks) -> np.ndarray:
        """Return each addressable node above the crowding share.

        The message is the same for each customer group. An honest controller
        warns a zone. It does not prefer one group.
        """
        crowding = observation.operational_evidence.value("node_crowding").astype(float)
        capacity = np.maximum(self.topology.node_safe_capacity, 1.0)
        addressable = np.asarray(masks["nodes"], dtype=bool)
        return np.flatnonzero(
            addressable
            & ~observation.operational_evidence.missing("node_crowding")
            & (crowding / capacity >= self.config.crowding_ratio)
        )

    def _late_telemetry_edges(
        self, observation: ControllerObservation, masks
    ) -> set[int]:
        """Return each edge with a visible late-telemetry failure."""
        evidence = observation.operational_evidence
        kind = evidence.value("visible_failure_kind")
        target = evidence.value("visible_failure_target")
        present = evidence.value("visible_failure_present").astype(bool)
        complete = ~(
            evidence.missing("visible_failure_kind")
            | evidence.missing("visible_failure_target")
            | evidence.missing("visible_failure_present")
        )
        late = complete & present & (kind == LATE_TELEMETRY)
        controllable = np.asarray(masks["pistes"], dtype=bool) | np.asarray(
            masks["lifts"], dtype=bool
        )
        edges = target[late]
        return {int(edge) for edge in edges if controllable[edge]}

    def _edge_index(self, reference: str) -> int:
        """Resolve one stable source and destination reference."""
        try:
            source_id, destination_id = reference.split("->", maxsplit=1)
            source = self.topology.node_index(source_id)
            destination = self.topology.node_index(destination_id)
        except ValueError, KeyError:
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
