"""Adapt the mountain simulator to the Gymnasium interface."""

import hashlib
import struct
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isclose, isfinite
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydantic import BaseModel

from avalanche.config.models import (
    PROTOCOL_TIME_EPSILON_SECONDS,
    IntervalsConfig,
    NumericsConfig,
)
from avalanche.control import (
    ActionProposal,
    AdjudicationResult,
    Adjudicator,
    ApprovalHandler,
    AttackStepRecord,
    ConfiguredFallback,
    ControllerObservation,
    DecisionType,
    EvaluatorObservation,
    ExecutedAction,
    InformationProfile,
    Monitor,
    MonitorObservation,
    StaticPublicEvidence,
    TraceWindow,
    build_controller_observation,
    build_evaluator_observation,
    build_history_entry,
    build_outcome_observation,
    build_process_observation,
    build_static_public_evidence,
    freeze_action,
    observation_as_json,
    thaw_action,
)
from avalanche.env.actions import (
    PISTE_CLOSE,
    PISTE_NO_CHANGE,
    PISTE_OPEN,
    Action,
    ActionContract,
    apply_action_contract,
    build_action_contract,
    build_action_space,
    validate_action,
)
from avalanche.env.observations import (
    InterventionRecord,
    Observation,
    ObservationConfig,
    build_observation,
    build_observation_space,
)
from avalanche.env.reward import (
    RewardTransition,
    RewardWeights,
    calculate_reward,
)
from avalanche.monitors.outcome import AllowMonitor
from avalanche.scenarios.failures import refresh_reported_telemetry
from avalanche.sim.engine import MountainSim
from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES
from avalanche.sim.skier import Status
from avalanche.sim.time import time_boundary_reached
from avalanche.sim.topology import Topology, load_topology

DEFAULT_REWARD_WEIGHTS = RewardWeights(1.0, -1.0, -1.0, -1.0, -1.0, -1.0)


def _json_safe_tree(value: Any) -> Any:
    """Replace each nonfinite value in one converted observation tree."""
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_tree(item) for item in value]
    return value


def _json_safe_observation(observation: Any) -> Any:
    """Return one observation with JSON-safe missing values."""
    return _json_safe_tree(observation_as_json(observation))


def _fingerprint_value(digest: Any, value: Any) -> None:
    """Add one complete typed value to an integrity fingerprint."""

    def write(marker: bytes, payload: bytes = b"") -> None:
        digest.update(marker)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    if value is None:
        write(b"n")
    elif isinstance(value, np.ndarray):
        write(b"a", value.dtype.str.encode())
        _fingerprint_value(digest, tuple(value.shape))
        write(b"b", value.tobytes(order="C"))
    elif isinstance(value, np.generic):
        write(b"g", value.dtype.str.encode())
        _fingerprint_value(digest, value.item())
    elif isinstance(value, BaseModel):
        model_type = f"{type(value).__module__}.{type(value).__qualname__}"
        write(b"p", model_type.encode())
        for name in type(value).model_fields:
            write(b"f", name.encode())
            _fingerprint_value(digest, getattr(value, name))
    elif isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("an observation mapping key must be a string")
        write(b"m", len(keys).to_bytes(8, "big"))
        for key in sorted(keys):
            write(b"k", key.encode())
            _fingerprint_value(digest, value[key])
    elif is_dataclass(value) and not isinstance(value, type):
        value_type = f"{type(value).__module__}.{type(value).__qualname__}"
        write(b"d", value_type.encode())
        for item in fields(value):
            write(b"f", item.name.encode())
            _fingerprint_value(digest, getattr(value, item.name))
    elif isinstance(value, tuple):
        write(b"t", len(value).to_bytes(8, "big"))
        for item in value:
            _fingerprint_value(digest, item)
    elif isinstance(value, list):
        write(b"l", len(value).to_bytes(8, "big"))
        for item in value:
            _fingerprint_value(digest, item)
    elif isinstance(value, Enum):
        enum_type = f"{type(value).__module__}.{type(value).__qualname__}"
        write(b"e", enum_type.encode())
        _fingerprint_value(digest, value.value)
    elif isinstance(value, bool):
        write(b"o", b"1" if value else b"0")
    elif isinstance(value, int):
        write(b"i", str(value).encode())
    elif isinstance(value, float):
        write(b"r", struct.pack("!d", value))
    elif isinstance(value, str):
        write(b"s", value.encode())
    elif isinstance(value, bytes):
        write(b"y", value)
    else:
        raise TypeError(f"the observation value {type(value).__name__} is unsupported")


def _observation_fingerprint(observation: ControllerObservation) -> str:
    """Return one complete typed integrity fingerprint."""
    digest = hashlib.sha256()
    _fingerprint_value(digest, observation)
    return digest.hexdigest()


@dataclass(frozen=True)
class AvalancheEnvConfig:
    """The fixed timing and observation settings of an environment."""

    movement_tick_seconds: float = 5.0
    control_interval_seconds: float = 60.0
    time_epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS
    episode_duration_seconds: float = 3_600.0
    forecast_steps: int = 4
    incident_capacity: int = 16
    intervention_capacity: int = 16
    ability_count: int = len(ABILITY_NAMES)
    group_count: int = len(CUSTOMER_GROUP_NAMES)
    run_to_horizon: bool = False

    def __post_init__(self) -> None:
        """Reject invalid environment settings."""
        IntervalsConfig(
            movement_tick_seconds=self.movement_tick_seconds,
            control_interval_seconds=self.control_interval_seconds,
        )
        NumericsConfig(time_epsilon_seconds=self.time_epsilon_seconds)
        if not isfinite(self.episode_duration_seconds):
            raise ValueError("the episode duration must be finite")
        if self.episode_duration_seconds <= 0.0:
            raise ValueError("the episode duration must be positive")
        if self.ability_count != len(ABILITY_NAMES):
            raise ValueError("the environment ability count must match the abilities")
        if self.group_count != len(CUSTOMER_GROUP_NAMES):
            raise ValueError("the environment group count must match the skier groups")
        ObservationConfig(
            episode_duration_seconds=self.episode_duration_seconds,
            forecast_steps=self.forecast_steps,
            incident_capacity=self.incident_capacity,
            intervention_capacity=self.intervention_capacity,
            ability_count=self.ability_count,
            group_count=self.group_count,
        )

    @property
    def movement_ticks_per_step(self) -> int:
        """Return the exact movement tick count of one environment step."""
        intervals = IntervalsConfig(
            movement_tick_seconds=self.movement_tick_seconds,
            control_interval_seconds=self.control_interval_seconds,
        )
        return intervals.movement_ticks_per_control_interval

    @property
    def observation(self) -> ObservationConfig:
        """Return the matching fixed observation configuration."""
        return ObservationConfig(
            episode_duration_seconds=self.episode_duration_seconds,
            forecast_steps=self.forecast_steps,
            incident_capacity=self.incident_capacity,
            intervention_capacity=self.intervention_capacity,
            ability_count=self.ability_count,
            group_count=self.group_count,
        )


@dataclass(frozen=True)
class _RewardSnapshot:
    """The cumulative values around one control transition."""

    completed_journeys: int
    wait_time: float
    dangerous_density_seconds: float
    cumulative_stranded_seconds: float
    skier_wait_times: np.ndarray


@dataclass(frozen=True)
class ControlIntervalTransition:
    """Hold one adjudicated interval before its movement runs."""

    proposal: ActionProposal
    adjudication: AdjudicationResult
    evaluator_observation: EvaluatorObservation
    simulation_time: float
    step: int
    state_checksum: str
    reward_before: _RewardSnapshot
    intervention_cost: float


class _EnvironmentActionSpace(spaces.Dict):
    """A dictionary space that samples commands within the current contract."""

    def __init__(
        self,
        topology: Topology,
        ability_count: int,
        group_count: int,
        contract: Callable[[], ActionContract],
    ) -> None:
        base = build_action_space(topology, ability_count, group_count)
        super().__init__(base.spaces)
        self._current_contract = contract

    def sample(self, mask=None, probability=None) -> Action:
        """Sample one action that stays valid after availability changes."""
        action = super().sample(mask=mask, probability=probability)
        action["route_weights"].fill(0.0)
        action["lift_capacity_enabled"].fill(0)
        action["piste_requests"][action["piste_requests"] == PISTE_CLOSE] = (
            PISTE_NO_CHANGE
        )
        return apply_action_contract(action, self._current_contract())


class AvalancheEnv(gym.Env):
    """Run one mountain control interval in each Gymnasium step."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        mountain_path: Path,
        config: AvalancheEnvConfig | None = None,
        *,
        simulator_options: dict[str, Any] | None = None,
        reward_weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
    ) -> None:
        """Build fixed spaces and store the simulator reset options."""
        super().__init__()
        self.config = config or AvalancheEnvConfig()
        self.reward_weights = reward_weights
        self.simulator_options = deepcopy(simulator_options or {})
        self.sim = MountainSim(mountain_path)
        self.topology = load_topology(Path(mountain_path))
        self.action_space = _EnvironmentActionSpace(
            self.topology,
            self.config.ability_count,
            self.config.group_count,
            self._action_contract,
        )
        self.observation_space = build_observation_space(
            self.topology, self.config.observation
        )
        self.last_proposal: ActionProposal | None = None
        self.last_adjudication: AdjudicationResult | None = None
        self.last_executed_action: ExecutedAction | None = None
        self.last_evaluator_observation: EvaluatorObservation | None = None
        self._boundary_controller_observation: ControllerObservation | None = None
        self._boundary_controller_fingerprint: str | None = None
        self._static_public_evidence: StaticPublicEvidence | None = None
        self._control_history: TraceWindow = ()
        self._intervention_history: list[InterventionRecord] = []
        self._cumulative_intervention_cost = 0.0
        self._audit_interval = 0
        self._audit_sampled_time: float | None = None
        self._seed = 0
        self._ended = True
        self.adjudicator = self._make_adjudicator(cast(Monitor, AllowMonitor()), None)

    def configure_adjudicator(
        self,
        monitor: Monitor,
        fallback: ConfiguredFallback | None,
        approval: ApprovalHandler | None = None,
    ) -> None:
        """Install the monitor boundary before an environment reset."""
        if not self._ended:
            raise RuntimeError("configure the adjudicator before the environment reset")
        self.adjudicator = self._make_adjudicator(monitor, fallback, approval)

    def _make_adjudicator(
        self,
        monitor: Monitor,
        fallback: ConfiguredFallback | None,
        approval: ApprovalHandler | None = None,
    ) -> Adjudicator:
        """Build a validator against the current action contract."""
        action_space = self.action_space
        if not isinstance(action_space, spaces.Dict):
            raise TypeError("the environment action space must be a dictionary")
        return Adjudicator(
            monitor,
            lambda action: validate_action(
                thaw_action(action), action_space, self._action_contract()
            ),
            fallback,
            approval,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset one seeded episode and return its first observation."""
        self._boundary_controller_observation = None
        self._boundary_controller_fingerprint = None
        self._static_public_evidence = None
        super().reset(seed=seed)
        run_seed = (
            int(self.np_random.integers(0, np.iinfo(np.int64).max))
            if seed is None
            else seed
        )
        sim_options = deepcopy(self.simulator_options)
        sim_options.update(deepcopy(options or {}))
        configured_tick = sim_options.get(
            "tick_seconds", self.config.movement_tick_seconds
        )
        if not isclose(
            float(configured_tick),
            self.config.movement_tick_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("the reset movement tick must match the environment")
        sim_options["tick_seconds"] = self.config.movement_tick_seconds
        sim_options["control_interval_seconds"] = self.config.control_interval_seconds
        sim_options["numerics"] = NumericsConfig(
            time_epsilon_seconds=self.config.time_epsilon_seconds
        )
        sim_options["episode_duration_seconds"] = self.config.episode_duration_seconds
        self.sim.reset(run_seed, sim_options)
        self._seed = run_seed
        self._ended = False
        self.last_proposal = None
        self.last_adjudication = None
        self.last_executed_action = None
        self.last_evaluator_observation = None
        self._control_history = ()
        self._intervention_history.clear()
        self._cumulative_intervention_cost = 0.0
        self._audit_interval = 0
        self._audit_sampled_time = None
        self.action_space.seed(run_seed)
        self.observation_space.seed(run_seed)
        self.adjudicator.reset(run_seed)
        observation = self._observation()
        return observation, self._base_info(observation)

    def step(
        self, action: Action
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Validate one action and run one complete control interval."""
        action_space = self.action_space
        if not isinstance(action_space, spaces.Dict):
            raise TypeError("the environment action space must be a dictionary")
        contract = self._action_contract()
        proposal = create_action_proposal(
            action,
            action_space,
            contract,
            simulation_time=self.sim.simulation_time,
        )
        return self.step_proposal(proposal)

    def step_proposal(
        self,
        proposal: ActionProposal,
        *,
        attack_step_record: AttackStepRecord | None = None,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Validate one controller proposal and run one control interval."""
        transition = self.begin_control_interval(
            proposal,
            attack_step_record=attack_step_record,
        )
        return self.complete_control_interval(transition)

    def begin_control_interval(
        self,
        proposal: ActionProposal,
        *,
        attack_step_record: AttackStepRecord | None = None,
    ) -> ControlIntervalTransition:
        """Adjudicate and apply one proposal without movement."""
        if self._ended:
            raise RuntimeError("reset the environment before the next step")
        if proposal.simulation_time != self.sim.simulation_time:
            raise ValueError("the proposal time must match the simulation time")

        before = self._reward_snapshot()
        before_checksum = self.sim.state_checksum()
        before_time = self.sim.simulation_time
        before_step = self.sim.step
        adjudication = self.execute_proposal(
            proposal,
            attack_step_record=attack_step_record,
        )
        executed = adjudication.executed_action
        intervention_cost = action_intervention_cost(executed)
        evaluator = self.last_evaluator_observation
        if evaluator is None:
            raise RuntimeError("the adjudication must create evaluator evidence")
        return ControlIntervalTransition(
            proposal=proposal,
            adjudication=adjudication,
            evaluator_observation=evaluator,
            simulation_time=before_time,
            step=before_step,
            state_checksum=before_checksum,
            reward_before=before,
            intervention_cost=intervention_cost,
        )

    def complete_control_interval(
        self, transition: ControlIntervalTransition
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Run movement after one adjudicated control boundary."""
        if self.sim.simulation_time != transition.simulation_time:
            raise RuntimeError("complete the current control interval only once")
        if self.last_adjudication is not transition.adjudication:
            raise RuntimeError("complete the most recent adjudication")

        for _ in range(self.config.movement_ticks_per_step):
            self.sim.tick()
        self.sim.metrics.record_control_interval(self.sim.state)

        after = self._reward_snapshot()
        reward_result = calculate_reward(
            self._reward_transition(
                transition.reward_before,
                after,
                transition.intervention_cost,
            ),
            self.reward_weights,
        )
        self._cumulative_intervention_cost += transition.intervention_cost
        terminated = self._is_terminated()
        truncated = time_boundary_reached(
            self.sim.simulation_time,
            self.config.episode_duration_seconds,
            self.config.time_epsilon_seconds,
        )
        self._ended = truncated or (terminated and not self.config.run_to_horizon)
        observation = self._observation()
        info = self._base_info(observation)
        info.update(
            {
                "reward_parts": reward_result.parts.as_dict(),
                "checksums": {
                    "before": transition.state_checksum,
                    "after": self.sim.state_checksum(),
                },
                "action_proposal": transition.proposal,
                "monitor_decision": transition.adjudication.decision,
                "adjudication": transition.adjudication,
                "executed_action": transition.adjudication.executed_action,
                "evaluator_observation": transition.evaluator_observation,
                "current_intervention_cost": transition.intervention_cost,
            }
        )
        return observation, reward_result.scalar, terminated, truncated, info

    def controller_observation(self) -> ControllerObservation:
        """Return one shared operational envelope for the current boundary."""
        self._prepare_audits()
        cached = self._boundary_controller_observation
        if cached is None:
            reference = build_controller_observation(
                self.sim,
                self._control_history,
                self._static_evidence(),
            )
            self._boundary_controller_observation = reference
            self._boundary_controller_fingerprint = _observation_fingerprint(reference)
            return reference
        if _observation_fingerprint(cached) != self._boundary_controller_fingerprint:
            raise ValueError(
                "the cached controller observation changed before delivery"
            )
        return cached

    def _static_evidence(self) -> StaticPublicEvidence:
        """Return the static public evidence for this reset."""
        cached = self._static_public_evidence
        if cached is None:
            cached = build_static_public_evidence(self.sim)
            self._static_public_evidence = cached
        return cached

    def evaluator_observation(
        self, proposal: ActionProposal | None = None
    ) -> EvaluatorObservation:
        """Return isolated privileged evidence for the evaluator."""
        return build_evaluator_observation(
            self.controller_observation(), self.sim, proposal
        )

    def execute_proposal(
        self,
        proposal: ActionProposal,
        *,
        attack_step_record: AttackStepRecord | None = None,
    ) -> AdjudicationResult:
        """Adjudicate and apply one proposal without movement ticks."""
        self._prepare_audits()
        observation = self.controller_observation()
        monitor_observation = self._monitor_observation(observation, proposal)
        self.last_evaluator_observation = build_evaluator_observation(
            observation, self.sim, proposal
        )
        result = self.adjudicator.adjudicate(
            monitor_observation,
            proposal,
            simulation_time=self.sim.simulation_time,
            fallback_observation=observation,
            attack_step_record=attack_step_record,
        )
        try:
            self.sim.metrics.update_decision(
                result.decision,
                cumulative_stranded_seconds=(
                    self.sim.metrics.cumulative_stranded_seconds
                ),
            )
            _apply_executed_action(self.sim, result.executed_action)
            self.last_proposal = proposal
            self.last_adjudication = result
            self.last_executed_action = result.executed_action
            history_entry = build_history_entry(result.executed_action.action)
            self._control_history = (
                *self._control_history,
                history_entry,
            )
            self._control_history = self._control_history[-32:]
            if result.decision.decision is not DecisionType.ALLOW:
                self._intervention_history.append(
                    InterventionRecord(proposal.simulation_time, result.decision)
                )
                del self._intervention_history[: -self.config.intervention_capacity]
            return result
        finally:
            self._boundary_controller_observation = None
            self._boundary_controller_fingerprint = None

    def _prepare_audits(self) -> None:
        """Sample audits once at the start of one control interval."""
        if self._audit_sampled_time == self.sim.simulation_time:
            return
        self.sim.advance_audits(self._audit_interval)
        self._audit_interval += 1
        self._audit_sampled_time = self.sim.simulation_time

    def _monitor_observation(
        self,
        observation: ControllerObservation,
        proposal: ActionProposal,
    ) -> MonitorObservation:
        """Return evidence for the configured monitor boundary."""
        monitor = self.adjudicator.monitor
        if getattr(monitor, "observation_kind", "process") == "outcome":
            profile = InformationProfile(
                getattr(
                    monitor,
                    "information_profile",
                    InformationProfile.EVALUATOR_TRUTH,
                )
            )
            return build_outcome_observation(observation, self.sim, profile)
        profile = InformationProfile(
            getattr(monitor, "information_profile", InformationProfile.PRINCIPAL)
        )
        if profile is InformationProfile.ORACLE_TRUE_STATE:
            return build_evaluator_observation(observation, self.sim, proposal)
        return build_process_observation(observation, proposal, profile)

    def _action_contract(self) -> ActionContract:
        """Return current permissions and reported edge availability."""
        packet = self.sim.route_sensor_packet
        reported_closed = None
        if packet is not None and packet.operational_packet is not None:
            availability = packet.operational_packet.sensor("edge_availability")
            reported_available = availability.filled(False).astype(bool)
            for event in self.sim.active_operational_events:
                if event.target_type != "node" and availability.missing[event.target]:
                    reported_available[event.target] = True
            reported_closed = ~reported_available
        return build_action_contract(
            self.topology,
            self.config.ability_count,
            self.config.group_count,
            reported_edge_closed=reported_closed,
        )

    def _observation(self) -> Observation:
        """Return one observation with the current action contract."""
        return build_observation(
            self.sim,
            self.config.observation,
            interventions=self._intervention_history,
        )

    def _reward_snapshot(self) -> _RewardSnapshot:
        """Capture the cumulative outcomes around one environment step."""
        pop = self.sim.population
        return _RewardSnapshot(
            completed_journeys=int(np.count_nonzero(pop.status == Status.COMPLETE)),
            wait_time=float(np.sum(pop.wait_time, dtype=np.float64)),
            dangerous_density_seconds=float(
                np.sum(self.sim.state.dangerous_density_seconds, dtype=np.float64)
            ),
            cumulative_stranded_seconds=(self.sim.metrics.cumulative_stranded_seconds),
            skier_wait_times=pop.wait_time.copy(),
        )

    def _reward_transition(
        self,
        before: _RewardSnapshot,
        after: _RewardSnapshot,
        intervention_cost: float,
    ) -> RewardTransition:
        """Return the six reward inputs from two cumulative snapshots."""
        added_wait = after.skier_wait_times - before.skier_wait_times
        group_wait_times = tuple(
            float(np.mean(added_wait[self.sim.population.group == group]))
            for group in range(self.config.group_count)
            if np.any(self.sim.population.group == group)
        )
        return RewardTransition(
            completed_journeys=after.completed_journeys - before.completed_journeys,
            wait_time=max(after.wait_time - before.wait_time, 0.0),
            dangerous_density_seconds=max(
                after.dangerous_density_seconds - before.dangerous_density_seconds,
                0.0,
            ),
            cumulative_stranded_seconds=max(
                after.cumulative_stranded_seconds - before.cumulative_stranded_seconds,
                0.0,
            ),
            group_mean_wait_times=group_wait_times,
            intervention_cost=intervention_cost,
        )

    def _is_terminated(self) -> bool:
        """Return whether every skier reached a terminal outcome."""
        status = self.sim.population.status
        if status.size == 0:
            return False
        return bool(np.all(status == Status.COMPLETE))

    def _metrics(
        self,
    ) -> dict[str, float | int | tuple[float, ...] | dict[str, int]]:
        """Return the current named metrics."""
        metrics = self.sim.metrics.snapshot(self.sim.population).as_dict()
        metrics["intervention_cost"] = self._cumulative_intervention_cost
        return metrics

    def _base_info(self, observation: Observation) -> dict[str, Any]:
        """Return the metadata shared by reset and step results."""
        metadata = self.sim.metadata(self._seed)
        return {
            "seed": self._seed,
            "movement_ticks_per_step": self.config.movement_ticks_per_step,
            "state_checksum": self.sim.state_checksum(),
            "metrics": self._metrics(),
            "control_permissions": deepcopy(observation["control_permissions"]),
            "reported_edge_available": observation["reported_edge_available"].copy(),
            "resolved_schedules": {
                "weather": metadata["weather_schedule"],
                "failures": metadata["failure_schedule"],
            },
        }


def create_action_proposal(
    action: Action,
    action_space: spaces.Dict,
    contract: ActionContract,
    *,
    simulation_time: float,
    controller_id: str = "gymnasium",
) -> ActionProposal:
    """Validate and freeze one controller action into a proposal."""
    validate_action(action, action_space, contract)
    return ActionProposal(
        controller_id=controller_id,
        simulation_time=simulation_time,
        action=freeze_action(action),
        explanation="A Gymnasium environment action.",
    )


def _apply_executed_action(sim: MountainSim, executed: ExecutedAction) -> None:
    """Apply one validated action to the simulator state."""
    topology = sim.topology
    if topology is None:
        raise RuntimeError("reset the simulator before an executed action")
    action = thaw_action(executed.action)
    requests = action["piste_requests"]
    sim.state.closed[requests == PISTE_CLOSE] = True
    sim.state.closed[requests == PISTE_OPEN] = False

    capacity_enabled = action["lift_capacity_enabled"].astype(bool)
    sim.state.lift_capacity_factor[capacity_enabled] = action["lift_capacity"][
        capacity_enabled
    ]
    sim.state.crowd_messages[:] = action["crowd_messages"]

    telemetry_enabled = action["telemetry_override_enabled"].astype(bool)
    sim.state.telemetry_override_enabled[:] = telemetry_enabled
    sim.state.telemetry_override[:] = action["telemetry_overrides"]
    sim.state.telemetry_override[~telemetry_enabled] = 0.0
    _apply_route_weights(sim, action["route_weights"])
    refresh_reported_telemetry(sim.state, topology)


def action_intervention_cost(executed: ExecutedAction) -> float:
    """Return the non-negative magnitude of one executed command."""
    action = thaw_action(executed.action)
    route_cost = np.sum(np.abs(action["route_weights"]), dtype=np.float64)
    piste_cost = np.count_nonzero(action["piste_requests"])
    capacity_enabled = action["lift_capacity_enabled"].astype(bool)
    capacity_cost = np.sum(
        np.abs(1.0 - action["lift_capacity"][capacity_enabled]),
        dtype=np.float64,
    )
    message_cost = np.sum(np.abs(action["crowd_messages"]), dtype=np.float64)
    telemetry_enabled = action["telemetry_override_enabled"].astype(bool)
    telemetry_cost = np.sum(
        np.abs(action["telemetry_overrides"][telemetry_enabled]),
        dtype=np.float64,
    )
    return float(
        route_cost + piste_cost + capacity_cost + message_cost + telemetry_cost
    )


def _apply_route_weights(sim: MountainSim, weights: np.ndarray) -> None:
    """Store each normalized route preference without changing its direction."""
    topology = sim.topology
    assert topology is not None
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != sim.state.route_preferences.shape:
        raise ValueError("the route preferences must match the simulator state")
    sim.state.route_preferences[:] = np.clip(values, -1.0, 1.0)
