"""Adapt the mountain simulator to the Gymnasium interface."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
    ConfiguredFallback,
    ControllerObservation,
    DecisionType,
    EvaluatorObservation,
    ExecutedAction,
    InformationProfile,
    Monitor,
    MonitorObservation,
    ProcessObservation,
    build_controller_observation,
    build_evaluator_observation,
    build_outcome_observation,
    build_process_observation,
    freeze_action,
    thaw_action,
)
from avalanche.env.actions import (
    PISTE_CLOSE,
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
from avalanche.sim.movement import effective_closed
from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES
from avalanche.sim.routes import NO_EDGE
from avalanche.sim.skier import Status
from avalanche.sim.time import time_boundary_reached
from avalanche.sim.topology import Topology, load_topology

DEFAULT_REWARD_WEIGHTS = RewardWeights(1.0, -1.0, -1.0, -1.0, -1.0, -1.0)


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
    dangerous_density: float
    stranded_skiers: int
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
        """Sample one action and neutralise each unavailable command."""
        action = super().sample(mask=mask, probability=probability)
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
        self._control_history: list[dict[str, Any]] = []
        self._intervention_history: list[InterventionRecord] = []
        self._cumulative_intervention_cost = 0.0
        self._audit_interval = 0
        self._audit_sampled_time: float | None = None
        self._seed = 0
        self._ended = True
        self.adjudicator = self._make_adjudicator(AllowMonitor(), None)

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
        self._control_history.clear()
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
        self, proposal: ActionProposal
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Validate one controller proposal and run one control interval."""
        transition = self.begin_control_interval(proposal)
        return self.complete_control_interval(transition)

    def begin_control_interval(
        self, proposal: ActionProposal
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
        adjudication = self.execute_proposal(proposal)
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
        self._ended = terminated or truncated
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
        """Return the isolated reported state for one controller."""
        self._prepare_audits()
        return build_controller_observation(
            self._observation(),
            self.sim.simulation_time,
            self.sim.delivered_audits,
            tuple(
                event.public(self.sim.simulation_time)
                for event in self.sim.active_operational_events
            ),
        )

    def evaluator_observation(
        self, proposal: ActionProposal | None = None
    ) -> EvaluatorObservation:
        """Return isolated privileged evidence for the evaluator."""
        return build_evaluator_observation(
            self.controller_observation(), self.sim, proposal
        )

    def execute_proposal(self, proposal: ActionProposal) -> AdjudicationResult:
        """Adjudicate and apply one proposal without movement ticks."""
        self._prepare_audits()
        observation = self.controller_observation()
        monitor_observation = self._monitor_observation(observation)
        self.last_evaluator_observation = build_evaluator_observation(
            observation, self.sim, proposal
        )
        result = self.adjudicator.adjudicate(
            monitor_observation,
            proposal,
            tuple(self._control_history),
            simulation_time=self.sim.simulation_time,
        )
        self.sim.metrics.update_decision(
            result.decision,
            harm_count=float(np.sum(self.sim.state.harm_count, dtype=np.int64)),
        )
        _apply_executed_action(self.sim, result.executed_action)
        self.last_proposal = proposal
        self.last_adjudication = result
        self.last_executed_action = result.executed_action
        self._control_history.append(
            {
                "proposal": proposal.model_dump(mode="json"),
                "decision": result.decision.model_dump(mode="json"),
            }
        )
        del self._control_history[:-32]
        if result.decision.decision is not DecisionType.ALLOW:
            self._intervention_history.append(
                InterventionRecord(proposal.simulation_time, result.decision)
            )
            del self._intervention_history[: -self.config.intervention_capacity]
        return result

    def _prepare_audits(self) -> None:
        """Sample audits once at the start of one control interval."""
        if self._audit_sampled_time == self.sim.simulation_time:
            return
        self.sim.advance_audits(self._audit_interval)
        self._audit_interval += 1
        self._audit_sampled_time = self.sim.simulation_time

    def _monitor_observation(
        self, observation: ControllerObservation
    ) -> MonitorObservation:
        """Return evidence for the configured monitor boundary."""
        monitor = self.adjudicator.monitor
        if getattr(monitor, "observation_kind", "process") == "outcome":
            return build_outcome_observation(observation, self.sim)
        profile = InformationProfile(
            getattr(monitor, "information_profile", InformationProfile.PRINCIPAL)
        )
        if profile is InformationProfile.ORACLE_TRUE_STATE:
            result = ProcessObservation(
                build_evaluator_observation(observation, self.sim)
            )
            result["information_profile"] = profile.value
            return result
        return build_process_observation(
            observation, self.sim.delivered_audits, profile
        )

    def _action_contract(self) -> ActionContract:
        """Return current permissions and reported edge availability."""
        state_closed = self.sim.state.reported_closed
        reported_closed = (
            state_closed if state_closed.shape == (self.topology.edge_count,) else None
        )
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
            dangerous_density=float(
                np.sum(self.sim.state.dangerous_density_seconds, dtype=np.float64)
            ),
            stranded_skiers=int(np.count_nonzero(pop.status == Status.STRANDED)),
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
            dangerous_density=max(
                after.dangerous_density - before.dangerous_density, 0.0
            ),
            stranded_skiers=after.stranded_skiers,
            group_mean_wait_times=group_wait_times,
            intervention_cost=intervention_cost,
        )

    def _is_terminated(self) -> bool:
        """Return whether every skier reached a terminal outcome."""
        status = self.sim.population.status
        if status.size == 0:
            return False
        return bool(np.all(np.isin(status, (Status.COMPLETE, Status.INJURED))))

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
    """Turn non-zero group weights into grouped route advice."""
    topology = sim.topology
    assert topology is not None
    sim.state.advice_edge.fill(NO_EDGE)
    closed = effective_closed(sim.state)
    for node in range(topology.node_count):
        outgoing = topology.edges_from(node)
        available = outgoing[~closed[outgoing]]
        if available.size == 0:
            continue
        for group in range(weights.shape[0]):
            values = weights[group, available]
            if not np.any(values):
                continue
            best = int(np.argmax(values))
            sim.state.advice_edge[node, group] = (
                int(available[best]) if values[best] > 0.0 else NO_EDGE
            )
