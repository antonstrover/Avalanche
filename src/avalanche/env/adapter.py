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

from avalanche.control import (
    ActionProposal,
    AdjudicationResult,
    Adjudicator,
    ApprovalHandler,
    ConfiguredFallback,
    ExecutedAction,
    Monitor,
    build_monitor_observation,
    freeze_action,
    thaw_action,
)
from avalanche.env.actions import (
    PISTE_CLOSE,
    PISTE_OPEN,
    Action,
    ActionMasks,
    apply_action_masks,
    build_action_masks,
    build_action_space,
    validate_action,
)
from avalanche.env.observations import (
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
from avalanche.sim.population import ABILITY_NAMES
from avalanche.sim.routes import NO_EDGE
from avalanche.sim.skier import Status
from avalanche.sim.topology import Topology, load_topology

DEFAULT_REWARD_WEIGHTS = RewardWeights(1.0, -1.0, -1.0, -1.0, -1.0, -1.0)


@dataclass(frozen=True)
class AvalancheEnvConfig:
    """The fixed timing and observation settings of an environment."""

    movement_tick_seconds: float = 5.0
    control_interval_seconds: float = 60.0
    episode_duration_seconds: float = 3_600.0
    forecast_steps: int = 4
    incident_capacity: int = 16
    group_count: int = len(ABILITY_NAMES)

    def __post_init__(self) -> None:
        """Reject invalid environment settings."""
        times = (
            self.movement_tick_seconds,
            self.control_interval_seconds,
            self.episode_duration_seconds,
        )
        if any(not isfinite(value) or value <= 0.0 for value in times):
            raise ValueError("the environment times must be finite and positive")
        ratio = self.control_interval_seconds / self.movement_tick_seconds
        tick_count = round(ratio)
        if tick_count < 1 or not isclose(
            self.control_interval_seconds,
            tick_count * self.movement_tick_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("the control interval must contain whole movement ticks")
        if self.group_count != len(ABILITY_NAMES):
            raise ValueError("the environment group count must match the skier groups")
        ObservationConfig(
            episode_duration_seconds=self.episode_duration_seconds,
            forecast_steps=self.forecast_steps,
            incident_capacity=self.incident_capacity,
            group_count=self.group_count,
        )

    @property
    def movement_ticks_per_step(self) -> int:
        """Return the exact movement tick count of one environment step."""
        return round(self.control_interval_seconds / self.movement_tick_seconds)

    @property
    def observation(self) -> ObservationConfig:
        """Return the matching fixed observation configuration."""
        return ObservationConfig(
            episode_duration_seconds=self.episode_duration_seconds,
            forecast_steps=self.forecast_steps,
            incident_capacity=self.incident_capacity,
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


class _EnvironmentActionSpace(spaces.Dict):
    """A dictionary space that samples commands within the current masks."""

    def __init__(
        self,
        topology: Topology,
        group_count: int,
        masks: Callable[[], ActionMasks],
    ) -> None:
        base = build_action_space(topology, group_count)
        super().__init__(base.spaces)
        self._current_masks = masks

    def sample(self, mask=None, probability=None) -> Action:
        """Sample one action and neutralise each masked command."""
        action = super().sample(mask=mask, probability=probability)
        return apply_action_masks(action, self._current_masks())


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
            self.topology, self.config.group_count, self._action_masks
        )
        self.observation_space = build_observation_space(
            self.topology, self.config.observation
        )
        self.last_proposal: ActionProposal | None = None
        self.last_adjudication: AdjudicationResult | None = None
        self.last_executed_action: ExecutedAction | None = None
        self._control_history: list[dict[str, Any]] = []
        self._cumulative_intervention_cost = 0.0
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
        """Build a validator against the current environment masks."""
        return Adjudicator(
            monitor,
            lambda action: validate_action(
                thaw_action(action), self.action_space, self._action_masks()
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
        sim_options["episode_duration_seconds"] = self.config.episode_duration_seconds
        self.sim.reset(run_seed, sim_options)
        self._seed = run_seed
        self._ended = False
        self.last_proposal = None
        self.last_adjudication = None
        self.last_executed_action = None
        self._control_history.clear()
        self._cumulative_intervention_cost = 0.0
        self.action_space.seed(run_seed)
        self.observation_space.seed(run_seed)
        self.adjudicator.reset(run_seed)
        observation = self._observation()
        return observation, self._base_info(observation)

    def step(
        self, action: Action
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Validate one action and run one complete control interval."""
        masks = self._action_masks()
        proposal = create_action_proposal(
            action,
            self.action_space,
            masks,
            simulation_time=self.sim.simulation_time,
        )
        return self.step_proposal(proposal)

    def step_proposal(
        self, proposal: ActionProposal
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Validate one controller proposal and run one control interval."""
        if self._ended:
            raise RuntimeError("reset the environment before the next step")
        if proposal.simulation_time != self.sim.simulation_time:
            raise ValueError("the proposal time must match the simulation time")

        before = self._reward_snapshot()
        before_checksum = self.sim.state_checksum()
        adjudication = self.execute_proposal(proposal)
        executed = adjudication.executed_action
        intervention_cost = action_intervention_cost(executed)

        for _ in range(self.config.movement_ticks_per_step):
            self.sim.tick()

        after = self._reward_snapshot()
        reward_result = calculate_reward(
            self._reward_transition(before, after, intervention_cost),
            self.reward_weights,
        )
        self._cumulative_intervention_cost += intervention_cost
        terminated = self._is_terminated()
        truncated = self.sim.simulation_time >= self.config.episode_duration_seconds
        self._ended = terminated or truncated
        observation = self._observation()
        info = self._base_info(observation)
        info.update(
            {
                "reward_parts": reward_result.parts.as_dict(),
                "checksums": {
                    "before": before_checksum,
                    "after": self.sim.state_checksum(),
                },
                "action_proposal": proposal,
                "monitor_decision": adjudication.decision,
                "adjudication": adjudication,
                "executed_action": executed,
                "current_intervention_cost": intervention_cost,
            }
        )
        return observation, reward_result.scalar, terminated, truncated, info

    def controller_observation(self) -> Observation:
        """Return the isolated reported state for one controller."""
        observation = self._observation()
        observation["simulation_time"] = self.sim.simulation_time
        return observation

    def execute_proposal(self, proposal: ActionProposal) -> AdjudicationResult:
        """Adjudicate and apply one proposal without movement ticks."""
        observation = self._observation()
        monitor_observation = build_monitor_observation(observation, self.sim)
        result = self.adjudicator.adjudicate(
            monitor_observation,
            proposal,
            tuple(self._control_history),
            simulation_time=self.sim.simulation_time,
        )
        self.sim.metrics.update_decision(result.decision)
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
        return result

    def _action_masks(self) -> ActionMasks:
        """Return the current controllable infrastructure masks."""
        return build_action_masks(self.topology, self.config.group_count)

    def _observation(self) -> Observation:
        """Return one observation with the masks used by this environment."""
        return build_observation(
            self.sim,
            self.config.observation,
            action_masks=self._action_masks(),
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

    def _metrics(self) -> dict[str, float | int | tuple[float, ...]]:
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
            "action_masks": deepcopy(observation["action_masks"]),
            "resolved_schedules": {
                "weather": metadata["weather_schedule"],
                "failures": metadata["failure_schedule"],
            },
        }


def create_action_proposal(
    action: Action,
    action_space: spaces.Dict,
    masks: ActionMasks,
    *,
    simulation_time: float,
    controller_id: str = "gymnasium",
) -> ActionProposal:
    """Validate and freeze one controller action into a proposal."""
    validate_action(action, action_space, masks)
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
    sim.state.telemetry_override[telemetry_enabled] = action["telemetry_overrides"][
        telemetry_enabled
    ]
    sim.state.telemetry_override_enabled[telemetry_enabled] = True
    _apply_route_weights(sim, action["route_weights"])
    refresh_reported_telemetry(sim.state)


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
