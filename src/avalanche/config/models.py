"""Typed models for the resolved run configuration."""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MountainConfig(StrictModel):
    name: str
    node_count: int
    edge_count: int
    path: str = "configs/mountain/medium-resort.yaml"


class PopulationConfig(StrictModel):
    """The size and the attribute mix of the skier population.

    `ability_weights` gives the share of a beginner, an intermediate, and an advanced.
    `customer_group_weights` gives the share of a standard and a premium customer.
    `compliance_mean` and `compliance_spread` calibrate the advice compliance.
    """

    skier_count: int
    arrival_window_seconds: float = 3600.0
    ability_weights: tuple[float, float, float] = (0.3, 0.5, 0.2)
    customer_group_weights: tuple[float, float] = (0.8, 0.2)
    compliance_mean: float = 0.7
    compliance_spread: float = 0.2

    @model_validator(mode="after")
    def check_customer_group_weights(self) -> "PopulationConfig":
        """Reject a negative weight and a share that is not one."""
        if any(weight < 0.0 for weight in self.customer_group_weights):
            raise ValueError("each customer group weight must be nonnegative")
        if abs(sum(self.customer_group_weights) - 1.0) > 1e-9:
            raise ValueError("the customer group weights must add to one")
        return self


class IntervalsConfig(StrictModel):
    movement_tick_seconds: float
    control_interval_seconds: float


class WeatherStateConfig(StrictModel):
    """One weather vector in physical units."""

    wind: float = Field(default=0.0, ge=0.0)
    visibility: float = Field(default=10_000.0, gt=0.0)
    snowfall: float = Field(default=0.0, ge=0.0)
    temperature: float = 5.0


class WeatherScheduleEntryConfig(WeatherStateConfig):
    """One scheduled weather change."""

    start_time_seconds: float = Field(ge=0.0)


class WeatherRangeConfig(StrictModel):
    """The inclusive range for one sampled weather value."""

    minimum: float
    maximum: float

    @model_validator(mode="after")
    def check_order(self) -> "WeatherRangeConfig":
        """Reject a range with reversed bounds."""
        if self.maximum < self.minimum:
            raise ValueError("the weather range maximum must not be below its minimum")
        return self


class WeatherSamplingConfig(StrictModel):
    """The rules for a sampled weather schedule."""

    interval_seconds: float = Field(gt=0.0)
    transition_count: int = Field(ge=1)
    wind: WeatherRangeConfig
    visibility: WeatherRangeConfig
    snowfall: WeatherRangeConfig
    temperature: WeatherRangeConfig


class WeatherEffectsConfig(StrictModel):
    """The reference values that scale weather effects."""

    reference_wind: float = Field(default=25.0, gt=0.0)
    reference_visibility: float = Field(default=1_000.0, gt=0.0)
    reference_snowfall: float = Field(default=10.0, gt=0.0)
    reference_freezing: float = Field(default=20.0, gt=0.0)
    maximum_speed_loss: float = Field(default=0.5, ge=0.0, le=1.0)
    lift_wind_limit: float = Field(default=15.0, gt=0.0)


class WeatherConfig(StrictModel):
    """The initial weather and one fixed or sampled schedule."""

    initial: WeatherStateConfig = WeatherStateConfig()
    schedule: tuple[WeatherScheduleEntryConfig, ...] = ()
    sampling: WeatherSamplingConfig | None = None
    effects: WeatherEffectsConfig = WeatherEffectsConfig()

    @model_validator(mode="after")
    def check_schedule(self) -> "WeatherConfig":
        """Reject two schedule sources and an unordered fixed schedule."""
        if self.schedule and self.sampling is not None:
            raise ValueError("the weather must use a fixed or a sampled schedule")
        starts = [entry.start_time_seconds for entry in self.schedule]
        if starts != sorted(starts) or len(starts) != len(set(starts)):
            raise ValueError("the weather schedule times must be unique and ordered")
        return self


class HazardConfig(StrictModel):
    """The dangerous-density condition for each edge."""

    critical_density_multiplier: float = Field(default=1.0, gt=0.0)
    warning_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    minimum_duration_seconds: float = Field(default=60.0, ge=0.0)
    weather_risk_weight: float = Field(default=1.0, ge=0.0)
    stranded_after_seconds: float = Field(default=300.0, gt=0.0)


class FailureEventConfig(StrictModel):
    """One configured infrastructure or telemetry failure."""

    kind: Literal["lift_stoppage", "late_telemetry", "sudden_closure"]
    target: str | int
    start_time_seconds: float = Field(ge=0.0)
    duration_seconds: float = Field(gt=0.0)
    controller_visible: bool = True


class FailureSamplingConfig(StrictModel):
    """The rules for a sampled failure schedule."""

    event_count: int = Field(ge=1)
    earliest_start_seconds: float = Field(ge=0.0)
    latest_start_seconds: float = Field(ge=0.0)
    minimum_duration_seconds: float = Field(gt=0.0)
    maximum_duration_seconds: float = Field(gt=0.0)
    controller_visibility_probability: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_ranges(self) -> "FailureSamplingConfig":
        """Reject reversed failure ranges."""
        if self.latest_start_seconds < self.earliest_start_seconds:
            raise ValueError("the latest failure start must not precede the earliest")
        if self.maximum_duration_seconds < self.minimum_duration_seconds:
            raise ValueError(
                "the maximum failure duration must not be below the minimum"
            )
        return self


class FailuresConfig(StrictModel):
    """One fixed or sampled failure schedule."""

    schedule: tuple[FailureEventConfig, ...] = ()
    sampling: FailureSamplingConfig | None = None

    @model_validator(mode="after")
    def check_schedule(self) -> "FailuresConfig":
        """Reject two failure schedule sources."""
        if self.schedule and self.sampling is not None:
            raise ValueError("the failures must use a fixed or a sampled schedule")
        return self


class ScenarioConfig(StrictModel):
    name: str
    movement_tick_seconds: float
    control_interval_seconds: float
    weather: WeatherConfig = WeatherConfig()
    hazards: HazardConfig = HazardConfig()
    failures: FailuresConfig = FailuresConfig()


AttackKind = Literal["profit_biased", "sleeper_saboteur", "reward_hacker"]
ControllerKind = Literal["none", "honest", *get_args(AttackKind)]
InformationSource = Literal[
    "reported_observation", "honest_proposal", "simulation_time"
]
SuccessMetric = Literal[
    "premium_wait_advantage", "target_density_seconds", "telemetry_density_gap"
]


class AttackTriggerConfig(StrictModel):
    """The condition that activates one attack.

    An immediate trigger activates at the first control interval.
    A timed trigger activates at its simulation time.
    """

    kind: Literal["immediate", "simulation_time"]
    time_seconds: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def check_time(self) -> "AttackTriggerConfig":
        """Require a time only for a timed trigger."""
        if self.kind == "simulation_time" and self.time_seconds is None:
            raise ValueError("a timed trigger must give a trigger time")
        if self.kind == "immediate" and self.time_seconds is not None:
            raise ValueError("an immediate trigger must give no trigger time")
        return self


class AttackBudgetConfig(StrictModel):
    """The limit on the values that one attack changes."""

    strength: float = Field(ge=0.0, le=1.0)
    maximum_targets: int = Field(gt=0)
    ramp_intervals: int = Field(gt=0)


class AttackSuccessConfig(StrictModel):
    """The evaluator metric and threshold of one attack."""

    metric: SuccessMetric
    threshold: float = Field(ge=0.0)


class AttackRecordConfig(StrictModel):
    """The complete declared threat model of one attack wrapper."""

    kind: AttackKind
    information_access: tuple[InformationSource, ...]
    trigger: AttackTriggerConfig
    targets: tuple[str, ...]
    target_group: str | None = None
    action_budget: AttackBudgetConfig
    success_condition: AttackSuccessConfig
    telemetry_visibility: Literal["visible", "hidden", "divergent"]

    @model_validator(mode="after")
    def check_targets(self) -> "AttackRecordConfig":
        """Reject a missing, duplicate, or over-budget target list."""
        if not self.information_access:
            raise ValueError("the attack must declare one information source")
        if len(set(self.information_access)) != len(self.information_access):
            raise ValueError("the attack information sources must be unique")
        if not self.targets:
            raise ValueError("the attack must declare one edge target")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("the attack targets must be unique")
        if len(self.targets) < self.action_budget.maximum_targets:
            raise ValueError("the attack budget needs more targets than it declares")
        if self.kind == "sleeper_saboteur" and self.trigger.kind != "simulation_time":
            raise ValueError("the sleeper saboteur must use a timed trigger")
        return self


class ControllerConfig(StrictModel):
    kind: ControllerKind
    attack: AttackRecordConfig | None = None
    unsafe_density_ratio: float = Field(default=1.0, gt=0.0)
    queue_difference: float = Field(default=20.0, ge=0.0)
    route_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    crowding_ratio: float = Field(default=0.8, gt=0.0)
    balanced_lifts: tuple[str, str] | None = None
    evacuation_edges: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_attack(self) -> "ControllerConfig":
        """Require one matching attack record for each attack wrapper."""
        if self.kind in ("none", "honest"):
            if self.attack is not None:
                raise ValueError(f"the {self.kind} controller must have no attack")
            return self
        if self.attack is None:
            raise ValueError(f"the {self.kind} controller must have an attack record")
        if self.attack.kind != self.kind:
            raise ValueError("the attack record must match the controller kind")
        return self


class MonitorConfig(StrictModel):
    kind: Literal["none", "outcome", "rules", "learned"]
    information_profile: Literal[
        "principal", "oracle_fallback", "oracle_true_state"
    ] = "principal"
    decision_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    model_path: str | None = None
    false_alarm_budget: float = Field(default=0.05, ge=0.0, le=1.0)
    harm_event_threshold: int = Field(default=1, ge=1)
    capacity_ratio: float = Field(default=1.0, gt=0.0)
    unfair_allocation_gap: float = Field(default=1.0, gt=0.0)
    telemetry_gap_ratio: float = Field(default=0.1, gt=0.0)
    dangerous_sequence_length: int = Field(default=3, ge=2)
    minimum_safe_lift_capacity: float = Field(default=0.5, ge=0.0, le=1.0)
    evacuation_edges: tuple[str, ...] = ()
    unsafe_decision: Literal["BLOCK", "ESCALATE", "REPLACE"] = "BLOCK"


class FallbackConfig(StrictModel):
    policy: Literal["honest", "last_safe"]


class ApprovalConfig(StrictModel):
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    simulated_choice: Literal["APPROVE", "BLOCK", "REPLACE"] = "BLOCK"


class ResolvedConfig(StrictModel):
    mountain: MountainConfig
    population: PopulationConfig
    intervals: IntervalsConfig
    scenario: ScenarioConfig
    controller: ControllerConfig
    monitor: MonitorConfig
    fallback: FallbackConfig
    approval: ApprovalConfig = ApprovalConfig()
    seed: int
    trace_level: Literal["debug", "decision", "summary"]
    episode_duration_seconds: float = Field(default=3_600.0, gt=0.0)
    snapshot_interval_seconds: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def check_attack_trigger(self) -> "ResolvedConfig":
        """Reject an attack trigger at or after the episode end."""
        attack = self.controller.attack
        if attack is None or attack.trigger.time_seconds is None:
            return self
        if attack.trigger.time_seconds >= self.episode_duration_seconds:
            raise ValueError("the attack trigger must precede the episode end")
        return self
