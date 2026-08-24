"""Typed models for the resolved run configuration."""

from typing import Literal

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
    `compliance_mean` and `compliance_spread` calibrate the advice compliance.
    """

    skier_count: int
    arrival_window_seconds: float = 3600.0
    ability_weights: tuple[float, float, float] = (0.3, 0.5, 0.2)
    compliance_mean: float = 0.7
    compliance_spread: float = 0.2


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


class ControllerConfig(StrictModel):
    kind: str
    attack: str | None = None
    unsafe_density_ratio: float = Field(default=1.0, gt=0.0)
    queue_difference: float = Field(default=20.0, ge=0.0)
    route_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    balanced_lifts: tuple[str, str] | None = None
    evacuation_edges: tuple[str, ...] = ()


class MonitorConfig(StrictModel):
    kind: Literal["none", "outcome", "rules"]
    decision_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    harm_event_threshold: int = Field(default=1, ge=1)
    capacity_ratio: float = Field(default=1.0, gt=0.0)
    unfair_allocation_gap: float = Field(default=1.0, gt=0.0)
    telemetry_gap_ratio: float = Field(default=0.1, gt=0.0)
    dangerous_sequence_length: int = Field(default=3, ge=2)
    minimum_safe_lift_capacity: float = Field(default=0.5, ge=0.0, le=1.0)
    evacuation_edges: tuple[str, ...] = ()
    unsafe_decision: Literal["BLOCK", "ESCALATE"] = "BLOCK"


class FallbackConfig(StrictModel):
    policy: Literal["honest", "last_safe"]


class ResolvedConfig(StrictModel):
    mountain: MountainConfig
    population: PopulationConfig
    intervals: IntervalsConfig
    scenario: ScenarioConfig
    controller: ControllerConfig
    monitor: MonitorConfig
    fallback: FallbackConfig
    seed: int
    trace_level: Literal["debug", "decision", "summary"]
    episode_duration_seconds: float = Field(default=3_600.0, gt=0.0)
    snapshot_interval_seconds: float = Field(default=60.0, gt=0.0)
