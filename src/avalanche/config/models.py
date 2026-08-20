"""Typed models for the resolved run configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MountainConfig(StrictModel):
    name: str
    node_count: int
    edge_count: int


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


class ScenarioConfig(StrictModel):
    name: str
    movement_tick_seconds: float
    control_interval_seconds: float


class ControllerConfig(StrictModel):
    kind: str
    attack: str | None = None


class MonitorConfig(StrictModel):
    kind: str
    threshold: float


class FallbackConfig(StrictModel):
    policy: str


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
