"""Serve the application API and live simulator streams."""

import asyncio
from contextlib import asynccontextmanager
from math import isfinite
from typing import Any, Literal

import msgpack
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, model_validator

from avalanche.api.sessions import (
    MAX_SKIERS,
    STREAM_VERSION,
    manager,
    snapshot_message,
    validate_replacement_action,
)
from avalanche.config import ResolvedConfig, load_yaml, merge_configs
from avalanche.config.models import (
    ControllerConfig,
    FallbackConfig,
    IntervalsConfig,
    MonitorConfig,
    MountainConfig,
    PopulationConfig,
    ScenarioConfig,
)
from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import ApprovalChoice
from avalanche.sim import load_topology


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Stop each live worker during application shutdown."""
    yield
    manager.close()


app = FastAPI(title="avalanche", lifespan=lifespan)
CONFIG_ROOT = REPO_ROOT / "configs"


class SessionCreate(BaseModel):
    """Validate the inputs of a live session."""

    seed: int = 0
    skier_count: int = Field(default=5000, ge=1, le=MAX_SKIERS)
    demo_failure: bool = False
    demo_monitor: bool = False
    demo_approval: bool = False
    config: ResolvedConfig | None = None


class SessionResponse(BaseModel):
    """Describe one live session."""

    session_id: str
    status: str
    skier_count: int
    simulation_speed: float
    frame_interval_ms: int
    topology_version: str
    demo_failure: bool
    demo_monitor: bool
    demo_approval: bool
    resolved_config: ResolvedConfig


class ApprovalResponseRequest(BaseModel):
    """Validate one response to a pending escalation."""

    choice: ApprovalChoice
    replacement_action: dict[str, Any] | None = None


class SessionCommandRequest(BaseModel):
    """Validate one live session command."""

    command: Literal["pause", "resume", "step", "set_speed"]
    speed: float | None = None

    @model_validator(mode="after")
    def check_speed(self) -> "SessionCommandRequest":
        """Require a valid speed only for a speed command."""
        if self.command == "set_speed":
            if self.speed is None or not isfinite(self.speed) or self.speed <= 0.0:
                raise ValueError("the session speed must be finite and positive")
        elif self.speed is not None:
            raise ValueError("only a speed command can contain a speed")
        return self


class MountainOption(BaseModel):
    """Describe one available mountain configuration."""

    id: str
    label: str
    mountain: MountainConfig
    population: PopulationConfig
    topology: dict[str, Any]


class ScenarioOption(BaseModel):
    """Describe one available scenario configuration."""

    id: str
    label: str
    scenario: ScenarioConfig
    intervals: IntervalsConfig
    episode_duration_seconds: float
    snapshot_interval_seconds: float


class ControllerOption(BaseModel):
    """Describe one available controller configuration."""

    id: str
    label: str
    compatible_mountain_ids: tuple[str, ...]
    controller: ControllerConfig


class MonitorOption(BaseModel):
    """Describe one available monitor configuration."""

    id: str
    label: str
    monitor: MonitorConfig
    fallback: FallbackConfig
    trace_level: Literal["debug", "decision", "summary"]


class ConfigOptionsResponse(BaseModel):
    """List each validated live configuration choice."""

    mountains: list[MountainOption]
    scenarios: list[ScenarioOption]
    controllers: list[ControllerOption]
    monitors: list[MonitorOption]


class LiveConfigSelection(BaseModel):
    """Select each component of one resolved live configuration."""

    mountain: str = "medium-resort"
    scenario: str = "default"
    controller: str = "honest"
    monitor: str = "none"
    seed: int = 0
    skier_count: int = Field(default=5000, ge=1, le=MAX_SKIERS)


def _label(value: str) -> str:
    """Change one stable identifier into a display label."""
    return value.replace("-", " ").replace("_", " ").title()


def _mountain_options() -> list[MountainOption]:
    """Load each mountain and its static scene topology."""
    defaults = load_yaml(CONFIG_ROOT / "mountain" / "default.yaml")
    population = PopulationConfig.model_validate(defaults["population"])
    choices = []
    for path in sorted((CONFIG_ROOT / "mountain").glob("*-resort.yaml")):
        data = load_yaml(path)
        topology = load_topology(path)
        nodes = sorted(data["nodes"], key=lambda node: node["node_id"])
        edges = sorted(
            data["edges"],
            key=lambda edge: (edge["source"], edge["destination"]),
        )
        identifier = path.stem
        name = topology.name
        choices.append(
            MountainOption(
                id=identifier,
                label=_label(name),
                mountain=MountainConfig(
                    name=name,
                    node_count=topology.node_count,
                    edge_count=topology.edge_count,
                    path=str(path.relative_to(REPO_ROOT)),
                ),
                population=population,
                topology={"name": name, "nodes": nodes, "edges": edges},
            )
        )
    return choices


def _scenario_options() -> list[ScenarioOption]:
    """Load each validated scenario configuration."""
    choices = []
    defaults = load_yaml(CONFIG_ROOT / "scenarios" / "default.yaml")
    for path in sorted((CONFIG_ROOT / "scenarios").glob("*.yaml")):
        values = merge_configs(defaults, load_yaml(path))
        scenario = ScenarioConfig.model_validate(values["scenario"])
        choices.append(
            ScenarioOption(
                id=path.stem,
                label=_label(scenario.name),
                scenario=scenario,
                intervals=IntervalsConfig.model_validate(values["intervals"]),
                episode_duration_seconds=values.get(
                    "episode_duration_seconds", 28_800.0
                ),
                snapshot_interval_seconds=values.get("snapshot_interval_seconds", 60.0),
            )
        )
    return choices


def _controller_options() -> list[ControllerOption]:
    """Load each validated controller configuration."""
    mountain_ids = tuple(option.id for option in _mountain_options())
    paths = [
        (path, path.stem, "medium-resort")
        for path in (CONFIG_ROOT / "controllers").glob("*.yaml")
    ]
    paths.extend(
        (path, f"small-resort/{path.stem}", "small-resort")
        for path in (CONFIG_ROOT / "controllers" / "small-resort").glob("*.yaml")
    )
    choices = []
    for path, identifier, mountain_id in paths:
        controller = ControllerConfig.model_validate(load_yaml(path)["controller"])
        choices.append(
            ControllerOption(
                id=identifier,
                label=_label(path.stem),
                compatible_mountain_ids=(
                    mountain_ids if controller.kind == "none" else (mountain_id,)
                ),
                controller=controller,
            )
        )
    return sorted(choices, key=lambda option: option.id)


def _monitor_options() -> list[MonitorOption]:
    """Load each validated monitor configuration."""
    choices = []
    for path in sorted((CONFIG_ROOT / "monitors").glob("*.yaml")):
        values = load_yaml(path)
        choices.append(
            MonitorOption(
                id=path.stem,
                label=_label(path.stem),
                monitor=MonitorConfig.model_validate(values["monitor"]),
                fallback=FallbackConfig.model_validate(values["fallback"]),
                trace_level=values["trace_level"],
            )
        )
    return choices


def _find_option(options: list[Any], identifier: str, kind: str) -> Any:
    """Return one named option or reject an unknown identifier."""
    for option in options:
        if option.id == identifier:
            return option
    raise HTTPException(status_code=422, detail=f"the {kind} choice is unknown")


def resolve_live_config(selection: LiveConfigSelection) -> ResolvedConfig:
    """Resolve one validated live configuration selection."""
    mountain = _find_option(_mountain_options(), selection.mountain, "mountain")
    scenario = _find_option(_scenario_options(), selection.scenario, "scenario")
    controller = _find_option(_controller_options(), selection.controller, "controller")
    monitor = _find_option(_monitor_options(), selection.monitor, "monitor")
    if mountain.id not in controller.compatible_mountain_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "the controller is incompatible with the mountain",
                "mountain": selection.mountain,
                "controller": selection.controller,
            },
        )
    population = mountain.population.model_copy(
        update={"skier_count": selection.skier_count}
    )
    return ResolvedConfig(
        mountain=mountain.mountain,
        population=population,
        intervals=scenario.intervals,
        scenario=scenario.scenario,
        controller=controller.controller,
        monitor=monitor.monitor,
        fallback=monitor.fallback,
        seed=selection.seed,
        trace_level=monitor.trace_level,
        episode_duration_seconds=scenario.episode_duration_seconds,
        snapshot_interval_seconds=scenario.snapshot_interval_seconds,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config-options", response_model=ConfigOptionsResponse)
def config_options() -> ConfigOptionsResponse:
    """Return each validated live configuration choice."""
    return ConfigOptionsResponse(
        mountains=_mountain_options(),
        scenarios=_scenario_options(),
        controllers=_controller_options(),
        monitors=_monitor_options(),
    )


@app.post("/api/config-options/resolve", response_model=ResolvedConfig)
def resolve_config(selection: LiveConfigSelection) -> ResolvedConfig:
    """Return the exact configuration for one live selection."""
    return resolve_live_config(selection)


@app.post("/api/sessions", status_code=201, response_model=SessionResponse)
def create_session(request: SessionCreate) -> dict[str, object]:
    """Start an isolated live simulator session."""
    resolved = request.config or resolve_live_config(
        LiveConfigSelection(seed=request.seed, skier_count=request.skier_count)
    )
    return manager.create(
        resolved.seed,
        resolved.population.skier_count,
        request.demo_failure,
        request.demo_monitor,
        request.demo_approval,
        resolved,
    ).response()


@app.post("/api/sessions/{session_id}/approvals/{decision_id}")
def resolve_approval(
    session_id: str,
    decision_id: str,
    request: ApprovalResponseRequest,
) -> dict[str, str]:
    """Resolve one pending live escalation."""
    if request.choice is ApprovalChoice.REPLACE:
        if request.replacement_action is None:
            raise HTTPException(
                status_code=422, detail="a replacement action is required"
            )
        try:
            validate_replacement_action(request.replacement_action)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    elif request.replacement_action is not None:
        raise HTTPException(
            status_code=422,
            detail="only a replace response can contain an action",
        )
    result = manager.respond(
        session_id,
        decision_id,
        request.choice,
        request.replacement_action,
    )
    if result == "missing_session":
        raise HTTPException(status_code=404, detail="the session does not exist")
    if result == "missing_decision":
        raise HTTPException(status_code=404, detail="the approval does not exist")
    if result == "resolved":
        raise HTTPException(status_code=409, detail="the approval is resolved")
    return {"status": "accepted"}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, object]:
    """Return the current session state."""
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="the session does not exist")
    return session.response()


@app.post(
    "/api/sessions/{session_id}/commands",
    response_model=SessionResponse,
)
def command_session(
    session_id: str, request: SessionCommandRequest
) -> dict[str, object]:
    """Apply one command to an isolated live session."""
    result, session = manager.command(session_id, request.command, request.speed)
    if result == "missing_session":
        raise HTTPException(status_code=404, detail="the session does not exist")
    if result == "invalid_state":
        raise HTTPException(status_code=409, detail="the command is invalid now")
    if result == "timeout":
        raise HTTPException(status_code=504, detail="the command did not finish")
    assert session is not None
    return session.response()


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    """Stop and remove one live session."""
    if not manager.delete(session_id):
        raise HTTPException(status_code=404, detail="the session does not exist")


@app.websocket("/api/sessions/{session_id}/stream")
async def stream_session(websocket: WebSocket, session_id: str) -> None:
    """Send complete MessagePack skier frames."""
    await websocket.accept()
    session = manager.get(session_id)
    if session is None:
        await websocket.close(code=4404)
        return

    sent_sequence = -1
    sent_initial = False
    try:
        while True:
            with session.lock:
                packed = session.latest
                sequence = session.latest_sequence
            if packed is not None and (not sent_initial or sequence != sent_sequence):
                if not sent_initial:
                    packed = snapshot_message(packed)
                    sent_initial = True
                await websocket.send_bytes(packed)
                sent_sequence = sequence

            try:
                request = await asyncio.wait_for(
                    websocket.receive_bytes(), timeout=0.03
                )
            except TimeoutError:
                continue
            envelope = msgpack.unpackb(request, raw=False)
            valid_request = (
                envelope.get("version") == STREAM_VERSION
                and envelope.get("type") == "snapshot_request"
            )
            if not valid_request:
                await websocket.close(code=4400)
                return
            with session.lock:
                packed = session.latest
            if packed is not None:
                await websocket.send_bytes(snapshot_message(packed))
    except (WebSocketDisconnect, RuntimeError):
        return
    except (msgpack.UnpackException, ValueError, TypeError):
        await websocket.close(code=4400)
