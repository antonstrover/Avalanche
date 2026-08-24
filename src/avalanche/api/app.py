"""Serve the application API and live simulator streams."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import msgpack
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from avalanche.api.sessions import (
    MAX_SKIERS,
    STREAM_VERSION,
    manager,
    snapshot_message,
    validate_replacement_action,
)
from avalanche.config import ResolvedConfig
from avalanche.control import ApprovalChoice


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Stop each live worker during application shutdown."""
    yield
    manager.close()


app = FastAPI(title="avalanche", lifespan=lifespan)


class SessionCreate(BaseModel):
    """Validate the inputs of a live session."""

    seed: int = 0
    skier_count: int = Field(default=5000, ge=1, le=MAX_SKIERS)
    demo_failure: bool = False
    demo_monitor: bool = False
    demo_approval: bool = False


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


class ApprovalResponseRequest(BaseModel):
    """Validate one response to a pending escalation."""

    choice: ApprovalChoice
    replacement_action: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config-options")
def config_options() -> dict[str, object]:
    return {"schema": ResolvedConfig.model_json_schema()}


@app.post("/api/sessions", status_code=201, response_model=SessionResponse)
def create_session(request: SessionCreate) -> dict[str, object]:
    """Start an isolated live simulator session."""
    return manager.create(
        request.seed,
        request.skier_count,
        request.demo_failure,
        request.demo_monitor,
        request.demo_approval,
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
