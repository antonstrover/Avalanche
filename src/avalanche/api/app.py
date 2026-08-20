"""Serve the application API and live simulator streams."""

import asyncio
from contextlib import asynccontextmanager

import msgpack
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from avalanche.api.sessions import MAX_SKIERS, manager, snapshot_message
from avalanche.config import ResolvedConfig


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Stop each live worker during application shutdown."""
    yield
    manager.close()


app = FastAPI(title="avalanche", lifespan=lifespan)


class SessionCreate(BaseModel):
    """Validate the inputs of a Stage 3 live session."""

    seed: int = 0
    skier_count: int = Field(default=5000, ge=1, le=MAX_SKIERS)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config-options")
def config_options() -> dict[str, object]:
    return {"schema": ResolvedConfig.model_json_schema()}


@app.post("/api/sessions", status_code=201)
def create_session(request: SessionCreate) -> dict[str, object]:
    """Start an isolated live simulator session."""
    return manager.create(request.seed, request.skier_count).response()


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
                envelope.get("version") == 1
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
