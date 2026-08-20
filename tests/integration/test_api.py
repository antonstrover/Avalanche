import base64
from pathlib import Path

import msgpack
from fastapi.testclient import TestClient

from avalanche.api.app import app

client = TestClient(app)
CONTRACT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "live-frame-v1.msgpack.b64"
)


def test_health_reports_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_options_serves_the_resolved_config_schema():
    response = client.get("/api/config-options")
    assert response.status_code == 200
    body = response.json()
    assert "schema" in body
    assert body["schema"]["title"] == "ResolvedConfig"
    assert "seed" in body["schema"]["properties"]


def test_openapi_document_is_generated():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/config-options" in response.json()["paths"]


def test_live_session_streams_a_complete_population():
    response = client.post("/api/sessions", json={"seed": 7, "skier_count": 5000})
    assert response.status_code == 201
    session = response.json()
    session_id = session["session_id"]
    assert session["skier_count"] == 5000
    assert session["frame_interval_ms"] == 250

    with client.websocket_connect(f"/api/sessions/{session_id}/stream") as websocket:
        first = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        assert first["version"] == 1
        assert first["type"] == "snapshot"
        assert first["session_id"] == session_id
        assert len(first["payload"]["location_kind"]) == 5000
        assert len(first["payload"]["location_index"]) == 5000 * 4
        assert len(first["payload"]["progress"]) == 5000 * 4

        websocket.send_bytes(
            msgpack.packb({"version": 1, "type": "snapshot_request"}, use_bin_type=True)
        )
        recovered = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        assert recovered["type"] == "snapshot"
        assert recovered["sequence"] >= first["sequence"]

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_live_session_rejects_an_invalid_population_size():
    response = client.post("/api/sessions", json={"skier_count": 10001})
    assert response.status_code == 422


def test_unknown_live_session_is_not_found():
    assert client.get("/api/sessions/missing").status_code == 404
    with client.websocket_connect("/api/sessions/missing/stream") as websocket:
        assert websocket.receive()["type"] == "websocket.close"


def test_python_decodes_the_stream_contract_fixture():
    packed = base64.b64decode(CONTRACT_FIXTURE.read_text().strip())
    frame = msgpack.unpackb(packed, raw=False)

    assert frame["version"] == 1
    assert frame["session_id"] == "fixture-session"
    assert frame["payload"]["skier_count"] == 2
    assert len(frame["payload"]["location_index"]) == 8
