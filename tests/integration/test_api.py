import base64
import json
from pathlib import Path

import msgpack
from fastapi.testclient import TestClient

from avalanche.api.app import app
from avalanche.api.sessions import (
    DEMO_FAILURE_TARGET,
    MOUNTAIN_PATH,
    TIMELINE_LIMIT,
    display_state,
)
from avalanche.sim import MountainSim, load_topology

client = TestClient(app)
CONTRACT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "live-frame-v3.msgpack.b64"
)
SCENE_RESORT = (
    Path(__file__).resolve().parents[2]
    / "dashboard"
    / "src"
    / "mountain"
    / "resort.json"
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
        assert first["version"] == 4
        assert first["type"] == "snapshot"
        assert first["session_id"] == session_id
        assert len(first["payload"]["location_kind"]) == 5000
        assert len(first["payload"]["location_index"]) == 5000 * 4
        assert len(first["payload"]["progress"]) == 5000 * 4
        assert set(first["payload"]["display"]) == {
            "weather",
            "failures",
            "hazards",
            "closures",
            "timeline",
            "decision",
            "telemetry",
        }
        assert (
            first["payload"]["display"]["decision"]["proposal"]["controller_id"]
            == "honest"
        )

        websocket.send_bytes(
            msgpack.packb({"version": 4, "type": "snapshot_request"}, use_bin_type=True)
        )
        recovered = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        assert recovered["type"] == "snapshot"
        assert recovered["sequence"] >= first["sequence"]

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_the_scene_indices_match_the_live_topology():
    stored = json.loads(SCENE_RESORT.read_text())
    topology = load_topology(MOUNTAIN_PATH)

    assert [node["node_id"] for node in stored["nodes"]] == list(topology.node_ids)
    assert [(edge["source"], edge["destination"]) for edge in stored["edges"]] == [
        (
            topology.node_ids[topology.edge_source[index]],
            topology.node_ids[topology.edge_destination[index]],
        )
        for index in range(topology.edge_count)
    ]


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

    assert frame["version"] == 3
    assert frame["session_id"] == "fixture-session"
    assert frame["payload"]["skier_count"] == 2
    assert len(frame["payload"]["location_index"]) == 8
    assert frame["payload"]["display"]["weather"]["wind"] == 12.0
    assert (
        frame["payload"]["display"]["decision"]["proposal"]["controller_id"] == "honest"
    )


def test_failure_demo_streams_one_stable_timeline_marker():
    response = client.post(
        "/api/sessions",
        json={"seed": 7, "skier_count": 20, "demo_failure": True},
    )
    session = response.json()
    session_id = session["session_id"]

    with client.websocket_connect(f"/api/sessions/{session_id}/stream") as websocket:
        marker = None
        for _ in range(12):
            frame = msgpack.unpackb(websocket.receive_bytes(), raw=False)
            timeline = frame.get("payload", {}).get("display", {}).get("timeline", [])
            marker = next(
                (
                    event
                    for event in timeline
                    if event["event_type"] == "failure_started"
                ),
                None,
            )
            if marker is not None:
                break
        assert marker is not None
        assert marker["event_id"].endswith(":start")
        assert marker["target"] == DEMO_FAILURE_TARGET

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204


def test_monitor_demo_streams_one_blocked_rule():
    response = client.post(
        "/api/sessions",
        json={"seed": 7, "skier_count": 20, "demo_monitor": True},
    )
    session_id = response.json()["session_id"]

    with client.websocket_connect(f"/api/sessions/{session_id}/stream") as websocket:
        frame = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        decision = frame["payload"]["display"]["decision"]["monitor_decision"]
        assert decision["decision"] == "BLOCK"
        assert "EVACUATION_ROUTE_CLOSURE" in decision["reason_codes"]

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204


def test_the_timeline_window_is_bounded_and_has_unique_identities():
    failures = {
        "schedule": [
            {
                "kind": "sudden_closure",
                "target": 0,
                "start_time_seconds": float(index),
                "duration_seconds": 0.5,
                "controller_visible": True,
            }
            for index in range(40)
        ]
    }
    sim = MountainSim(MOUNTAIN_PATH)
    sim.reset(1, {"failures": failures})
    sim.simulation_time = 100.0

    timeline = display_state(sim)["timeline"]
    identities = {event["event_id"] for event in timeline}
    assert len(timeline) == TIMELINE_LIMIT
    assert len(identities) == TIMELINE_LIMIT
