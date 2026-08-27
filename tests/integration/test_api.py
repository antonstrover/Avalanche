import base64
import json
import queue
import threading
import time
from pathlib import Path

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient

from avalanche.api.app import app
from avalanche.api.sessions import (
    DEMO_FAILURE_TARGET,
    MOUNTAIN_PATH,
    STREAM_VERSION,
    TIMELINE_LIMIT,
    attack_state,
    display_state,
    manager,
    pack_frame,
    run_session,
    topology_version,
)
from avalanche.config import load_yaml
from avalanche.config.models import ControllerConfig
from avalanche.sim import (
    LocationKind,
    MountainSim,
    load_topology,
    population_from_starts,
)

client = TestClient(app)
CONTRACT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "live-frame-v3.msgpack.b64"
)
CURRENT_CONTRACT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "live-frame-v5.msgpack.b64"
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


def test_config_options_serves_each_validated_configuration_choice():
    response = client.get("/api/config-options")
    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["mountains"]] == [
        "medium-resort",
        "small-resort",
    ]
    assert {item["id"] for item in body["scenarios"]} == {
        "attack-profit-biased",
        "attack-reward-hacker",
        "attack-sleeper-saboteur",
        "default",
        "failure-examples",
        "family-busy-weekend",
        "family-calm",
        "family-lift-failure",
        "family-storm",
        "honest-baseline",
    }
    assert {item["controller"]["kind"] for item in body["controllers"]} == {
        "honest",
        "none",
        "profit_biased",
        "reward_hacker",
        "sleeper_saboteur",
    }
    assert [item["id"] for item in body["controllers"]] == sorted(
        item["id"] for item in body["controllers"]
    )
    none = next(item for item in body["controllers"] if item["id"] == "none")
    assert none["compatible_mountain_ids"] == ["medium-resort", "small-resort"]
    assert next(item for item in body["controllers"] if item["id"] == "honest")[
        "compatible_mountain_ids"
    ] == ["medium-resort"]
    assert next(
        item for item in body["controllers"] if item["id"] == "small-resort/honest"
    )["compatible_mountain_ids"] == ["small-resort"]
    assert {item["monitor"]["kind"] for item in body["monitors"]} == {
        "none",
        "outcome",
        "rules",
    }
    small = next(item for item in body["mountains"] if item["id"] == "small-resort")
    assert small["mountain"]["node_count"] == 10
    assert len(small["topology"]["edges"]) == 12
    for mountain in body["mountains"]:
        assert mountain["mountain"]["node_count"] == len(mountain["topology"]["nodes"])
        assert mountain["mountain"]["edge_count"] == len(mountain["topology"]["edges"])


def test_openapi_document_is_generated():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/config-options" in response.json()["paths"]
    assert "/api/config-options/resolve" in response.json()["paths"]
    assert "/api/demo-sessions" in response.json()["paths"]
    session_fields = response.json()["components"]["schemas"]["SessionCreate"][
        "properties"
    ]
    assert not {"demo_failure", "demo_monitor", "demo_approval"} & set(session_fields)


def test_live_configuration_resolution_combines_every_selected_part():
    response = client.post(
        "/api/config-options/resolve",
        json={
            "mountain": "small-resort",
            "scenario": "failure-examples",
            "controller": "small-resort/honest",
            "monitor": "none",
            "seed": 17,
            "population": {"skier_count": 20},
        },
    )

    assert response.status_code == 200
    resolved = response.json()
    assert resolved["mountain"]["name"] == "small-resort"
    assert resolved["scenario"]["name"] == "failure-examples"
    assert resolved["controller"]["kind"] == "honest"
    assert resolved["monitor"]["kind"] == "none"
    assert resolved["seed"] == 17
    assert resolved["population"]["skier_count"] == 20
    assert resolved["intervals"] == {
        "movement_tick_seconds": 5.0,
        "control_interval_seconds": 60.0,
    }
    assert "movement_tick_seconds" not in resolved["scenario"]
    assert "control_interval_seconds" not in resolved["scenario"]


def test_display_timing_does_not_change_configuration_identity():
    first = client.post(
        "/api/config-options/resolve",
        json={"frame_interval_ms": 16, "simulation_speed": 1.0},
    )
    second = client.post(
        "/api/config-options/resolve",
        json={"frame_interval_ms": 1000, "simulation_speed": 80.0},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert (
        first.json()["resolved_configuration_sha256"]
        == second.json()["resolved_configuration_sha256"]
    )
    assert (
        first.json()["scientific_configuration_sha256"]
        == second.json()["scientific_configuration_sha256"]
    )


def test_the_api_rejects_client_supplied_resolved_configuration():
    resolved = client.post(
        "/api/config-options/resolve",
        json={
            "mountain": "small-resort",
            "controller": "none",
            "population": {"skier_count": 20},
        },
    ).json()
    resolved["monitor"] = {
        "kind": "rules",
        "unsafe_decision": "REPLACE",
    }

    response = client.post("/api/sessions", json={"config": resolved})

    assert response.status_code == 422
    assert "Extra inputs are not permitted" in response.text


@pytest.mark.parametrize("field", ["demo_failure", "demo_monitor", "demo_approval"])
def test_a_formal_live_session_rejects_demo_fields(field):
    response = client.post("/api/sessions", json={field: True})

    assert response.status_code == 422
    assert "Extra inputs are not permitted" in response.text


@pytest.mark.parametrize(("field", "loaded"), [("node_count", 10), ("edge_count", 12)])
def test_the_api_rejects_a_forbidden_live_field(field, loaded):
    resolved = client.post(
        "/api/config-options/resolve",
        json={
            "mountain": "small-resort",
            "controller": "none",
            "population": {"skier_count": 20},
        },
    ).json()
    resolved["mountain"][field] = loaded + 1

    response = client.post("/api/sessions", json={"mountain_config": resolved})

    assert response.status_code == 422
    assert response.status_code == 422
    assert "Extra inputs are not permitted" in response.text


def test_live_configuration_resolution_rejects_an_unknown_choice():
    response = client.post("/api/config-options/resolve", json={"mountain": "missing"})
    assert response.status_code == 422
    assert response.json()["detail"] == "the mountain choice is unknown"


def test_live_configuration_resolution_rejects_an_incompatible_pair():
    response = client.post(
        "/api/config-options/resolve",
        json={"mountain": "small-resort", "controller": "honest"},
    )

    assert response.status_code == 422
    assert "unknown edge" in response.text


@pytest.mark.parametrize("mountain", ["medium-resort", "small-resort"])
def test_the_shared_controller_resolves_for_each_mountain(mountain: str):
    response = client.post(
        "/api/config-options/resolve",
        json={
            "mountain": mountain,
            "controller": "none",
            "population": {"skier_count": 20},
        },
    )

    assert response.status_code == 200
    resolved = response.json()
    assert resolved["mountain"]["path"].endswith(f"/{mountain}.yaml")
    assert resolved["controller"]["kind"] == "none"
    session = client.post(
        "/api/sessions",
        json={
            "mountain": mountain,
            "controller": "none",
            "population": {"skier_count": 20},
        },
    )
    assert session.status_code == 201
    deleted = client.delete(f"/api/sessions/{session.json()['session_id']}")
    assert deleted.status_code == 204


def test_live_session_runs_the_explicit_resolved_configuration():
    resolved = client.post(
        "/api/config-options/resolve",
        json={
            "mountain": "small-resort",
            "scenario": "default",
            "controller": "none",
            "monitor": "none",
            "seed": 17,
            "population": {"skier_count": 20},
        },
    ).json()
    response = client.post(
        "/api/sessions",
        json={
            "mountain": "small-resort",
            "scenario": "default",
            "controller": "none",
            "monitor": "none",
            "seed": 17,
            "population": {"skier_count": 20},
        },
    )
    assert response.status_code == 201
    session = response.json()
    session_id = session["session_id"]
    assert session["resolved_config"] == resolved
    assert session["topology_version"] == topology_version(
        Path(resolved["mountain"]["path"])
    )

    try:
        with client.websocket_connect(
            f"/api/sessions/{session_id}/stream"
        ) as websocket:
            frame = msgpack.unpackb(websocket.receive_bytes(), raw=False)
            proposal = frame["payload"]["display"]["decision"]["proposal"]
            assert proposal["controller_id"] == "none"
    finally:
        client.delete(f"/api/sessions/{session_id}")


def test_live_version_five_derives_legacy_progress():
    """Keep bounded display progress outside the formal skier state."""
    sim = MountainSim(
        Path(__file__).resolve().parents[2] / "configs/mountain/small-resort.yaml"
    )
    sim.reset(7)
    pop = population_from_starts([0], 1)
    pop.location_kind[0] = LocationKind.PISTE
    pop.location_index[0] = 0
    pop.required_travel_seconds[0] = 120.0
    pop.remaining_travel_seconds[0] = 30.0
    sim.population = pop

    frame = msgpack.unpackb(pack_frame(sim, "test", 1, "topology"), raw=False)
    progress = np.frombuffer(frame["payload"]["progress"], dtype="<f4")

    np.testing.assert_array_equal(progress, [0.75])


def test_live_session_streams_a_complete_population():
    response = client.post(
        "/api/sessions", json={"seed": 7, "population": {"skier_count": 5000}}
    )
    assert response.status_code == 201
    session = response.json()
    session_id = session["session_id"]
    assert session["skier_count"] == 5000
    assert session["frame_interval_ms"] == 250

    with client.websocket_connect(f"/api/sessions/{session_id}/stream") as websocket:
        first = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        assert first["version"] == 5
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
            "attack",
        }
        assert (
            first["payload"]["display"]["decision"]["proposal"]["controller_id"]
            == "honest"
        )

        websocket.send_bytes(
            msgpack.packb(
                {"version": STREAM_VERSION, "type": "snapshot_request"},
                use_bin_type=True,
            )
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
    response = client.post("/api/sessions", json={"population": {"skier_count": 10001}})
    assert response.status_code == 422


def test_unknown_live_session_is_not_found():
    assert client.get("/api/sessions/missing").status_code == 404
    with client.websocket_connect("/api/sessions/missing/stream") as websocket:
        assert websocket.receive()["type"] == "websocket.close"


def test_live_commands_change_only_the_addressed_session():
    first = client.post(
        "/api/sessions", json={"seed": 7, "population": {"skier_count": 20}}
    ).json()
    second = client.post(
        "/api/sessions", json={"seed": 8, "population": {"skier_count": 20}}
    ).json()
    first_id = first["session_id"]
    second_id = second["session_id"]

    try:
        first_session = manager.get(first_id)
        second_session = manager.get(second_id)
        assert first_session is not None
        assert second_session is not None
        for _ in range(100):
            if first_session.status == "running" and second_session.status == "running":
                break
            time.sleep(0.01)
        paused = client.post(
            f"/api/sessions/{first_id}/commands", json={"command": "pause"}
        )
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"
        with first_session.lock:
            paused_sequence = first_session.latest_sequence
        with second_session.lock:
            second_sequence = second_session.latest_sequence
        time.sleep(0.35)
        with first_session.lock:
            assert first_session.latest_sequence == paused_sequence
        with second_session.lock:
            assert second_session.latest_sequence > second_sequence

        with first_session.lock:
            before = msgpack.unpackb(first_session.latest, raw=False)
        stepped = client.post(
            f"/api/sessions/{first_id}/commands", json={"command": "step"}
        )
        assert stepped.status_code == 200
        with first_session.lock:
            packed = first_session.latest
        assert packed is not None
        frame = msgpack.unpackb(packed, raw=False)
        # A step advances the paused session by exactly one control interval.
        assert frame["simulation_time"] == before["simulation_time"] + 60.0

        speed = client.post(
            f"/api/sessions/{first_id}/commands",
            json={"command": "set_speed", "speed": 40.0},
        )
        assert speed.status_code == 200
        assert speed.json()["simulation_speed"] == 40.0
        resumed = client.post(
            f"/api/sessions/{first_id}/commands", json={"command": "resume"}
        )
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "running"
    finally:
        client.delete(f"/api/sessions/{first_id}")
        client.delete(f"/api/sessions/{second_id}")


def test_live_commands_reject_invalid_values_and_states():
    response = client.post("/api/sessions", json={"population": {"skier_count": 20}})
    session_id = response.json()["session_id"]
    try:
        assert (
            client.post(
                f"/api/sessions/{session_id}/commands",
                json={"command": "set_speed", "speed": 0.0},
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/sessions/{session_id}/commands",
                json={"command": "step"},
            ).status_code
            == 409
        )
    finally:
        client.delete(f"/api/sessions/{session_id}")


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
        "/api/demo-sessions",
        json={
            "seed": 7,
            "population": {"skier_count": 20},
            "demo_failure": True,
        },
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
        "/api/demo-sessions",
        json={
            "seed": 7,
            "population": {"skier_count": 20},
            "demo_monitor": True,
        },
    )
    session_id = response.json()["session_id"]

    with client.websocket_connect(f"/api/sessions/{session_id}/stream") as websocket:
        frame = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        decision = frame["payload"]["display"]["decision"]["monitor_decision"]
        assert decision["decision"] == "BLOCK"
        assert "EVACUATION_ROUTE_CLOSURE" in decision["reason_codes"]

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204


def test_an_approval_demo_pauses_and_accepts_the_proposal():
    response = client.post(
        "/api/demo-sessions",
        json={
            "seed": 7,
            "population": {"skier_count": 20},
            "demo_approval": True,
        },
    )
    session_id = response.json()["session_id"]

    with client.websocket_connect(f"/api/sessions/{session_id}/stream") as websocket:
        pending = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        approval = pending["payload"]["display"]["decision"]["approval"]
        assert approval["status"] == "pending"
        decision_id = approval["decision_id"]

        accepted = client.post(
            f"/api/sessions/{session_id}/approvals/{decision_id}",
            json={"choice": "APPROVE"},
        )
        assert accepted.status_code == 200

        resolved = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        decision = resolved["payload"]["display"]["decision"]
        assert decision["approval"]["status"] == "resolved"
        assert decision["executed_action"]["controller_id"] == "rule-demo"

        duplicate = client.post(
            f"/api/sessions/{session_id}/approvals/{decision_id}",
            json={"choice": "BLOCK"},
        )
        assert duplicate.status_code == 409

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204


def test_an_invalid_manual_replacement_is_rejected():
    response = client.post(
        "/api/demo-sessions",
        json={
            "seed": 7,
            "population": {"skier_count": 20},
            "demo_approval": True,
        },
    )
    session_id = response.json()["session_id"]
    with client.websocket_connect(f"/api/sessions/{session_id}/stream") as websocket:
        pending = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        decision_id = pending["payload"]["display"]["decision"]["approval"][
            "decision_id"
        ]
        invalid = client.post(
            f"/api/sessions/{session_id}/approvals/{decision_id}",
            json={"choice": "REPLACE", "replacement_action": {}},
        )
        assert invalid.status_code == 422
        client.post(
            f"/api/sessions/{session_id}/approvals/{decision_id}",
            json={"choice": "BLOCK"},
        )
    assert client.delete(f"/api/sessions/{session_id}").status_code == 204


def test_an_approval_timeout_uses_the_safe_fallback():
    output = queue.Queue()
    approval_input = queue.Queue()
    stop = threading.Event()
    stop.set()
    run_session(
        "timeout-test",
        7,
        20,
        topology_version(),
        output,
        stop,
        False,
        False,
        True,
        approval_input,
        0.01,
    )
    frames = []
    while not output.empty():
        frames.append(msgpack.unpackb(output.get(), raw=False))

    pending = frames[0]["payload"]["display"]["decision"]["approval"]
    resolved = frames[-1]["payload"]["display"]["decision"]
    assert pending["status"] == "pending"
    assert resolved["approval"]["choice"] == "BLOCK"
    assert resolved["executed_action"]["controller_id"] == "honest-fallback"


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


def test_python_decodes_the_current_stream_contract_fixture():
    packed = base64.b64decode(CURRENT_CONTRACT_FIXTURE.read_text().strip())
    frame = msgpack.unpackb(packed, raw=False)
    attack = frame["payload"]["display"]["attack"]

    assert frame["version"] == STREAM_VERSION == 5
    assert frame["session_id"] == "fixture-session"
    assert attack["kind"] == "reward_hacker"
    assert attack["active"] is True
    assert attack["targets"]
    assert attack["divergent_edges"]
    reported = frame["payload"]["display"]["telemetry"]["reported_density"]
    true = frame["payload"]["display"]["telemetry"]["true_density"]
    for edge in attack["divergent_edges"]:
        assert reported[edge] != true[edge]


def reward_hacker_config() -> ControllerConfig:
    """Return the configured reward-hacker controller."""
    path = MOUNTAIN_PATH.parents[1] / "controllers" / "reward-hacker.yaml"
    return ControllerConfig.model_validate(load_yaml(path)["controller"])


def test_the_attack_state_reports_no_attack_without_a_controller():
    sim = MountainSim(MOUNTAIN_PATH)
    sim.reset(3)

    state = attack_state(sim, None)

    assert state == {
        "kind": "none",
        "active": False,
        "targets": [],
        "divergent_edges": [],
    }


def test_the_attack_state_reports_the_configured_targets():
    sim = MountainSim(MOUNTAIN_PATH)
    sim.reset(3)

    state = attack_state(sim, reward_hacker_config())

    assert state["kind"] == "reward_hacker"
    assert state["active"] is True
    assert len(state["targets"]) == 2
    assert state["divergent_edges"] == []


def test_the_attack_state_finds_each_divergent_edge():
    sim = MountainSim(MOUNTAIN_PATH)
    sim.reset(3)
    sim.state.density_ratio[4] = 0.9
    sim.state.reported_density_ratio[4] = 0.4

    state = attack_state(sim, reward_hacker_config())

    assert state["divergent_edges"] == [4]


def test_an_honest_controller_reports_no_divergent_edge():
    sim = MountainSim(MOUNTAIN_PATH)
    sim.reset(3)
    sim.state.density_ratio[4] = 0.9
    sim.state.reported_density_ratio[4] = 0.4
    honest = ControllerConfig(kind="honest")

    state = attack_state(sim, honest)

    assert state["kind"] == "honest"
    assert state["active"] is False
    assert state["divergent_edges"] == []
