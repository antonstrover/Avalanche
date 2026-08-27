from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from avalanche.config import load_yaml
from avalanche.sim import build_graph, validate_graph

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
MOUNTAIN_FIXTURES = sorted(FIXTURE.parent.glob("*-resort.yaml"))
NODE_FIELDS = ("node_id", "node_type", "x", "y", "elevation", "capacity")
EDGE_FIELDS = (
    "source",
    "destination",
    "edge_type",
    "difficulty",
    "length",
    "nominal_travel_time",
    "safe_capacity",
    "critical_density",
    "lift_throughput",
    "wind_sensitivity",
    "visibility_sensitivity",
    "snow_sensitivity",
)
POSITIVE_EDGE_FIELDS = (
    "length",
    "nominal_travel_time",
    "safe_capacity",
    "critical_density",
)
SENSITIVITY_FIELDS = (
    "wind_sensitivity",
    "visibility_sensitivity",
    "snow_sensitivity",
)
INVALID_NUMBERS = (
    pytest.param("invalid", id="string"),
    pytest.param(None, id="null"),
    pytest.param(True, id="Boolean"),
    pytest.param(float("inf"), id="infinite"),
    pytest.param(float("nan"), id="NaN"),
)


def build_broken(tmp_path: Path, change) -> Path:
    """Copy the fixture, apply one change, and return the new file path."""
    data = load_yaml(FIXTURE)
    change(data)
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_the_fixture_builds_and_validates():
    graph = build_graph(FIXTURE)
    assert graph.number_of_nodes() == 10
    assert graph.number_of_edges() == 12
    validate_graph(graph)


@pytest.mark.parametrize("path", MOUNTAIN_FIXTURES, ids=lambda path: path.stem)
def test_each_raw_record_survives_graph_construction(path):
    data = load_yaml(path)
    graph = build_graph(path)

    assert len(data["nodes"]) == graph.number_of_nodes()
    assert len(data["edges"]) == graph.number_of_edges()


def test_a_duplicate_node_is_rejected(tmp_path):
    def change(data):
        data["nodes"].append(deepcopy(data["nodes"][0]))

    path = build_broken(tmp_path, change)
    node_id = load_yaml(path)["nodes"][0]["node_id"]

    with pytest.raises(ValueError, match=rf"{path}.*node.*{node_id}"):
        build_graph(path)


def test_a_duplicate_directed_edge_is_rejected(tmp_path):
    def change(data):
        data["edges"].append(deepcopy(data["edges"][0]))

    path = build_broken(tmp_path, change)
    edge = load_yaml(path)["edges"][0]

    with pytest.raises(
        ValueError,
        match=rf"{path}.*edge.*{edge['source']}.*{edge['destination']}",
    ):
        build_graph(path)


def test_opposite_directed_edges_are_accepted(tmp_path):
    def change(data):
        edge = deepcopy(data["edges"][0])
        edge["source"], edge["destination"] = edge["destination"], edge["source"]
        data["edges"].append(edge)

    graph = build_graph(build_broken(tmp_path, change))

    assert graph.number_of_edges() == 13


@pytest.mark.parametrize("field", NODE_FIELDS)
def test_a_missing_node_field_is_rejected(field, tmp_path):
    def change(data):
        data["nodes"][0].pop(field)

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*{field}"):
        build_graph(path)


@pytest.mark.parametrize("field", EDGE_FIELDS)
def test_a_missing_edge_field_is_rejected(field, tmp_path):
    def change(data):
        data["edges"][0].pop(field)

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*{field}"):
        build_graph(path)


@pytest.mark.parametrize("value", INVALID_NUMBERS)
@pytest.mark.parametrize("field", ("x", "y", "elevation", "capacity"))
def test_an_invalid_node_number_is_rejected(field, value, tmp_path):
    def change(data):
        data["nodes"][0][field] = value

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*{field}"):
        build_graph(path)


@pytest.mark.parametrize("value", INVALID_NUMBERS)
@pytest.mark.parametrize("field", (*POSITIVE_EDGE_FIELDS, *SENSITIVITY_FIELDS))
def test_an_invalid_edge_number_is_rejected(field, value, tmp_path):
    def change(data):
        data["edges"][0][field] = value

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*{field}"):
        build_graph(path)


@pytest.mark.parametrize("value", INVALID_NUMBERS)
def test_an_invalid_lift_throughput_is_rejected(value, tmp_path):
    def change(data):
        lift = next(edge for edge in data["edges"] if edge["edge_type"] == "lift")
        lift["lift_throughput"] = value

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*lift_throughput"):
        build_graph(path)


@pytest.mark.parametrize("value", [0, -1])
def test_a_nonpositive_node_capacity_is_rejected(value, tmp_path):
    def change(data):
        data["nodes"][0]["capacity"] = value

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*capacity"):
        build_graph(path)


@pytest.mark.parametrize("field", POSITIVE_EDGE_FIELDS)
@pytest.mark.parametrize("value", [0, -1])
def test_a_nonpositive_edge_limit_is_rejected(field, value, tmp_path):
    def change(data):
        data["edges"][0][field] = value

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*{field}"):
        build_graph(path)


@pytest.mark.parametrize("value", [0, -1])
def test_a_nonpositive_lift_throughput_is_rejected(value, tmp_path):
    def change(data):
        lift = next(edge for edge in data["edges"] if edge["edge_type"] == "lift")
        lift["lift_throughput"] = value

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*lift_throughput"):
        build_graph(path)


@pytest.mark.parametrize("field", ["capacity", "safe_capacity", "lift_throughput"])
def test_a_fractional_capacity_is_rejected(field, tmp_path):
    def change(data):
        if field == "capacity":
            data["nodes"][0][field] = 1.5
            return
        edge = next(value for value in data["edges"] if value["edge_type"] == "lift")
        edge[field] = 1.5

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*{field}"):
        build_graph(path)


@pytest.mark.parametrize("field", SENSITIVITY_FIELDS)
def test_a_negative_sensitivity_is_rejected(field, tmp_path):
    def change(data):
        data["edges"][0][field] = -0.1

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*{field}"):
        build_graph(path)


def test_zero_sensitivities_are_accepted(tmp_path):
    def change(data):
        for field in SENSITIVITY_FIELDS:
            data["edges"][0][field] = 0.0

    graph = build_graph(build_broken(tmp_path, change))

    assert graph.number_of_edges() == 12


def test_negative_coordinates_are_accepted(tmp_path):
    def change(data):
        data["nodes"][0]["x"] = -10.0
        data["nodes"][0]["y"] = -20.0
        data["nodes"][0]["elevation"] = -30.0

    graph = build_graph(build_broken(tmp_path, change))

    assert graph.nodes["base_village"]["elevation"] == -30.0


def test_a_piste_with_a_throughput_is_rejected(tmp_path):
    def change(data):
        data["edges"][0]["lift_throughput"] = 100

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*lift_throughput"):
        build_graph(path)


def test_a_piste_with_an_explicit_null_throughput_is_accepted():
    graph = build_graph(FIXTURE)

    piste = next(
        data for *_, data in graph.edges(data=True) if data["edge_type"] == "piste"
    )
    assert piste["lift_throughput"] is None


def test_an_unknown_difficulty_is_rejected(tmp_path):
    def change(data):
        data["edges"][0]["difficulty"] = "extreme"

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*difficulty"):
        build_graph(path)


@pytest.mark.parametrize(
    ("kind", "field"),
    [("nodes", "controllable"), ("edges", "controllable")],
)
def test_a_nonboolean_control_flag_is_rejected(kind, field, tmp_path):
    def change(data):
        data[kind][0][field] = 1

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*{field}"):
        build_graph(path)


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_a_missing_record_list_is_rejected(kind, tmp_path):
    def change(data):
        data.pop(kind)

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*{kind}"):
        build_graph(path)


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_a_nonmapping_record_is_rejected(kind, tmp_path):
    def change(data):
        data[kind][0] = "invalid"

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*mapping"):
        build_graph(path)


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_a_nonlist_record_collection_is_rejected(kind, tmp_path):
    def change(data):
        data[kind] = {}

    path = build_broken(tmp_path, change)

    with pytest.raises(ValueError, match=rf"{path}.*list"):
        build_graph(path)


def test_the_lifts_go_up_and_the_pistes_go_down():
    graph = build_graph(FIXTURE)
    for source, destination, edge_type in graph.edges(data="edge_type"):
        rise = graph.nodes[destination]["elevation"] - graph.nodes[source]["elevation"]
        if edge_type == "lift":
            assert rise > 0
        else:
            assert rise <= 0


def test_no_entrance_is_rejected(tmp_path):
    def change(data):
        for node in data["nodes"]:
            if node["node_type"] == "entrance":
                node["node_type"] = "junction"

    with pytest.raises(ValueError, match="no entrance node"):
        validate_graph(build_graph(build_broken(tmp_path, change)))


def test_no_exit_is_rejected(tmp_path):
    def change(data):
        for node in data["nodes"]:
            if node["node_type"] == "exit":
                node["node_type"] = "junction"

    with pytest.raises(ValueError, match="no exit node"):
        validate_graph(build_graph(build_broken(tmp_path, change)))


def test_an_orphan_node_is_rejected(tmp_path):
    def change(data):
        data["nodes"].append(
            {
                "node_id": "lonely_hut",
                "node_type": "safe_zone",
                "x": 5.0,
                "y": 5.0,
                "elevation": 1400.0,
                "capacity": 20,
            }
        )

    with pytest.raises(ValueError, match="orphan node"):
        validate_graph(build_graph(build_broken(tmp_path, change)))


def test_an_unreachable_exit_is_rejected(tmp_path):
    def change(data):
        data["edges"] = [e for e in data["edges"] if e["destination"] != "base_exit"]
        data["edges"].append(
            {
                "source": "base_exit",
                "destination": "valley_junction",
                "edge_type": "piste",
                "difficulty": "green",
                "length": 190.0,
                "nominal_travel_time": 120.0,
                "safe_capacity": 240,
                "critical_density": 1.6,
                "lift_throughput": None,
                "wind_sensitivity": 0.2,
                "visibility_sensitivity": 0.2,
                "snow_sensitivity": 0.2,
            }
        )

    with pytest.raises(ValueError, match="no entrance reaches the exit node"):
        validate_graph(build_graph(build_broken(tmp_path, change)))


def test_an_unknown_node_in_an_edge_is_rejected(tmp_path):
    def change(data):
        data["edges"][0]["destination"] = "ghost_station"

    with pytest.raises(ValueError, match="unknown node"):
        validate_graph(build_graph(build_broken(tmp_path, change)))


def test_an_unknown_node_type_is_rejected(tmp_path):
    def change(data):
        data["nodes"][0]["node_type"] = "helipad"

    with pytest.raises(ValueError, match="unknown type"):
        validate_graph(build_graph(build_broken(tmp_path, change)))


def test_an_unknown_edge_type_is_rejected(tmp_path):
    def change(data):
        data["edges"][0]["edge_type"] = "gondola"

    with pytest.raises(ValueError, match="unknown type"):
        validate_graph(build_graph(build_broken(tmp_path, change)))
