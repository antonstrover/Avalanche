import hashlib
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from avalanche.sim import (
    DIFFICULTY_NAMES,
    EDGE_TYPE_NAMES,
    NODE_TYPE_NAMES,
    build_graph,
    load_topology,
)
from avalanche.sim.topology import immutable_array

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)

EDGE_FIELDS = {
    "edge_length": "length",
    "edge_nominal_travel_time": "nominal_travel_time",
    "edge_safe_capacity": "safe_capacity",
    "edge_critical_density": "critical_density",
    "edge_wind_sensitivity": "wind_sensitivity",
    "edge_visibility_sensitivity": "visibility_sensitivity",
    "edge_snow_sensitivity": "snow_sensitivity",
}

EXPECTED_DTYPES = {
    "node_x": "<f4",
    "node_y": "<f4",
    "node_elevation": "<f4",
    "node_type": "|i1",
    "node_capacity": "<i4",
    "node_controllable": "|b1",
    "edge_source": "<i4",
    "edge_destination": "<i4",
    "edge_type": "|i1",
    "edge_difficulty": "|i1",
    "edge_length": "<f4",
    "edge_nominal_travel_time": "<f4",
    "edge_safe_capacity": "<i4",
    "edge_critical_density": "<f4",
    "edge_lift_throughput": "<f4",
    "edge_wind_sensitivity": "<f4",
    "edge_visibility_sensitivity": "<f4",
    "edge_snow_sensitivity": "<f4",
    "edge_controllable": "|b1",
    "edge_offsets": "<i4",
    "outgoing_edges": "<i4",
}


@pytest.fixture(scope="module")
def topology():
    return load_topology(FIXTURE)


@pytest.fixture(scope="module")
def graph():
    return build_graph(FIXTURE)


def test_the_array_shapes_match_the_counts(topology):
    assert topology.node_count == 10
    assert topology.edge_count == 12
    for name in (
        "node_x",
        "node_y",
        "node_elevation",
        "node_type",
        "node_capacity",
        "node_controllable",
    ):
        assert getattr(topology, name).shape == (topology.node_count,)
    for name in (
        "edge_source",
        "edge_destination",
        "edge_type",
        "edge_controllable",
        *EDGE_FIELDS,
    ):
        assert getattr(topology, name).shape == (topology.edge_count,)
    assert topology.edge_offsets.shape == (topology.node_count + 1,)
    assert topology.outgoing_edges.shape == (topology.edge_count,)


def test_the_node_index_is_sorted_and_consistent(topology):
    assert list(topology.node_ids) == sorted(topology.node_ids)
    for index, node_id in enumerate(topology.node_ids):
        assert topology.node_index[node_id] == index


def test_the_offsets_are_monotonic_and_complete(topology):
    offsets = topology.edge_offsets
    assert offsets[0] == 0
    assert offsets[-1] == topology.edge_count
    assert np.all(np.diff(offsets) >= 0)
    assert sorted(topology.outgoing_edges.tolist()) == list(range(topology.edge_count))


def test_each_outgoing_edge_starts_at_its_node(topology):
    for node in range(topology.node_count):
        edges = topology.edges_from(node)
        assert np.all(topology.edge_source[edges] == node)


def test_the_edges_round_trip_against_the_graph(topology, graph):
    expected = {
        (source, destination): attributes
        for source, destination, attributes in graph.edges(data=True)
    }
    seen = set()
    for edge in range(topology.edge_count):
        source = topology.node_ids[topology.edge_source[edge]]
        destination = topology.node_ids[topology.edge_destination[edge]]
        attributes = expected[(source, destination)]
        seen.add((source, destination))
        assert EDGE_TYPE_NAMES[topology.edge_type[edge]] == attributes["edge_type"]
        assert (
            DIFFICULTY_NAMES[topology.edge_difficulty[edge]] == attributes["difficulty"]
        )
        for name, field in EDGE_FIELDS.items():
            assert getattr(topology, name)[edge] == pytest.approx(attributes[field])
        throughput = attributes["lift_throughput"] or 0.0
        assert topology.edge_lift_throughput[edge] == pytest.approx(throughput)
    assert seen == set(expected)


def test_the_nodes_round_trip_against_the_graph(topology, graph):
    for index, node_id in enumerate(topology.node_ids):
        attributes = graph.nodes[node_id]
        assert NODE_TYPE_NAMES[topology.node_type[index]] == attributes["node_type"]
        assert topology.node_x[index] == pytest.approx(attributes["x"])
        assert topology.node_y[index] == pytest.approx(attributes["y"])
        assert topology.node_elevation[index] == pytest.approx(attributes["elevation"])
        assert topology.node_capacity[index] == attributes["capacity"]


def test_every_topology_array_is_read_only(topology):
    for field_info in fields(topology):
        values = getattr(topology, field_info.name)
        if not isinstance(values, np.ndarray):
            continue
        before = values.tobytes()
        assert not values.flags.writeable, field_info.name
        assert not values.flags.owndata, field_info.name
        with pytest.raises(ValueError, match="read-only"):
            values.flat[0] = values.flat[0]
        with pytest.raises(ValueError):
            values.setflags(write=True)
        assert values.tobytes() == before, field_info.name


def test_every_topology_array_has_one_declared_dtype(topology):
    for name, dtype in EXPECTED_DTYPES.items():
        assert getattr(topology, name).dtype.str == dtype, name


def test_every_topology_array_uses_immutable_bytes(topology):
    for name in EXPECTED_DTYPES:
        owner = getattr(topology, name)
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        assert isinstance(owner.base, bytes), name


def test_the_mountain_digest_covers_the_source_file(topology):
    assert topology.mountain_sha256 == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def test_an_object_static_array_is_rejected(topology):
    values = np.asarray(topology.node_x, dtype=object)

    with pytest.raises(TypeError, match="object dtype"):
        replace(topology, node_x=values)


def test_a_platform_dtype_declaration_is_rejected():
    values = np.arange(3, dtype=np.int64)

    with pytest.raises(TypeError, match="byte order"):
        immutable_array(values, np.int64)


def test_the_node_index_is_read_only(topology):
    with pytest.raises(TypeError):
        topology.node_index["new-node"] = topology.node_count


def test_an_outgoing_edge_view_is_read_only(topology):
    source = topology.node_index["base_village"]
    outgoing = topology.edges_from(source)

    assert not outgoing.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        outgoing[0] = outgoing[0]
