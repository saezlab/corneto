"""Tests for CORNETO and AnnNet graph conversion."""

import pytest

from corneto.contrib.annnet import from_annnet, to_annnet
from corneto.graph import Attr, EdgeType, Graph

annnet = pytest.importorskip("annnet")


def test_directed_hypergraph_roundtrip():
    """Directed hyperedges and annotations survive a round-trip."""
    graph = Graph(name="example")
    graph.add_vertex("A", kind="gene")
    graph.add_edge(
        {"A": -2.0, "B": -1.0},
        {"C": 3.0},
        relation="reaction",
    )

    converted = to_annnet(graph)

    assert converted.vertices() == ["A", "B", "C"]
    assert converted.get_edge("corneto_edge_0") == (
        frozenset({"A", "B"}),
        frozenset({"C"}),
    )
    assert converted.get_edges_by_direction(True) == ["corneto_edge_0"]
    assert converted.attrs.get_vertex_attrs("A")["kind"] == "gene"
    assert converted.attrs.get_edge_attrs("corneto_edge_0")["relation"] == "reaction"
    assert converted.uns["name"] == "example"

    restored = from_annnet(converted)
    source, target = restored.get_edge(0)
    attributes = restored.get_attr_edge(0)

    assert source == frozenset({"A", "B"})
    assert target == frozenset({"C"})
    assert attributes.get_attr(Attr.EDGE_TYPE) == EdgeType.DIRECTED.value
    assert attributes.get_attr(Attr.SOURCE_ATTR)["A"].get_attr(Attr.VALUE) == 2.0
    assert attributes.get_attr(Attr.TARGET_ATTR)["C"].get_attr(Attr.VALUE) == 3.0
    assert attributes["relation"] == "reaction"
    assert attributes["_annnet_edge_id"] == "corneto_edge_0"
    assert restored.get_graph_attributes()["name"] == "example"


def test_binary_parallel_edges_keep_distinct_ids():
    """Explicit AnnNet IDs keep parallel CORNETO edges separate."""
    graph = Graph()
    graph.add_edge("A", "B", label="first")
    graph.add_edge("A", "B", label="second")

    converted = to_annnet(graph)

    assert converted.edges() == ["corneto_edge_0", "corneto_edge_1"]
    assert converted.attrs.get_edge_attrs("corneto_edge_0")["label"] == "first"
    assert converted.attrs.get_edge_attrs("corneto_edge_1")["label"] == "second"


def test_undirected_hyperedge_is_canonicalized_to_member_set():
    """Undirected hyperedges retain members but not endpoint partitions."""
    graph = Graph()
    graph.add_edge(
        {"A", "B"},
        {"C"},
        type=EdgeType.UNDIRECTED,
    )

    converted = to_annnet(graph)
    members = frozenset({"A", "B", "C"})

    assert converted.get_edge("corneto_edge_0") == (members, members)
    assert converted.get_edges_by_direction(False) == ["corneto_edge_0"]

    restored = from_annnet(converted)
    assert restored.get_edge(0) == (members, members)
    assert (
        restored.get_attr_edge(0).get_attr(Attr.EDGE_TYPE)
        == EdgeType.UNDIRECTED.value
    )


def test_non_string_vertex_ids_are_converted_to_strings():
    """CORNETO vertex identifiers are stringified for AnnNet."""
    graph = Graph()
    graph.add_edge(1, 2)

    converted = to_annnet(graph)

    assert converted.vertices() == ["1", "2"]


def test_string_conversion_rejects_vertex_id_collisions():
    """Stringification cannot silently merge distinct CORNETO vertices."""
    graph = Graph()
    graph.add_vertex(1)
    graph.add_vertex("1")

    with pytest.raises(ValueError, match="map to AnnNet vertex"):
        to_annnet(graph)


def test_empty_endpoint_set_is_not_supported():
    """AnnNet conversion rejects CORNETO boundary edges clearly."""
    graph = Graph()
    graph.add_edge("A", ())

    with pytest.raises(ValueError, match="empty source or target"):
        to_annnet(graph)
