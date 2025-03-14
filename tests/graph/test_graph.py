"""Tests for the Graph class and related functionality in corneto.

This module contains unit tests that verify the behavior of the Graph class,
including vertex and edge operations, graph traversal algorithms, and attribute handling.
"""

from copy import deepcopy

from corneto.graph import (
    EdgeType,
    Graph,
)
from corneto.graph._base import _fset, _tpl, unique_iter
from corneto.io import load_graph_from_sif_tuples
from corneto.utils import Attr, Attributes


def test_fset():
    """Test the _fset helper function that converts inputs to frozensets."""
    assert _fset("aaa") == frozenset({"aaa"})
    assert _fset(1) == frozenset((1,))
    assert _fset(["a", 1]) == frozenset(("a", 1))


def test_tpl():
    """Test the _tpl helper function that converts inputs to tuples."""
    assert _tpl("aaa") == ("aaa",)
    assert _tpl(1) == (1,)
    assert _tpl(["a", 1]) == ("a", 1)


def test_unique_iter():
    """Test the unique_iter function that returns unique elements from an iterable."""
    assert set(unique_iter(())) == set()
    assert set(unique_iter([1, 2, "a", 2, "b", "a"])) == {1, 2, "a", "b"}


def test_attributes():
    """Test basic attribute creation and access."""
    attr = Attributes()
    attr["a"] = 1
    assert attr.a == 1


def test_attributes_dict():
    """Test attribute initialization from dictionary."""
    attr = Attributes(**{"a": 1, "b": 2})
    assert attr.a == 1
    assert attr.b == 2


def test_attributes_deepcopy():
    """Test that attributes are properly deep copied."""
    attr = Attributes()
    attr.a = 1
    attr.b = 2
    attr2 = deepcopy(attr)
    assert attr is not attr2
    assert attr.a == attr2.a
    assert attr.b == attr2.b


def test_add_single_vertex():
    """Test adding a single vertex to an empty graph."""
    g = Graph()
    index = g.add_vertex("v")
    assert g.num_vertices == 1
    assert g.num_edges == 0
    assert index == 0


def test_add_single_vertex_with_attributes():
    """Test adding a vertex with custom attributes."""
    g = Graph()
    g.add_vertex("v", name="vertex", id="V")
    assert g.get_attr_vertex("v").name == "vertex"
    assert g.get_attr_vertex("v").id == "V"


def test_add_vertexset_with_attributes():
    """Test adding a vertex set (multiple vertices) with attributes."""
    g = Graph()
    v = frozenset((1, 2, 3))
    g.add_vertex(v, name="vertex", id="V")
    attr = g._vertex_attr[v]
    assert attr["name"] == "vertex"
    assert attr["id"] == "V"


def test_add_single_edge():
    """Test adding a single edge to an empty graph."""
    g = Graph()
    i1 = g.add_edge("a", "b")
    assert "a" in g.V
    assert "b" in g.V
    assert i1 == 0


def test_add_simple_edges():
    """Test adding multiple simple edges to a graph."""
    g = Graph()
    i1 = g.add_edge("a", "b")
    i2 = g.add_edge(1, 2)
    assert "a" in g.V
    assert "b" in g.V
    assert 1 in g.V
    assert 2 in g.V
    assert i1 == 0
    assert i2 == 1


def test_add_edges():
    """Test adding multiple edges and verifying edge counts."""
    g = Graph()
    i1 = g.add_edge("a", "b")
    i2 = g.add_edge(1, 2)
    assert i1 == 0
    assert i2 == 1
    assert g.num_edges == 2


def test_get_edge():
    """Test retrieving an edge by its index."""
    g = Graph()
    i1 = g.add_edge("a", "b")
    g.add_edge(1, 2)
    assert g.get_edge(i1) == (frozenset({"a"}), frozenset({"b"}))


def test_edge_reserved_attributes():
    """Test that edges have the required reserved attributes after creation."""
    g = Graph()
    idx = g.add_edge("a", "b")
    assert Attr.EDGE_TYPE.value in g._edge_attr[idx]
    assert Attr.SOURCE_ATTR.value in g._edge_attr[idx]
    assert Attr.TARGET_ATTR.value in g._edge_attr[idx]


def test_add_edge_with_vertex_attributes():
    """Test adding an edge with vertex attributes in dictionary form."""
    g = Graph()
    idx = g.add_edge({"a": -1}, {"b": 1, "c": 2})
    attr = g.get_attr_edge(idx)
    assert attr.get_attr(Attr.SOURCE_ATTR)["a"].get_attr(Attr.VALUE) == -1
    assert attr.get_attr(Attr.TARGET_ATTR)["b"].get_attr(Attr.VALUE) == 1
    assert attr.get_attr(Attr.TARGET_ATTR)["c"].get_attr(Attr.VALUE) == 2


def test_add_edge_with_dupl_vertex_attributes():
    """Test handling of duplicate vertex attributes when adding edges."""
    g = Graph()
    g.add_vertex("a", id="v_a")
    g.add_edge({"a": -1}, {"b": 1, "c": 2})
    idx1 = g.add_edge({"a": 1}, {"a": -1})  # self loop
    # in second edge, there are two diff. attributes for a
    attr = g.get_attr_edge(idx1)
    assert attr.get_attr(Attr.SOURCE_ATTR)["a"].get_attr(Attr.VALUE) == 1
    assert attr.get_attr(Attr.TARGET_ATTR)["a"].get_attr(Attr.VALUE) == -1


def test_add_edge_with_edge_attributes():
    """Test adding an edge with custom edge attributes."""
    g = Graph()
    idx = g.add_edge(1, 2, name="edge_name")
    assert g._edge_attr[idx]["name"] == "edge_name"


def test_edges_by_single_vertex():
    """Test retrieving edges connected to a single vertex."""
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    e = list(g.edges(vertices=("a",)))
    assert len(e) == 2


def test_edges_by_multiple_vertices():
    """Test retrieving edges connected to multiple vertices."""
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    g.add_edge("c", "d")
    g.add_edge("b", "c")
    g.add_edge("d", "e")
    e = list(g.edges(vertices=("a", "c")))
    assert len(e) == 4


def test_parallel_edges():
    """Test handling of parallel edges between the same vertices."""
    g = Graph()
    g.add_edge(1, 2, attr1="x")
    g.add_edge(1, 2, attr2="y")
    g.add_edge(1, 3, attr2="y")
    assert g.num_edges == 3
    assert len(list(g.edges())) == 3


def test_in_edges_directed():
    """Test incoming edges in a directed graph."""
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 2)
    assert len(list(g.in_edges(2))) == 2


def test_in_edges_undirected():
    """Test incoming edges in an undirected graph."""
    g = Graph()
    g.add_edge(1, 2, type=EdgeType.UNDIRECTED)
    g.add_edge(2, 3, type=EdgeType.UNDIRECTED)
    g.add_edge(3, 2)
    assert len(list(g.in_edges(2))) == 3


def test_out_edges_directed():
    """Test outgoing edges in a directed graph."""
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 2)
    assert len(list(g.out_edges(2))) == 1


def test_out_edges_undirected():
    """Test outgoing edges in an undirected graph."""
    g = Graph()
    g.add_edge(1, 2, type=EdgeType.UNDIRECTED)
    g.add_edge(2, 3, type=EdgeType.UNDIRECTED)
    g.add_edge(3, 2)
    assert len(list(g.out_edges(2))) == 2


def test_succesors_directed():
    """Test successor vertices in a directed graph."""
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(1, 1)
    g.add_edge(1, 2)
    g.add_edge(1, 4)
    g.add_edge(2, 3)
    g.add_edge(3, 2)
    g.add_edge(3, 4)
    g.add_edge(4, 1)
    assert set(g.successors(1)) == {1, 2, 4}
    assert set(g.successors(2)) == {3}


def test_succesors_directed_hyperedge():
    """Test successor vertices with hyperedges in a directed graph."""
    g = Graph()
    g.add_edge({1, 2}, {3, 4})
    g.add_edge(1, 1)
    g.add_edge(3, 1)
    g.add_edge(4, 2)
    g.add_edge({3}, {5, 6})
    assert set(g.successors(1)) == {3, 4, 1}
    assert set(g.successors(3)) == {1, 5, 6}


def test_succesors_undirected():
    """Test successor vertices in an undirected graph."""
    g = Graph()
    g.add_edge(1, 2, type=EdgeType.UNDIRECTED)
    g.add_edge(2, 3, type=EdgeType.UNDIRECTED)
    g.add_edge(3, 2)
    assert set(g.successors(2)) == {1, 3}


def test_predecessors_directed():
    """Test predecessor vertices in a directed graph."""
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 2)
    g.add_edge(2, 2)
    g.add_edge(3, 4)
    assert set(g.predecessors(2)) == {2, 1, 3}


def test_predecessors_undirected():
    """Test predecessor vertices in an undirected graph."""
    g = Graph()
    g.add_edge(1, 2, type=EdgeType.UNDIRECTED)
    g.add_edge(2, 3, type=EdgeType.UNDIRECTED)
    g.add_edge(3, 2)
    assert set(g.successors(2)) == {1, 3}


# TODO: add tests for subgraph, edge_subgraph, filter_graph making sure that
# edge attributes are preserved


def test_graph_bfs():
    """Test breadth-first search traversal."""
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (1, 3), (3, 4), (4, 1), (3, 1), (4, 5)])
    dist = g.bfs(2)
    assert dist[1] == 2


def test_graph_bfs_rev():
    """Test reverse breadth-first search traversal."""
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (1, 3), (3, 4), (4, 1), (3, 1), (4, 5)])
    dist = g.bfs(2, reverse=True)
    assert dist[1] == 1
    assert dist[4] == 2
    assert 5 not in dist


def test_graph_toposort():
    """Test topological sorting of vertices."""
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("c", "b")
    g.add_edge("c", "d")
    g.add_edge("c", "e")
    g.add_edge("b", "d")
    g.add_edge("d", "e")
    order = g.toposort()
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("c") < order.index("d")
    assert order.index("d") < order.index("e")
    assert order.index("b") < order.index("d")


def test_incidence_single_edge_single_source_vertex():
    """Test incidence matrix for a graph with single edge and source vertex."""
    g = Graph()
    g.add_edge(1, ())
    A = g.vertex_incidence_matrix()
    assert A.shape == (1, 1)
    assert A[0, 0] == -1


def test_incidence_single_edge_single_target_vertex():
    """Test incidence matrix for a graph with single edge and target vertex."""
    g = Graph()
    g.add_edge((), 1)
    A = g.vertex_incidence_matrix()
    assert A.shape == (1, 1)
    assert A[0, 0] == 1


def test_incidence_two_edges_single_vertex():
    """Test incidence matrix for a graph with two edges and a single vertex."""
    g = Graph()
    g.add_edge(1, ())
    g.add_edge((), 2)
    A = g.vertex_incidence_matrix()
    assert A.shape == (2, 2)


def test_E_right_order():
    """Test that edges are maintained in the correct order."""
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (3, 4)])
    assert g.E[0] == ({1}, {2})
    assert g.E[-1] == ({3}, {4})


def test_V_right_order():
    """Test that vertices are maintained in the correct order."""
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (3, {4, 5})])
    assert g.V[0] == 1
    assert g.V[-1] == 5


def test_edge_subgraph():
    """Test creating a subgraph based on selected edges."""
    g = Graph(name="graph")
    g.add_edge({1, 2}, {3, 4}, custom="custom")
    g.add_edge(1, 1, type=EdgeType.UNDIRECTED)
    g.add_edge(3, 1)
    g.add_edge(4, 2)
    g.add_edge({3}, {5, 6})
    g.add_edge(7, 8)
    gs = g.edge_subgraph([0, 1, 4])
    assert gs.num_edges == 3
    assert gs.num_vertices == 6


def test_edge_subgraph_attributes():
    """Test that attributes are preserved when creating an edge subgraph."""
    g = Graph(name="graph")
    g.add_edge({1, 2}, {3, 4}, custom="e1")
    g.add_edge(1, 1, type=EdgeType.UNDIRECTED, custom="e2")
    g.add_edge(3, 1, custom="e3")
    g.add_edge(4, 2, custom="e4")
    g.add_edge({3}, {5, 6}, custom="e5")
    g.add_edge(7, 8, custom="e6")
    gs = g.edge_subgraph([0, 1, 4])
    # Graph contains the graph attribute name graph
    assert gs.get_graph_attributes().name == "graph"
    # Selected edges contain the attributes
    assert len(gs.get_attr_edges()) == 3
    assert "custom" in gs.get_attr_edge(0)
    assert gs.get_attr_edge(0)["custom"] == "e1"
    assert "custom" in gs.get_attr_edge(1)
    assert gs.get_attr_edge(1)["custom"] == "e2"
    assert "custom" in gs.get_attr_edge(2)
    assert gs.get_attr_edge(2)["custom"] == "e5"


def test_import_from_tuples():
    """Test graph creation from SIF format tuples."""
    tpl = [("A", 1, "B"), ("B", -1, "C")]
    g = load_graph_from_sif_tuples(tpl)
    assert g.get_attr_edge(0).interaction == 1
    assert g.get_attr_edge(1).interaction == -1
    assert g.num_edges == 2


def test_prune_directed():
    """Test graph pruning in a directed graph."""
    G = Graph()
    G.add_edges(
        [
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("D", "C"),
            ("D", "E"),
            ("B", "E"),
            ("E", "F"),
            ("A", "F"),
            ("F", "G"),
            ("F", "H"),
            ("H", "E"),
            ("I", "F"),
            ("D", "I"),
            ("J", "C"),
            ("J", "G"),
            ("C", "F"),
            ("J", "A"),
            ("I", "K"),
            ("H", "K"),
            ("B", "K"),
        ],
        type=EdgeType.DIRECTED,
    )
    assert set(G.prune(["E"], ["K"]).V) == {"E", "F", "H", "K"}


def test_graph_hash():
    """Test that graph hashing changes when the graph structure or attributes change."""
    G = Graph()
    G.add_edges(
        [
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("D", "C"),
            ("D", "E"),
            ("B", "E"),
            ("E", "F"),
            ("A", "F"),
        ],
        type=EdgeType.DIRECTED,
    )
    h1 = G.hash()
    G.add_edge("F", "G")
    h2 = G.hash()
    G._edge_attr[0]["x"] = ""
    h3 = G.hash()
    assert h1 != h2
    assert h1 != h3
    assert h2 != h3
