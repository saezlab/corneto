import pytest
import pathlib
from corneto._core import Graph


def test_add_simple_edges():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge(1, 2)
    assert "a" in g.vertices
    assert "b" in g.vertices
    assert 1 in g.vertices
    assert 2 in g.vertices

def test_graph_bfs():
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (1, 3), (3, 4), (4, 1), (3, 1), (4, 5)])
    dist = g.bfs(2)
    assert dist[1] == 2

def test_graph_bfs_rev():
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (1, 3), (3, 4), (4, 1), (3, 1), (4, 5)])
    dist = g.bfs(2, rev=True)
    assert dist[1] == 1
    assert dist[4] == 2
    assert 5 not in dist

def test_get_edges_with_source_vertex():
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (1, 3), (3, 4), (4, 1), (3, 1), (4, 5)])
    E = g.get_edges_with_source_vertex(4)
    assert (frozenset({4}), frozenset({1})) in E
    assert (frozenset({4}), frozenset({5})) in E
    assert (frozenset({3}), frozenset({4})) not in E


def test_get_edges_with_target_vertex():
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (1, 3), (3, 4), (4, 1), (3, 1), (4, 5)])
    E = g.get_edges_with_target_vertex(4)
    assert (frozenset({3}), frozenset({4})) in E
    assert (frozenset({4}), frozenset({1})) not in E


def test_add_edge_single_vertex():
    g = Graph()
    g.add_edge(1, ())
    g.add_edge((), 2)


def test_incidence_single_edge_single_source_vertex():
    g = Graph()
    g.add_edge(1, ())
    A = g.vertex_incidence_matrix()
    assert A.shape == (1,1)
    assert A[0, 0] == -1


def test_incidence_single_edge_single_target_vertex():
    g = Graph()
    g.add_edge((), 1)
    A = g.vertex_incidence_matrix()
    assert A.shape == (1,1)
    assert A[0, 0] == 1
    

def test_incidence_two_edges_single_vertex():
    g = Graph()
    g.add_edge(1, ())
    g.add_edge((), 2)
    A = g.vertex_incidence_matrix()
    assert A.shape == (2, 2)


def test_add_hyperedges():
    g = Graph()
    g.add_edge((1, 2), (3, 4))
    g.add_edge(("a", 1), ("b", 2))
    assert g.edges[0] == ({1, 2}, {3, 4})
    assert g.edges[1] == ({"a", 1}, {"b", 2})


def test_edge_vertex_properties():
    g = Graph()
    g.add_edge({1: 10, "a": -1}, {2: -10, "b": 5})
    props, = g.get_vertex_properties_for_edge(g.edges[0])
    assert "v" in props[1] and props[1]["v"] == 10
    assert "v" in props["a"] and props["a"]["v"] == -1
    assert "v" in props[2] and props[2]["v"] == -10
    assert "v" in props["b"] and props["b"]["v"] == 5

def test_source_vertices():
    g = Graph()
    assert g.get_source_vertices() == set()
    g.add_vertex('s')
    assert g.get_source_vertices() == {'s'}
    g.add_edge('s', 't')
    assert g.get_source_vertices() == {'s'}
    g.add_edge('u', 's')
    assert g.get_source_vertices() == {'u'}


def test_sink_vertices():
    g = Graph()
    assert g.get_sink_vertices() == set()
    g.add_vertex('s')
    assert g.get_sink_vertices() == {'s'}
    g.add_edge('s', 't')
    assert g.get_sink_vertices() == {'t'}
    g.add_edge('t', 'u')
    assert g.get_sink_vertices() == {'u'}

# TODO: predecessor/successor of a vertex without edges