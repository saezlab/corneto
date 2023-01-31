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

def test_get_edge():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge((1, 2), (3, 4))
    g.add_edge(1, 2, prop = True)
    assert g.get_edge(("a", "b")) == g.get_edge((("a",), ("b",)))
    assert g.get_edge(((1, 2), (3, 4))) == g.get_edge(({1, 2}, {3, 4}))
    assert g.get_edge((1, 2)) is not None


def test_add_hyperedges():
    g = Graph()
    g.add_edge((1, 2), (3, 4))
    g.add_edge(("a", 1), ("b", 2))
    assert g.edges[0] == ({1, 2}, {3, 4})
    assert g.edges[1] == ({"a", 1}, {"b", 2})


def test_edge_vertex_properties():
    g = Graph()
    g.add_edge({1: 10, "a": -1}, {2: -10, "b": 5})
    props = g.get_edge(g.edges[0])
    assert "v" in props[1] and props[1]["v"] == 10
    assert "v" in props["a"] and props["a"]["v"] == -1
    assert "v" in props[2] and props[2]["v"] == -10
    assert "v" in props["b"] and props["b"]["v"] == 5

def test_edge_properties():
    g = Graph()
    g.add_edge(1, 2, prop=True)
    g.get_edge(({1}, {2}))