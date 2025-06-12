from copy import deepcopy

from corneto._graph import (
    Attr,
    Attributes,
    EdgeType,
    Graph,
    _fset,
    _tpl,
    unique_iter,
)


def test_fset():
    assert _fset("aaa") == frozenset({"aaa"})
    assert _fset(1) == frozenset((1,))
    assert _fset(["a", 1]) == frozenset(("a", 1))


def test_tpl():
    assert _tpl("aaa") == ("aaa",)
    assert _tpl(1) == (1,)
    assert _tpl(["a", 1]) == ("a", 1)


def test_unique_iter():
    assert set(unique_iter(())) == set()
    assert set(unique_iter([1, 2, "a", 2, "b", "a"])) == {1, 2, "a", "b"}


def test_attributes():
    attr = Attributes()
    attr["a"] = 1
    assert attr.a == 1


def test_attributes_dict():
    attr = Attributes(**{"a": 1, "b": 2})
    assert attr.a == 1
    assert attr.b == 2


def test_attributes_deepcopy():
    attr = Attributes()
    attr.a = 1
    attr.b = 2
    attr2 = deepcopy(attr)
    assert attr is not attr2
    assert attr.a == attr2.a
    assert attr.b == attr2.b


def test_add_single_vertex():
    g = Graph()
    index = g.add_vertex("v")
    assert g.num_vertices == 1
    assert g.num_edges == 0
    assert index == 0


def test_add_single_vertex_with_attributes():
    g = Graph()
    g.add_vertex("v", name="vertex", id="V")
    assert g.get_attr_vertex("v").name == "vertex"
    assert g.get_attr_vertex("v").id == "V"


def test_add_vertexset_with_attributes():
    g = Graph()
    v = frozenset((1, 2, 3))
    g.add_vertex(v, name="vertex", id="V")
    attr = g._vertex_attr[v]
    assert attr["name"] == "vertex"
    assert attr["id"] == "V"


def test_add_single_edge():
    g = Graph()
    i1 = g.add_edge("a", "b")
    assert "a" in g.V
    assert "b" in g.V
    assert i1 == 0


def test_add_simple_edges():
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
    g = Graph()
    i1 = g.add_edge("a", "b")
    i2 = g.add_edge(1, 2)
    assert i1 == 0
    assert i2 == 1
    assert g.num_edges == 2


def test_get_edge():
    g = Graph()
    i1 = g.add_edge("a", "b")
    g.add_edge(1, 2)
    assert g.get_edge(i1) == (frozenset({"a"}), frozenset({"b"}))


def test_edge_reserved_attributes():
    g = Graph()
    idx = g.add_edge("a", "b")
    assert Attr.EDGE_TYPE.value in g._edge_attr[idx]
    assert Attr.SOURCE_ATTR.value in g._edge_attr[idx]
    assert Attr.TARGET_ATTR.value in g._edge_attr[idx]


def test_add_edge_with_vertex_attributes():
    g = Graph()
    idx = g.add_edge({"a": -1}, {"b": 1, "c": 2})
    attr = g.get_attr_edge(idx)
    assert attr.get_attr(Attr.SOURCE_ATTR)["a"].get_attr(Attr.VALUE) == -1
    assert attr.get_attr(Attr.TARGET_ATTR)["b"].get_attr(Attr.VALUE) == 1
    assert attr.get_attr(Attr.TARGET_ATTR)["c"].get_attr(Attr.VALUE) == 2


def test_add_edge_with_dupl_vertex_attributes():
    g = Graph()
    g.add_vertex("a", id="v_a")
    g.add_edge({"a": -1}, {"b": 1, "c": 2})
    idx1 = g.add_edge({"a": 1}, {"a": -1})  # self loop
    # in second edge, there are two diff. attributes for a
    attr = g.get_attr_edge(idx1)
    assert attr.get_attr(Attr.SOURCE_ATTR)["a"].get_attr(Attr.VALUE) == 1
    assert attr.get_attr(Attr.TARGET_ATTR)["a"].get_attr(Attr.VALUE) == -1


def test_add_edge_with_edge_attributes():
    g = Graph()
    idx = g.add_edge(1, 2, name="edge_name")
    assert g._edge_attr[idx]["name"] == "edge_name"


def test_edges_by_single_vertex():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    e = list(g.edges(vertices=("a",)))
    assert len(e) == 2


def test_edges_by_multiple_vertices():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    g.add_edge("c", "d")
    g.add_edge("b", "c")
    g.add_edge("d", "e")
    e = list(g.edges(vertices=("a", "c")))
    assert len(e) == 4


def test_parallel_edges():
    g = Graph()
    g.add_edge(1, 2, attr1="x")
    g.add_edge(1, 2, attr2="y")
    g.add_edge(1, 3, attr2="y")
    assert g.num_edges == 3
    assert len(list(g.edges())) == 3


def test_in_edges_directed():
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 2)
    assert len(list(g.in_edges(2))) == 2


def test_in_edges_undirected():
    g = Graph()
    g.add_edge(1, 2, type=EdgeType.UNDIRECTED)
    g.add_edge(2, 3, type=EdgeType.UNDIRECTED)
    g.add_edge(3, 2)
    assert len(list(g.in_edges(2))) == 3


def test_out_edges_directed():
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 2)
    assert len(list(g.out_edges(2))) == 1


def test_out_edges_undirected():
    g = Graph()
    g.add_edge(1, 2, type=EdgeType.UNDIRECTED)
    g.add_edge(2, 3, type=EdgeType.UNDIRECTED)
    g.add_edge(3, 2)
    assert len(list(g.out_edges(2))) == 2


def test_succesors_directed():
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
    g = Graph()
    g.add_edge({1, 2}, {3, 4})
    g.add_edge(1, 1)
    g.add_edge(3, 1)
    g.add_edge(4, 2)
    g.add_edge({3}, {5, 6})
    assert set(g.successors(1)) == {3, 4, 1}
    assert set(g.successors(3)) == {1, 5, 6}


def test_succesors_undirected():
    g = Graph()
    g.add_edge(1, 2, type=EdgeType.UNDIRECTED)
    g.add_edge(2, 3, type=EdgeType.UNDIRECTED)
    g.add_edge(3, 2)
    assert set(g.successors(2)) == {1, 3}


def test_predecessors_directed():
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 2)
    g.add_edge(2, 2)
    g.add_edge(3, 4)
    assert set(g.predecessors(2)) == {2, 1, 3}


def test_predecessors_undirected():
    g = Graph()
    g.add_edge(1, 2, type=EdgeType.UNDIRECTED)
    g.add_edge(2, 3, type=EdgeType.UNDIRECTED)
    g.add_edge(3, 2)
    assert set(g.successors(2)) == {1, 3}


# TODO: add tests for subgraph, edge_subgraph, filter_graph making sure that
# edge attributes are preserved


def test_graph_bfs():
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (1, 3), (3, 4), (4, 1), (3, 1), (4, 5)])
    dist = g.bfs(2)
    assert dist[1] == 2


def test_graph_bfs_rev():
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (1, 3), (3, 4), (4, 1), (3, 1), (4, 5)])
    dist = g.bfs(2, reverse=True)
    assert dist[1] == 1
    assert dist[4] == 2
    assert 5 not in dist


def test_graph_toposort():
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
    g = Graph()
    g.add_edge(1, ())
    A = g.vertex_incidence_matrix()
    assert A.shape == (1, 1)
    assert A[0, 0] == -1


def test_incidence_single_edge_single_target_vertex():
    g = Graph()
    g.add_edge((), 1)
    A = g.vertex_incidence_matrix()
    assert A.shape == (1, 1)
    assert A[0, 0] == 1


def test_incidence_two_edges_single_vertex():
    g = Graph()
    g.add_edge(1, ())
    g.add_edge((), 2)
    A = g.vertex_incidence_matrix()
    assert A.shape == (2, 2)


def test_E_right_order():
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (3, 4)])
    assert g.E[0] == ({1}, {2})
    assert g.E[-1] == ({3}, {4})


def test_V_right_order():
    g = Graph()
    g.add_edges([(1, 2), (2, 3), (3, {4, 5})])
    assert g.V[0] == 1
    assert g.V[-1] == 5


def test_edge_subgraph():
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
    tpl = [("A", 1, "B"), ("B", -1, "C")]
    g = Graph.from_sif_tuples(tpl)
    assert g.get_attr_edge(0).interaction == 1
    assert g.get_attr_edge(1).interaction == -1
    assert g.num_edges == 2


def test_prune_directed():
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
