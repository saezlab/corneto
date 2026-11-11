import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from corneto.backend import CvxpyBackend
from corneto.graph import EdgeType, Graph
from corneto.methods.steiner import solve_steiner_tree


@pytest.fixture(params=["corneto", "steinerpy"])
def engine_options(request):
    if request.param == "steinerpy":
        pytest.importorskip("steinerpy")
        return {"engine": "steinerpy", "solver": "highs"}
    return {"engine": "corneto", "solver": "SCIPY", "backend": CvxpyBackend()}


def _graph(edge_type):
    graph = Graph()
    edges = [("A", "B"), ("A", "C"), ("C", "B"), ("B", "D"), ("A", "D")]
    for index, (source, target) in enumerate(edges):
        if edge_type == EdgeType.UNDIRECTED and index == 2:
            source, target = target, source
        graph.add_edge(source, target, type=edge_type, weight=100 if index == 4 else 1)
    if edge_type == EdgeType.DIRECTED:
        graph.add_edge("D", "A", type=edge_type)
    return graph


@pytest.mark.parametrize("edge_type", [EdgeType.UNDIRECTED, EdgeType.DIRECTED])
@pytest.mark.parametrize("cheap_cost", [0, 1])
def test_steiner_helper_preserves_costs_direction_and_original_indices(engine_options, edge_type, cheap_cost):
    graph = _graph(edge_type)
    original = graph.to_dict()
    costs = {0: 7, 1: cheap_cost, 2: cheap_cost, 3: cheap_cost, 4: 10}
    if edge_type == EdgeType.DIRECTED:
        costs[5] = 0.01  # This cheap reverse arc cannot connect A to D.

    indices, result = solve_steiner_tree(
        graph, terminals=["B", "D"], root_vertex="A", edge_costs=costs, **engine_options
    )

    assert isinstance(indices, np.ndarray)
    np.testing.assert_array_equal(indices, [1, 2, 3])
    assert sum(costs[index] for index in indices) == 3 * cheap_cost
    assert {"A", "B", "D"} <= set(graph.edge_subgraph(indices).bfs(["A"]))
    assert result is not None
    objective = result.objectives[0].value if engine_options["engine"] == "corneto" else result.objective
    assert float(np.asarray(objective).item()) == pytest.approx(3 * cheap_cost)
    assert graph.to_dict() == original


def test_steiner_helper_defaults_to_unit_costs_and_ignores_weight_attributes(engine_options):
    graph = _graph(EdgeType.UNDIRECTED)
    indices, _ = solve_steiner_tree(graph, terminals=["A", "D"], **engine_options)
    np.testing.assert_array_equal(indices, [4])


@pytest.mark.parametrize("edge_type", [EdgeType.UNDIRECTED, EdgeType.DIRECTED])
def test_steiner_helper_single_terminal_still_connects_a_distinct_root(engine_options, edge_type):
    graph = _graph(edge_type)
    indices, result = solve_steiner_tree(graph, terminals=["D"], root_vertex="A", **engine_options)
    np.testing.assert_array_equal(indices, [4])
    assert result is not None


@pytest.mark.parametrize("engine", ["corneto", "steinerpy"])
def test_steiner_helper_single_vertex_needs_no_solver(engine, monkeypatch):
    monkeypatch.setitem(sys.modules, "steinerpy", None)
    graph = Graph()
    graph.add_vertex("A")
    indices, result = solve_steiner_tree(graph, terminals=["A", "A"], engine=engine)
    assert isinstance(indices, np.ndarray)
    assert indices.size == 0
    assert result is None


@pytest.mark.parametrize("engine", ["corneto", "steinerpy"])
def test_steiner_helper_rejects_disconnected_terminals_before_import(engine, monkeypatch):
    monkeypatch.setitem(sys.modules, "steinerpy", None)
    graph = _graph(EdgeType.UNDIRECTED)
    graph.add_vertex("isolated")
    with pytest.raises(ValueError, match=r"[Uu]nreachable|[Dd]isconnected|[Rr]each"):
        solve_steiner_tree(graph, terminals=["A", "isolated"], engine=engine)


@pytest.mark.parametrize(
    "edges",
    [
        pytest.param([(["A", "C"], "B", EdgeType.DIRECTED)], id="hyperedge"),
        pytest.param([([], "A", EdgeType.DIRECTED)], id="boundary"),
        pytest.param([("A", "A", EdgeType.DIRECTED)], id="self-loop"),
        pytest.param([("A", "B", EdgeType.DIRECTED), ("A", "B", EdgeType.DIRECTED)], id="parallel-directed"),
        pytest.param([("A", "B", EdgeType.UNDIRECTED), ("B", "A", EdgeType.UNDIRECTED)], id="parallel-undirected"),
        pytest.param([("A", "B", EdgeType.DIRECTED), ("B", "C", EdgeType.UNDIRECTED)], id="mixed"),
    ],
)
def test_steiner_helper_rejects_unsupported_graphs_before_import(edges, monkeypatch):
    monkeypatch.setitem(sys.modules, "steinerpy", None)
    graph = Graph()
    for vertex in ["A", "B", "C"]:
        graph.add_vertex(vertex)
    for source, target, edge_type in edges:
        graph.add_edge(source, target, type=edge_type)
    with pytest.raises(ValueError):
        solve_steiner_tree(graph, terminals=["A", "B"], root_vertex="A", engine="steinerpy")


@pytest.mark.parametrize(
    "arguments",
    [
        {"terminals": []},
        {"terminals": "A"},
        {"terminals": {"A": 1, "D": 2}},
        {"terminals": ["missing"]},
        {"root_vertex": "missing"},
        {"edge_costs": {0: -1}},
        {"edge_costs": {0: np.inf}},
        {"edge_costs": {0: np.nan}},
        {"edge_costs": {0: True}},
        {"edge_costs": {-1: 1}},
        {"edge_costs": {5: 1}},
        {"edge_costs": [1, 2, 3, 4, 5]},
    ],
)
def test_steiner_helper_validates_inputs_before_import(arguments, monkeypatch):
    monkeypatch.setitem(sys.modules, "steinerpy", None)
    graph = _graph(EdgeType.UNDIRECTED)
    kwargs = {"terminals": ["A", "D"], "engine": "steinerpy", **arguments}
    with pytest.raises((TypeError, ValueError)):
        solve_steiner_tree(graph, **kwargs)


def test_steiner_helper_directed_graph_requires_explicit_root():
    with pytest.raises(ValueError, match="root"):
        solve_steiner_tree(_graph(EdgeType.DIRECTED), terminals=["A", "D"])


def test_steiner_helper_optional_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "steinerpy", None)
    with pytest.raises(ImportError, match=r"corneto\[steinerpy\]"):
        solve_steiner_tree(_graph(EdgeType.UNDIRECTED), terminals=["A", "D"], engine="steinerpy")


@pytest.mark.parametrize("arguments", [{"engine": "auto"}, {"engine": "steinerpy", "backend": CvxpyBackend()}])
def test_steiner_helper_rejects_invalid_engine_configuration(arguments):
    with pytest.raises((TypeError, ValueError)):
        solve_steiner_tree(_graph(EdgeType.UNDIRECTED), terminals=["A", "D"], **arguments)


def test_steinerpy_helper_forwards_heuristic_option():
    pytest.importorskip("steinerpy")
    graph = _graph(EdgeType.UNDIRECTED)
    indices, result = solve_steiner_tree(
        graph,
        terminals=["A", "D"],
        edge_costs={0: 7, 1: 1, 2: 1, 3: 1, 4: 10},
        engine="steinerpy",
        solver_kwargs={"exact": False},
    )
    np.testing.assert_array_equal(indices, [1, 2, 3])
    assert result.objective == pytest.approx(3)


@pytest.mark.parametrize(
    ("objective", "edges"),
    [(np.inf, [("A", "D")]), (1, [("A", "B")])],
    ids=["no-incumbent", "disconnected-incumbent"],
)
def test_steinerpy_helper_rejects_unusable_solution(monkeypatch, objective, edges):
    pytest.importorskip("networkx")
    problem = Mock()
    problem.get_solution.return_value = SimpleNamespace(objective=objective, edges=edges)
    monkeypatch.setitem(sys.modules, "steinerpy", SimpleNamespace(Problem=Mock(return_value=problem)))
    with pytest.raises(ValueError, match="No solution"):
        solve_steiner_tree(_graph(EdgeType.UNDIRECTED), terminals=["A", "D"], engine="steinerpy")


def test_corneto_helper_rejects_missing_solver_values(monkeypatch):
    problem = Mock()
    problem.expr.with_flow.value = None
    method = Mock()
    method.build.return_value = problem
    monkeypatch.setattr("corneto.methods.steiner.SteinerTreeFlow", Mock(return_value=method))
    with pytest.raises(ValueError, match="No solution"):
        solve_steiner_tree(_graph(EdgeType.UNDIRECTED), terminals=["A", "D"])
