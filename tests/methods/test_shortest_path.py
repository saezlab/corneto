"""Tests for single- and multi-condition shortest paths."""

import numpy as np
import pytest

from corneto._constants import VarType
from corneto.graph import EdgeType, Graph
from corneto.methods.shortest_path import (
    create_multisample_shortest_path,
    shortest_path,
    solve_shortest_path,
)

EDGE_CASES = [
    pytest.param(EdgeType.DIRECTED, False, id="directed"),
    pytest.param(EdgeType.UNDIRECTED, False, id="undirected-forward"),
    pytest.param(EdgeType.UNDIRECTED, True, id="undirected-reverse"),
    pytest.param(None, False, id="mixed-forward"),
    pytest.param(None, True, id="mixed-reverse"),
]


def _weighted_graph(edge_type, reverse):
    """Offer a cheap three-edge path and more expensive shorter alternatives."""
    graph = Graph()
    edges = [("A", "B", 8), ("A", "C", 1), ("C", "B", 1), ("B", "D", 1), ("A", "D", 9)]
    for index, (source, target, weight) in enumerate(edges):
        kind = edge_type if edge_type is not None else EdgeType.DIRECTED
        if edge_type is None and index == 2:
            kind = EdgeType.UNDIRECTED
        if index == 2 and reverse:
            source, target = target, source
            kind = EdgeType.UNDIRECTED
        graph.add_edge(source, target, type=kind, weight=weight)
    # A backwards directed shortcut must not become traversable in a mixed graph.
    if edge_type is None:
        graph.add_edge("D", "A", type=EdgeType.DIRECTED, weight=0.1)
    return graph


def _assert_cost(problem, graph, expected, weights=None):
    """Check traversal costs using original edges, including reverse traversal."""
    assert problem.expr.flow.value is not None
    flow = np.asarray(problem.expr.flow.value).reshape(graph.num_edges, -1)
    if weights is None:
        weights = np.array([graph.get_attr_edge(i).get("weight", 0) for i in range(graph.num_edges)])
    weights = np.asarray(weights)
    if weights.ndim == 1:
        weights = weights[:, None]
    assert np.allclose((np.abs(flow) * weights).sum(axis=0), expected)


@pytest.mark.parametrize(("edge_type", "reverse"), EDGE_CASES)
@pytest.mark.parametrize("integral_path", [False, True], ids=["lp", "integral"])
def test_shortest_path_weighted_edge_types(backend, edge_type, reverse, integral_path):
    graph = _weighted_graph(edge_type, reverse)
    original = graph.to_dict()
    problem, flow_graph = shortest_path(graph, "A", "D", integral_path=integral_path, backend=backend)
    problem.solve()

    _assert_cost(problem, flow_graph, [3])
    assert np.isclose(problem.objectives[0].value, 3)
    assert graph.to_dict() == original
    assert flow_graph.E[: graph.num_edges] == graph.E


@pytest.mark.parametrize(("edge_type", "reverse"), EDGE_CASES)
def test_solve_shortest_path_selects_original_edges(backend, edge_type, reverse):
    graph = _weighted_graph(edge_type, reverse)
    edges, problem, flow_graph = solve_shortest_path(graph, "A", "D", backend=backend)

    _assert_cost(problem, flow_graph, [3])
    assert set(edges[edges < graph.num_edges]) == {1, 2, 3}


@pytest.mark.parametrize(("edge_type", "reverse"), EDGE_CASES)
def test_multisample_shortest_path_weighted_edge_types(backend, edge_type, reverse):
    graph = _weighted_graph(edge_type, reverse)
    original = graph.to_dict()
    problem, flow_graph = create_multisample_shortest_path(graph, [("A", "D"), ("C", "D")], backend=backend)
    problem.solve()

    _assert_cost(problem, flow_graph, [3, 2])
    assert np.allclose([objective.value for objective in problem.objectives], [3, 2])
    assert graph.to_dict() == original
    assert flow_graph.E[: graph.num_edges] == graph.E


@pytest.mark.parametrize("method", ["builder", "solve", "multisample"])
def test_shortest_path_custom_weights_override_attributes(backend, method):
    graph = _weighted_graph(EdgeType.DIRECTED, False)
    # Attribute weights favor A-C-B-D; explicit weights favor A-B-D.
    weights = np.array([1, 8, 8, 1, 9])
    if method == "multisample":
        problem, flow_graph = create_multisample_shortest_path(
            graph,
            [("A", "D"), ("C", "D")],
            edge_weights=np.stack([weights, weights]),
            backend=backend,
        )
        problem.solve()
        expected = [2, 9]
    elif method == "builder":
        problem, flow_graph = shortest_path(graph, "A", "D", edge_weights=weights, integral_path=False, backend=backend)
        problem.solve()
        expected = [2]
    else:
        edges, problem, flow_graph = solve_shortest_path(graph, "A", "D", edge_weights=weights, backend=backend)
        assert set(edges[edges < graph.num_edges]) == {0, 3}
        expected = [2]
    padded_weights = np.pad(weights, (0, flow_graph.num_edges - graph.num_edges))
    _assert_cost(problem, flow_graph, expected, padded_weights)


def test_multisample_shortest_path_uses_condition_specific_weights(backend):
    graph = _weighted_graph(EdgeType.DIRECTED, False)
    weights = np.array([[8, 1, 1, 1, 9], [1, 8, 8, 1, 9]])
    problem, flow_graph = create_multisample_shortest_path(
        graph, [("A", "D"), ("A", "D")], edge_weights=weights, backend=backend
    )
    problem.solve()

    padded_weights = np.pad(weights, ((0, 0), (0, flow_graph.num_edges - graph.num_edges)))
    _assert_cost(problem, flow_graph, [3, 2], padded_weights.T)
    flow = np.asarray(problem.expr.flow.value)
    assert set(np.flatnonzero(np.abs(flow[: graph.num_edges, 0]) > 0.5)) == {1, 2, 3}
    assert set(np.flatnonzero(np.abs(flow[: graph.num_edges, 1]) > 0.5)) == {0, 3}


def test_multisample_shortest_path_accepts_one_condition(backend):
    graph = _weighted_graph(EdgeType.DIRECTED, False)
    problem, flow_graph = create_multisample_shortest_path(graph, [("A", "D")], backend=backend)
    problem.solve()

    _assert_cost(problem, flow_graph, [3])


def _reported_graph(reverse_middle):
    graph = Graph()
    middle = ("D", "F") if reverse_middle else ("F", "D")
    for source, target in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("E", "F"), middle, ("F", "G")]:
        graph.add_edge(source, target, type=EdgeType.UNDIRECTED, weight=1)
    return graph


@pytest.mark.parametrize(
    ("source", "target", "expected", "reverse_middle"),
    [
        pytest.param("A", "B", 1, False, id="FD-A-B"),
        pytest.param("A", "F", 3, False, id="FD-A-F"),
        pytest.param("A", "D", 2, False, id="FD-A-D"),
        pytest.param("B", "F", 2, False, id="FD-B-F"),
        pytest.param("B", "D", 1, False, id="FD-B-D"),
        pytest.param("F", "D", 1, False, id="FD-F-D"),
        pytest.param("A", "B", 1, True, id="DF-A-B"),
        pytest.param("A", "F", 3, True, id="DF-A-F"),
        pytest.param("A", "D", 2, True, id="DF-A-D"),
        pytest.param("B", "F", 2, True, id="DF-B-F"),
        pytest.param("B", "D", 1, True, id="DF-B-D"),
        pytest.param("F", "D", 1, True, id="DF-F-D"),
    ],
)
def test_solve_shortest_path_reported_endpoint_ordering(backend, source, target, expected, reverse_middle):
    graph = _reported_graph(reverse_middle)
    edges, problem, flow_graph = solve_shortest_path(graph, source, target, backend=backend)

    _assert_cost(problem, flow_graph, [expected])
    # Both equally short A-F paths are valid; do not depend on the solver's tie breaking.
    assert len(edges[edges < graph.num_edges]) == expected


@pytest.mark.parametrize("reverse_middle", [False, True], ids=["FD", "DF"])
def test_multisample_shortest_path_opposite_traversals(backend, reverse_middle):
    graph = _reported_graph(reverse_middle)
    problem, flow_graph = create_multisample_shortest_path(graph, [("D", "F"), ("F", "D")], backend=backend)
    problem.solve()

    _assert_cost(problem, flow_graph, [1, 1])
    flow = np.asarray(problem.expr.flow.value)
    assert np.allclose(flow[5, :], [1, -1] if reverse_middle else [-1, 1])


def test_solve_shortest_path_unreachable_target(backend):
    graph = Graph()
    graph.add_edge("A", "B", weight=1)
    graph.add_vertex("C")

    with pytest.raises(ValueError, match="No solution"):
        solve_shortest_path(graph, "A", "C", backend=backend)


def test_solve_shortest_path_missing_solver_values(backend, monkeypatch):
    graph = Graph()
    graph.add_edge("A", "B", weight=1)
    # An interrupted solve can leave values unset even on a connected graph.
    monkeypatch.setattr(backend, "_solve", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="No solution"):
        solve_shortest_path(graph, "A", "B", backend=backend)


def test_solve_shortest_path_identical_terminals(backend):
    graph = Graph()
    graph.add_vertex("A")
    edges, problem, flow_graph = solve_shortest_path(graph, "A", "A", backend=backend)

    assert set(edges) == {0, 1}  # Only the two boundary edges; the biological path is empty.
    _assert_cost(problem, flow_graph, [0])


@pytest.mark.parametrize("method", ["single", "single-default", "multisample"])
def test_unregularized_shortest_path_has_no_integer_variables(backend, method):
    graph = _weighted_graph(EdgeType.DIRECTED, False)
    if method == "single":
        problem, _ = shortest_path(graph, "A", "D", integral_path=False, backend=backend)
    elif method == "single-default":
        problem, _ = shortest_path(graph, "A", "D", backend=backend)
    else:
        problem, _ = create_multisample_shortest_path(graph, [("A", "D"), ("C", "D")], backend=backend)

    assert all(symbol._vartype == VarType.CONTINUOUS for symbol in problem.symbols.values())


@pytest.mark.parametrize("zero_cost", [False, True], ids=["weighted-tie", "zero-cost-tie"])
def test_solve_shortest_path_extracts_fractional_tie_without_integer_solve(backend, monkeypatch, zero_cost):
    graph = Graph()
    for source, target in [("C", "A"), ("C", "D"), ("A", "B"), ("B", "D")]:
        graph.add_edge(source, target, type=EdgeType.UNDIRECTED, weight=10)
    weights = np.zeros(4) if zero_cost else np.array([2, 1, 1, 2])
    solve = backend._solve
    calls = []

    def solve_fractional_optimum(problem, **kwargs):
        calls.append(problem)
        assert all(symbol._vartype == VarType.CONTINUOUS for symbol in problem.symbols.values())
        # Force a feasible mixture of the two optimal paths rather than relying
        # on a particular solver algorithm or its tie breaking.
        problem += problem.expr.flow[:4] == np.array([-0.5, 0.5, 0.5, 0.5])
        return solve(problem, **kwargs)

    monkeypatch.setattr(backend, "_solve", solve_fractional_optimum)
    edges, problem, flow_graph = solve_shortest_path(graph, "A", "D", edge_weights=weights, backend=backend)

    assert len(calls) == 1
    assert set(edges[edges < graph.num_edges]) in ({0, 1}, {2, 3})
    assert np.allclose(np.asarray(problem.expr.flow.value).reshape(-1)[:4], [-0.5, 0.5, 0.5, 0.5])
    assert np.isclose(weights[edges[edges < graph.num_edges]].sum(), 0 if zero_cost else 3)
    assert flow_graph.num_edges == graph.num_edges + 2


def test_solve_shortest_path_discards_zero_cost_circulation(backend, monkeypatch):
    graph = Graph()
    for source, target, weight in [("A", "B", 1), ("B", "D", 1), ("B", "C", 0), ("C", "B", 0)]:
        graph.add_edge(source, target, weight=weight)
    solve = backend._solve

    def solve_with_circulation(problem, **kwargs):
        problem += problem.expr.flow[:4] == np.array([1, 1, 0.5, 0.5])
        return solve(problem, **kwargs)

    monkeypatch.setattr(backend, "_solve", solve_with_circulation)
    edges, problem, flow_graph = solve_shortest_path(graph, "A", "D", backend=backend)

    assert set(edges[edges < graph.num_edges]) == {0, 1}
    _assert_cost(problem, flow_graph, [2])


def test_shortest_path_reuses_augmented_graph_with_custom_weights(backend):
    graph = _weighted_graph(None, True)
    _, flow_graph = shortest_path(graph, "A", "D", backend=backend)
    # Select the path that requires reverse traversal, overriding attributes.
    weights = np.array([8, 1, 1, 1, 9, 0.1, 0, 0])
    problem, reused = shortest_path(
        flow_graph, "A", "D", edge_weights=weights, create_flow_graph=False, backend=backend
    )
    problem.solve()

    assert reused is flow_graph
    _assert_cost(problem, flow_graph, [3], weights)


@pytest.mark.parametrize("method", ["single", "multisample"])
@pytest.mark.parametrize("weight", [-1, np.inf, np.nan])
def test_shortest_path_rejects_invalid_costs(backend, method, weight):
    graph = Graph()
    graph.add_edge("A", "B", weight=weight)
    with pytest.raises(ValueError, match="finite, nonnegative"):
        if method == "single":
            shortest_path(graph, "A", "B", backend=backend)
        else:
            create_multisample_shortest_path(graph, [("A", "B")], backend=backend)


def test_multisample_shortest_path_l0_penalty_shares_edges(backend):
    graph = Graph()
    for source, target, weight in [
        ("A", "X", 1),
        ("X", "D", 1),
        ("A", "Y", 1.5),
        ("Y", "D", 1),
        ("B", "Y", 1),
        ("B", "X", 1.5),
    ]:
        graph.add_edge(source, target, weight=weight)

    independent, _ = create_multisample_shortest_path(graph, [("A", "D"), ("B", "D")], backend=backend)
    independent.solve()
    independent_flow = np.asarray(independent.expr.flow.value)[: graph.num_edges, :]
    assert np.count_nonzero(np.any(np.abs(independent_flow) > 0.5, axis=1)) == 4

    shared, _ = create_multisample_shortest_path(graph, [("A", "D"), ("B", "D")], lam=2, backend=backend)
    shared.solve()
    shared_flow = np.asarray(shared.expr.flow.value)[: graph.num_edges, :]
    used = np.any(np.abs(shared_flow) > 0.5, axis=1)
    assert np.count_nonzero(used) == 3
    assert np.allclose(np.asarray(shared.expr.active_edge.value).reshape(-1)[: graph.num_edges], used)
    weights = np.array([graph.get_attr_edge(i)["weight"] for i in range(graph.num_edges)])
    assert np.isclose((np.abs(shared_flow) * weights[:, None]).sum(), 4.5)


def test_multisample_shortest_path_l0_counts_opposite_traversals_once(backend):
    graph = Graph()
    graph.add_edge("A", "B", type=EdgeType.UNDIRECTED, weight=1)
    problem, flow_graph = create_multisample_shortest_path(graph, [("A", "B"), ("B", "A")], lam=1, backend=backend)
    problem.solve()

    _assert_cost(problem, flow_graph, [1, 1])
    assert np.allclose(np.asarray(problem.expr.flow.value)[0, :], [1, -1])
    assert np.isclose(np.asarray(problem.expr.active_edge.value).reshape(-1)[0], 1)


def test_multisample_shortest_path_isolates_condition_terminals(backend):
    """Each flow must use only the source and sink assigned to its condition."""
    graph = Graph.from_tuples([("A", 1, "B"), ("C", 1, "D")])

    problem, flow_graph = create_multisample_shortest_path(
        graph,
        [("A", "B"), ("C", "D")],
        backend=backend,
    )
    result = problem.solve()

    flow = np.asarray(problem.expr.flow.value)
    artificial_edges = {
        (tuple(source), tuple(target)): index
        for index, (source, target) in enumerate(flow_graph.E)
        if not source or not target
    }

    assert result.status == "optimal"
    assert flow.shape == (6, 2)
    assert np.isclose(flow[artificial_edges[((), ("A",))], 0], 1)
    assert np.isclose(flow[artificial_edges[(("B",), ())], 0], 1)
    assert np.isclose(flow[artificial_edges[((), ("C",))], 0], 0)
    assert np.isclose(flow[artificial_edges[(("D",), ())], 0], 0)
    assert np.isclose(flow[artificial_edges[((), ("A",))], 1], 0)
    assert np.isclose(flow[artificial_edges[(("B",), ())], 1], 0)
    assert np.isclose(flow[artificial_edges[((), ("C",))], 1], 1)
    assert np.isclose(flow[artificial_edges[(("D",), ())], 1], 1)
