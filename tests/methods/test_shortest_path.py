"""Tests for single- and multi-condition shortest paths."""

import numpy as np

from corneto.graph import Graph
from corneto.methods.shortest_path import create_multisample_shortest_path


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
