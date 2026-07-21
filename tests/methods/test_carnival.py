"""Tests for the retained public CARNIVAL implementations."""

import numpy as np

from corneto.data import Data
from corneto.graph import Graph
from corneto.methods import milp_carnival
from corneto.methods.carnival import CarnivalILP
from corneto.methods.carnival import milp_carnival as module_milp_carnival


def test_milp_carnival_is_public_and_builds(backend):
    """The documented compatibility formulation remains a supported entry point."""
    graph = Graph.from_tuples([("input", 1, "output")])

    problem = milp_carnival(
        graph,
        {"input": 1},
        {"output": 1},
        beta_weight=1e-3,
        backend=backend,
    )

    assert milp_carnival is module_milp_carnival
    assert "vertex_values" in problem.expr
    assert "edge_values" in problem.expr


def test_carnival_ilp_supports_multiple_conditions(backend):
    """Each sample gets an independent signaling state on the shared graph."""
    graph = Graph.from_tuples([("input", 1, "output")])
    data = Data.from_cdict(
        {
            "activated": {
                "input": {"value": 1, "role": "input", "mapping": "vertex"},
                "output": {"value": 1, "role": "output", "mapping": "vertex"},
            },
            "inhibited": {
                "input": {"value": -1, "role": "input", "mapping": "vertex"},
                "output": {"value": -1, "role": "output", "mapping": "vertex"},
            },
        }
    )

    method = CarnivalILP(beta_weight=0, backend=backend)
    problem = method.build(graph, data)
    result = problem.solve()

    vertex_values = np.asarray(problem.expr.vertex_values.value)
    input_index = method.processed_graph.V.index("input")
    output_index = method.processed_graph.V.index("output")

    assert result.status == "optimal"
    assert problem.expr.vertex_values.shape == (2, 2)
    assert np.allclose(vertex_values[input_index], [1, -1])
    assert np.allclose(vertex_values[output_index], [1, -1])
