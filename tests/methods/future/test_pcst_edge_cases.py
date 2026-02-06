import numpy as np
import pytest

from corneto._data import Data
from corneto._graph import EdgeType
from corneto.backend import PicosBackend
from corneto.graph import Graph
from corneto.methods.future.steiner import SteinerTreeFlow


def _small_undirected_path_graph():
    g = Graph()
    g.add_edge("A", "B", type=EdgeType.UNDIRECTED, value=1.0)
    g.add_edge("B", "C", type=EdgeType.UNDIRECTED, value=1.0)
    return g


def _two_sample_terminal_data():
    return Data.from_dict(
        {
            "s1": {
                "features": [
                    {"id": "A", "mapping": "vertex", "role": "terminal"},
                    {"id": "C", "mapping": "vertex", "role": "terminal"},
                    {"id": 0, "mapping": "edge", "value": 1.0},
                    {"id": 1, "mapping": "edge", "value": 1.0},
                ]
            },
            "s2": {
                "features": [
                    {"id": "A", "mapping": "vertex", "role": "terminal"},
                    {"id": "C", "mapping": "vertex", "role": "terminal"},
                    {"id": 0, "mapping": "edge", "value": 1.0},
                    {"id": 1, "mapping": "edge", "value": 1.0},
                ]
            },
        }
    )


def test_steiner_mixed_none_and_fixed_roots_is_feasible(backend, request):
    if isinstance(backend, PicosBackend) and not request.config.getoption("--run-optional"):
        pytest.skip("PicosBackend is optional (use --run-optional)")

    g = _small_undirected_path_graph()
    d = _two_sample_terminal_data()

    # Sample 1 should choose best root, sample 2 uses fixed root.
    method = SteinerTreeFlow(
        root_vertex=[None, "A"],
        root_selection_strategy="best",
        strict_acyclic=False,
        lambda_reg=0.0,
        backend=backend,
    )
    problem = method.build(g, d)
    problem.solve()

    # Feasible expected: each sample should connect A<->C with cost 2.
    edge_cost_values = [float(np.asarray(o.value).reshape(-1)[0]) for o in problem.objectives if o.name == "edge_cost"]
    assert len(edge_cost_values) == 2
    assert np.isclose(sum(edge_cost_values), 4.0, atol=1e-6)


def test_steiner_different_fixed_roots_per_sample_is_feasible(backend, request):
    if isinstance(backend, PicosBackend) and not request.config.getoption("--run-optional"):
        pytest.skip("PicosBackend is optional (use --run-optional)")

    g = _small_undirected_path_graph()
    d = _two_sample_terminal_data()

    # Sample-specific roots should both be feasible on this tiny path.
    method = SteinerTreeFlow(
        root_vertex=["A", "C"],
        strict_acyclic=False,
        lambda_reg=0.0,
        backend=backend,
    )
    problem = method.build(g, d)
    problem.solve()

    edge_cost_values = [float(np.asarray(o.value).reshape(-1)[0]) for o in problem.objectives if o.name == "edge_cost"]
    assert len(edge_cost_values) == 2
    assert np.isclose(sum(edge_cost_values), 4.0, atol=1e-6)


def test_steiner_lambda_reg_promotes_edge_sharing_across_samples(backend, request):
    if isinstance(backend, PicosBackend) and not request.config.getoption("--run-optional"):
        pytest.skip("PicosBackend is optional (use --run-optional)")

    g = Graph()
    # Shared path: A-B-E
    # Sample-specific alternatives: A-C-E (sample 1), A-D-E (sample 2)
    edges = [("A", "B"), ("B", "E"), ("A", "C"), ("C", "E"), ("A", "D"), ("D", "E")]
    for u, v in edges:
        g.add_edge(u, v, type=EdgeType.UNDIRECTED, value=1.0)

    s1_features = [
        {"id": "A", "mapping": "vertex", "role": "terminal"},
        {"id": "E", "mapping": "vertex", "role": "terminal"},
        {"id": 0, "mapping": "edge", "value": 0.6},
        {"id": 1, "mapping": "edge", "value": 0.6},
        {"id": 2, "mapping": "edge", "value": 0.4},
        {"id": 3, "mapping": "edge", "value": 0.4},
        {"id": 4, "mapping": "edge", "value": 5.0},
        {"id": 5, "mapping": "edge", "value": 5.0},
    ]
    s2_features = [
        {"id": "A", "mapping": "vertex", "role": "terminal"},
        {"id": "E", "mapping": "vertex", "role": "terminal"},
        {"id": 0, "mapping": "edge", "value": 0.6},
        {"id": 1, "mapping": "edge", "value": 0.6},
        {"id": 2, "mapping": "edge", "value": 5.0},
        {"id": 3, "mapping": "edge", "value": 5.0},
        {"id": 4, "mapping": "edge", "value": 0.4},
        {"id": 5, "mapping": "edge", "value": 0.4},
    ]
    d = Data.from_dict({"s1": {"features": s1_features}, "s2": {"features": s2_features}})

    low_reg = SteinerTreeFlow(
        root_vertex="A",
        strict_acyclic=False,
        lambda_reg=0.0,
        backend=backend,
    )
    p_low = low_reg.build(g, d)
    p_low.solve()
    wf_low = np.array(p_low.expr.with_flow.value)[: g.num_edges, :]
    unique_low = np.flatnonzero(wf_low.max(axis=1) > 0.5)

    high_reg = SteinerTreeFlow(
        root_vertex="A",
        strict_acyclic=False,
        lambda_reg=2.0,
        backend=backend,
    )
    p_high = high_reg.build(g, d)
    p_high.solve()
    wf_high = np.array(p_high.expr.with_flow.value)[: g.num_edges, :]
    unique_high = np.flatnonzero(wf_high.max(axis=1) > 0.5)

    # With no coupling each sample picks its own cheapest path (4 unique edges).
    assert set(unique_low.tolist()) == {2, 3, 4, 5}
    # With strong coupling both samples share A-B-E (2 unique edges).
    assert set(unique_high.tolist()) == {0, 1}
    assert len(unique_high) < len(unique_low)
