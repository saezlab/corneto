import os

import numpy as np
import pytest

from corneto._data import Data
from corneto.backend import Backend, CvxpyBackend, PicosBackend
from corneto.graph import Graph
from corneto.methods.future.steiner import SteinerTreeFlow


@pytest.fixture
def undirected_steiner():
    G = Graph.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "undirected_steiner.pkl.xz",
        )
    )
    features = []
    for i, attr in enumerate(G.get_attr_edges()):
        features.append(dict(id=i, mapping="edge", value=attr.value))

    for v in G.V:
        attr = G.get_attr_vertex(v)
        if attr["terminal"]:
            features.append(dict(id=v, mapping="vertex", role="terminal"))

    D = Data.from_dict({"s1": {"features": features}})
    return G, D

def _run_steiner_test(undirected_steiner, backend, strict_acyclic=False):
    G, D = undirected_steiner
    m = SteinerTreeFlow(lambda_reg=0, strict_acyclic=strict_acyclic, backend=backend)
    #m.disable_structured_sparsity = True
    P = m.build(G, D)
    P.solve()
    sol = m.processed_graph.edge_subgraph(np.flatnonzero(P.expr.with_flow.value > 0.5))
    cost = sum([attr.get("value", 0) for attr in sol.get_attr_edges()])
    assert np.isclose(P.objectives[0].value, 45.00, atol=1e-2)
    assert np.isclose(cost, 45.00, atol=1e-2)


def test_undirected_steiner_single_sample(undirected_steiner, backend, request):
    if isinstance(backend, PicosBackend) and not request.config.getoption(
        "--run-optional"
    ):
        pytest.skip("PicosBackend with strict_acyclic is optional (use --run-optional)")
    _run_steiner_test(undirected_steiner, backend, strict_acyclic=False)


def test_undirected_steiner_single_sample_strict_acyclic(
    undirected_steiner, backend, request
):
    if isinstance(backend, PicosBackend) and not request.config.getoption(
        "--run-optional"
    ):
        pytest.skip("PicosBackend with strict_acyclic is optional (use --run-optional)")
    _run_steiner_test(undirected_steiner, backend, strict_acyclic=True)
