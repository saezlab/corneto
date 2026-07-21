import os

import pytest

from corneto.backend import PicosBackend
from corneto.data import Data, GraphData
from corneto.graph import Graph
from corneto.methods.pcst import PrizeCollectingSteinerTree
from corneto.methods.steiner import SteinerTreeFlow


@pytest.fixture
def directed_steiner():
    data = GraphData.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "directed_steiner.zip",
        )
    )
    return data.graph, data.data


def test_pcst_directed(directed_steiner, backend):
    if backend is None:
        pytest.skip("Backend is required")

    G, D = directed_steiner
    # PCST test:
    # We use the same graph as steiner, but we can play with prizes.

    # Create an instance
    pcst = PrizeCollectingSteinerTree(
        # We don't force include all terminals, so it's a PCST
        include_all_terminals=False,
        lambda_reg=0.0,
        backend=backend,
    )

    # Run build/solve
    problem = pcst.build(G, D)
    problem.solve()

    # Just check it solved and returned something.
    assert problem.objectives[0].value is not None


@pytest.mark.parametrize("method_cls", [PrizeCollectingSteinerTree, SteinerTreeFlow])
def test_edge_feature_id_sparse_breaks_cost_indexing(method_cls, backend, request):
    if isinstance(backend, PicosBackend) and not request.config.getoption("--run-optional"):
        pytest.skip("PicosBackend is optional (use --run-optional)")

    G = Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")

    # Edge feature id is intentionally sparse/out-of-range with respect to the
    # base graph edge vector (len=2), which currently crashes during build().
    D = Data.from_dict(
        {
            "s1": {
                "features": [
                    {"id": "A", "mapping": "vertex", "role": "terminal"},
                    {"id": "C", "mapping": "vertex", "role": "terminal"},
                    {"id": 10, "mapping": "edge", "value": 3.0},
                ]
            }
        }
    )

    method = method_cls(backend=backend, strict_acyclic=False)
    with pytest.raises(ValueError, match="Invalid edge feature id"):
        method.build(G, D)
