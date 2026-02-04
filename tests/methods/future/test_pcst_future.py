import os

import pytest

from corneto._data import GraphData
from corneto.methods.future.pcst import PrizeCollectingSteinerTree


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
