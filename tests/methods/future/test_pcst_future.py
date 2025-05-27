import os

import pytest


@pytest.fixture
def directed_steiner():
    from corneto._data import GraphData

    data = GraphData.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "directed_steiner.zip",
        )
    )
    return data.graph, data.data
