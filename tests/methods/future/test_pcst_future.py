import os

import numpy as np
import pytest

from corneto._data import Data
from corneto.backend import Backend, CvxpyBackend, PicosBackend
from corneto.graph import Graph
from corneto.methods.future.pcst import PrizeCollectingSteinerTree as PCST



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