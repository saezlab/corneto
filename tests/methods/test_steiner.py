import pytest
import pathlib
from corneto.methods.steiner import exact_steiner_tree
from corneto.backend import PicosBackend, CvxpyBackend
from corneto._graph import BaseGraph
from corneto.backend import Backend
import numpy as np


@pytest.fixture(params=[CvxpyBackend, PicosBackend])
def backend(request):
    K: Backend = request.param()
    # TODO: Unify solver names
    if isinstance(K, CvxpyBackend):
        K._default_solver = "GLPK_MI"
    elif isinstance(K, PicosBackend):
        K._default_solver = "glpk"
    return K


@pytest.fixture
def steiner_graph() -> BaseGraph:
    file = pathlib.Path(__file__).parent.joinpath("test_steiner_graph.pkl.gz")
    return BaseGraph.load(str(file))


def test_steiner(backend, steiner_graph):
    terminals = [2, 6, 21, 23, 1, 7]
    P, _ = exact_steiner_tree(steiner_graph, terminals, backend=backend)
    P.solve(verbosity=1)
    assert np.isclose(P.objectives[0].value, 36.0)
