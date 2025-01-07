import numpy as np
import pytest

import corneto as cn
from corneto.backend import Backend, CvxpyBackend, PicosBackend
from corneto.methods.signal.cellnopt_ilp import cellnoptILP


# PICOS does not work
@pytest.fixture(params=[CvxpyBackend, PicosBackend])
def backend(request):
    opt: Backend = request.param()
    if isinstance(opt, CvxpyBackend):
        opt._default_solver = "SCIPY"
    elif isinstance(opt, PicosBackend):
        opt._default_solver = "gurobi"
    return opt


def get_test_graph_1():
    G1 = cn.Graph.from_sif_tuples(
        [
            ("EGF", 1, "AND1"),
            ("TNFa", 1, "AND1"),
            ("AND1", 1, "Ras"),
            ("EGF", 1, "Ras"),
            ("TNFa", 1, "Ras"),
        ]
    )
    G1.add_edge((), "EGF")
    G1.add_edge((), "TNFa")
    G1.add_edge("Ras", ())

    return G1

@pytest.mark.skip(reason="not compatible with picos")
def test_cellnoptILP_AND(backend):
    G1 = get_test_graph_1()

    # RAS is only active iff both EGF and TNFa are active -> we need to identify the AND gate
    exp_list_G1_and = {
        "exp0": {"input": {"EGF": 0, "TNFa": 0}, "output": {"Ras": 0}},
        "exp1": {"input": {"EGF": 1, "TNFa": 0}, "output": {"Ras": 0}},
        "exp2": {"input": {"EGF": 0, "TNFa": 1}, "output": {"Ras": 0}},
        "exp3": {"input": {"EGF": 1, "TNFa": 1}, "output": {"Ras": 1}},
    }

    P = cellnoptILP(
        G1, exp_list_G1_and, verbose=True, alpha_flow=0.001, backend=backend
    )
    expected_edge_values = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [-0.0, -0.0, -0.0, 1.0],
            [0.0, -0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # vertices do not have a specific order (set of vertices is a list, but the order is not fixed)
    expected_vertex_values = np.array(
        [
            [-0.0, -0.0, -0.0, 1.0],
            [-0.0, 1.0, -0.0, 1.0],
            [-0.0, -0.0, 1.0, 1.0],
            [0.0, -0.0, -0.0, 1.0],
        ]
    )
    obj = sum([o.value for o in P.objectives])
    assert np.isclose(sum([o.value for o in P.objectives]), 0.006)
    assert np.isclose(P.expr.edge_activates.value, expected_edge_values).all()
    assert np.isclose(
        np.sum(P.expr.vertex_value.value, axis=0), expected_vertex_values.sum(axis=0)
    ).all()

@pytest.mark.skip(reason="not compatible with picos")
def test_cellnoptILP_OR(backend):
    G1 = get_test_graph_1()

    # RAS is only active iff both EGF and TNFa are active -> we need to identify the AND gate
    exp_list_G1_or = {
        "exp0": {"input": {"EGF": 0, "TNFa": 0}, "output": {"Ras": 0}},
        "exp1": {"input": {"EGF": 1, "TNFa": 0}, "output": {"Ras": 1}},
        "exp2": {"input": {"EGF": 0, "TNFa": 1}, "output": {"Ras": 1}},
        "exp3": {"input": {"EGF": 1, "TNFa": 1}, "output": {"Ras": 1}},
    }

    P = cellnoptILP(G1, exp_list_G1_or, verbose=True, alpha_flow=0.001, backend=backend)
    expected_edge_values = np.array(
        [
            [0.0, -0.0, 0.0, -0.0],
            [0.0, 0.0, -0.0, -0.0],
            [-0.0, -0.0, -0.0, -0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )

    # vertices do not have a specific order (set of vertices is a list, but the order is not fixed)
    expected_vertex_values = np.array(
        [
            [-0.0, -0.0, -0.0, -0.0],
            [-0.0, 1.0, -0.0, 1.0],
            [-0.0, -0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )

    assert np.isclose(sum([o.value for o in P.objectives]), 0.005)
    assert np.isclose(P.expr.edge_activates.value, expected_edge_values).all()
    assert np.isclose(
        np.sum(P.expr.vertex_value.value, axis=0), expected_vertex_values.sum(axis=0)
    ).all()
