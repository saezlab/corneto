import pathlib

import cvxpy as cp
import numpy as np
import pytest

from corneto._graph import Graph
from corneto.backend import Backend, CvxpyBackend, PicosBackend, VarType


@pytest.fixture(params=[CvxpyBackend, PicosBackend])
def backend(request):
    K: Backend = request.param()
    if isinstance(K, CvxpyBackend):
        K._default_solver = "SCIPY"
    elif isinstance(K, PicosBackend):
        K._default_solver = "glpk"
    return K


@pytest.fixture
def mitocore_small():
    from corneto._io import _load_compressed_gem

    file = pathlib.Path(__file__).parent.joinpath("gem", "mitocore_small.xz")
    S, R, M = _load_compressed_gem(file)
    return S, R, M


def test_picos_convex():
    backend = PicosBackend()
    P = backend.Problem()
    A = np.array([[0.12, 0.92, 0.76, 0.98, 0.79], [0.58, 0.57, 0.53, 0.71, 0.55]])
    b = np.array([1, 0])
    x = backend.Variable("x", (A.shape[1],))
    P += sum(x) == 1, x >= 0
    # Convex optimization problem
    P.add_objectives(abs(A @ x - b), inplace=True)
    P.solve(solver="cvxopt", verbosity=1)
    assert np.all(np.array(x.value).T < np.array([1e-6, 0.64, 0.37, 1e-6, 1e-6]))
    assert np.all(np.array(x.value).T > np.array([-1e-6, 0.62, 0.36, -1e-6, -1e-6]))


def test_cvxpy_convex():
    backend = CvxpyBackend()
    P = backend.Problem()
    A = np.array([[0.12, 0.92, 0.76, 0.98, 0.79], [0.58, 0.57, 0.53, 0.71, 0.55]])
    b = np.array([1, 0])
    x = backend.Variable("x", (A.shape[1],))
    P += sum(x) == 1, x >= 0  # type: ignore
    # Convex optimization problem
    P.add_objectives(
        cp.sum_squares((A @ x - b).e), inplace=True
    )  # TODO: add sum squares
    P.solve(solver="osqp", verbosity=1)
    assert np.all(np.array(x.value) < np.array([1e-6, 0.64, 0.37, 1e-6, 1e-6]))
    assert np.all(np.array(x.value) > np.array([-1e-6, 0.62, 0.36, -1e-6, -1e-6]))


def test_cvxpy_convex_apply():
    backend = CvxpyBackend()
    P = backend.Problem()
    A = np.array([[0.12, 0.92, 0.76, 0.98, 0.79], [0.58, 0.57, 0.53, 0.71, 0.55]])
    b = np.array([1, 0])
    x = backend.Variable("x", (A.shape[1],))
    P += sum(x) == 1, x >= 0  # type: ignore
    # Convex optimization problem
    P.add_objectives((A @ x - b).apply(cp.sum_squares), inplace=True)
    P.solve(solver="osqp", verbosity=1)
    assert np.all(np.array(x.value) < np.array([1e-6, 0.64, 0.37, 1e-6, 1e-6]))
    assert np.all(np.array(x.value) > np.array([-1e-6, 0.62, 0.36, -1e-6, -1e-6]))


def test_delegate_multiply_shape(backend):
    V = backend.Variable(shape=(2, 3))
    V = V.multiply(np.ones((2, 3)))
    assert V.shape == (2, 3)


def test_delegate_sum_axis0_shape(backend):
    V = backend.Variable(shape=(2, 3))
    V = V.sum(axis=0)
    if len(V.shape) == 1:
        # Cvxpy assumes keepdims=False
        assert V.shape == (3,)
    else:
        # Picos assumes keepdims=True
        assert V.shape == (1, 3)


def test_delegate_sum_axis1_shape(backend):
    V = backend.Variable(shape=(2, 3))
    V = V.sum(axis=1)
    if len(V.shape) == 1:
        # Cvxpy assumes keepdims=False
        assert V.shape == (2,)
    else:
        # Picos assumes keepdims=True
        assert V.shape == (2, 1)


def test_opt_delegate_sum_axis0(backend):
    x = backend.Variable("x", (2, 3))
    e = x.sum(axis=0)
    P = backend.Problem()
    P += x <= 10
    esum = e[0] + e[1] + e[2]
    P.add_objectives(esum, weights=-1)
    P.solve()
    assert np.isclose(esum.value, 60)


def test_opt_delegate_sum_axis1(backend):
    x = backend.Variable("x", (2, 3))
    e = x.sum(axis=1)
    P = backend.Problem()
    P += x <= 10
    esum = e[0] + e[1]
    P.add_objectives(esum, weights=-1)
    P.solve()
    assert np.isclose(esum.value, 60)


def test_cexpression_name(backend):
    x = backend.Variable("x")
    e = x <= 10
    e._name = "x_leq_10"
    assert e.name == "x_leq_10"


def test_lb_symbol(backend):
    x = backend.Variable("x", lb=-10, ub=10)
    P = backend.Problem()
    P.add_objectives(2 * x, inplace=True)
    P.solve()
    assert x.value is not None
    assert np.isclose(x.value, -10)


def test_ub_symbol(backend):
    x = backend.Variable("x", lb=-10, ub=10)
    P = backend.Problem()
    P.add_objectives(-2 * x, inplace=True)
    P.solve()
    assert x.value is not None
    assert np.isclose(x.value, 10)


def test_expr_symbols(backend):
    x = backend.Variable("x", lb=-10, ub=10)
    y = backend.Variable("y", lb=10, ub=100)
    z = backend.Variable("z", lb=-5, ub=5)
    e0 = x + y >= 10
    assert e0._proxy_symbols == {x, y}
    e1 = x + x + 2 * y
    assert e1._proxy_symbols == {x, y}
    e2 = 2 + x + 1 + y
    assert e2._proxy_symbols == {x, y}
    e3 = (2 * x + y * 2) / 2 + z * 2
    assert e3._proxy_symbols == {x, y, z}
    e4 = e3 + e2 <= 10
    assert e4._proxy_symbols == {x, y, z}
    e5 = e1 + 2 * e3
    assert e5._proxy_symbols == {x, y, z}


def test_register(backend):
    P = backend.Problem()
    x = backend.Variable("x", lb=-10, ub=10)
    P += x >= 0
    P.register("1-x", 1 - x)
    assert "1-x" in P.expressions


def test_register_merge(backend):
    P1 = backend.Problem()
    x = backend.Variable("x", lb=-10, ub=10)
    P1 += x >= 0
    P1.register("1-x", 1 - x)
    P2 = backend.Problem()
    y = backend.Variable("y", lb=-10, ub=10)
    P2 += y >= 0
    P2.register("1-y", 1 - y)
    P = P1.merge(P2)
    assert "1-x" in P.expressions
    assert "1-y" in P.expressions


def test_symbol_only_in_objective(backend):
    x = backend.Variable("x", lb=-10, ub=10)
    P = backend.Problem()
    P.add_objectives(x, inplace=True)
    assert "x" in P.symbols


def test_cvxpy_custom_expr_symbols():
    backend = CvxpyBackend()
    P = backend.Problem()
    A = np.zeros((2, 5))
    b = np.array([1, 0])
    x = backend.Variable("x", (A.shape[1],))
    P.add_objectives((A @ x - b).apply(cp.sum_squares), inplace=True)
    assert "x" in P.symbols


def test_picos_custom_expr_symbols():
    backend = PicosBackend()
    P = backend.Problem()
    A = np.zeros((2, 5))
    b = np.array([1, 0])
    x = backend.Variable("x", (A.shape[1],))
    P.add_objectives(abs(A @ x - b), inplace=True)
    assert "x" in P.symbols


def test_matmul_symbols(backend):
    P = backend.Problem()
    A = np.zeros((5, 5))
    x = backend.Variable("x", A.shape)
    P.add_objectives(A @ x, inplace=True)
    assert "x" in P.symbols


def test_rmatmul_symbols(backend):
    P = backend.Problem()
    A = np.zeros((5, 5))
    x = backend.Variable("x", A.shape)
    P.add_objectives(x @ A, inplace=True)
    assert "x" in P.symbols


def test_linearized_or_axis0(backend):
    P = backend.Problem()
    X = backend.Variable("X", (2, 3), vartype=VarType.BINARY)
    P += backend.linear_or(X, axis=0, varname="v_or")
    # Force X to have at least a 1 in the first column
    P += P.expr.v_or[0] == 1
    P.add_objectives(sum(X[:, 0]))
    P.solve()
    assert np.isclose(np.sum(X[:, 0].value), 1.0)


def test_linearized_or_axis1(backend):
    P = backend.Problem()
    X = backend.Variable("X", (2, 3), vartype=VarType.BINARY)
    P += backend.linear_or(X, axis=1, varname="v_or")
    # Force X to have at least a 1 in the first row
    P += P.expr.v_or[0] == 1
    P.add_objectives(sum(X[0, :]))
    P.solve()
    assert np.isclose(np.sum(X[0, :].value), 1.0)


def test_linearized_and_axis0(backend):
    P = backend.Problem()
    X = backend.Variable("X", (2, 3), vartype=VarType.BINARY)
    P += backend.linear_and(X, axis=0, varname="v_and")
    P += P.expr.v_and[0] == 1
    P.add_objectives(sum(X[:, 0]))
    P.solve()
    assert np.isclose(np.sum(X[:, 0].value), 2.0)


def test_linearized_and_axis1(backend):
    P = backend.Problem()
    X = backend.Variable("X", (2, 3), vartype=VarType.BINARY)
    P += backend.linear_and(X, axis=1, varname="v_and")
    P += P.expr.v_and[0] == 1
    P.add_objectives(sum(X[0, :]))
    P.solve()
    assert np.isclose(np.sum(X[0, :].value), 3.0)


def test_undirected_flow(backend):
    g = Graph()
    g.add_edges([((), "A"), ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", ())])
    P = backend.Flow(g, lb=-1000, ub=1000)
    P += P.expr.flow[1] >= 10
    P += P.expr.flow[2] >= -10
    P.add_objectives(sum(P.expr.flow), weights=1)
    P.solve()
    assert np.isclose(P.objectives[0].value, 0)


def test_undirected_flow_unbounded(backend):
    from corneto._graph import Graph

    g = Graph()
    g.add_edges([((), "A"), ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", ())])
    P = backend.Flow(g, lb=None, ub=None)
    P += P.expr.flow[1] >= 10
    P += P.expr.flow[2] >= -10
    P.add_objectives(sum(P.expr.flow), weights=1)
    P.solve()
    assert np.isclose(P.objectives[0].value, 0)


def test_fba_flow(backend, mitocore_small):
    S, R, M = mitocore_small
    reaction_id = np.flatnonzero(R["id"] == "EX_biomass_e")[0]
    G = Graph.from_vertex_incidence(S, M["id"], R["id"])
    P = backend.Flow(G, lb=R["lb"], ub=R["ub"], values=True, n_flows=1)
    obj = P.expr.flow[reaction_id]
    P.add_objectives(-obj)  # maximize
    P.solve()
    assert np.isclose(np.round(obj.value, 3), 100.257)


def test_fba_flow_with_weighted_obj(backend, mitocore_small):
    S, R, M = mitocore_small
    reaction_id = np.flatnonzero(R["id"] == "EX_biomass_e")[0]
    G = Graph.from_vertex_incidence(S, M["id"], R["id"])
    P = backend.Flow(G, lb=R["lb"], ub=R["ub"], values=True, n_flows=1)
    obj = P.expr.flow[reaction_id]
    P.add_objectives(obj, weights=-1)  # maximize
    P.solve()
    assert np.isclose(np.round(obj.value, 3), 100.257)


def test_fba_multiflow(backend, mitocore_small):
    S, R, M = mitocore_small
    reaction_id = np.flatnonzero(R["id"] == "EX_biomass_e")[0]
    G = Graph.from_vertex_incidence(S, M["id"], R["id"])
    P = backend.Flow(G, lb=R["lb"], ub=R["ub"], values=True, n_flows=2)
    obj = P.expr.flow[reaction_id]
    P.add_objectives(-sum(obj))
    P.solve()
    assert np.allclose(np.round(obj.value, 3), [100.257, 100.257])


def test_acyclic_flow_directed_graph(backend):
    G = Graph.from_sif_tuples(
        [
            ("v1", 1, "v2"),
            ("v2", 1, "v2"),
            ("v2", 1, "v3"),
            ("v3", -1, "v1"),
            ("v1", -1, "v2"),
            ("v2", 1, "v4"),
            ("v4", -1, "v3"),
            ("v4", 1, "v5"),
            ("v5", 1, "v3"),
            ("v5", -1, "v6"),
            ("v3", 1, "v5"),
            ("v3", 1, "v6"),
        ]
    )
    G.add_edge((), "v1")
    G.add_edge("v6", ())
    P = backend.AcyclicFlow(G, lb=0, ub=10)
    # P.add_objectives(-sum(P.get_symbol(VAR_FLOW + "_ipos")))
    P.add_objectives(-sum(P.expr.with_flow))
    solver = None
    P.solve(solver=solver)
    # Check that the maximum acyclic graph has no cycles
    # TODO: picos returns a column vector [[1],[0],...] homogeneise outputs
    # sol = np.round(P.get_symbol(VAR_FLOW + "_ipos").value).ravel()
    sol = np.round(P.expr.with_flow.value).ravel()
    assert np.allclose(
        sol, [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_acyclic_flow_undirected_edge(backend):
    G = Graph.from_sif_tuples(
        [
            ("v1", 1, "v2"),
            ("v2", 1, "v2"),
            ("v2", 1, "v3"),
            ("v3", -1, "v1"),
            ("v1", -1, "v2"),
            ("v2", 1, "v4"),
            ("v4", -1, "v3"),
            ("v4", 1, "v5"),
            ("v5", 1, "v3"),
            ("v5", -1, "v6"),
            ("v3", 1, "v5"),
            ("v3", 1, "v6"),
        ]
    )
    G.add_edge((), "v1")
    G.add_edge("v6", ())
    # TODO: fix lb/ub being a list instead of a numpy array
    lb = np.array([0] * G.ne)
    ub = np.array([10] * G.ne)
    lb[3] = -10  # reversible v3 <-> v1
    P = backend.AcyclicFlow(G, lb=lb, ub=ub)
    # obj = P.get_symbol(VAR_FLOW + "_ipos") + P.get_symbol(VAR_FLOW + "_ineg")
    P.add_objectives(-sum(P.expr.with_flow))
    solver = None
    if isinstance(backend, PicosBackend):
        # Picos choses ECOS solver for this...
        # TODO: overwrite solver preferences
        solver = "glpk"
    P.solve(solver=solver)
    sol = np.round(P.expr.with_flow.value).ravel()
    vsol1 = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    vsol2 = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert np.allclose(sol, vsol1) or np.allclose(sol, vsol2)


def test_feasible_loop(backend):
    G = Graph()
    G.add_edge((), "A")
    G.add_edge("A", ())
    G.add_edge("A", "B")
    G.add_edge("B", "A")
    G.add_edge((), "B")
    G.add_edge("B", ())
    P = backend.Flow(G)
    P += P.expr.flow[2] >= 1
    P += P.expr.flow[3] >= 1
    if isinstance(backend, PicosBackend):
        P.solve(solver="glpk")
    else:
        P.solve()
    assert np.sum(P.expr.flow.value) >= 2


def test_acyclic_unfeasible_loop(backend):
    G = Graph()
    G.add_edge((), "A")
    G.add_edge("A", ())
    G.add_edge("A", "B")
    G.add_edge("B", "A")
    G.add_edge((), "B")
    G.add_edge("B", ())
    P = backend.AcyclicFlow(G)
    P += P.expr.with_flow[2] == 1
    P += P.expr.with_flow[3] == 1
    if isinstance(backend, PicosBackend):
        P.solve(solver="glpk", primals=False)
    else:
        P.solve()
    assert np.all(P.expr.flow.value == None)


@pytest.mark.skip(reason="Only a small subset of solvers support this")
def test_l2_norm(backend):
    x = np.array([1, 2])
    y = np.array([3, 4])
    expected_result = np.linalg.norm(x - y)
    P = backend.Problem()
    diff = backend.Variable("diff", x.shape)
    n = diff.norm()
    P += [diff == x - y]
    P.add_objectives(n)
    P.solve()
    assert np.isclose(n.value, expected_result, rtol=1e-5)
