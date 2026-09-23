import pathlib

import cvxpy as cp
import numpy as np
import pytest

from corneto.backend import CvxpyBackend, PicosBackend, VarType
from corneto.graph import Graph


@pytest.fixture
def mitocore_small():
    from corneto.io._metabolism import _load_compressed_gem

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
    P.add_objectives(cp.sum_squares((A @ x - b).e), inplace=True)  # TODO: add sum squares
    P.solve(solver="cvxopt", verbosity=1)
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
    P.solve(solver="CVXOPT", verbosity=1)
    assert np.all(np.array(x.value) < np.array([1e-6, 0.64, 0.37, 1e-6, 1e-6]))
    assert np.all(np.array(x.value) > np.array([-1e-6, 0.62, 0.36, -1e-6, -1e-6]))


def test_add_suffix(backend):
    A = backend.Variable("A", shape=(2, 3))
    B = backend.Constant(np.zeros((2, 3)), name="B")
    C = A + B
    P = backend.Problem()
    P.register("C", C)
    P.add_objectives(C.sum())
    P.add_suffix("_1", inplace=True)
    assert "A_1" in P.expr
    assert "B_1" in P.expr
    assert "C_1" in P.expr
    assert "A" not in P.expr
    assert "B" not in P.expr
    assert "C" not in P.expr


def test_add_sufix_and_merge(backend):
    # Create a decision variable and a constant
    A = backend.Variable("A", shape=(2, 3))
    B = backend.Constant(np.zeros((2, 3)), name="B")
    C = (A + B) @ np.ones((3, 1))

    # Create the first problem with an objective and a suffix
    P1 = backend.Problem()
    P1.register("C", C)
    P1 += C >= 0
    # Define the objective as the sum of all elements of C.
    P1.add_objectives(C.sum())
    # Add a suffix for later inspection/tracking.
    P1.add_suffix("_1", inplace=True)

    # Create a second problem with the same structure
    P2 = backend.Problem()
    A = backend.Variable("A", shape=(2, 3))
    B = backend.Constant(np.zeros((2, 3)), name="B")
    C = (A + B) @ np.ones((3, 1))
    P2 += C >= 0
    P2.register("C", C)
    P2.add_objectives(C.sum().sum())
    P2.add_suffix("_2", inplace=True)

    merged_problem = P1.merge(P2)
    merged_problem.add_constraints(A >= 0)
    merged_problem.add_constraints(C.sum().sum() == 1)
    # Make sure it compiles
    merged_problem.solve()
    assert "A_1" in merged_problem.expr
    assert "B_1" in merged_problem.expr
    assert "C_1" in merged_problem.expr
    assert "A_2" in merged_problem.expr
    assert "B_2" in merged_problem.expr
    assert "C_2" in merged_problem.expr


def test_sum_expr_shape(backend):
    A = backend.Variable(shape=(2, 3))
    B = backend.Variable(shape=(2, 3))
    C = A + B
    assert C.shape == (2, 3)


def test_mul_expr_shape(backend):
    A = backend.Variable(shape=(2, 3))
    B = np.ones((3, 2))
    C = A @ B
    assert C.shape == (2, 2)


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


def test_vstack_matrix(backend):
    x = backend.Variable("x", (2, 2))
    y = backend.Variable("y", (3, 2))
    z = x.vstack(y)
    assert z.shape == (5, 2)


def test_vstack_col_transposed(backend):
    x = backend.Variable("x", (5, 1))
    y = backend.Variable("y", (5,))
    assert x.T.vstack(y).shape == (2, 5)


def test_vstack_1d(backend):
    x = backend.Variable("x", (5,))
    y = None
    for _ in range(5):
        if y is None:
            y = x
        else:
            y = y.vstack(x)
    assert y.shape == (5, 5)


def test_vstack_1d_col(backend):
    y = None
    for i in range(5):
        x = backend.Variable("x", (1, 3))
        if y is None:
            y = x
        else:
            y = y.vstack(x)
    assert y.shape == (5, 3)


def test_vstack_1d_2d(backend):
    x = backend.Variable("x", (5,))
    y = backend.Variable("y", (2, 5))
    z = x.vstack(y)
    assert z.shape == (3, 5)


def test_vstack_2d_1d(backend):
    x = backend.Variable("x", (5,))
    y = backend.Variable("y", (2, 5))
    z = y.vstack(x)
    assert z.shape == (3, 5)


def test_vstack_left_vec_expression(backend):
    left = backend.Variable("x", (5,)) + backend.Variable("y", (5,))
    right = backend.Variable("z", (2, 5))
    assert left.vstack(right).shape == (3, 5)


def test_vstack_right_vec_expression(backend):
    left = backend.Variable("x", (2, 5))
    right = backend.Variable("y", (5,)) + backend.Variable("z", (5,))
    assert left.vstack(right).shape == (3, 5)


def test_vstack_invalid_left(backend):
    left = backend.Variable("y", (1, 5))
    right = backend.Variable("x", (5,))
    with pytest.raises(Exception):
        left.T.vstack(right)


def test_vstack_invalid_right(backend):
    left = backend.Variable("x", (5,))
    right = backend.Variable("y", (1, 5))
    with pytest.raises(Exception):
        left.vstack(right.T)


def test_vstack_backend(backend):
    x = backend.Variable("x", (3, 1))
    y = backend.Variable("y", (6, 1))
    z = backend.Variable("z", (1, 1))
    t = backend.vstack([x, y, z])
    assert t.shape == (10, 1)


def test_vstack_backend_1d(backend):
    x = backend.Variable("x", (3,))
    y = backend.Variable("y", (3,))
    z = backend.Variable("z", (3,))
    t = backend.vstack([x, y, z])
    assert t.shape == (3, 3)


def test_hstack_matrix(backend):
    x = backend.Variable("x", (2, 2))
    y = backend.Variable("y", (2, 3))
    z = x.hstack(y)
    assert z.shape == (2, 5)


# @pytest.mark.skip(reason="PICOS Backend fails on  this")
def test_hstack_1d(backend):
    # PICOS treats this as (5,1)
    x = backend.Variable("x", (5,))
    y = None
    for _ in range(5):
        if y is None:
            y = x
        else:
            y = y.hstack(x)
    assert y.shape == (25,)


# Skip test for now
# @pytest.mark.skip(reason="To be fixed")
def test_hstack_2d_1d(backend):
    x = backend.Variable("x", (5,))
    y = backend.Variable("y", (1, 5))
    with pytest.raises(Exception):
        y.hstack(x)


def test_hstack_rowvec_2d(backend):
    x = backend.Variable("x", (1, 5))
    y = None
    for _ in range(5):
        if y is None:
            y = x
        else:
            y = y.hstack(x)
    assert y.shape == (1, 25)


def test_invalid_hstack_1d_2d(backend):
    x = backend.Variable("x", (5,))
    y = backend.Variable("y", (1, 5))
    with pytest.raises(Exception):
        x.hstack(y)


def test_hstack_left_vec_expression(backend):
    left = backend.Variable("x", (5,)) + backend.Variable("y", (5,))
    right = backend.Variable("z", (2, 5))
    assert left.vstack(right).shape == (3, 5)


def test_hstack_right_vec_expression(backend):
    left = backend.Variable("x", (2, 5))
    right = backend.Variable("y", (5,)) + backend.Variable("z", (5,))
    assert left.vstack(right).shape == (3, 5)


def test_hstack_backend(backend):
    x = backend.Variable("x", (1, 3))
    y = backend.Variable("y", (1, 6))
    z = backend.Variable("z", (1, 1))
    t = backend.hstack([x, y, z])
    assert t.shape == (1, 10)


def test_reshape(backend):
    x = backend.Variable("x", (2, 3))
    y = x.reshape((3, 2))
    assert y.shape == (3, 2)


def test_reshape_onedim(backend):
    x = backend.Variable("x", (6,))
    y = x.reshape((3, 2))
    assert y.shape == (3, 2)


def test_reshape_negative_one_one(backend):
    x = backend.Variable("x", (4, 1))
    reshaped = x.reshape((-1, 1))
    expected_shape = (4, 1)
    assert reshaped.shape == expected_shape


def test_reshape_one_negative_one(backend):
    x = backend.Variable("x", (1, 4))
    reshaped = x.reshape((1, -1))
    expected_shape = (1, 4)
    assert reshaped.shape == expected_shape


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


def test_parameter(backend):
    P = backend.Problem()
    x = backend.Variable("x")
    p = backend.Parameter("p")
    P += x >= p
    # Change value after the constraint was created
    p.value = 2
    P.add_objectives(x)
    P.solve()
    assert np.isclose(x.value, 2)


def test_parameter_cvxpy():
    backend = CvxpyBackend()
    P = backend.Problem()
    x = backend.Variable("x")
    p = backend.Parameter("p")
    P += x >= p
    p.value = 2
    P.add_objectives(x)
    P.solve()
    assert np.isclose(x.value, 2)
    p.value = 3
    P.solve()
    assert np.isclose(x.value, 3)


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


def test_linearized_xor_axis0(backend):
    P = backend.Problem()
    X = backend.Variable("X", (2, 3), vartype=VarType.BINARY)
    # Apply linear_xor along axis 0
    P += backend.linear_xor(X, axis=0, varname="v_xor")
    # Force the XOR of the first column to be 1 (odd number of ones)
    P += P.expr.v_xor[0] == 1
    # Objective: minimize the number of ones in the first column
    P.add_objectives(sum(X[:, 0]))
    P.solve()
    # Retrieve the total number of ones in the first column
    total = np.sum(X[:, 0].value)
    # Check that the sum is odd and minimal
    assert total % 2 == 1, f"Sum along axis 0 should be odd, got {total}"
    assert np.isclose(total, 1.0), f"Sum along axis 0 should be minimal, got {total}"


def test_linearized_xor_axis1(backend):
    P = backend.Problem()
    X = backend.Variable("X", (2, 3), vartype=VarType.BINARY)
    # Apply linear_xor along axis 1
    P += backend.linear_xor(X, axis=1, varname="v_xor")
    # Force the XOR of the first row to be 1 (odd number of ones)
    P += P.expr.v_xor[0] == 1
    # Objective: minimize the number of ones in the first row
    P.add_objectives(sum(X[0, :]))
    P.solve()
    # Retrieve the total number of ones in the first row
    total = np.sum(X[0, :].value)
    # Check that the sum is odd and minimal
    assert total % 2 == 1, f"Sum along axis 1 should be odd, got {total}"
    assert np.isclose(total, 1.0), f"Sum along axis 1 should be minimal, got {total}"


def test_indexing_axis0(backend):
    P = backend.Problem()
    X = backend.Variable("X", (3, 4), vartype=VarType.BINARY)
    idx = [0, 2]
    P += X[idx, :] == 1
    P.add_objectives(sum(sum(X[idx, :])))
    P.solve()
    assert np.isclose(np.sum(np.sum(X.value[idx, :])), 8.0)


def test_indexing_axis1(backend):
    P = backend.Problem()
    X = backend.Variable("X", (3, 4), vartype=VarType.BINARY)
    idx = [0, 2]
    P += X[:, idx] == 1
    P.add_objectives(sum(sum(X[:, idx])))
    P.solve()
    assert np.isclose(np.sum(np.sum(X.value[:, idx])), 6.0)


def test_zero_function(backend):
    assert backend.zero_function().value == 0
    # TODO: Fix this for picos, as it returns (1,1)
    # assert backend.zero_function().shape == ()


def test_undirected_flow(backend):
    g = Graph()
    g.add_edges([((), "A"), ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", ())])
    P = backend.Flow(g, lb=-1000, ub=1000)
    P += P.expr.flow[1] >= 10
    P += P.expr.flow[2] >= -10
    P.add_objectives(sum(P.expr.flow), weights=1)
    P.solve()
    assert np.isclose(P.objectives[0].value, 0)


def test_complex_chaining_with_indexing(backend):
    Ah = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]])
    Int = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]])
    V = backend.Variable("V", (4, 3))
    idx = [0, 2]
    # Matrix multiplication Ah.T (3x4) @ V (4x3) -> (3x3) matrix
    # X is converted to a 2x3 matrix
    indicator = Int[idx, :] < 1  # This creates issues in PICOS
    X = (Ah.T @ V)[idx, :].multiply(indicator.astype(int))
    P = backend.Problem()
    P += X >= 0
    P.add_objectives(sum(sum(X)))
    P.solve()
    assert np.isclose(np.sum(np.sum(X.value)), 0.0)


def test_undirected_flow_unbounded(backend):
    from corneto.graph import Graph

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
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    b = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert np.allclose(sol, a) or np.allclose(sol, b)


def test_flow_plus_acyclic_directed_graph(backend):
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
    P = backend.Flow(G, lb=0, ub=10)
    P += backend.NonZeroIndicator(P.expr._flow, tolerance=1e-4)
    P += backend.Acyclic(
        G,
        P,
        indicator_positive_var_name="_flow_ipos",
        indicator_negative_var_name="_flow_ineg",
    )
    with_flow = P.expr._flow_ipos + P.expr._flow_ineg
    # P.add_objectives(-with_flow.sum())
    P.add_objectives(-sum(with_flow))
    P.solve()
    sol = np.round(with_flow.value).ravel()
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    b = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert np.allclose(sol, a) or np.allclose(sol, b)


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
    P.solve()
    sol = np.round(P.expr.with_flow.value).ravel()
    vsol1 = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    vsol2 = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert np.allclose(sol, vsol1) or np.allclose(sol, vsol2)


def test_flow_plus_acyclic_undirected_edge(backend):
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
    lb = np.array([0] * G.ne)
    ub = np.array([10] * G.ne)
    lb[3] = -10  # reversible v3 <-> v1
    P = backend.Flow(G, lb=lb, ub=ub)
    P += backend.NonZeroIndicator(P.expr._flow, tolerance=1e-6)
    P += backend.Acyclic(
        G,
        P,
        indicator_positive_var_name="_flow_ipos",
        indicator_negative_var_name="_flow_ineg",
    )
    with_flow = P.expr._flow_ipos + P.expr._flow_ineg
    P.add_objectives(-with_flow.sum())
    P.solve()
    sol = np.round(with_flow.value).ravel()
    vsol1 = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    vsol2 = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert np.allclose(sol, vsol1) or np.allclose(sol, vsol2)


def test_two_sample_flow_plus_acyclic_undirected_edge(backend):
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
    lb = np.array([0] * G.ne)
    ub = np.array([10] * G.ne)
    lb[3] = -10  # reversible v3 <-> v1
    P = backend.Flow(G, lb=lb, ub=ub, n_flows=2)
    P += backend.NonZeroIndicator(P.expr._flow, tolerance=1e-6)
    P += backend.Acyclic(
        G,
        P,
        indicator_positive_var_name="_flow_ipos",
        indicator_negative_var_name="_flow_ineg",
    )
    with_flow = P.expr._flow_ipos + P.expr._flow_ineg
    P.add_objectives(-with_flow.sum().sum())
    P.solve()
    sol_s1 = np.round(with_flow.value[:, 0]).ravel()
    sol_s2 = np.round(with_flow.value[:, 1]).ravel()
    vsol1 = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    vsol2 = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert np.allclose(sol_s1, vsol1) or np.allclose(sol_s1, vsol2)
    assert np.allclose(sol_s2, vsol1) or np.allclose(sol_s2, vsol2)


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
    assert np.all(P.expr.flow.value is None)


def test_l2_norm(backend):
    x = np.array([1, 2])
    y = np.array([3, 4])
    expected_result = np.linalg.norm(x - y)
    P = backend.Problem()
    diff = backend.Variable("diff", x.shape)
    n = diff.norm()
    P += [diff == x - y]
    P.add_objectives(n)
    P.solve(solver="cvxopt")
    assert np.isclose(n.value, expected_result, rtol=1e-5)


def test_variable_matrix_bounds(backend):
    """Test that Variables with matrix bounds (lb/ub with 2 columns) work correctly.

    Creates a Variable with different bounds for each column and verifies that
    constraints are properly applied, forcing different values in each column.
    """
    # Create a 3x2 variable with matrix bounds
    n_rows = 3
    n_cols = 2

    # Set up different bounds for each column
    # Column 0: lower bound = -5, upper bound = 5
    # Column 1: lower bound = 0, upper bound = 0 (fixed at 0)
    lb = np.array([[-5] * n_rows, [0] * n_rows]).T  # Shape: (3, 2)
    ub = np.array([[5] * n_rows, [0] * n_rows]).T  # Shape: (3, 2)

    # Create variable with matrix bounds
    v = backend.Variable("v", shape=(n_rows, n_cols), lb=lb, ub=ub)

    # Create optimization problem
    P = backend.Problem()

    # Objective: maximize values in column 0 (should reach upper bound of 5)
    P.add_objectives(-v[:, 0].sum())  # Negative sign for maximization

    # Solve the problem
    P.solve()

    # Verify results
    assert v.value is not None, "Solution should exist"

    # Column 0 should be at upper bound (5) for all rows
    col0_msg = f"Column 0 should be 5, got {v.value[:, 0]}"
    assert np.allclose(v.value[:, 0], 5.0, atol=1e-6), col0_msg

    # Column 1 should be exactly 0 (both lower and upper bound are 0)
    col1_msg = f"Column 1 should be 0, got {v.value[:, 1]}"
    assert np.allclose(v.value[:, 1], 0.0, atol=1e-6), col1_msg

    # Verify that values in column 0 are different from values in column 1
    for i in range(n_rows):
        diff_msg = f"v[{i}, 0] = {v.value[i, 0]} should differ from v[{i}, 1] = {v.value[i, 1]}"
        assert not np.isclose(v.value[i, 0], v.value[i, 1]), diff_msg


def test_indicator_matrix_bounds_block_per_entry(backend):
    """Indicator constraints should honor blocked entries per (row, col) in 2D bounds."""
    lb = np.array(
        [
            [-5.0, -5.0],
            [0.0, -5.0],
            [-5.0, 0.0],
        ]
    )
    ub = np.array(
        [
            [5.0, 5.0],
            [0.0, 5.0],
            [5.0, 0.0],
        ]
    )

    v = backend.Variable("v_ind", shape=lb.shape, lb=lb, ub=ub)
    P = backend.Problem()
    P += backend.Indicator(v, name="v_ind_i")
    indicator = P.expr.v_ind_i

    # Maximize number of active indicators: all unblocked entries should be 1.
    P.add_objectives(-indicator.sum())
    P.solve()

    indicator_value = np.asarray(indicator.value)
    v_val = np.asarray(v.value)

    # Blocked entries (lb=ub=0) must be inactive and fixed to zero.
    assert np.isclose(indicator_value[1, 0], 0.0, atol=1e-9)
    assert np.isclose(indicator_value[2, 1], 0.0, atol=1e-9)
    assert np.isclose(v_val[1, 0], 0.0, atol=1e-9)
    assert np.isclose(v_val[2, 1], 0.0, atol=1e-9)

    # Unblocked entries should be active under the maximizing objective.
    assert np.isclose(indicator_value[0, 0], 1.0, atol=1e-9)
    assert np.isclose(indicator_value[0, 1], 1.0, atol=1e-9)
    assert np.isclose(indicator_value[1, 1], 1.0, atol=1e-9)
    assert np.isclose(indicator_value[2, 0], 1.0, atol=1e-9)


def test_nonzero_indicator_matrix_bounds_block_and_sign_per_entry(backend):
    """NonZeroIndicator should respect blocked and sign-restricted entries in 2D bounds."""
    tol = 1e-3
    lb = np.array(
        [
            [0.0, -5.0],
            [-5.0, 0.0],
            [-5.0, 0.0],
        ]
    )
    ub = np.array(
        [
            [0.0, 5.0],
            [0.0, 5.0],
            [5.0, 0.0],
        ]
    )

    v = backend.Variable("v_nz", shape=lb.shape, lb=lb, ub=ub)
    P = backend.Problem()
    P += backend.NonZeroIndicator(v, tolerance=tol)

    I_pos = P.expr.v_nz_ipos
    I_neg = P.expr.v_nz_ineg
    I_tot = I_pos + I_neg

    # Maximize activity to force all feasible indicators on.
    P.add_objectives(-I_tot.sum())
    P.solve()

    pos_val = np.asarray(I_pos.value)
    neg_val = np.asarray(I_neg.value)
    tot_val = np.asarray(I_tot.value)
    v_val = np.asarray(v.value)

    # Blocked entries.
    assert np.isclose(pos_val[0, 0], 0.0, atol=1e-9)
    assert np.isclose(neg_val[0, 0], 0.0, atol=1e-9)
    assert np.isclose(v_val[0, 0], 0.0, atol=1e-9)

    assert np.isclose(pos_val[2, 1], 0.0, atol=1e-9)
    assert np.isclose(neg_val[2, 1], 0.0, atol=1e-9)
    assert np.isclose(v_val[2, 1], 0.0, atol=1e-9)

    # Sign-constrained entries.
    # ub <= 0 => positive indicator disabled.
    assert np.isclose(pos_val[1, 0], 0.0, atol=1e-9)
    assert np.isclose(neg_val[1, 0], 1.0, atol=1e-9)
    assert v_val[1, 0] <= -tol + 1e-8

    # lb >= 0 => negative indicator disabled.
    assert np.isclose(neg_val[1, 1], 0.0, atol=1e-9)
    assert np.isclose(pos_val[1, 1], 1.0, atol=1e-9)
    assert v_val[1, 1] >= tol - 1e-8

    # Free entries should have exactly one active sign indicator.
    assert np.isclose(tot_val[0, 1], 1.0, atol=1e-9)
    assert np.isclose(tot_val[2, 0], 1.0, atol=1e-9)


def test_exact_support_reuses_nonnegative_selector(backend):
    """ExactSupport should couple a supplied selector without another binary."""
    value = backend.Variable("supported_value", (2,), lb=0, ub=5)
    selected = backend.Variable("supplied_selector", (2,), vartype=VarType.BINARY)
    problem = backend.Problem()
    problem += backend.ExactSupport(
        value,
        selected=selected,
        epsilon=1,
        nonnegative=True,
        name="exact_support",
    )
    problem += selected == np.array([0, 1])
    problem.add_objective(value.sum())
    problem.solve()

    assert np.allclose(np.asarray(selected.value).reshape(-1), [0, 1], atol=1e-8)
    assert np.allclose(np.asarray(value.value).reshape(-1), [0, 1], atol=1e-8)
    binary_symbols = [symbol for symbol in problem.symbols.values() if symbol._vartype == VarType.BINARY]
    assert [symbol.name for symbol in binary_symbols] == ["supplied_selector"]


def test_exact_support_broadcasts_per_entry_epsilon(backend):
    """ExactSupport should apply explicitly broadcastable gaps elementwise."""
    epsilon = np.array([[0.25, 0.5]])
    value = backend.Variable("array_supported_value", (2, 2), lb=0, ub=5)
    problem = backend.Problem()
    problem += backend.ExactSupport(
        value,
        epsilon=epsilon,
        nonnegative=True,
        name="array_support",
    )
    problem += problem.expr.array_support == np.ones((2, 2))
    problem += value == np.broadcast_to(epsilon, value.shape)
    problem.solve()

    assert np.allclose(np.asarray(value.value), np.broadcast_to(epsilon, value.shape), atol=1e-8)


def test_exact_support_rejects_nonbroadcastable_epsilon_shape(backend):
    """ExactSupport should not reinterpret same-sized epsilon arrays by reshaping."""
    value = backend.Variable("shape_checked_value", (3, 2), lb=-5, ub=5)

    with pytest.raises(ValueError, match="broadcast"):
        backend.ExactSupport(value, epsilon=np.ones((2, 3)))


@pytest.mark.parametrize("lower, upper", [(-np.inf, 5), (-5, np.inf)])
def test_exact_support_requires_finite_bounds(backend, lower, upper):
    """ExactSupport should reject infinite bounds even when they were supplied."""
    value = backend.Variable("finite_bound_value", (2,), lb=lower, ub=upper)

    with pytest.raises(ValueError, match="finite"):
        backend.ExactSupport(value, epsilon=1)


@pytest.mark.parametrize("nonnegative", [False, True])
def test_exact_support_requires_binary_supplied_selector(backend, nonnegative):
    """A supplied selector must not relax ExactSupport into a continuous formulation."""
    value = backend.Variable("selector_value", (2,), lb=0 if nonnegative else -5, ub=5)
    selector = backend.Variable("continuous_selector", (2,), lb=0, ub=1)

    with pytest.raises(TypeError, match="binary"):
        backend.ExactSupport(value, selected=selector, epsilon=1, nonnegative=nonnegative)


def test_exact_support_rejects_fractional_selector_expression(backend):
    """A fractional selector expression must be linked to a binary selector."""
    value = backend.Variable("fractional_selector_value", (1,), lb=0, ub=5)
    binary = backend.Variable("selector_binary", (1,), vartype=VarType.BINARY)

    problem = backend.Problem()
    problem += backend.ExactSupport(value, selected=0.5 * binary, epsilon=1, nonnegative=True)
    problem += binary == 1
    if isinstance(backend, PicosBackend):
        result = problem.solve(solver="glpk", primals=None)
    else:
        result = problem.solve()

    assert str(result.status).lower() in {"infeasible", "infeasible_or_unbounded"}


def test_exact_support_signed_value(backend):
    """Signed exact support should expose mutually exclusive directions."""
    value = backend.Variable("signed_value", (2,), lb=-4, ub=4)
    problem = backend.Problem()
    problem += backend.ExactSupport(
        value,
        epsilon=0.5,
        name="signed_support",
    )
    problem += problem.expr.signed_support == np.array([1, 1])
    problem += value[0] >= 0
    problem += value[1] <= 0
    problem.add_objective(value[0] - value[1])
    problem.solve()

    assert np.allclose(np.asarray(value.value).reshape(-1), [0.5, -0.5], atol=1e-7)
    assert np.allclose(np.asarray(problem.expr.signed_support.value).reshape(-1), [1, 1], atol=1e-8)


def test_selected_flow_uses_negative_direction_when_required(backend):
    """Signed SelectedFlow should expose the actual traversal direction."""
    graph = Graph()
    graph.add_edge((), "B")
    graph.add_edge("A", "B")
    graph.add_edge("A", ())

    problem = backend.SelectedFlow(
        graph,
        lb=np.array([0, -5, 0]),
        ub=np.array([5, 5, 5]),
        edge_indices=[1],
        epsilon=1,
    )
    problem += problem.expr.selected_any == 1
    problem.solve()

    assert -5 - 1e-7 <= problem.expr.flow.value[1, 0] <= -1 + 1e-7
    assert np.isclose(np.asarray(problem.expr.selected_by_flow.value).reshape(-1)[0], 1, atol=1e-8)
    assert np.isclose(np.asarray(problem.expr.selected_by_flow_positive.value).reshape(-1)[0], 0, atol=1e-8)
    assert np.isclose(np.asarray(problem.expr.selected_by_flow_negative.value).reshape(-1)[0], 1, atol=1e-8)


def test_selected_flow_mixed_bounds_uses_minimal_binary_types():
    """Only edge rows that permit negative flow should get direction binaries."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "A")])
    problem = CvxpyBackend().SelectedFlow(
        graph,
        lb=np.array([[0, 0], [-5, -5]]),
        ub=np.full((2, 2), 5),
        n_flows=2,
        edge_indices=[0, 1],
        epsilon=1,
    )
    cvxpy_problem = problem.solve(solver="SCIPY")

    boolean_count = sum(variable.size for variable in cvxpy_problem.variables() if variable.attributes["boolean"])
    # Nonnegative edge: 1*2 support binaries. Signed edge: 2*2 direction
    # binaries. Shared structural union: 2 binaries.
    assert boolean_count == 2 + 4 + 2
    assert problem.expr.selected_by_flow.shape == (2, 2)
    assert problem.expr.selected_by_flow_positive.shape == (2, 2)
    assert problem.expr.selected_by_flow_negative.shape == (2, 2)


def test_selected_flow_signed_shared_dag_rejects_opposite_directions():
    """Different flows cannot traverse one edge oppositely in a shared DAG."""
    flow_graph = Graph()
    flow_graph.add_edge((), "A")
    flow_graph.add_edge((), "B")
    flow_graph.add_edge("A", "B")
    flow_graph.add_edge("A", ())
    flow_graph.add_edge("B", ())
    acyclic_graph = Graph.from_tuples([("A", 1, "B")])

    problem = CvxpyBackend().SelectedFlow(
        flow_graph,
        lb=np.array(
            [
                [0, 0],
                [0, 0],
                [-5, -5],
                [0, 0],
                [0, 0],
            ]
        ),
        ub=np.array(
            [
                [5, 0],
                [0, 5],
                [5, 5],
                [0, 5],
                [5, 0],
            ]
        ),
        n_flows=2,
        edge_indices=[2],
        epsilon=1,
        acyclic_graph=acyclic_graph,
    )
    problem += problem.expr.flow[2, 0] >= 1
    problem += problem.expr.flow[2, 1] <= -1
    cvxpy_problem = problem.solve(solver="SCIPY")

    assert cvxpy_problem.status == cp.INFEASIBLE
    boolean_count = sum(variable.size for variable in cvxpy_problem.variables() if variable.attributes["boolean"])
    # Two directions per flow plus two direction-union binaries. There is no
    # redundant structural-union binary.
    assert boolean_count == 2 * 2 + 2


def test_acyclic_signed_max_parents_counts_reversed_edges():
    """A negative edge contributes a parent at its structural source."""
    graph = Graph.from_tuples([("A", 1, "C"), ("C", 1, "B")])
    backend = CvxpyBackend()
    problem = backend.Problem()
    positive = backend.Variable("positive", (2,), vartype=VarType.BINARY)
    negative = backend.Variable("negative", (2,), vartype=VarType.BINARY)
    problem.register("positive", positive)
    problem.register("negative", negative)
    problem += positive == np.array([1, 0])
    problem += negative == np.array([0, 1])
    backend.Acyclic(
        graph,
        problem,
        indicator_positive_var_name="positive",
        indicator_negative_var_name="negative",
        max_parents={"C": 1},
    )

    cvxpy_problem = problem.solve(solver="SCIPY")
    assert cvxpy_problem.status == cp.INFEASIBLE
