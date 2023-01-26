import pytest
import pathlib
import numpy as np
from corneto.backend import PicosBackend, CvxpyBackend
import cvxpy as cp


@pytest.fixture(params=[CvxpyBackend, PicosBackend])
def backend(request):
    return request.param()


def test_picos_convex():
    backend = PicosBackend()
    P = backend.Problem()
    A = np.array([[0.12, 0.92, 0.76, 0.98, 0.79], [0.58, 0.57, 0.53, 0.71, 0.55]])
    b = np.array([1, 0])
    x = backend.Variable("x", A.shape[1])
    # P += x # not needed since 0.9.1
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
    x = backend.Variable("x", A.shape[1])
    # P += x # not needed since 0.9.1
    P += sum(x) == 1, x >= 0
    # Convex optimization problem
    P.add_objectives(cp.sum_squares((A @ x - b).e), inplace=True)
    P.solve(solver="osqp", verbosity=1)
    assert np.all(np.array(x.value) < np.array([1e-6, 0.64, 0.37, 1e-6, 1e-6]))
    assert np.all(np.array(x.value) > np.array([-1e-6, 0.62, 0.36, -1e-6, -1e-6]))


def test_cvxpy_convex_apply():
    backend = CvxpyBackend()
    P = backend.Problem()
    A = np.array([[0.12, 0.92, 0.76, 0.98, 0.79], [0.58, 0.57, 0.53, 0.71, 0.55]])
    b = np.array([1, 0])
    x = backend.Variable("x", A.shape[1])
    # P += x # not needed since 0.9.1
    P += sum(x) == 1, x >= 0
    # Convex optimization problem
    P.add_objectives((A @ x - b).apply(cp.sum_squares), inplace=True)
    P.solve(solver="osqp", verbosity=1)
    assert np.all(np.array(x.value) < np.array([1e-6, 0.64, 0.37, 1e-6, 1e-6]))
    assert np.all(np.array(x.value) > np.array([-1e-6, 0.62, 0.36, -1e-6, -1e-6]))


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
    x = backend.Variable("x", A.shape[1])
    P.add_objectives((A @ x - b).apply(cp.sum_squares), inplace=True)
    assert "x" in P.symbols


def test_picos_custom_expr_symbols():
    backend = PicosBackend()
    P = backend.Problem()
    A = np.zeros((2, 5))
    b = np.array([1, 0])
    x = backend.Variable("x", A.shape[1])
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
