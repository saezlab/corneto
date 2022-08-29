import pytest
import pathlib
import numpy as np
from corneto.backend import PicosBackend, CvxpyBackend
import cvxpy as cp


def test_picos_convex():
    backend = PicosBackend()
    P = backend.Problem()
    A = np.array([[0.12, 0.92, 0.76, 0.98, 0.79], [0.58, 0.57, 0.53, 0.71, 0.55]])
    b = np.array([1, 0])
    x = backend.Variable("x", A.shape[1])
    P += x
    P += sum(x) == 1, x >= 0
    # Convex optimization problem
    P.add_objectives(abs(A @ x - b), inplace=True)
    P.solve(solver="cvxopt", verbosity=1)
    print(x.value)
    assert np.all(np.array(x.value).T < np.array([1e-6, 0.64, 0.37, 1e-6, 1e-6]))
    assert np.all(np.array(x.value).T > np.array([-1e-6, 0.62, 0.36, -1e-6, -1e-6]))


def test_cvxpy_convex():
    backend = CvxpyBackend()
    P = backend.Problem()
    A = np.array([[0.12, 0.92, 0.76, 0.98, 0.79], [0.58, 0.57, 0.53, 0.71, 0.55]])
    b = np.array([1, 0])
    x = backend.Variable("x", A.shape[1])
    P += x
    P += sum(x) == 1, x >= 0
    # Convex optimization problem
    P.add_objectives(cp.sum_squares((A @ x - b).e), inplace=True)
    P.solve(solver="osqp", verbosity=1)
    assert np.all(np.array(x.value) < np.array([1e-6, 0.64, 0.37, 1e-6, 1e-6]))
    assert np.all(np.array(x.value) > np.array([-1e-6, 0.62, 0.36, -1e-6, -1e-6]))
