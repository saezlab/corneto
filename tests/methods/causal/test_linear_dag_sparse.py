"""Sparse affine-operator checks for linear DAG discovery."""

import numpy as np
from scipy.sparse import eye, issparse

from corneto._constants import VarType
from corneto.backend import PicosBackend
from corneto.backend._base import _sparse_vector_entry_repetition, _sparse_vector_replication
from corneto.data import Data
from corneto.graph import Graph
from corneto.methods.causal import LinearDAGDiscovery


def _is_sparse_constant(symbol) -> bool:
    """Recognize sparse constants in both backend expression implementations."""
    value = getattr(symbol.e, "value", None)
    if issparse(value):
        return True
    picos_constant = getattr(symbol.e, "_cached__constant_coef", None)
    return picos_constant is not None and hasattr(picos_constant, "CCS")


def _single_edge_data() -> Data:
    return Data.from_cdict(
        {
            "obs": {
                "A": {"mapping": "vertex", "value": 0.0},
                "B": {"mapping": "vertex", "value": 0.0},
            },
            "do_A": {
                "A": {"mapping": "vertex", "value": 2.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 3.0},
            },
        }
    )


def test_sparse_vector_maps_preserve_column_major_layout():
    values = np.array([1.0, 2.0, 3.0])

    repeated_columns = _sparse_vector_replication(3, 2) @ values
    repeated_entries = _sparse_vector_entry_repetition(3, 2) @ values

    np.testing.assert_allclose(repeated_columns, [1, 2, 3, 1, 2, 3])
    np.testing.assert_allclose(repeated_entries, [1, 1, 2, 2, 3, 3])


def test_sparse_constants_do_not_allocate_dense_bound_metadata(backend):
    constant = backend.Constant(eye(1000, format="csr"))

    assert constant.lb is None
    assert constant.ub is None
    assert constant._lb is None
    assert constant._ub is None


def test_selected_flow_matrix_operators_are_sparse(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    problem = backend.SelectedFlow(
        graph,
        lb=np.zeros((1, 3)),
        ub=np.full((1, 3), 5.0),
        n_flows=3,
        edge_indices=[0],
        selector_groups=[0, 0, 1],
    )

    matrix_constants = [
        symbol
        for symbol in problem.symbols.values()
        if len(getattr(symbol.e, "shape", ())) == 2 and not symbol.is_variable
    ]

    assert matrix_constants
    assert all(_is_sparse_constant(symbol) for symbol in matrix_constants)


def test_linear_dag_prediction_uses_sparse_affine_operators(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    problem = LinearDAGDiscovery(fit_intercept=True, backend=backend).build(graph, _single_edge_data())

    prediction_constants = [symbol for symbol in problem.expr.prediction._proxy_symbols if not symbol.is_variable]

    assert prediction_constants
    assert all(_is_sparse_constant(symbol) for symbol in prediction_constants)


def test_linear_dag_coverage_uses_sparse_affine_operators(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    problem = LinearDAGDiscovery(
        fit_intercept=False,
        min_commodity_coverage=1.0,
        backend=backend,
    ).build(graph, _single_edge_data())

    matrix_constants = [
        symbol
        for symbol in problem.symbols.values()
        if not symbol.is_variable and len(getattr(symbol.e, "shape", ())) == 2
    ]

    assert matrix_constants
    assert all(_is_sparse_constant(symbol) for symbol in matrix_constants)


def test_picos_flow_conservation_keeps_sparse_incidence():
    graph = Graph.from_tuples([("A", 1, "B")])
    problem = PicosBackend().Flow(graph, n_flows=2)

    matrix_constants = [
        symbol
        for symbol in problem.constraints[0]._proxy_symbols
        if not symbol.is_variable and len(getattr(symbol.e, "shape", ())) == 2
    ]

    assert matrix_constants
    assert all(_is_sparse_constant(symbol) for symbol in matrix_constants)


def test_picos_axis_sums_keep_sparse_affine_coefficients():
    backend = PicosBackend()
    expression = backend.Variable("sum_value", (1000, 12))
    values = np.arange(12000, dtype=float).reshape((1000, 12), order="F")
    expression.e.value = values
    expected = {
        0: values.sum(axis=0, keepdims=True),
        1: values.sum(axis=1, keepdims=True),
        None: values.sum(),
    }
    shapes = {0: (1, 12), 1: (1000, 1), None: (1, 1)}

    for axis in (0, 1, None):
        reduced = expression.sum(axis=axis)
        coefficients = list(reduced.e._coefs.values())

        assert coefficients
        assert all(hasattr(coefficient, "CCS") for coefficient in coefficients)
        assert reduced.shape == shapes[axis]
        np.testing.assert_allclose(np.asarray(reduced.value), expected[axis])


def test_picos_axis_sums_preserve_explicit_vector_orientation():
    backend = PicosBackend()
    expression = backend.Variable("orientation_value", (3,))
    problem = backend.Problem()
    problem += expression == np.array([1.0, 2.0, 4.0])

    row_sum = expression.reshape((1, 3)).sum(axis=0)
    column_sum = expression.reshape((3, 1)).sum(axis=1)
    transposed_sum = expression.T.sum(axis=0)
    problem.solve(solver="glpk")

    assert row_sum.shape == (1, 3)
    assert column_sum.shape == (3, 1)
    assert transposed_sum.shape == (1, 3)
    np.testing.assert_allclose(np.asarray(row_sum.value), [[1.0, 2.0, 4.0]])
    np.testing.assert_allclose(np.asarray(column_sum.value), [[1.0], [2.0], [4.0]])
    np.testing.assert_allclose(np.asarray(transposed_sum.value), [[1.0, 2.0, 4.0]])


def test_acyclic_parent_limits_are_applied_vertexwise(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    problem = backend.Problem()
    positive = backend.Variable("positive", (1,), vartype=VarType.BINARY)
    problem.register("positive", positive)
    problem += positive == np.array([1])
    backend.Acyclic(
        graph,
        problem,
        indicator_positive_var_name="positive",
        max_parents={"A": 0, "B": 1},
    )

    solved = problem.solve()
    assert str(solved.status).lower() == "optimal"
