import numpy as np
import pytest

from corneto.backend import PicosBackend, VarType


def _solve(problem):
    if isinstance(problem.backend, PicosBackend):
        return problem.solve(primals=None)
    return problem.solve()


def test_total_variation_weighted_matrix_and_registered_shapes(backend):
    x = backend.Variable("x", shape=(2, 4))
    tv = backend.TotalVariation(
        x,
        pairs=[(0, 1), (1, 3)],
        axis=1,
        weights=[1.0, 2.0],
        name="conditions",
    )
    assert tv.objectives == []
    fixed = np.array([[0.0, 1.0, 3.0, 2.0], [4.0, 2.0, 1.0, 3.0]])
    problem = backend.Problem(x == fixed) + tv
    problem.add_objective(problem.expr.conditions_total_variation)
    _solve(problem)

    assert problem.expr.conditions_difference.shape == (2, 2)
    assert problem.expr.conditions_abs_bound.shape == (2, 2)
    assert problem.expr.conditions_variation_by_pair.shape == (2,)
    assert problem.expr.conditions_total_variation.shape == ()
    np.testing.assert_allclose(problem.expr.conditions_difference.value, [[1, 1], [-2, 1]])
    np.testing.assert_allclose(np.asarray(problem.expr.conditions_variation_by_pair.value).ravel(), [3, 2])
    assert np.isclose(float(problem.expr.conditions_total_variation.value), 7)


@pytest.mark.parametrize("axis", [0, -2])
def test_total_variation_axis_zero_and_single_pair(backend, axis):
    x = backend.Variable("x", shape=(3, 2))
    tv = backend.TotalVariation(x, pairs=[(0, 2)], axis=axis, name="axis_zero")
    fixed = np.array([[0, 2], [4, 1], [3, -2]])
    problem = backend.Problem(x == fixed) + tv
    problem.add_objective(problem.expr.axis_zero_total_variation)
    _solve(problem)

    assert problem.expr.axis_zero_difference.shape == (2, 1)
    np.testing.assert_allclose(problem.expr.axis_zero_difference.value, [[3], [-4]])
    assert np.isclose(float(problem.expr.axis_zero_total_variation.value), 7)


def test_total_variation_vector_and_singleton(backend):
    x = backend.Variable("x", shape=(1, 3))
    tv = backend.TotalVariation(x, pairs=[(0, 2)], axis=-1, name="singleton")
    problem = backend.Problem(x == np.array([[-2, 1, 3]])) + tv
    problem.add_objective(problem.expr.singleton_total_variation)
    _solve(problem)

    assert problem.expr.singleton_difference.shape == (1, 1)
    np.testing.assert_allclose(problem.expr.singleton_difference.value, [[5]])
    assert np.isclose(float(problem.expr.singleton_total_variation.value), 5)


def test_total_variation_vector_signed_difference(backend):
    x = backend.Variable("x", shape=(3,))
    tv = backend.TotalVariation(x, pairs=[(2, 0)], weights=[3], name="vector")
    problem = backend.Problem(x == np.array([1, 4, -2])) + tv
    problem.add_objective(problem.expr.vector_total_variation)
    _solve(problem)

    assert problem.expr.vector_difference.shape == (1, 1)
    np.testing.assert_allclose(problem.expr.vector_difference.value, [[3]])
    assert np.isclose(float(problem.expr.vector_total_variation.value), 9)


def test_total_variation_affine_slice_tracks_source_and_bounds(backend):
    x = backend.Variable("x", shape=(2, 4), lb=0, ub=5)
    sliced = x[:, 1:4] + 1
    tv = backend.TotalVariation(sliced, pairs=[(0, 2)], name="slice")
    problem = tv
    problem.add_objective(-problem.expr.slice_difference.sum())
    result = _solve(problem)

    assert str(result.status).lower() == "optimal"
    assert problem.expr.slice_difference.shape == (2, 1)
    assert "x" in problem.symbols
    np.testing.assert_allclose(problem.expr.slice_difference.value, [[5], [5]])
    source_values = np.asarray(x.value).reshape(x.shape)
    np.testing.assert_allclose(source_values[:, 1], [0, 0])
    np.testing.assert_allclose(source_values[:, 3], [5, 5])


def test_total_variation_disconnected_duplicate_reversed_and_zero_weight(backend):
    x = backend.Variable("x", shape=(4,))
    pairs = [(0, 1), (0, 1), (1, 0), (2, 3)]
    tv = backend.TotalVariation(x, pairs=pairs, weights=[1, 2, 3, 0], name="graph")
    problem = backend.Problem(x == np.array([0, 1, 4, 2])) + tv
    problem += problem.expr.graph_abs_bound[0, 3] == 7
    problem.add_objective(problem.expr.graph_total_variation)
    _solve(problem)

    np.testing.assert_allclose(problem.expr.graph_difference.value.ravel(), [1, 1, -1, -2])
    pair_variation = np.asarray(problem.expr.graph_variation_by_pair.value).ravel()
    np.testing.assert_allclose(pair_variation[:3], [1, 1, 1])
    assert np.isclose(pair_variation[3], 7)
    assert np.isclose(float(problem.expr.graph_total_variation.value), 6)


def test_total_variation_objective_changes_fitted_solution(backend):
    x = backend.Variable("x", shape=(2,), lb=0, ub=2)
    baseline = backend.Problem(x[0] == 0)
    baseline += x[1] <= 2
    baseline.add_objective(x[1], weight=-1)
    _solve(baseline)
    assert np.isclose(x.value[1], 2)

    penalized_x = backend.Variable("penalized_x", shape=(2,), lb=0, ub=2)
    tv = backend.TotalVariation(penalized_x, pairs=[(0, 1)], name="fit")
    penalized = backend.Problem(penalized_x[0] == 0) + tv
    penalized += penalized_x[1] <= 2
    penalized.add_objective(penalized_x[1], weight=-1)
    penalized.add_objective(penalized.expr.fit_total_variation, weight=2)
    _solve(penalized)

    assert np.isclose(penalized_x.value[1], 0)


def test_total_variation_budget_is_exactly_projected_without_objective(backend):
    values = np.array([0.0, 1.0, 3.0, 2.0])
    for budget, expected_status in [(2.0, "infeasible"), (3.0, "feasible")]:
        x = backend.Variable("x", shape=(4,))
        tv = backend.TotalVariation(x, pairs=[(0, 1), (1, 3)], weights=[1, 2], name=f"budget_{budget}")
        problem = backend.Problem(x == values) + tv
        problem += problem.expr[f"budget_{budget}_total_variation"] <= budget
        result = _solve(problem)
        if expected_status == "infeasible":
            assert "infeasible" in str(result.status).lower()
        else:
            assert str(result.status).lower() == "optimal"
            assert np.isclose(float(problem.expr[f"budget_{budget}_total_variation"].value), 3)


@pytest.mark.parametrize("vartype", [VarType.INTEGER, VarType.BINARY])
def test_total_variation_discrete_input_adds_only_continuous_auxiliary(backend, vartype):
    x = backend.Variable("x", shape=(3,), vartype=vartype)
    tv = backend.TotalVariation(x, pairs=[(0, 1), (1, 2)], name="discrete")
    problem = backend.Problem(x == np.array([0, 1, 1])) + tv
    problem.add_objective(problem.expr.discrete_total_variation)
    _solve(problem)

    assert tv.expr.discrete_abs_bound._vartype == VarType.CONTINUOUS
    auxiliary_variables = [symbol for symbol in tv.symbols.values() if symbol.is_variable and symbol.name != "x"]
    assert [symbol.name for symbol in auxiliary_variables] == ["discrete_abs_bound"]
    assert all(symbol._vartype == VarType.CONTINUOUS for symbol in auxiliary_variables)
    assert len(tv.constraints) == 2
    np.testing.assert_allclose(problem.expr.discrete_difference.value.ravel(), [1, 0])
    assert np.isclose(float(problem.expr.discrete_total_variation.value), 1)


def test_total_variation_bound_can_be_slack_and_diagnostics_use_difference(backend):
    x = backend.Variable("x", shape=(2,))
    tv = backend.TotalVariation(x, pairs=[(0, 1)], weights=[2], name="slack")
    problem = backend.Problem(x == np.array([0.0, 2.0])) + tv
    problem += problem.expr.slack_abs_bound == 3
    result = _solve(problem)

    assert str(result.status).lower() == "optimal"
    assert np.isclose(float(problem.expr.slack_total_variation.value), 6)
    actual_variation = 2 * np.abs(problem.expr.slack_difference.value).sum()
    assert np.isclose(actual_variation, 4)


def test_total_variation_named_and_automatic_primitives_compose(backend):
    x = backend.Variable("x", shape=(3,))
    first = backend.TotalVariation(x, pairs=[(0, 1)], name="first")
    second = backend.TotalVariation(x, pairs=[(1, 2)], name="second")
    automatic_a = backend.TotalVariation(x, pairs=[(0, 2)])
    automatic_b = backend.TotalVariation(x, pairs=[(2, 0)])
    combined = first + second + automatic_a + automatic_b

    assert "first_total_variation" in combined.expr
    assert "second_total_variation" in combined.expr
    automatic_names = [
        key
        for key in combined.expr
        if key.endswith("_total_variation") and key not in {"first_total_variation", "second_total_variation"}
    ]
    assert len(set(automatic_names)) == 2
    auto_prefixes = {key.removesuffix("_total_variation") for key in automatic_names}
    assert {f"{prefix}_abs_bound" for prefix in auto_prefixes}.issubset(combined.symbols)


@pytest.mark.parametrize(
    "shape, kwargs, error",
    [
        ((3,), {"pairs": [(0, 1)], "axis": 1}, ValueError),
        ((3,), {"pairs": [(0, 1)], "axis": True}, TypeError),
        ((3,), {"pairs": []}, ValueError),
        ((3,), {"pairs": [(0,)]}, ValueError),
        ((3,), {"pairs": [(0, 1, 2)]}, ValueError),
        ((3,), {"pairs": [(True, 1)]}, TypeError),
        ((3,), {"pairs": [(0.0, 1)]}, TypeError),
        ((3,), {"pairs": [(-1, 0)]}, ValueError),
        ((3,), {"pairs": [(0, 3)]}, ValueError),
        ((3,), {"pairs": [(1, 1)]}, ValueError),
        ((3,), {"pairs": [(0, 1)], "weights": 1.0}, ValueError),
        ((3,), {"pairs": [(0, 1)], "weights": [[1.0]]}, ValueError),
        ((3,), {"pairs": [(0, 1)], "weights": []}, ValueError),
        ((3,), {"pairs": [(0, 1)], "weights": [-1.0]}, ValueError),
        ((3,), {"pairs": [(0, 1)], "weights": [np.inf]}, ValueError),
        ((3,), {"pairs": [(0, 1)], "weights": [np.nan]}, ValueError),
        ((3,), {"pairs": [(0, 1)], "weights": [1j]}, ValueError),
    ],
)
def test_total_variation_rejects_invalid_inputs(backend, shape, kwargs, error):
    x = backend.Variable("x", shape=shape)
    with pytest.raises(error):
        backend.TotalVariation(x, **kwargs)


@pytest.mark.parametrize("shape", [(), (2, 2, 2), (0, 2)])
def test_total_variation_rejects_unsupported_expression_shapes(backend, shape):
    # Override only shape metadata: some backends cannot construct these
    # dimensions, but validation belongs to TotalVariation itself.
    x = backend.Variable("x", shape=(1,))
    x._shape = shape

    with pytest.raises(ValueError, match="nonempty vector or matrix"):
        backend.TotalVariation(x, pairs=[(0, 1)])
