"""Behavior of optional threshold implications on bounded values."""

import numpy as np
import pytest


@pytest.mark.parametrize("values", [[-0.005, 0, 0.005], [-0.01, 0, 0.01], [-0.02, 0, 0.02]])
def test_threshold_indicator_rewards_only_reached_thresholds(backend, values):
    x = backend.Variable("x", (3,), -1, 1)
    problem = backend.ThresholdIndicator(x, epsilon=0.01)
    problem += x == np.array(values)
    problem.add_objective(-problem.expr.x_threshold.sum())
    problem.solve()

    np.testing.assert_allclose(np.asarray(x.value).reshape(-1), values, atol=1e-7)
    expected = np.asarray(values)
    np.testing.assert_allclose(
        np.asarray(problem.expr.x_threshold_positive.value).reshape(-1), expected >= 0.01, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(problem.expr.x_threshold_negative.value).reshape(-1), expected <= -0.01, atol=1e-7
    )


def test_threshold_indicator_declining_rewards_preserves_full_range(backend):
    x = backend.Variable("x", (3,), -1, 1)
    problem = backend.ThresholdIndicator(x, epsilon=0.01)
    problem += problem.expr.x_threshold_positive == 0
    problem += problem.expr.x_threshold_negative == 0
    problem += x == np.array([-1, 0.005, 1])
    problem.solve()
    np.testing.assert_allclose(np.asarray(x.value).reshape(-1), [-1, 0.005, 1], atol=1e-7)


def test_threshold_indicator_selected_matrix_bounds_and_thresholds(backend):
    x = backend.Variable("x", (2, 2), lb=np.array([[-1, -0.005], [0, 0]]), ub=np.ones((2, 2)))
    problem = backend.ThresholdIndicator(x, indexes=(np.array([0, 1]), 1), epsilon=[0.01, 0.02], name="agreement")
    problem += x[:, 1] == np.array([-0.005, 0.02])
    problem.add_objective(-problem.expr.agreement.sum())
    problem.solve()
    np.testing.assert_allclose(np.asarray(problem.expr.agreement.value).reshape(-1), [0, 1], atol=1e-7)


@pytest.mark.parametrize("epsilon", [0, -1, np.inf, np.nan])
def test_threshold_indicator_rejects_invalid_thresholds(backend, epsilon):
    x = backend.Variable("x", (2,), -1, 1)
    with pytest.raises(ValueError, match="epsilon"):
        backend.ThresholdIndicator(x, epsilon=epsilon)


@pytest.mark.parametrize("epsilon", [True, "0.1", 1j])
def test_threshold_indicator_rejects_nonnumeric_thresholds(backend, epsilon):
    x = backend.Variable("x", (2,), -1, 1)
    with pytest.raises(TypeError, match="epsilon"):
        backend.ThresholdIndicator(x, epsilon=epsilon)


def test_threshold_indicator_requires_finite_bounds_and_matching_shapes(backend):
    x = backend.Variable("x", (2,))
    with pytest.raises(ValueError, match="bounds"):
        backend.ThresholdIndicator(x)
    bounded = backend.Variable("bounded", (2,), -1, 1)
    with pytest.raises(ValueError, match="broadcastable"):
        backend.ThresholdIndicator(bounded, epsilon=[0.01, 0.02, 0.03])
