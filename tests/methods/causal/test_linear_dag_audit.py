"""Regression tests for findings from the linear DAG discovery audit."""

import numpy as np
import pytest

from corneto.backend import PicosBackend
from corneto.data import Data
from corneto.graph import Graph
from corneto.methods.causal import LinearDAGDiscovery


def _data(rows):
    return Data.from_cdict(
        {
            sample_name: {
                vertex: {
                    "mapping": "vertex",
                    "value": value,
                    **({"intervened": True} if vertex in intervened else {}),
                }
                for vertex, value in values.items()
            }
            for sample_name, values, intervened in rows
        }
    )


def _solve(problem, backend):
    if isinstance(backend, PicosBackend):
        return problem.solve(solver="glpk", primals=None)
    return problem.solve()


def _zero_connector_data(scale_b=1.0, scale_c=1.0):
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 1.0, "C": 2.0}, ()),
            ("do_A", {"A": 1.0, "B": 0.0, "C": 0.0}, ("A",)),
        ]
    )
    for sample in data.samples.values():
        for feature in sample.features:
            if feature.id == "B":
                feature.data["value"] *= scale_b
            elif feature.id == "C":
                feature.data["value"] *= scale_c
    return data


def test_cointervention_blocks_flow_through_intervened_target(backend):
    """A commodity cannot use an incoming mechanism removed by intervention."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0, "C": 0.0}, ()),
            ("do_AB", {"A": 1.0, "B": 2.0, "C": 4.0}, ("A", "B")),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        min_commodity_coverage=1.0,
        backend=backend,
    )

    problem = method.build(graph, data)
    result = _solve(problem, backend)

    assert str(result.status).lower() in {"infeasible", "infeasible_or_unbounded"}


def test_partial_coverage_keeps_nonblocked_cointervention_commodity(backend):
    """A co-intervened source remains usable when its route avoids the other target."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0, "C": 0.0}, ()),
            ("do_AB", {"A": 1.0, "B": 2.0, "C": 4.0}, ("A", "B")),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        min_commodity_coverage=0.5,
        backend=backend,
    )

    problem = method.build(graph, data)
    _solve(problem, backend)

    active = np.asarray(problem.expr.commodity_active.value).reshape(-1) > 0.5
    active_by_source = dict(zip(method._commodity_sources, active, strict=True))
    usage = method.get_edge_usage()
    usage_by_source = {source: usage[:, index] for index, source in enumerate(method._commodity_sources)}

    assert active_by_source == {"A": False, "B": True}
    assert not usage_by_source["A"].any()
    assert np.array_equal(np.flatnonzero(usage_by_source["B"]), np.array([1]))


def test_squared_loss_builds_with_multiple_residuals_and_weights(backend):
    """Squared loss uses a backend-portable vector operation."""
    graph = Graph.from_tuples([("A", 1, "B"), ("A", 1, "C")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0, "C": 0.0}, ()),
            ("do_A", {"A": 2.0, "B": 3.0, "C": 4.0}, ("A",)),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        loss="squared",
        vertex_weights={"B": 2.0, "C": 3.0},
        sample_weights={"do_A": 4.0},
        backend=backend,
    )

    problem = method.build(graph, data)

    assert "squared_fit" in {objective.name for objective in problem.objectives}


def test_default_flow_capacity_does_not_change_recovery_with_epsilon(backend):
    """Increasing flow epsilon must not impose an artificial branch limit."""
    graph = Graph.from_tuples([("A", 1, "B"), ("A", 1, "C")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0, "C": 0.0}, ()),
            ("do_A", {"A": 1.0, "B": 1.0, "C": 1.0}, ("A",)),
        ]
    )

    selected = []
    for epsilon in (1.0, 2.0):
        method = LinearDAGDiscovery(
            fit_intercept=False,
            flow_epsilon=epsilon,
            backend=backend,
        )
        problem = method.build(graph, data)
        _solve(problem, backend)
        selected.append(method.get_selected_edge_indices())

    assert np.array_equal(selected[0], np.array([0, 1]))
    assert np.array_equal(selected[1], np.array([0, 1]))


def test_explicit_flow_capacity_is_not_scaled_with_epsilon(backend):
    """An explicit capacity remains a genuine user-specified constraint."""
    graph = Graph.from_tuples([("A", 1, "B")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0}, ()),
            ("do_A", {"A": 1.0, "B": 1.0}, ("A",)),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        flow_capacity=3.0,
        flow_epsilon=2.0,
        backend=backend,
    )

    method.build(graph, data)

    assert method._resolved_flow_capacity == 3.0


def test_optional_coverage_allows_intervention_sample_without_sinks(backend):
    """A sample with only intervened measurements need not abort inference."""
    graph = Graph.from_tuples([("A", 1, "B")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0}, ()),
            ("do_AB", {"A": 1.0, "B": 2.0}, ("A", "B")),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        min_commodity_coverage=0.0,
        lambda_unexplained=0.0,
        backend=backend,
    )

    problem = method.build(graph, data)
    result = _solve(problem, backend)

    assert str(result.status).lower() == "optimal"
    assert method._commodity_sink_edges == [[], []]


def test_full_coverage_is_infeasible_with_intervention_without_sinks(backend):
    """A sinkless intervention remains represented but cannot satisfy full coverage."""
    graph = Graph.from_tuples([("A", 1, "B")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0}, ()),
            ("do_AB", {"A": 1.0, "B": 2.0}, ("A", "B")),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        min_commodity_coverage=1.0,
        backend=backend,
    )

    problem = method.build(graph, data)
    result = _solve(problem, backend)

    assert str(result.status).lower() in {"infeasible", "infeasible_or_unbounded"}


def test_exact_coefficient_support_has_nonzero_fitted_coefficients(backend):
    """Exact support keeps selected edges in the fitted coefficient support."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")])
    data = _zero_connector_data()
    method = LinearDAGDiscovery(
        fit_intercept=False,
        min_abs_coefficient=0.25,
        backend=backend,
    )
    assert method.coefficient_support == "exact"
    problem = method.build(graph, data)
    _solve(problem, backend)

    selected = method.get_selected_edge_indices()
    normalized = np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1)

    assert np.array_equal(selected, np.array([0, 1]))
    assert np.all(np.abs(normalized[selected]) >= 0.25 - 1e-8)


def test_default_exact_support_enforces_its_threshold(backend):
    """The default exact-support threshold must survive solver tolerances."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")])
    data = _zero_connector_data()
    method = LinearDAGDiscovery(fit_intercept=False, backend=backend)
    problem = method.build(graph, data)
    _solve(problem, backend)

    selected = method.get_selected_edge_indices()
    normalized = np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1)

    assert np.array_equal(selected, np.array([0, 1]))
    assert np.all(np.abs(normalized[selected]) >= method.min_abs_coefficient)


@pytest.mark.parametrize(
    ("interaction", "effect"),
    [(1, 1.5), (-1, -1.5), (None, -1.5)],
)
def test_exact_support_handles_positive_negative_and_unknown_signs(backend, interaction, effect):
    """Exact support preserves fitted sign information for every prior sign case."""
    graph = Graph.from_tuples([("A", interaction, "B")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0}, ()),
            ("do_A", {"A": 2.0, "B": 2.0 * effect}, ("A",)),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        enforce_signs=True,
        backend=backend,
    )
    problem = method.build(graph, data)
    _solve(problem, backend)

    selected = method.get_selected_edge_indices()
    coefficient = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)[0]
    normalized = np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1)[0]

    assert np.array_equal(selected, np.array([0]))
    assert np.sign(coefficient) == np.sign(effect)
    assert abs(normalized) >= method.min_abs_coefficient


@pytest.mark.parametrize("coefficient_support", ["exact", "structural"])
def test_forced_coverage_distinguishes_zero_response_support(backend, coefficient_support):
    """Exact support rejects zero effects while structural support permits connectors."""
    graph = Graph.from_tuples([("A", 1, "B")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0}, ()),
            ("do_A", {"A": 1.0, "B": 0.0}, ("A",)),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        coefficient_support=coefficient_support,
        min_commodity_coverage=1.0,
        backend=backend,
    )
    problem = method.build(graph, data)
    _solve(problem, backend)

    assert np.array_equal(method.get_selected_edge_indices(), np.array([0]))
    normalized = np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1)[0]
    if coefficient_support == "exact":
        assert abs(normalized) >= method.min_abs_coefficient
    else:
        assert np.isclose(normalized, 0.0, atol=1e-8)


@pytest.mark.parametrize("coefficient_support", ["exact", "structural"])
def test_unselected_edges_have_zero_fitted_coefficients(backend, coefficient_support):
    """Both support modes keep coefficients zero on unselected prior edges."""
    graph = Graph.from_tuples([("A", 1, "B"), ("A", 1, "C")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0, "C": 0.0}, ()),
            ("do_A", {"A": 1.0, "B": 2.0, "C": 0.0}, ("A",)),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        coefficient_support=coefficient_support,
        backend=backend,
    )
    problem = method.build(graph, data)
    _solve(problem, backend)

    selected = method.get_selected_edge_indices()
    normalized = np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1)

    assert np.array_equal(selected, np.array([0]))
    assert np.isclose(normalized[1], 0.0, atol=1e-8)


def test_exact_support_accepts_threshold_equal_to_coefficient_bound(backend):
    """The exact-support threshold may equal the configured coefficient bound."""
    graph = Graph.from_tuples([("A", 1, "B")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0}, ()),
            ("do_A", {"A": 2.0, "B": 2.0}, ("A",)),
        ]
    )
    method = LinearDAGDiscovery(
        coefficient_bound=0.25,
        min_abs_coefficient=0.25,
        fit_intercept=False,
        min_commodity_coverage=1.0,
        backend=backend,
    )
    problem = method.build(graph, data)
    _solve(problem, backend)

    selected = method.get_selected_edge_indices()
    coefficient = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)[0]
    normalized = np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1)[0]

    assert np.array_equal(selected, np.array([0]))
    assert np.isclose(abs(coefficient), 0.25, atol=1e-8)
    assert np.isclose(abs(normalized), 0.25, atol=1e-8)


def test_default_exact_support_is_invariant_to_measurement_rescaling(backend):
    """Normalized coefficient support is unchanged when one variable is rescaled."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")])
    selected = []
    normalized = []
    for scale_b in (1.0, 100.0):
        method = LinearDAGDiscovery(
            fit_intercept=True,
            coefficient_bound=100.0,
            intercept_bound=200.0,
            backend=backend,
        )
        problem = method.build(graph, _zero_connector_data(scale_b=scale_b))
        _solve(problem, backend)
        selected.append(method.get_selected_edge_indices())
        normalized.append(np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1))

    assert np.array_equal(selected[0], np.array([0, 1]))
    assert np.array_equal(selected[0], selected[1])
    assert np.all(np.abs(normalized[0][selected[0]]) >= 0.25)
    assert np.all(np.abs(normalized[1][selected[1]]) >= 0.25)


def test_exact_support_is_invariant_to_intervention_source_rescaling(backend):
    """A source's intervention measurements determine its predictor scale."""
    graph = Graph.from_tuples([("A", 1, "B")])
    selected = []
    normalized = []
    coefficients = []
    for source_scale in (1.0, 100.0):
        data = _data(
            [
                ("obs", {"A": 0.0, "B": 0.0}, ()),
                ("do_A", {"A": source_scale, "B": 1.0}, ("A",)),
            ]
        )
        method = LinearDAGDiscovery(
            fit_intercept=False,
            standardize_loss=False,
            backend=backend,
        )
        problem = method.build(graph, data)
        _solve(problem, backend)
        selected.append(method.get_selected_edge_indices())
        normalized.append(np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1))
        coefficients.append(np.asarray(problem.expr.edge_coefficient.value).reshape(-1))

    assert np.array_equal(selected[0], np.array([0]))
    assert np.array_equal(selected[0], selected[1])
    assert np.allclose(normalized[0], normalized[1], atol=1e-8)
    assert np.isclose(coefficients[0][0], 1.0, atol=1e-8)
    assert np.isclose(coefficients[1][0], 0.01, atol=1e-8)


def test_exact_support_is_invariant_to_leaf_response_rescaling(backend):
    """A terminal response's units must not change normalized support."""
    graph = Graph.from_tuples([("A", 1, "B")])
    selected = []
    coefficients = []
    for response_scale in (1.0, 0.01):
        data = _data(
            [
                ("obs", {"A": 0.0, "B": 0.0}, ()),
                ("do_A", {"A": 2.0, "B": 3.0 * response_scale}, ("A",)),
            ]
        )
        method = LinearDAGDiscovery(fit_intercept=False, backend=backend)
        problem = method.build(graph, data)
        _solve(problem, backend)
        selected.append(method.get_selected_edge_indices())
        coefficients.append(np.asarray(problem.expr.edge_coefficient.value).reshape(-1)[0])

    assert np.array_equal(selected[0], np.array([0]))
    assert np.array_equal(selected[1], selected[0])
    assert np.allclose(coefficients, [1.5, 0.015], atol=1e-8)


def test_tiny_exact_support_threshold_cannot_validate_zero(backend):
    """Validation must not erase a positive threshold through its tolerance."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")])
    method = LinearDAGDiscovery(
        fit_intercept=False,
        min_abs_coefficient=1e-12,
        backend=backend,
    )
    problem = method.build(graph, _zero_connector_data())
    _solve(problem, backend)

    with pytest.raises(ValueError, match="invalid"):
        method.get_selected_edge_indices()
    with pytest.raises(ValueError, match="invalid"):
        method.get_edge_usage()


def test_exact_coefficient_support_works_with_different_bounds(backend):
    """Exact support remains meaningful across raw coefficient bounds."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")])
    for coefficient_bound in (0.6, 4.0):
        method = LinearDAGDiscovery(
            fit_intercept=False,
            coefficient_bound=coefficient_bound,
            min_abs_coefficient=0.25,
            backend=backend,
        )
        problem = method.build(graph, _zero_connector_data())
        _solve(problem, backend)

        selected = method.get_selected_edge_indices()
        coefficients = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)
        normalized = np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1)

        assert np.array_equal(selected, np.array([0, 1]))
        assert np.all(np.abs(normalized[selected]) >= 0.25 - 1e-8)
    assert np.all(np.abs(coefficients) <= coefficient_bound + 1e-8)


def test_normalized_threshold_is_not_compared_to_raw_bound(backend):
    """A normalized threshold may exceed the raw bound when predictor units differ."""
    graph = Graph.from_tuples([("A", 1, "B")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0}, ()),
            ("do_A", {"A": 10.0, "B": 5.0}, ("A",)),
        ]
    )
    method = LinearDAGDiscovery(
        coefficient_bound=0.5,
        min_abs_coefficient=1.0,
        fit_intercept=False,
        standardize_loss=False,
        backend=backend,
    )
    problem = method.build(graph, data)
    _solve(problem, backend)

    selected = method.get_selected_edge_indices()
    coefficient = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)[0]
    normalized = np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1)[0]

    assert np.array_equal(selected, np.array([0]))
    assert np.isclose(coefficient, 0.5, atol=1e-8)
    assert np.isclose(normalized, 1.0, atol=1e-8)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True])
def test_rejects_invalid_min_abs_coefficient(value):
    """The exact-support threshold must be finite, positive, and non-boolean."""
    with pytest.raises((TypeError, ValueError)):
        LinearDAGDiscovery(coefficient_bound=5.0, min_abs_coefficient=value)


def test_structural_coefficient_support_allows_zero_connector(backend):
    """Structural support preserves the connector-edge interpretation."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 1.0, "C": 2.0}, ()),
            ("do_A", {"A": 1.0, "B": 0.0, "C": 0.0}, ("A",)),
        ]
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        coefficient_support="structural",
        backend=backend,
    )
    problem = method.build(graph, data)
    _solve(problem, backend)

    selected = method.get_selected_edge_indices()
    coefficients = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)

    assert np.array_equal(selected, np.array([0, 1]))
    assert np.isclose(coefficients[0], 0.0, atol=1e-8)

    solution_edges = method.get_solution_graph().get_attr_edges()
    assert solution_edges[0]["connector"] is True
    assert solution_edges[0]["interaction"] == 0
    assert solution_edges[0]["inferred_interaction"] == 0
    assert solution_edges[0]["prior_interaction"] == 1
    assert solution_edges[1]["connector"] is False


def test_solution_graph_does_not_retain_conflicting_prior_sign(backend):
    """Solution metadata must expose fitted effects separately from prior signs."""
    graph = Graph.from_tuples([("A", 1, "B")])
    data = _data(
        [
            ("obs", {"A": 0.0, "B": 0.0}, ()),
            ("do_A", {"A": 2.0, "B": -3.0}, ("A",)),
        ]
    )
    method = LinearDAGDiscovery(fit_intercept=False, backend=backend)
    problem = method.build(graph, data)
    _solve(problem, backend)

    coefficient = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)[0]
    normalized = np.asarray(problem.expr.edge_coefficient_normalized.value).reshape(-1)[0]
    solution_edge = method.get_solution_graph().get_attr_edge(0)

    assert coefficient < 0
    assert solution_edge["interaction"] < 0
    assert solution_edge["inferred_interaction"] == -1
    assert np.isclose(solution_edge["coefficient"], coefficient, atol=1e-8)
    assert np.isclose(solution_edge["normalized_coefficient"], normalized, atol=1e-8)
    assert solution_edge["prior_interaction"] == 1
