"""Tests for flow-supported linear DAG discovery."""

from pathlib import Path

import nbformat
import numpy as np
import pytest
from nbclient import NotebookClient

from corneto.backend import CvxpyBackend
from corneto.data import Data
from corneto.graph import Graph
from corneto.methods.causal import LinearDAGDiscovery


def _single_edge_data(effect: float) -> Data:
    return Data.from_cdict(
        {
            "obs": {
                "A": {"mapping": "vertex", "value": 0.0},
                "B": {"mapping": "vertex", "value": 0.0},
            },
            "do_A": {
                "A": {"mapping": "vertex", "value": 2.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 2.0 * effect},
            },
        }
    )


def _four_node_interventional_data() -> Data:
    """Identifiable noiseless SEM data with interventions on every parent."""
    samples = {}

    def add_sample(name, a, b=None, c=None, *, intervened=()):
        if b is None:
            b = 1.2 * a
        if c is None:
            c = -0.8 * a
        d = b + 0.7 * c
        values = {"A": a, "B": b, "C": c, "D": d}
        samples[name] = {
            vertex: {
                "mapping": "vertex",
                "value": value,
                **({"intervened": True} if vertex in intervened else {}),
            }
            for vertex, value in values.items()
        }

    for index, a in enumerate((-1.5, -0.4, 0.8, 1.7)):
        add_sample(f"obs_{index}", a)
    add_sample("do_A_low", -2.0, intervened={"A"})
    add_sample("do_A_high", 2.2, intervened={"A"})
    add_sample("do_B_high", -0.7, b=2.1, intervened={"B"})
    add_sample("do_B_low", 1.1, b=-1.4, intervened={"B"})
    add_sample("do_C_high", -1.0, c=1.8, intervened={"C"})
    add_sample("do_C_low", 0.6, c=-1.7, intervened={"C"})
    return Data.from_cdict(samples)


def test_recovers_supported_linear_edge(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    method = LinearDAGDiscovery(
        lambda_edges=0.01,
        fit_intercept=False,
        enforce_signs=True,
        backend=backend,
    )
    problem = method.build(graph, _single_edge_data(effect=1.5))
    problem.solve()

    assert np.array_equal(method.get_selected_edge_indices(), np.array([0]))
    coefficient = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)
    assert np.isclose(coefficient[0], 1.5, atol=1e-7)
    assert np.array_equal(method.get_edge_usage(), np.array([[True]]))
    assert np.isclose(problem.expr.flow.value[0, 0], 1.0, atol=1e-7)


def test_recovers_identifiable_dag_from_cyclic_prior(backend):
    """Recover a multi-level SEM despite reverse and cross-edge distractors."""
    edges = [
        ("A", 1, "B"),
        ("A", -1, "C"),
        ("B", 1, "D"),
        ("C", 1, "D"),
        ("B", 1, "A"),
        ("C", 1, "A"),
        ("D", 1, "B"),
        ("D", 1, "C"),
        ("B", 1, "C"),
        ("C", 1, "B"),
    ]
    graph = Graph.from_tuples(edges)
    method = LinearDAGDiscovery(
        lambda_edges=0.01,
        coefficient_bound=3,
        fit_intercept=False,
        max_parents=2,
        enforce_signs=True,
        backend=backend,
    )
    problem = method.build(graph, _four_node_interventional_data())
    problem.solve()

    selected = set(method.get_selected_edge_indices())
    assert selected == {0, 1, 2, 3}
    method.get_solution_graph().toposort()

    coefficient = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)
    assert np.allclose(coefficient[:4], [1.2, -0.8, 1.0, 0.7], atol=1e-7)
    assert np.allclose(coefficient[4:], 0, atol=1e-8)

    usage = method.get_edge_usage()
    selected_values = np.asarray(problem.expr.edge_selected.value).reshape(-1) > 0.5
    assert np.array_equal(usage.any(axis=1), selected_values)
    assert np.all(usage[selected_values].sum(axis=1) >= 1)


def test_optional_coverage_does_not_force_no_effect_edge(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    method = LinearDAGDiscovery(
        lambda_edges=0.1,
        fit_intercept=False,
        backend=backend,
    )
    problem = method.build(graph, _single_edge_data(effect=0.0))
    problem.solve()

    assert method.get_selected_edge_indices().size == 0
    coefficient = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)
    assert np.isclose(coefficient[0], 0.0, atol=1e-8)
    assert np.allclose(problem.expr.flow.value, 0.0, atol=1e-8)
    assert "commodity_active" not in problem.expr


def test_one_commodity_per_intervened_vertex(backend):
    graph = Graph.from_tuples([("A", 1, "C"), ("B", 1, "C")])
    data = Data.from_cdict(
        {
            "condition": {
                "A": {"mapping": "vertex", "value": 1.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 2.0, "intervened": True},
                "C": {"mapping": "vertex", "value": 3.0},
            }
        }
    )
    method = LinearDAGDiscovery(
        lambda_edges=0.01,
        fit_intercept=False,
        min_commodity_coverage=1.0,
        backend=backend,
    )
    problem = method.build(graph, data)
    problem.solve()

    assert method._commodity_sources == ["A", "B"]
    assert method.get_edge_usage().shape == (2, 2)
    assert np.all(np.asarray(problem.expr.commodity_active.value) > 0.5)


def test_coverage_distinguishes_unreachable_intervention(backend):
    """Partial coverage drops an isolated commodity; strict coverage is infeasible."""
    graph = Graph.from_tuples([("A", 1, "C"), ("D", 1, "B")])
    data = Data.from_cdict(
        {
            "condition": {
                "A": {"mapping": "vertex", "value": 2.0, "intervened": True},
                "B": {"mapping": "vertex", "value": -1.0, "intervened": True},
                "C": {"mapping": "vertex", "value": 4.0},
                "D": {"mapping": "vertex", "value": 0.0},
            }
        }
    )
    method = LinearDAGDiscovery(
        lambda_edges=0.01,
        fit_intercept=False,
        min_commodity_coverage=0.5,
        backend=backend,
    )
    problem = method.build(graph, data)
    problem.solve()

    active = np.asarray(problem.expr.commodity_active.value).reshape(-1) > 0.5
    active_by_source = dict(zip(method._commodity_sources, active, strict=True))
    assert active_by_source == {"A": True, "B": False}
    assert np.array_equal(method.get_selected_edge_indices(), np.array([0]))

    strict_problem = LinearDAGDiscovery(
        fit_intercept=False,
        min_commodity_coverage=1,
        backend=backend,
    ).build(graph, data)
    solve_options = {"primals": None} if str(backend) == "PICOS" else {}
    result = strict_problem.solve(**solve_options)
    assert str(result.status).lower() in {"infeasible", "infeasible_or_unbounded"}


def test_recovers_chain_with_missing_response(backend):
    """A missing sink response is omitted without losing other equations."""
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")])
    data = Data.from_cdict(
        {
            "obs": {
                "A": {"mapping": "vertex", "value": 0.0},
                "B": {"mapping": "vertex", "value": 0.0},
                "C": {"mapping": "vertex", "value": 0.0},
            },
            "do_A_1": {
                "A": {"mapping": "vertex", "value": 1.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 2.0},
                "C": {"mapping": "vertex", "value": 6.0},
            },
            "do_A_missing_C": {
                "A": {"mapping": "vertex", "value": 2.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 4.0},
                "C": {"mapping": "vertex", "value": None},
            },
            "do_A_missing_B": {
                "A": {"mapping": "vertex", "value": 2.5, "intervened": True},
                "B": {"mapping": "vertex", "value": None},
                "C": {"mapping": "vertex", "value": 999.0},
            },
            "do_A_3": {
                "A": {"mapping": "vertex", "value": 3.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 6.0},
                "C": {"mapping": "vertex", "value": 18.0},
            },
        }
    )
    method = LinearDAGDiscovery(
        fit_intercept=False,
        coefficient_bound=4,
        backend=backend,
    )
    problem = method.build(graph, data)
    problem.solve()

    assert np.array_equal(method.get_selected_edge_indices(), np.array([0, 1]))
    coefficient = np.asarray(problem.expr.edge_coefficient.value).reshape(-1)
    assert np.allclose(coefficient, [2, 3], atol=1e-7)
    c_index = method._vertex_index["C"]
    missing_sample = method._sample_names.index("do_A_missing_C")
    assert not method._valid_residual[c_index, missing_sample]
    missing_parent_sample = method._sample_names.index("do_A_missing_B")
    b_index = method._vertex_index["B"]
    assert not method._valid_residual[b_index, missing_parent_sample]
    assert not method._valid_residual[c_index, missing_parent_sample]


def test_standardized_weighted_loss_is_unit_invariant(backend):
    """Response units must not change discovery under standardized loss."""

    def scaled_data(scale):
        data = _single_edge_data(effect=1.5)
        for sample in data.samples.values():
            for feature in sample.features:
                if feature.id == "B":
                    feature.data["value"] *= scale
        return data

    selected = []
    scales = []
    for unit_scale in (1.0, 1000.0):
        method = LinearDAGDiscovery(
            lambda_edges=0.5,
            coefficient_bound=2000,
            fit_intercept=False,
            standardize_loss=True,
            sample_weights={"do_A": 2.0},
            backend=backend,
        )
        problem = method.build(
            Graph.from_tuples([("A", 1, "B")]),
            scaled_data(unit_scale),
        )
        problem.solve()
        selected.append(method.get_selected_edge_indices())
        scales.append(np.asarray(problem.expr.loss_scale.value).reshape(-1)[1])

    assert np.array_equal(selected[0], np.array([0]))
    assert np.array_equal(selected[0], selected[1])
    assert np.isclose(scales[1] / scales[0], 1000)


def test_rejects_fully_latent_vertex(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    data = Data.from_cdict(
        {
            "condition": {
                "A": {"mapping": "vertex", "value": 1.0, "intervened": True},
            }
        }
    )
    with pytest.raises(ValueError, match="latent vertices"):
        LinearDAGDiscovery(backend=backend).build(graph, data)


def test_cvxpy_formulation_uses_only_required_integer_variables():
    """Boundary edges and commodity unions must not introduce extra binaries."""
    graph = Graph.from_tuples([("A", 1, "C"), ("B", 1, "C")])
    data = Data.from_cdict(
        {
            "condition": {
                "A": {"mapping": "vertex", "value": 1.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 2.0, "intervened": True},
                "C": {"mapping": "vertex", "value": 3.0},
            }
        }
    )
    problem = LinearDAGDiscovery(
        fit_intercept=False,
        backend=CvxpyBackend(),
    ).build(graph, data)
    cvxpy_problem = problem.solve(solver="SCIPY")

    boolean_count = sum(variable.size for variable in cvxpy_problem.variables() if variable.attributes["boolean"])
    integer_count = sum(variable.size for variable in cvxpy_problem.variables() if variable.attributes["integer"])
    continuous_count = cvxpy_problem.size_metrics.num_scalar_variables - boolean_count - integer_count

    # E shared selectors + E*K edge/commodity selectors + two coefficient-sign
    # binaries per edge from the default exact coefficient support. Optional
    # commodity activity binaries are omitted when no coverage term is requested.
    assert boolean_count == 2 + 2 * 2 + 2 * 2
    assert integer_count == 0
    # (E + 3 boundary edges)*K flows + V DAG layers + E coefficients
    # + one L1 residual for the sole non-intervened equation.
    assert continuous_count == 5 * 2 + 3 + 2 + 1
    assert cvxpy_problem.size_metrics.num_scalar_variables == 26


def test_linear_dag_guide_notebook_executes_every_cell():
    notebook_path = Path(__file__).parents[3] / "docs/guide/signaling/linear-dag-discovery.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)

    executed = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    ).execute()

    errors = [
        output for cell in executed.cells for output in cell.get("outputs", []) if output.get("output_type") == "error"
    ]
    assert errors == []
