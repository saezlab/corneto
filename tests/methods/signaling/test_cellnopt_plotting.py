from pathlib import Path

import nbformat
import numpy as np
import pytest
from nbclient import NotebookClient

from corneto._plotting import to_dot_source
from corneto.graph import Graph
from corneto.methods.signaling import (
    CellNOptDAG,
    plot_cellnopt_fit,
    plot_cellnopt_model,
)
from corneto.methods.signaling.cellnopt_plotting import (
    _build_cellnopt_model_plot,
)


def _solve(backend, graph, *, inputs, measurements, inhibitors=None, lambda_reg=1e-3):
    method = CellNOptDAG(lambda_reg=lambda_reg, backend=backend)
    problem = method.build_many(
        graph,
        inputs=inputs,
        measurements=measurements,
        inhibitors=inhibitors,
    )
    result = problem.solve()
    assert result.status == "optimal"
    return method, problem


def test_model_plot_expands_and_preserves_or_and_inhibition(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "AND1"),
            ("B", -1, "AND1"),
            ("AND1", 1, "Y"),
            ("C", 1, "Y"),
        ]
    )
    method, problem = _solve(
        backend,
        graph,
        inputs={
            "and_route": {"A": 1, "B": 0, "C": 0},
            "or_route": {"A": 0, "B": 0, "C": 1},
        },
        measurements={
            "and_route": {"Y": 1},
            "or_route": {"Y": 1},
        },
    )

    spec = _build_cellnopt_model_plot(method, problem, width_by="flow")
    and_nodes = [vertex for vertex in spec.graph.V if str(vertex).startswith("__cellnopt_and_")]
    assert len(and_nodes) == 1
    and_node = and_nodes[0]
    assert any(source == frozenset({"C"}) and target == frozenset({"Y"}) for source, target in spec.graph.E)
    assert any(source == frozenset({"A"}) and target == frozenset({and_node}) for source, target in spec.graph.E)
    assert any(source == frozenset({and_node}) and target == frozenset({"Y"}) for source, target in spec.graph.E)

    negative_edges = [
        index for index, attrs in enumerate(spec.graph.get_attr_edges()) if attrs.get("interaction") == -1
    ]
    assert len(negative_edges) == 1
    assert spec.edge_attributes[negative_edges[0]]["arrowhead"] == "tee"
    assert spec.edge_attributes[negative_edges[0]]["color"] == "#C43C39"
    aggregate_edges = [
        attrs for attrs in spec.edge_attributes.values() if "aggregate structural flow" in attrs["tooltip"]
    ]
    assert len(aggregate_edges) == 1
    assert np.isfinite(float(aggregate_edges[0]["penwidth"]))

    dot = to_dot_source(
        spec.graph,
        graph_attr=spec.graph_attributes,
        node_attr=spec.node_attributes,
        custom_edge_attr=spec.edge_attributes,
        custom_vertex_attr=spec.vertex_attributes,
    )
    assert 'label="AND"' in dot
    assert 'arrowhead="tee"' in dot
    assert dot.count('-> "Y"') == 2


def test_condition_plot_distinguishes_active_and_selected_inactive_reactions(backend):
    graph = Graph.from_tuples([("A", 1, "Y")])
    method, problem = _solve(
        backend,
        graph,
        inputs={"off": {"A": 0}, "on": {"A": 1}},
        measurements={"off": {"Y": 0}, "on": {"Y": 1}},
    )

    off = _build_cellnopt_model_plot(method, problem, condition="off")
    on = _build_cellnopt_model_plot(method, problem, condition="on")
    hidden = _build_cellnopt_model_plot(
        method,
        problem,
        condition="off",
        show_inactive=False,
    )

    assert off.edge_attributes[0]["style"] == "dashed"
    assert "active=0" in off.edge_attributes[0]["tooltip"]
    assert on.edge_attributes[0]["style"] == "solid"
    assert "active=1" in on.edge_attributes[0]["tooltip"]
    assert hidden.graph.num_edges == 0
    assert set(hidden.graph.V) == {"A", "Y"}


def test_model_plot_can_reveal_unselected_reactions(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "Y"),
            ("C", 1, "D"),
        ]
    )
    method, problem = _solve(
        backend,
        graph,
        inputs={"condition": {"A": 1, "C": 1}},
        measurements={"condition": {"Y": 1}},
    )

    selected = _build_cellnopt_model_plot(method, problem)
    complete = _build_cellnopt_model_plot(method, problem, show_unselected=True)

    assert selected.graph.num_edges == 1
    assert complete.graph.num_edges == 2
    unselected = [attrs for attrs in complete.edge_attributes.values() if "selected=0" in attrs["tooltip"]]
    assert len(unselected) == 1
    assert unselected[0]["style"] == "dotted"


def test_fit_views_support_uneven_measurements_and_named_subsets(backend):
    pytest.importorskip("matplotlib")
    graph = Graph.from_tuples(
        [
            ("A", 1, "Y"),
            ("A", 1, "Z"),
        ]
    )
    method, problem = _solve(
        backend,
        graph,
        inputs={
            "very_long_enabled_condition": {"A": 1},
            "blocked": {"A": 1},
        },
        inhibitors={
            "very_long_enabled_condition": {},
            "blocked": {"Y": 1},
        },
        measurements={
            "very_long_enabled_condition": {"Y": 0.8},
            "blocked": {"Z": 1},
        },
    )

    grid_figure, grid_axes = plot_cellnopt_fit(method, problem)
    inferred_figure, inferred_axes = plot_cellnopt_fit(
        method,
        problem,
        signals=["A"],
    )
    heatmap_figure, heatmap_axes = plot_cellnopt_fit(
        method,
        problem,
        view="heatmap",
        conditions=["blocked"],
        signals=["Z"],
    )

    assert grid_axes.shape == (2, 3)
    assert inferred_axes.shape == (2, 2)
    assert heatmap_axes.shape == (4,)
    assert grid_figure is grid_axes[0, 0].figure
    assert inferred_figure is inferred_axes[0, 0].figure
    assert heatmap_figure is heatmap_axes[0].figure

    measured_axis = grid_axes[0, 0]
    assert [line.get_label() for line in measured_axis.lines] == [
        "_nolegend_",
        "Model",
        "Observed",
    ]
    assert all(np.allclose(line.get_xdata(), 0) for line in measured_axis.lines)
    assert np.allclose(measured_axis.lines[-2].get_ydata(), [1])
    assert np.allclose(measured_axis.lines[-1].get_ydata(), [0.8])
    assert [tick.get_text() for tick in grid_axes[-1, 0].get_xticklabels()] == ["Endpoint"]

    prediction_only_axis = grid_axes[0, 1]
    assert [line.get_label() for line in prediction_only_axis.lines] == ["Model"]
    assert all([line.get_label() for line in inferred_axes[row, 0].lines] == ["Model"] for row in range(2))
    assert grid_axes[1, 0].patch.get_hatch() == "//"
    assert prediction_only_axis.patch.get_hatch() == "//"


def test_fit_grid_handles_one_condition_and_one_signal(backend):
    pytest.importorskip("matplotlib")
    method, problem = _solve(
        backend,
        Graph.from_tuples([("A", 1, "Y")]),
        inputs={"single_endpoint": {"A": 1}},
        measurements={"single_endpoint": {"Y": 0.2}},
    )

    figure, axes = plot_cellnopt_fit(method, problem)

    assert axes.shape == (1, 2)
    assert figure is axes[0, 0].figure
    assert [tick.get_text() for tick in axes[0, 0].get_xticklabels()] == ["Endpoint"]
    assert [line.get_label() for line in axes[0, 0].lines] == [
        "_nolegend_",
        "Model",
        "Observed",
    ]
    assert np.allclose(axes[0, 0].lines[-2].get_ydata(), [0])
    assert np.allclose(axes[0, 0].lines[-1].get_ydata(), [0.2])


def test_networkx_model_plot_returns_real_figure(backend):
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")
    method, problem = _solve(
        backend,
        Graph.from_tuples([("A", -1, "Y")]),
        inputs={"condition": {"A": 0}},
        measurements={"condition": {"Y": 1}},
    )

    figure = plot_cellnopt_model(method, problem, renderer="networkx")

    assert len(figure.axes) == 1
    assert {"A", "Y"} <= {text.get_text() for text in figure.axes[0].texts}
    assert len(figure.axes[0].patches) == 1


def test_plotting_requires_a_solved_problem():
    method = CellNOptDAG(lambda_reg=0)
    method.build(
        Graph.from_tuples([("A", 1, "Y")]),
        inputs={"A": 1},
        measurements={"Y": 1},
    )

    with pytest.raises(ValueError, match="Solve a feasible problem"):
        plot_cellnopt_model(method, renderer="networkx")
    with pytest.raises(ValueError, match="Solve a feasible problem"):
        plot_cellnopt_fit(method)


def test_cellnopt_worked_notebook_executes_every_cell():
    notebook_path = Path(__file__).parents[3] / "docs/guide/signaling/cellnopt_dag.ipynb"
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
