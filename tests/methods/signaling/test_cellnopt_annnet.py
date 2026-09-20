import pytest

from corneto.methods.signaling import CellNOptDAG
from corneto.methods.signaling.annnet import (
    add_cellnopt_conditions,
    add_cellnopt_results,
    build_cellnopt_from_annnet,
)

annnet = pytest.importorskip("annnet")


def test_cellnopt_reads_conditions_and_adds_results_to_annnet(backend):
    graph = annnet.AnnNet(directed=True)
    graph.history.enable(True)
    graph.slices.add("prior")
    graph.add_edges(
        [
            {"source": "L", "target": "A", "edge_id": "e0"},
            {"source": "A", "target": "Y", "edge_id": "e1"},
            {"source": "L", "target": "Y", "edge_id": "shortcut"},
        ],
        slice="prior",
        default_edge_directed=True,
    )
    for edge_id in graph.edges():
        graph.attrs.set_edge_attrs(edge_id, interaction=1)
    layers = add_cellnopt_conditions(
        graph,
        inputs={"off": {"L": 0}, "on": {"L": 1}, "blocked": {"L": 1}},
        inhibitors={"off": {}, "on": {}, "blocked": {"A": 1}},
        measurements={"off": {"Y": 0}, "on": {"Y": 1}, "blocked": {"Y": 0}},
    )

    method = CellNOptDAG(lambda_reg=1e-3, backend=backend)
    problem = build_cellnopt_from_annnet(method, graph, network_slice="prior")
    solution = problem.solve()
    summary = add_cellnopt_results(graph, method, problem, solution=solution)

    assert solution.status == "optimal"
    assert layers == {"off": ("off",), "on": ("on",), "blocked": ("blocked",)}
    assert graph.layers.get_vertex_layer_attrs("L", ("on",))["input"] == 1
    assert graph.layers.get_vertex_layer_attrs("A", ("blocked",))["inhibited"] == 1
    assert graph.layers.get_vertex_layer_attrs("Y", ("on",))["predicted"] == 1
    assert graph.layers.get_layer_attrs(("blocked",))["endpoint_absolute_error"] == 0
    assert graph.slices.exists("cellnopt_selected")
    assert summary["selected_reactions"] == 2
    assert summary["condition_errors"] == {"off": 0.0, "on": 0.0, "blocked": 0.0}
