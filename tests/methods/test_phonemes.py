import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from corneto import Data, Graph
from corneto.graph import EdgeType
from corneto.methods import PHONEMeS


def _is_acyclic(vertices, selected_edges):
    successors = {vertex: [] for vertex in vertices}
    indegree = {vertex: 0 for vertex in vertices}
    for source, target in selected_edges:
        successors[source].append(target)
        indegree[target] += 1
    queue = [vertex for vertex, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        source = queue.pop()
        visited += 1
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(vertices)


def _feasible_subsets(graph, targets, measured):
    edges = [(next(iter(source)), next(iter(target))) for source, target in graph.E]
    target_set = set(targets)
    measured_set = set(measured)
    internal_set = set(graph.V) - target_set - measured_set
    feasible = []
    for bits in itertools.product((0, 1), repeat=graph.num_edges):
        selected_edges = [edge for edge, selected in zip(edges, bits) if selected]
        selected_vertices = target_set | {vertex for edge in selected_edges for vertex in edge}
        if not _is_acyclic(graph.V, selected_edges):
            continue
        incoming = {vertex: 0 for vertex in graph.V}
        outgoing = {vertex: 0 for vertex in graph.V}
        for source, target in selected_edges:
            outgoing[source] += 1
            incoming[target] += 1
        if any(outgoing[target] < 1 for target in target_set):
            continue
        if any(incoming[vertex] < 1 or outgoing[vertex] < 1 for vertex in internal_set & selected_vertices):
            continue
        if any(incoming[vertex] < 1 for vertex in measured_set & selected_vertices):
            continue
        feasible.append((np.array(bits), selected_vertices))
    return feasible


def _brute_force_objective(graph, perturbations, scores, edge_costs):
    conditions = tuple(perturbations)
    candidates = [_feasible_subsets(graph, perturbations[condition], scores[condition]) for condition in conditions]
    best = np.inf
    for selection in itertools.product(*candidates):
        edge_union = np.maximum.reduce([candidate[0] for candidate in selection])
        node_cost = sum(
            sum(value for vertex, value in scores[condition].items() if vertex in selection[index][1])
            for index, condition in enumerate(conditions)
        )
        best = min(best, node_cost + edge_costs @ edge_union)
    return best


def _objective_value(problem):
    return sum(float(objective.value) * float(weight) for objective, weight in zip(problem.objectives, problem.weights))


def _solve(problem):
    problem.solve()
    assert all(objective.value is not None for objective in problem.objectives)


def test_phonemes_matches_reference_mtor_case(backend):
    fixture_path = Path(__file__).parent / "data" / "phonemes_mtor.json"
    fixture = json.loads(fixture_path.read_text())
    graph = Graph()
    graph.add_edges(fixture["edges"])

    assert graph.num_edges == 229
    assert len(fixture["phosphosite_scores"]) == 17
    assert fixture["provenance"]["commit"] == "1117662a0128454a66e250100c613e6caf34dd69"

    problem = PHONEMeS(
        default_edge_cost=fixture["edge_cost"],
        backend=backend,
    ).build(
        graph,
        perturbations=[fixture["target"]],
        phosphosite_scores=fixture["phosphosite_scores"],
    )
    _solve(problem)

    assert round(_objective_value(problem), 2) == round(fixture["expected_objective"], 2)
    selected = np.asarray(problem.expr.edge_selected.value)[:, 0] > 0.5
    assert selected.any()

    reachable = {fixture["target"]}
    selected_edges = [edge for edge, is_selected in zip(fixture["edges"], selected) if is_selected]
    while True:
        previous_size = len(reachable)
        reachable.update(target for source, target in selected_edges if source in reachable)
        if len(reachable) == previous_size:
            break
    assert all(source in reachable and target in reachable for source, target in selected_edges)


def test_phonemes_matches_single_condition_bruteforce(backend):
    graph = Graph()
    graph.add_edges([("r", "a"), ("r", "b"), ("a", "m"), ("b", "m"), ("a", "b")])
    scores = {"condition": {"m": -4.0}}
    perturbations = {"condition": ["r"]}
    costs = np.array([0.7, 0.6, 0.3, 0.4, -0.2])

    method = PHONEMeS(backend=backend)
    problem = method.build(
        graph,
        perturbations=perturbations["condition"],
        phosphosite_scores=scores["condition"],
        edge_costs=dict(enumerate(costs)),
    )
    _solve(problem)

    expected = _brute_force_objective(graph, perturbations, scores, costs)
    assert _objective_value(problem) == pytest.approx(expected)


def test_phonemes_matches_multicondition_bruteforce_and_union_cost(backend):
    graph = Graph()
    graph.add_edges([("r1", "a"), ("r2", "a"), ("a", "m")])
    perturbations = {"first": ["r1"], "second": ["r2"]}
    scores = {"first": {"m": -3.0}, "second": {"m": -3.0}}
    costs = np.array([0.4, 0.5, 2.0])

    method = PHONEMeS(backend=backend)
    problem = method.build_many(
        graph,
        perturbations=perturbations,
        phosphosite_scores=scores,
        edge_costs=dict(enumerate(costs)),
    )
    _solve(problem)

    expected = _brute_force_objective(graph, perturbations, scores, costs)
    assert _objective_value(problem) == pytest.approx(expected)
    assert np.asarray(problem.expr.edge_selected_any.value).reshape(-1).tolist() == pytest.approx([1, 1, 1])
    assert float(problem.objectives[1].value) == pytest.approx(costs.sum())


def test_phonemes_supports_convergent_dag_and_rejects_cycle(backend):
    graph = Graph()
    graph.add_edges([("r", "a"), ("r", "b"), ("a", "m"), ("b", "m"), ("a", "b"), ("b", "a")])
    method = PHONEMeS(default_edge_cost=-0.1, backend=backend)
    problem = method.build(graph, perturbations=["r"], phosphosite_scores={"m": -5})
    _solve(problem)

    selected = np.asarray(problem.expr.edge_selected.value)[:, 0]
    assert selected[2] + selected[3] == pytest.approx(2)
    assert selected[4] + selected[5] <= 1 + 1e-6


def test_internal_nodes_cannot_extract_flow(backend):
    graph = Graph()
    graph.add_edges([("r", "m"), ("r", "dead")])
    method = PHONEMeS(default_edge_cost=0, backend=backend)
    problem = method.build(
        graph,
        perturbations=["r"],
        phosphosite_scores={"m": -1},
        edge_costs={1: -10},
    )
    _solve(problem)

    assert np.asarray(problem.expr.edge_selected.value)[1, 0] == pytest.approx(0)
    assert "dead" not in method._phosphosite_outflow_edges


def test_zero_scored_site_is_sink_and_measured_site_can_be_intermediate(backend):
    graph = Graph()
    graph.add_edges([("r", "middle"), ("middle", "sink")])
    method = PHONEMeS(default_edge_cost=0, backend=backend)
    problem = method.build(
        graph,
        perturbations=["r"],
        phosphosite_scores={"middle": 0, "sink": -1},
    )
    _solve(problem)

    vertex_index = {vertex: index for index, vertex in enumerate(graph.V)}
    selected = np.asarray(problem.expr.vertex_selected.value)[:, 0]
    assert selected[vertex_index["middle"]] == pytest.approx(1)
    assert selected[vertex_index["sink"]] == pytest.approx(1)
    assert set(method._phosphosite_outflow_edges) == {"middle", "sink"}


def test_target_phosphosite_overlap_satisfies_both_roles(backend):
    graph = Graph()
    graph.add_edges([("r1", "r2"), ("r2", "m")])
    method = PHONEMeS(default_edge_cost=0, backend=backend)
    problem = method.build(
        graph,
        perturbations=["r1", "r2"],
        phosphosite_scores={"r2": 0, "m": -1},
    )
    _solve(problem)

    assert np.asarray(problem.expr.edge_selected.value)[:, 0].tolist() == pytest.approx([1, 1])


def test_auxiliary_edges_are_excluded_and_shapes_are_stable(backend):
    graph = Graph()
    graph.add_edges([("r", "a"), ("a", "m")])
    method = PHONEMeS(backend=backend)
    problem = method.build(graph, perturbations=["r"], phosphosite_scores={"m": -1})

    assert problem.expr.flow.shape == (4, 1)
    assert problem.expr.edge_selected.shape == (2, 1)
    assert problem.expr.vertex_selected.shape == (3, 1)
    assert problem.expr.edge_selected_any.shape in {(2,), (2, 1)}
    assert problem.expr.dag_layer.shape == (3, 1)
    assert method.processed_graph.num_edges == graph.num_edges + 2


def test_constraint_blocks_do_not_scale_with_graph_or_conditions(backend):
    def build_problem(num_internal):
        graph = Graph()
        path = ["r", *(f"v{i}" for i in range(num_internal)), "m"]
        graph.add_edges(list(itertools.pairwise(path)))
        return PHONEMeS(backend=backend).build_many(
            graph,
            perturbations={"a": ["r"], "b": ["r"], "c": ["r"]},
            phosphosite_scores={"a": {"m": -1}, "b": {"m": -1}, "c": {"m": -1}},
        )

    assert len(build_problem(1).constraints) == len(build_problem(12).constraints)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"perturbations": [], "phosphosite_scores": {"m": 0}}, "must not be empty"),
        ({"perturbations": ["missing"], "phosphosite_scores": {"m": 0}}, "Unknown vertex"),
        ({"perturbations": ["r"], "phosphosite_scores": {}}, "must not be empty"),
        ({"perturbations": ["r"], "phosphosite_scores": {"m": np.inf}}, "must be finite"),
        (
            {"perturbations": ["r"], "phosphosite_scores": {"m": 0}, "edge_costs": {99: 1}},
            "Invalid edge index",
        ),
    ],
)
def test_phonemes_validates_explicit_inputs(kwargs, message, backend):
    graph = Graph()
    graph.add_edge("r", "m")
    with pytest.raises((TypeError, ValueError), match=message):
        PHONEMeS(backend=backend).build(graph, **kwargs)


def test_phonemes_rejects_invalid_graphs_and_sink_targets(backend):
    undirected = Graph()
    undirected.add_edge("r", "m", type=EdgeType.UNDIRECTED)
    with pytest.raises(ValueError, match="requires directed edges"):
        PHONEMeS(backend=backend).build(
            undirected,
            perturbations=["r"],
            phosphosite_scores={"m": 0},
        )

    directed = Graph()
    directed.add_edge("r", "m")
    with pytest.raises(ValueError, match="has no outgoing PKN interaction"):
        PHONEMeS(backend=backend).build(
            directed,
            perturbations=["m"],
            phosphosite_scores={"m": 0},
        )


def test_build_from_data_requires_global_edge_costs(backend):
    graph = Graph()
    graph.add_edge("r", "m")
    data = Data.from_dict(
        {
            "one": {
                "features": [
                    {"id": "r", "mapping": "vertex", "role": "perturbation"},
                    {"id": "m", "mapping": "vertex", "role": "phosphosite", "value": 0},
                    {"id": 0, "mapping": "edge", "value": 1},
                ]
            },
            "two": {
                "features": [
                    {"id": "r", "mapping": "vertex", "role": "perturbation"},
                    {"id": "m", "mapping": "vertex", "role": "phosphosite", "value": 0},
                    {"id": 0, "mapping": "edge", "value": 2},
                ]
            },
        }
    )
    with pytest.raises(ValueError, match="edge costs are global"):
        PHONEMeS(backend=backend).build_from_data(graph, data)
