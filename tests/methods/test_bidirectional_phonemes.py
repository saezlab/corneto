import itertools

import numpy as np
import pytest

from corneto import Data, Graph
from corneto.methods import BidirectionalPHONEMeS, PHONEMeS


def _solve(problem):
    problem.solve()
    assert all(objective.value is not None for objective in problem.objectives)


def _objective_value(problem):
    return sum(float(objective.value) * float(weight) for objective, weight in zip(problem.objectives, problem.weights))


def _is_dag(vertices, edges):
    successors = {vertex: [] for vertex in vertices}
    indegree = {vertex: 0 for vertex in vertices}
    for source, target in edges:
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


def _direction_feasible(edges, selected, anchor, measured, *, reverse):
    traversal = [
        (target, source) if reverse else (source, target) for (source, target), active in zip(edges, selected) if active
    ]
    if not traversal:
        return False
    selected_vertices = {vertex for edge in traversal for vertex in edge}
    if anchor not in selected_vertices:
        return False
    incoming = {vertex: 0 for vertex in selected_vertices}
    outgoing = {vertex: 0 for vertex in selected_vertices}
    for source, target in traversal:
        outgoing[source] += 1
        incoming[target] += 1
    for vertex in selected_vertices:
        if vertex == anchor and outgoing[vertex] < 1:
            return False
        if vertex in measured and vertex != anchor and incoming[vertex] < 1:
            return False
        if vertex != anchor and vertex not in measured:
            if incoming[vertex] < 1 or outgoing[vertex] < 1:
                return False

    reachable = {anchor}
    while True:
        previous = len(reachable)
        reachable.update(target for source, target in traversal if source in reachable)
        if len(reachable) == previous:
            break
    if any(source not in reachable or target not in reachable for source, target in traversal):
        return False
    return any(vertex in measured and vertex in reachable for vertex in selected_vertices)


def _brute_force_both(graph, anchor, scores, costs):
    edges = [(next(iter(source)), next(iter(target))) for source, target in graph.E]
    best = np.inf
    for downstream in itertools.product((0, 1), repeat=len(edges)):
        if not _direction_feasible(
            edges,
            downstream,
            anchor,
            scores,
            reverse=False,
        ):
            continue
        for upstream in itertools.product((0, 1), repeat=len(edges)):
            if not _direction_feasible(
                edges,
                upstream,
                anchor,
                scores,
                reverse=True,
            ):
                continue
            union = np.maximum(downstream, upstream)
            selected_edges = [edge for edge, active in zip(edges, union) if active]
            if not _is_dag(graph.V, selected_edges):
                continue
            vertices = {anchor} | {vertex for edge in selected_edges for vertex in edge}
            objective = sum(score for vertex, score in scores.items() if vertex in vertices) + costs @ union
            best = min(best, objective)
    return best


def test_bidirectional_matches_independent_global_bruteforce(backend):
    graph = Graph()
    graph.add_edges(
        [
            ("u", "K"),
            ("u", "x"),
            ("x", "K"),
            ("K", "d"),
            ("K", "y"),
            ("y", "d"),
        ]
    )
    scores = {"u": -3.0, "d": -4.0}
    costs = np.array([1.1, 0.2, 0.2, 1.0, 0.3, 0.3])

    method = BidirectionalPHONEMeS(
        anchor_policy="both",
        backend=backend,
    )
    problem = method.build(
        graph,
        regulated_kinases=["K"],
        phosphosite_scores=scores,
        edge_costs=dict(enumerate(costs)),
    )
    _solve(problem)

    assert _objective_value(problem) == pytest.approx(_brute_force_both(graph, "K", scores, costs))


def test_bidirectional_selects_both_sides_and_reports_biological_orientation(
    backend,
):
    graph = Graph()
    graph.add_edges([("A", "K"), ("K", "B")])
    method = BidirectionalPHONEMeS(
        default_edge_cost=0,
        anchor_policy="both",
        backend=backend,
    )
    problem = method.build(
        graph,
        regulated_kinases=["K"],
        phosphosite_scores={"A": -2, "B": -2},
    )
    _solve(problem)

    assert np.asarray(problem.expr.edge_selected_upstream.value)[:, 0].tolist() == pytest.approx([1, 0])
    assert np.asarray(problem.expr.edge_selected_downstream.value)[:, 0].tolist() == pytest.approx([0, 1])
    assert np.asarray(problem.expr.edge_selected.value)[:, 0].tolist() == pytest.approx([1, 1])
    selected = method.processed_graph.edge_subgraph([0, 1])
    assert list(selected.E) == list(graph.E)


@pytest.mark.parametrize(
    ("policy", "scores", "expected_down", "expected_up"),
    [
        ("either", {"A": -2, "B": 1}, 0, 1),
        ("both", {"A": -2, "B": 1}, 1, 1),
        ("downstream", {"A": 1, "B": -2}, 1, 0),
    ],
)
def test_anchor_policies(policy, scores, expected_down, expected_up, backend):
    graph = Graph()
    graph.add_edges([("A", "K"), ("K", "B")])
    problem = BidirectionalPHONEMeS(
        default_edge_cost=0,
        anchor_policy=policy,
        backend=backend,
    ).build(
        graph,
        regulated_kinases=["K"],
        phosphosite_scores=scores,
    )
    _solve(problem)
    vertex_index = {vertex: index for index, vertex in enumerate(graph.V)}
    assert problem.expr.anchor_active_downstream.value[vertex_index["K"], 0] == pytest.approx(expected_down)
    assert problem.expr.anchor_active_upstream.value[vertex_index["K"], 0] == pytest.approx(expected_up)


def test_multicondition_union_cost_is_paid_once(backend):
    graph = Graph()
    graph.add_edges([("A", "K"), ("K", "B")])
    method = BidirectionalPHONEMeS(
        default_edge_cost=0,
        anchor_policy="either",
        backend=backend,
    )
    problem = method.build_many(
        graph,
        regulated_kinases={"one": ["K"], "two": ["K"]},
        phosphosite_scores={
            "one": {"A": -3, "B": 2},
            "two": {"A": -3, "B": 2},
        },
        edge_costs={0: 1.5, 1: 1.5},
    )
    _solve(problem)

    assert np.asarray(problem.expr.edge_selected_upstream.value)[0].tolist() == pytest.approx([1, 1])
    assert np.asarray(problem.expr.edge_selected_any.value).reshape(-1).tolist() == pytest.approx([1, 0])
    assert float(problem.objectives[1].value) == pytest.approx(1.5)


def test_global_solution_improves_on_two_independent_directional_solves(backend):
    graph = Graph()
    graph.add_edges(
        [
            ("u", "K1"),
            ("K1", "x"),
            ("x", "K2"),
            ("K2", "d"),
        ]
    )
    anchors = ["K1", "K2"]
    costs = dict.fromkeys(range(4), 1.0)

    global_problem = BidirectionalPHONEMeS(
        anchor_policy="both",
        backend=backend,
    ).build(
        graph,
        regulated_kinases=anchors,
        phosphosite_scores={"u": -10, "d": -10},
        edge_costs=costs,
    )
    _solve(global_problem)

    downstream = PHONEMeS(backend=backend).build(
        graph,
        perturbations=anchors,
        phosphosite_scores={"d": -10},
        edge_costs=costs,
    )
    _solve(downstream)
    reversed_graph = Graph()
    reversed_graph.add_edges([(next(iter(target)), next(iter(source))) for source, target in graph.E])
    upstream = PHONEMeS(backend=backend).build(
        reversed_graph,
        perturbations=anchors,
        phosphosite_scores={"u": -10},
        edge_costs=costs,
    )
    _solve(upstream)

    assert _objective_value(global_problem) == pytest.approx(-16)
    assert _objective_value(global_problem) < (_objective_value(downstream) + _objective_value(upstream))


def test_anchor_phosphosite_overlap_is_scored_once(backend):
    graph = Graph()
    graph.add_edges([("A", "K"), ("K", "B")])
    problem = BidirectionalPHONEMeS(
        default_edge_cost=0,
        anchor_policy="both",
        backend=backend,
    ).build(
        graph,
        regulated_kinases=["K"],
        phosphosite_scores={"A": -1, "K": -5, "B": -1},
    )
    _solve(problem)

    assert np.asarray(problem.expr.vertex_selected.value)[:, 0].tolist() == pytest.approx([1, 1, 1])
    assert float(problem.objectives[0].value) == pytest.approx(-7)


def test_combined_direction_selection_is_acyclic(backend):
    graph = Graph()
    graph.add_edges([("A", "K"), ("K", "A")])
    method = BidirectionalPHONEMeS(
        default_edge_cost=-1,
        anchor_policy="either",
        backend=backend,
    )
    problem = method.build(
        graph,
        regulated_kinases=["K"],
        phosphosite_scores={"A": -4},
    )
    _solve(problem)

    assert np.asarray(problem.expr.edge_selected.value)[:, 0].sum() == pytest.approx(1)


def test_exact_pruning_retains_non_shortest_paths(backend):
    graph = Graph()
    graph.add_edges(
        [
            ("K", "m"),
            ("K", "a"),
            ("a", "b"),
            ("b", "m"),
        ]
    )
    method = BidirectionalPHONEMeS(
        anchor_policy="downstream",
        backend=backend,
    )
    problem = method.build(
        graph,
        regulated_kinases=["K"],
        phosphosite_scores={"m": -5},
        edge_costs={0: 4, 1: 0.1, 2: 0.1, 3: 0.1},
    )
    _solve(problem)

    assert method._biological_num_edges == 4
    assert np.asarray(problem.expr.edge_selected_downstream.value)[:, 0].tolist() == pytest.approx([0, 1, 1, 1])


def test_exposed_shapes_and_auxiliary_edges(backend):
    graph = Graph()
    graph.add_edges([("A", "K"), ("K", "B")])
    method = BidirectionalPHONEMeS(backend=backend)
    problem = method.build(
        graph,
        regulated_kinases=["K"],
        phosphosite_scores={"A": -2, "B": -2},
    )

    assert problem.expr.flow.shape == (7, 2)
    assert problem.expr.flow_downstream.shape == (2, 1)
    assert problem.expr.flow_upstream.shape == (2, 1)
    assert problem.expr.edge_selected_downstream.shape == (2, 1)
    assert problem.expr.edge_selected_upstream.shape == (2, 1)
    assert problem.expr.edge_selected.shape == (2, 1)
    assert problem.expr.edge_selected_any.shape in {(2,), (2, 1)}
    assert problem.expr.vertex_selected.shape == (3, 1)
    assert problem.expr.dag_layer.shape == (3, 1)
    assert method.processed_graph.num_edges == 7


def test_constraint_blocks_are_vectorized(backend):
    def build_problem(length, conditions):
        graph = Graph()
        graph.add_edges(
            list(itertools.pairwise(["u", *(f"a{i}" for i in range(length)), "K"]))
            + list(itertools.pairwise(["K", *(f"b{i}" for i in range(length)), "d"]))
        )
        names = [f"c{i}" for i in range(conditions)]
        return BidirectionalPHONEMeS(
            anchor_policy="both",
            backend=backend,
        ).build_many(
            graph,
            regulated_kinases={name: ["K"] for name in names},
            phosphosite_scores={name: {"u": -2, "d": -2} for name in names},
        )

    small = build_problem(1, 2)
    large = build_problem(10, 4)
    assert len(small.constraints) == len(large.constraints)
    assert len(small.constraints) == len({id(constraint) for constraint in small.constraints})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"regulated_kinases": [], "phosphosite_scores": {"m": 0}},
            "must not be empty",
        ),
        (
            {"regulated_kinases": ["missing"], "phosphosite_scores": {"m": 0}},
            "Unknown vertex",
        ),
        (
            {"regulated_kinases": ["K"], "phosphosite_scores": {}},
            "must not be empty",
        ),
        (
            {"regulated_kinases": ["K"], "phosphosite_scores": {"m": np.inf}},
            "must be finite",
        ),
        (
            {
                "regulated_kinases": ["K"],
                "phosphosite_scores": {"m": 0},
                "edge_costs": {99: 1},
            },
            "Invalid edge index",
        ),
    ],
)
def test_bidirectional_validates_explicit_inputs(kwargs, message, backend):
    graph = Graph()
    graph.add_edge("K", "m")
    with pytest.raises((TypeError, ValueError), match=message):
        BidirectionalPHONEMeS(backend=backend).build(graph, **kwargs)


def test_anchor_path_validation_and_data_edge_costs(backend):
    graph = Graph()
    graph.add_edges([("K", "m"), ("u", "K")])
    with pytest.raises(ValueError, match="no upstream path"):
        BidirectionalPHONEMeS(
            anchor_policy="both",
            backend=backend,
        ).build(
            graph,
            regulated_kinases=["K"],
            phosphosite_scores={"m": -1},
        )

    data = Data.from_dict(
        {
            "one": {
                "features": [
                    {
                        "id": "K",
                        "mapping": "vertex",
                        "role": "regulated_kinase",
                    },
                    {
                        "id": "m",
                        "mapping": "vertex",
                        "role": "phosphosite",
                        "value": -1,
                    },
                    {"id": 0, "mapping": "edge", "value": 1},
                ]
            },
            "two": {
                "features": [
                    {
                        "id": "K",
                        "mapping": "vertex",
                        "role": "regulated_kinase",
                    },
                    {
                        "id": "m",
                        "mapping": "vertex",
                        "role": "phosphosite",
                        "value": -1,
                    },
                    {"id": 0, "mapping": "edge", "value": 2},
                ]
            },
        }
    )
    with pytest.raises(ValueError, match="edge costs are global"):
        BidirectionalPHONEMeS(backend=backend).build_from_data(graph, data)


def test_invalid_anchor_policy():
    with pytest.raises(ValueError, match="anchor_policy"):
        BidirectionalPHONEMeS(anchor_policy="unknown")
