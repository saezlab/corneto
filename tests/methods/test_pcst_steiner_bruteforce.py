import itertools
from collections import defaultdict, deque

import numpy as np
import pytest

from corneto.backend import PicosBackend
from corneto.data import Data
from corneto.graph import EdgeType, Graph
from corneto.methods.pcst import PrizeCollectingSteinerTree
from corneto.methods.steiner import SteinerTreeFlow


def _all_subsets(n):
    for r in range(n + 1):
        for subset in itertools.combinations(range(n), r):
            yield set(subset)


def _reachable_from_root(vertices, edges, selected_edges, root):
    adj = defaultdict(set)
    for idx in selected_edges:
        u, v = edges[idx]
        adj[u].add(v)
        adj[v].add(u)

    seen = set()
    q = deque([root])
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        for v in adj[u]:
            if v not in seen:
                q.append(v)
    return seen.intersection(vertices)


def _steiner_bruteforce_best_cost(vertices, edges, edge_costs, required_terminals, root):
    best = np.inf
    for selected_edges in _all_subsets(len(edges)):
        reachable = _reachable_from_root(vertices, edges, selected_edges, root)
        if not required_terminals.issubset(reachable):
            continue
        cost = sum(edge_costs[i] for i in selected_edges)
        best = min(best, cost)
    return best


def _pcst_bruteforce_best_objective(vertices, edges, edge_costs, prizes, root):
    best = np.inf
    for selected_edges in _all_subsets(len(edges)):
        reachable = _reachable_from_root(vertices, edges, selected_edges, root)
        cost = sum(edge_costs[i] for i in selected_edges)
        prize = sum(prizes.get(v, 0.0) for v in reachable)
        best = min(best, cost - prize)
    return best


def _objective_value_by_name(problem, name):
    for obj in problem.objectives:
        if obj.name == name:
            return float(np.asarray(obj.value).reshape(-1)[0])
    raise KeyError(f"Objective named {name!r} not found")


def _tiny_graph_and_costs():
    # Undirected tiny graph:
    # A-B (2), B-C (2), A-C (5), C-D (1), B-D (4)
    g = Graph()
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("A", "C"),
        ("C", "D"),
        ("B", "D"),
    ]
    costs = [2.0, 2.0, 5.0, 1.0, 4.0]
    for (u, v), c in zip(edges, costs):
        g.add_edge(u, v, type=EdgeType.UNDIRECTED, value=c)
    return g, edges, costs


def test_steiner_matches_bruteforce_optimum(backend, request):
    if isinstance(backend, PicosBackend) and not request.config.getoption("--run-optional"):
        pytest.skip("PicosBackend is optional (use --run-optional)")

    g, edges, costs = _tiny_graph_and_costs()
    required_terminals = {"A", "C", "D"}
    root = "A"

    data = Data.from_dict(
        {
            "s1": {
                "features": [
                    {"id": "A", "mapping": "vertex", "role": "terminal"},
                    {"id": "C", "mapping": "vertex", "role": "terminal"},
                    {"id": "D", "mapping": "vertex", "role": "terminal"},
                    *[{"id": i, "mapping": "edge", "value": c} for i, c in enumerate(costs)],
                ]
            }
        }
    )

    method = SteinerTreeFlow(root_vertex=root, strict_acyclic=False, lambda_reg=0.0, backend=backend)
    problem = method.build(g, data)
    problem.solve()

    brute_force_opt = _steiner_bruteforce_best_cost(set(g.V), edges, costs, required_terminals, root)
    corneto_opt = _objective_value_by_name(problem, "edge_cost")
    assert np.isclose(corneto_opt, brute_force_opt, atol=1e-6)


def test_pcst_matches_bruteforce_optimum(backend, request):
    if isinstance(backend, PicosBackend) and not request.config.getoption("--run-optional"):
        pytest.skip("PicosBackend is optional (use --run-optional)")

    g, edges, costs = _tiny_graph_and_costs()
    root = "A"
    prizes = {"C": 6.0, "D": 2.0}

    data = Data.from_dict(
        {
            "s1": {
                "features": [
                    {"id": "A", "mapping": "vertex", "role": "terminal"},
                    {"id": "C", "mapping": "vertex", "role": "terminal", "value": prizes["C"]},
                    {"id": "D", "mapping": "vertex", "role": "terminal", "value": prizes["D"]},
                    *[{"id": i, "mapping": "edge", "value": c} for i, c in enumerate(costs)],
                ]
            }
        }
    )

    method = PrizeCollectingSteinerTree(
        include_all_terminals=False,
        root_vertex=root,
        strict_acyclic=False,
        lambda_reg=0.0,
        backend=backend,
    )
    problem = method.build(g, data)
    problem.solve()

    brute_force_opt = _pcst_bruteforce_best_objective(set(g.V), edges, costs, prizes, root)
    corneto_opt = _objective_value_by_name(problem, "edge_cost") - _objective_value_by_name(problem, "prizes")
    assert np.isclose(corneto_opt, brute_force_opt, atol=1e-6)
