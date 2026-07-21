import numpy as np

from corneto.graph import EdgeType, Graph
from corneto.backend import PicosBackend


def _build_acyclic_problem(backend, graph, lb, ub):
    problem = backend.Flow(graph, lb=lb, ub=ub)
    problem += backend.NonZeroIndicator(problem.expr._flow, tolerance=1e-6)
    problem = backend.Acyclic(
        graph,
        problem,
        indicator_positive_var_name="_flow_ipos",
        indicator_negative_var_name="_flow_ineg",
    )
    return problem


def _solve_and_assert_infeasible(problem, backend):
    if isinstance(backend, PicosBackend):
        problem.solve(solver="glpk", primals=False)
    else:
        problem.solve()
    assert problem.expr._flow.value is None


def test_acyclic_blocks_forced_directed_2cycle(backend):
    graph = Graph()
    graph.add_edge("A", "B", type=EdgeType.DIRECTED)
    graph.add_edge("B", "A", type=EdgeType.DIRECTED)

    problem = _build_acyclic_problem(backend, graph, lb=0, ub=10)
    problem += problem.expr._flow_ipos[0] == 1
    problem += problem.expr._flow_ipos[1] == 1

    _solve_and_assert_infeasible(problem, backend)


def test_acyclic_blocks_forced_directed_plus_reversible_2cycle(backend):
    graph = Graph()
    graph.add_edge("A", "B", type=EdgeType.DIRECTED)
    graph.add_edge("A", "B", type=EdgeType.UNDIRECTED)

    lb = np.array([0, -10])
    ub = np.array([10, 10])
    problem = _build_acyclic_problem(backend, graph, lb=lb, ub=ub)
    problem += problem.expr._flow_ipos[0] == 1
    problem += problem.expr._flow_ineg[1] == 1

    _solve_and_assert_infeasible(problem, backend)


def test_acyclic_blocks_forced_mixed_3cycle(backend):
    graph = Graph()
    graph.add_edge("A", "B", type=EdgeType.DIRECTED)
    graph.add_edge("B", "C", type=EdgeType.DIRECTED)
    graph.add_edge("A", "C", type=EdgeType.UNDIRECTED)

    lb = np.array([0, 0, -10])
    ub = np.array([10, 10, 10])
    problem = _build_acyclic_problem(backend, graph, lb=lb, ub=ub)
    # Force A->B, B->C and C->A (negative use of A<->C).
    problem += problem.expr._flow_ipos[0] == 1
    problem += problem.expr._flow_ipos[1] == 1
    problem += problem.expr._flow_ineg[2] == 1

    _solve_and_assert_infeasible(problem, backend)


def test_acyclic_enforces_constraints_per_sample(backend):
    graph = Graph()
    graph.add_edge("A", "B", type=EdgeType.DIRECTED)
    graph.add_edge("A", "B", type=EdgeType.UNDIRECTED)

    lb = np.array([0, -10])
    ub = np.array([10, 10])
    problem = backend.Flow(graph, lb=lb, ub=ub, n_flows=2)
    problem += backend.NonZeroIndicator(problem.expr._flow, tolerance=1e-6)
    problem = backend.Acyclic(
        graph,
        problem,
        indicator_positive_var_name="_flow_ipos",
        indicator_negative_var_name="_flow_ineg",
    )

    # Force A->B and B->A (negative use of reversible edge) only in sample 0.
    problem += problem.expr._flow_ipos[0, 0] == 1
    problem += problem.expr._flow_ineg[1, 0] == 1

    _solve_and_assert_infeasible(problem, backend)
