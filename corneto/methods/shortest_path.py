from typing import Any, Optional

import numpy as np

from corneto._graph import BaseGraph
from corneto._settings import LOGGER
from corneto.backend import DEFAULT_BACKEND, Backend
from corneto.backend._base import DEFAULT_UB, Indicator


def shortest_path(
    G: BaseGraph,
    s: Any,
    t: Any,
    edge_weights=None,
    integral_path: bool = True,
    create_flow_graph: bool = True,
    backend: Backend = DEFAULT_BACKEND,
):
    # Transform to a flow problem
    if create_flow_graph:
        Gc = G.copy()
        e_start = Gc.add_edge((), s)
        e_end = Gc.add_edge(t, ())
    else:
        Gc = G
        e_start, (tail, head) = list(Gc.in_edges(s))[0]
        if tail != ():
            raise ValueError(
                f"Node {s} is not a source node. It has an incoming edge from {tail}."
            )

        e_end, (tail, head) = list(Gc.out_edges(t))[0]
        if head != ():
            raise ValueError(
                f"Node {t} is not a sink node. It has an outgoing edge to {head}."
            )
    if edge_weights is None:
        edge_weights = np.array(
            [Gc.get_attr_edge(i).get("weight", 0) for i in range(Gc.ne)]
        )
    if integral_path:
        P = backend.Flow(Gc, lb=0, ub=DEFAULT_UB)
        P += Indicator()
        selected = P.symbols["_flow_i"]
    else:
        P = backend.Flow(Gc, lb=0, ub=None)
        selected = P.symbols["_flow"]
    F = P.symbols["_flow"]
    P.add_objectives(selected @ edge_weights)
    P.add_constraints(F[e_start] == 1)
    P.add_constraints(F[e_end] == 1)
    return P, Gc


def solve_shortest_path(
    G: BaseGraph,
    s: Any,
    t: Any,
    edge_weights=None,
    solver: Optional[str] = None,
    backend: Backend = DEFAULT_BACKEND,
    integer_tolerance: float = 1e-6,
    solver_kwargs: Optional[dict] = None,
):
    P, Gc = shortest_path(
        G,
        s,
        t,
        edge_weights=edge_weights,
        integral_path=False,
        backend=backend,
    )
    if solver_kwargs is None:
        solver_kwargs = {}
    P.solve(solver=solver, **solver_kwargs)
    sol = np.array(P.expr.flow.value)
    if sol is None:
        raise ValueError("No solution found.")
    # Check if values are almost 0 or 1
    almost_zero = np.isclose(sol, 0, atol=integer_tolerance)
    almost_one = np.isclose(sol, 1, atol=integer_tolerance)
    almost_integral = almost_zero | almost_one
    solution = np.where(sol >= (1 - integer_tolerance))[0]
    if not np.all(almost_integral):
        LOGGER.warn(
            f"Number of non integral edges: {np.sum(~almost_integral)}. Solving again with integral constraints."
        )
        P, Gc = shortest_path(
            Gc, s, t, create_flow_graph=False, integral_path=True, backend=backend
        )
        P.solve(solver=solver, warm_start=True, **solver_kwargs)
        I = P.symbols["_flow_i"]
        solution = np.where(I.value > 0.5)[0]
    return solution, P, Gc
