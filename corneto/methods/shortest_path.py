from typing import Any, List, Optional

import numpy as np

from corneto._graph import BaseGraph
from corneto._settings import LOGGER
from corneto.backend import DEFAULT_BACKEND, Backend
from corneto.backend._base import DEFAULT_UB, Indicator

"""
class ShortestPath(CornetoMethod):
    def __init__(self, backend=DEFAULT_BACKEND):
        super().__init__(backend=backend)

    def create_problem(self, source_target_pairs: list, lambd: float = 0.0):
        P = self._backend.Flow(Gc, lb=0, ub=DEFAULT_UB, n_flows=len(source_target_nodes))
        # Now we add the objective and constraints for each sample
        for i, (s, t) in enumerate(source_target_pairs):
            weights = edge_weights[i, :]
            P.add_objectives(P.expr.flow[:, i] @ weights)
            # Now we inject/extract 1 unit flow from s to t
            P += P.expr.flow[inflow_edges[s]] == 1
            P += P.expr.flow[outflow_edges[t]] == 1
            # For the rest of inflow/outflow edges, we set the flow to 0
            for node in inflow_edges:
                if node != s:
                    P += P.expr.flow[inflow_edges[node]] == 0
            for node in outflow_edges:
                if node != t:
                    P += P.expr.flow[outflow_edges[node]] == 0
        # Add reg
        if lambd > 0:
            P += self._backend.linear_or(
                P.expr.flow, axis=1, ignore_type=True, varname="active_edge"
            )
            P.add_objectives(sum(P.expr.active_edge), weights=lambd)

"""


def create_multisample_shortest_path(
    G: BaseGraph,
    source_target_nodes: List[tuple],
    edge_weights=None,
    solver: Optional[str] = None,
    backend: Backend = DEFAULT_BACKEND,
    lam: float = 0.0,
):
    # Transform the graph into a flow problem
    Gc = G.copy()
    inflow_edges = dict()
    outflow_edges = dict()
    for s, t in source_target_nodes:
        if s not in inflow_edges:
            inflow_edges[s] = Gc.add_edge((), s)
        if t not in outflow_edges:
            outflow_edges[t] = Gc.add_edge(t, ())
    if edge_weights is None:
        edge_weights = np.array([Gc.get_attr_edge(i).get("weight", 0) for i in range(Gc.ne)])
        # The number of samples equals the number of source-target pairs.
        # We need to duplicate the edge weights for each sample.
        edge_weights = np.tile(edge_weights, (len(source_target_nodes), 1))
    else:
        # Verify that the number of edge weights is correct
        edge_weights = np.array(edge_weights)
        if edge_weights.shape[0] != len(source_target_nodes):
            raise ValueError("The number of edge weights must be equal to the number of source-target pairs.")
        # Add the weights for the extra edges, to be 0
        n_extra_edges = Gc.ne - G.ne
        edge_weights = np.concatenate([edge_weights, np.zeros((len(source_target_nodes), n_extra_edges))], axis=1)
    P = backend.Flow(Gc, lb=0, ub=DEFAULT_UB, n_flows=len(source_target_nodes))
    # Now we add the objective and constraints for each sample
    for i, (s, t) in enumerate(source_target_nodes):
        weights = edge_weights[i, :]
        P.add_objectives(P.expr.flow[:, i] @ weights)
        # Now we inject/extract 1 unit flow from s to t
        P += P.expr.flow[inflow_edges[s]] == 1
        P += P.expr.flow[outflow_edges[t]] == 1
        # For the rest of inflow/outflow edges, we set the flow to 0
        for node in inflow_edges:
            if node != s:
                P += P.expr.flow[inflow_edges[node]] == 0
        for node in outflow_edges:
            if node != t:
                P += P.expr.flow[outflow_edges[node]] == 0
    # Add reg
    if lam > 0:
        P += backend.linear_or(P.expr.flow, axis=1, ignore_type=True, varname="active_edge")
        P.add_objectives(sum(P.expr.active_edge), weights=lam)
    return P, Gc


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
            raise ValueError(f"Node {s} is not a source node. It has an incoming edge from {tail}.")

        e_end, (tail, head) = list(Gc.out_edges(t))[0]
        if head != ():
            raise ValueError(f"Node {t} is not a sink node. It has an outgoing edge to {head}.")
    if edge_weights is None:
        edge_weights = np.array([Gc.get_attr_edge(i).get("weight", 0) for i in range(Gc.ne)])
    else:
        edge_weights = np.array(edge_weights)
        # Add the weights for the extra edges, to be 0
        edge_weights = np.concatenate([edge_weights, [0, 0]])
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
        P, Gc = shortest_path(Gc, s, t, create_flow_graph=False, integral_path=True, backend=backend)
        P.solve(solver=solver, warm_start=True, **solver_kwargs)
        I = P.symbols["_flow_i"]
        solution = np.where(I.value > 0.5)[0]
    return solution, P, Gc
