"""Shortest-path formulations for one or multiple conditions."""

from collections import deque
from typing import Any, List, Optional

import numpy as np

from corneto.backend import DEFAULT_BACKEND, Backend, VarType
from corneto.graph import Attr, BaseGraph, EdgeType
from corneto.methods._network_utils import augment_with_boundaries

__all__ = [
    "create_multisample_shortest_path",
    "shortest_path",
    "solve_shortest_path",
]


def _validate_terminals(graph, pairs):
    if any(len(source) > 1 or len(target) > 1 for source, target in graph.E):
        raise ValueError("Shortest paths do not support hyperedges.")
    for source, target in pairs:
        if source not in graph.V or target not in graph.V:
            raise ValueError("Source and target must be vertices in the graph.")


def _path_weights(graph, edge_weights, shape):
    if edge_weights is None:
        weights = np.array([attr.get("weight", 0) for attr in graph.get_attr_edges()], dtype=float)
        weights = np.broadcast_to(weights, shape).copy()
    else:
        weights = np.asarray(edge_weights, dtype=float)
    if weights.shape != shape:
        raise ValueError(f"Expected edge_weights with shape {shape}, got {weights.shape}.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Shortest paths require finite, nonnegative edge weights.")
    return weights


def _flow_bounds(graph):
    return np.array(
        [-1.0 if attr.has_attr(Attr.EDGE_TYPE, EdgeType.UNDIRECTED) else 0.0 for attr in graph.get_attr_edges()]
    )


def _flow_magnitude(problem, backend):
    flow = problem.expr.flow
    magnitude = backend.Variable("_path_magnitude", flow.shape, lb=0, ub=1)
    problem += magnitude >= flow
    problem += magnitude >= -flow
    return magnitude


def create_multisample_shortest_path(
    G: BaseGraph,
    source_target_nodes: List[tuple],
    edge_weights=None,
    solver: Optional[str] = None,
    backend: Backend = DEFAULT_BACKEND,
    lam: float = 0.0,
):
    """Build unit shortest-path flows with finite nonnegative edge costs.

    With ``lam=0`` this is an LP. A positive ``lam`` adds binary support
    variables to penalize the union of edges used across conditions.
    Undirected edges support either traversal direction in each condition.
    """
    pairs = list(source_target_nodes)
    if not pairs:
        raise ValueError("At least one source-target pair is required.")
    _validate_terminals(G, pairs)
    weights = _path_weights(G, edge_weights, (len(pairs), G.ne))
    layout = augment_with_boundaries(
        G,
        inflow_vertices=[s for s, _ in pairs],
        outflow_vertices=[t for _, t in pairs],
    )
    Gc = layout.graph
    weights = np.pad(weights, ((0, 0), (0, Gc.ne - G.ne)))
    lb = _flow_bounds(Gc)
    if lam > 0:
        P = backend.SelectedFlow(
            Gc,
            lb=lb,
            ub=1,
            n_flows=len(pairs),
            epsilon=1,
            selected_by_flow_name="_path_selected",
            selected_any_name="active_edge",
        )
        magnitude = P.expr._path_selected
    else:
        P = backend.Flow(Gc, lb=lb, ub=1, n_flows=len(pairs), force_matrix=True)
        magnitude = _flow_magnitude(P, backend)
    for i, (s, t) in enumerate(pairs):
        P.add_objective(magnitude[:, i].multiply(weights[i]).sum(), name=f"path_cost_{i}")
        for node, edge in layout.inflow_edges.items():
            P += P.expr.flow[edge, i] == int(node == s)
        for node, edge in layout.outflow_edges.items():
            P += P.expr.flow[edge, i] == int(node == t)
    if lam > 0:
        P.add_objective(P.expr.active_edge.sum(), weight=lam, name="shared_edge_regularization")
    return P, Gc


def shortest_path(
    G: BaseGraph,
    s: Any,
    t: Any,
    edge_weights=None,
    integral_path: bool = False,
    create_flow_graph: bool = True,
    backend: Backend = DEFAULT_BACKEND,
):
    """Build a shortest-path LP with finite nonnegative edge costs.

    Undirected edges permit signed unit flow and cost its magnitude.
    Set ``integral_path=True`` to explicitly require binary edge support.
    With ``create_flow_graph=False``, weights must cover the existing graph,
    which must already contain source and sink boundary edges.
    """
    _validate_terminals(G, [(s, t)])
    weights = _path_weights(G, edge_weights, (G.ne,))
    if create_flow_graph:
        layout = augment_with_boundaries(G, inflow_vertices=[s], outflow_vertices=[t])
        Gc = layout.graph
        e_start = layout.inflow_edges[s]
        e_end = layout.outflow_edges[t]
        weights = np.pad(weights, (0, Gc.ne - G.ne))
    else:
        Gc = G
        e_start = next((i for i, (source, target) in enumerate(Gc.E) if not source and target == {s}), None)
        e_end = next((i for i, (source, target) in enumerate(Gc.E) if source == {t} and not target), None)
        if e_start is None or e_end is None:
            raise ValueError("The graph must contain source and sink boundary edges.")
    lb = _flow_bounds(Gc)
    P = backend.Flow(Gc, lb=lb, ub=1)
    if integral_path:
        selected = backend.Variable("_flow_i", P.expr.flow.shape, vartype=VarType.BINARY)
        P += backend.ExactSupport(
            P.symbols["_flow"],
            selected=selected,
            epsilon=1,
            nonnegative=bool(np.all(lb >= 0)),
            name="_path_support",
        )
        magnitude = selected
    else:
        magnitude = _flow_magnitude(P, backend)
    P.add_objectives(magnitude.multiply(weights).sum())
    P += P.expr.flow[e_start] == 1
    P += P.expr.flow[e_end] == 1
    return P, Gc


def _extract_path(graph, flow, source, target, tolerance):
    """Extract one simple path from an optimal flow, discarding ties and cycles."""
    adjacency = {vertex: [] for vertex in graph.V}
    for edge, ((tail, head), value) in enumerate(zip(graph.E, flow)):
        if not tail or not head or abs(value) <= tolerance:
            continue
        start, end = next(iter(tail)), next(iter(head))
        if value < 0:
            start, end = end, start
        adjacency[start].append((end, edge))
    parents = {source: None}
    pending = deque([source])
    while pending and target not in parents:
        vertex = pending.popleft()
        for neighbor, edge in adjacency[vertex]:
            if neighbor not in parents:
                parents[neighbor] = (vertex, edge)
                pending.append(neighbor)
    if target not in parents:
        raise ValueError("No solution path found in the solver flow.")
    path = []
    vertex = target
    while vertex != source:
        vertex, edge = parents[vertex]
        path.append(edge)
    return path


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
    """Solve an LP and return one path's edge indices, problem, and flow graph.

    A fractional mixture of tied optimal paths is resolved by extracting a
    simple path from its directional support, without an integer solve. Edge
    indices include the artificial source and sink edges for compatibility.
    The returned problem retains the original, possibly fractional LP flow.
    ``integer_tolerance`` is the threshold for ignoring numerical zero flows.
    """
    P, Gc = shortest_path(G, s, t, edge_weights=edge_weights, integral_path=False, backend=backend)
    # Report disconnected terminals consistently, including with backends that
    # raise their own exception instead of returning an infeasible problem.
    if t not in G.bfs([s]):
        raise ValueError("No solution found: target is unreachable from source.")
    if not np.isfinite(integer_tolerance) or integer_tolerance <= 0:
        raise ValueError("integer_tolerance must be finite and positive.")
    P.solve(solver=solver, **(solver_kwargs or {}))
    if P.expr.flow.value is None:
        raise ValueError("No solution found.")
    flow = np.asarray(P.expr.flow.value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(flow)):
        raise ValueError("No solution found: solver returned nonfinite flow.")
    path = _extract_path(G, flow[: G.ne], s, t, integer_tolerance)
    solution = np.array(sorted([*path, G.ne, G.ne + 1]), dtype=int)
    return solution, P, Gc
