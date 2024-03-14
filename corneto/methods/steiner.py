import numpy as np
from corneto._graph import BaseGraph, EdgeType, Attr
from corneto.backend import Backend, DEFAULT_BACKEND
from corneto._settings import sparsify
from corneto import VarType


def exact_steiner_tree(
    G: BaseGraph,
    terminals,
    edge_weights=None,
    root=None,
    tolerance=1e-3,
    strict_acyclic=False,
    backend: Backend = DEFAULT_BACKEND,
):
    prized_nodes, prizes = [], []
    if isinstance(terminals, dict):
        prized = {k: v for k, v in terminals.items() if v != 0}
        if len(prized) > 0:
            prized_nodes, prizes = zip(*prized.items())
        # non_zero_prizes = [k for k, v in terminals.items() if v != 0]
        terminals = list(terminals.keys())

    # V = {v: i for i, v in enumerate(G.V)}
    dummy_edges = dict()
    K = backend
    if K is None:
        raise ValueError("Invalid backend")
    # If root not provided, take the first terminal as a root.
    # Note that in undirected graphs, it doesn't matter the root node as long as
    # the graph is connected
    if root is None:
        root = terminals[0]
    Gc = G.copy()
    # TODO: If graph is directed, edges for inflow/outflow should be reversible
    eidx = Gc.add_edge(
        (), root, type=EdgeType.UNDIRECTED
    )  # () -> root (input flow, source flow node)
    dummy_edges[root] = eidx
    ids = []
    for v in terminals:
        if v != root:
            idx = Gc.add_edge(v, (), type=EdgeType.UNDIRECTED)
            ids.append(idx)  # terminal -> () (sink node, remove flow)
            dummy_edges[v] = idx
    ids = np.array(ids)
    # lower/upper bounds for flow. If directed, lb=0, ub>0, if undirected, lb<0, ub>0
    # NOTE: bounds are arbitrary, but very large/small numbers can introduce issues with integrality tolerances
    # TODO: lb/ub could be provided, or taken from the graph
    lb = np.array(
        [
            0 if prop.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED) else -10
            for prop in Gc.get_attr_edges()
        ]
    )
    if strict_acyclic:
        P = K.AcyclicFlow(Gc, lb=lb, ub=10)
    else:
        P = K.Flow(Gc, lb=lb, ub=10)
    F = P.symbols["_flow"]
    # Here there are differences, depending on if terminals have prizes or not.
    # If they have prizes, terminals are optional. In order to be optimal, in-out flow
    # through terminals have to be optional.
    ids_e = list(set(range(Gc.ne)) - set(ids + [eidx]))
    # Indicators for the edges (1=unconstrained, 0=blocked flow)
    P += K.Indicator(F, indexes=ids_e)
    Fi = P.symbols["_flow_i"]
    if strict_acyclic:
        P += P.symbols["_flow_ipos"] + P.symbols["_flow_ineg"] <= Fi

    # TODO: Take as argument, read from graph
    if edge_weights is None:
        edge_weights = np.array([prop.get("weight", 0) for prop in Gc.get_attr_edges()])
    elif isinstance(edge_weights, (list, tuple)):
        edge_weights = np.array(edge_weights)
    else:
        raise ValueError("Unknown type for edge_weights (list or tuple)")

    P.add_objectives(edge_weights @ Fi)  # sum the total cost of selected edges

    if len(prized_nodes) == 0:
        # If not prized
        P += F[eidx] == 10  # inject non-zero flow
        P += F[ids] >= 10 / (len(terminals) + 1)  # Force all terminals to be present
    else:
        id_edge_prized = np.array([dummy_edges[v] for v in prized_nodes])
        # P += K.NonZeroIndicator(
        #    F, indexes=np.array(id_edge_prized), tolerance=tolerance
        # )
        P += K.NonZeroIndicator(F, indexes=id_edge_prized, tolerance=tolerance)
        # I_prized_selected[i] corresponds to prized node non_zero_prizes[i]
        I_prized_selected = P.symbols["_flow_ipos"] + P.symbols["_flow_ineg"]
        P.add_objectives(np.array(prizes) @ I_prized_selected, weights=-1)

    # Add an objective for non-zero flow on prized nodes

    # For vertices, we need to check in which edges they appear.
    # If any of those edges is selected, then the node was collected
    return P, Gc
