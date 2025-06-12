import numpy as np

from corneto._constants import VAR_FLOW
from corneto._graph import Attr, BaseGraph, EdgeType
from corneto.backend import DEFAULT_BACKEND, Backend


def _exact_steiner_tree(
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
    eidx = Gc.add_edge((), root, type=EdgeType.UNDIRECTED)  # () -> root (input flow, source flow node)
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
    lb = np.array([0 if prop.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED) else -10 for prop in Gc.get_attr_edges()])
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


def exact_steiner_tree(
    G: BaseGraph,
    terminals,
    edge_weights=None,
    root=None,
    tolerance=1e-3,
    strict_acyclic=False,
    flow_name=VAR_FLOW,
    in_flow_edge_type=EdgeType.DIRECTED,
    out_flow_edge_type=EdgeType.DIRECTED,
    backend: Backend = DEFAULT_BACKEND,
    flow_injected: float = 10,
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
    eidx = Gc.add_edge((), root, type=in_flow_edge_type)  # () -> root (input flow, source flow node)
    dummy_edges[root] = eidx
    ids = []
    for v in terminals:
        if v != root:
            idx = Gc.add_edge(v, (), type=out_flow_edge_type)
            ids.append(idx)  # terminal -> () (sink node, remove flow)
            dummy_edges[v] = idx
    ids = np.array(ids)
    # lower/upper bounds for flow. If directed, lb=0, ub>0, if undirected, lb<0, ub>0
    # NOTE: bounds are arbitrary, but very large/small numbers can introduce issues with integrality tolerances
    # TODO: lb/ub could be provided, or taken from the graph
    lb = np.array([0 if prop.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED) else -10 for prop in Gc.get_attr_edges()])
    if strict_acyclic:
        P = K.AcyclicFlow(Gc, lb=lb, ub=flow_injected, varname=flow_name)
    else:
        P = K.Flow(Gc, lb=lb, ub=flow_injected, varname=flow_name)
    ids_e = list(set(range(Gc.ne)) - set(ids + [eidx]))
    # Indicators for the edges (1=unconstrained, 0=blocked flow)
    F = P.expr[flow_name]
    P += K.Indicator(F, indexes=ids_e)
    Fi = P.expr[f"{flow_name}_i"]
    if strict_acyclic:
        P += P.expr[f"{flow_name}_ipos"] + P.expr[f"{flow_name}_ineg"] <= Fi

    # TODO: Take as argument, read from graph
    if edge_weights is None:
        edge_weights = np.array([prop.get("weight", 0) for prop in Gc.get_attr_edges()])
    elif isinstance(edge_weights, (list, tuple, np.ndarray)):
        ew = np.zeros(Gc.ne)
        ew[: len(edge_weights)] = np.array(edge_weights)
        edge_weights = ew
    # If is a number, generate a list with the same value for all edges
    elif isinstance(edge_weights, (int, float)):
        edge_weights = np.array([edge_weights for _ in range(Gc.ne)])
    else:
        raise ValueError("Unknown type for edge_weights (list or tuple)")

    P.add_objectives(edge_weights @ Fi)  # sum the total cost of selected edges

    if len(prized_nodes) == 0:
        # If not prized
        P += F[eidx] == flow_injected  # inject non-zero flow
        P += F[ids] >= flow_injected / (len(terminals) + 1)  # Force all terminals to be present
    else:
        id_edge_prized = np.array([dummy_edges[v] for v in prized_nodes])
        P += K.NonZeroIndicator(
            F,
            indexes=id_edge_prized,
            tolerance=tolerance,
            suffix_pos="_nz_ipos",
            suffix_neg="_nz_ineg",
        )
        I_prized_selected = P.symbols[f"{flow_name}_nz_ipos"] + P.symbols[f"{flow_name}_nz_ineg"]
        P.add_objectives(np.array(prizes) @ I_prized_selected, weights=-1)

    # Add an objective for non-zero flow on prized nodes

    # For vertices, we need to check in which edges they appear.
    # If any of those edges is selected, then the node was collected
    return P, Gc


def create_exact_multi_steiner_tree(
    G: BaseGraph,
    terminal_per_condition,
    edge_weights_per_condition=None,
    root_vertices=None,
    tolerance=1e-3,
    strict_acyclic=False,
    lam=0.01,
    flow_name="flow",
    backend: Backend = DEFAULT_BACKEND,
):
    if backend is None:
        raise ValueError("Invalid backend")
    # Detect the number of conditions
    if isinstance(terminal_per_condition, list):
        # Get all the internal lists (conditions)
        conditions = [l for l in terminal_per_condition if isinstance(l, list)]
    elif isinstance(terminal_per_condition, dict):
        conditions = [d for d in terminal_per_condition.values() if isinstance(d, dict)]
    else:
        raise ValueError("Invalid terminals format")
    # Create the N problems
    num_vertices, num_edges = G.shape
    big_P = None
    for i in range(len(conditions)):
        terminals = conditions[i]
        P, Gc = exact_steiner_tree(
            G,
            terminals,
            edge_weights=edge_weights_per_condition[i] if edge_weights_per_condition is not None else None,
            root=root_vertices[i] if root_vertices is not None else None,
            tolerance=tolerance,
            strict_acyclic=strict_acyclic,
            backend=backend,
            flow_name=f"{flow_name}{i}",
        )
        if big_P is None:
            big_P = P
        else:
            big_P += P

    # We create a linking binary vector computing the or of the selected edges
    vars = [big_P.expr[f"{flow_name}{i}_i"][:num_edges] for i in range(len(conditions))]
    for v in vars:
        print(v.shape)
    I = backend.vstack(vars)
    print("I", I.shape)
    big_P += backend.linear_or(I, axis=0, varname="is_unblocked")
    print("OR", big_P.expr.is_unblocked.shape)
    # big_P.register("is_unblocked", v_or)
    big_P.add_objectives(sum(big_P.expr.is_unblocked), weights=lam)
    return big_P


def _exact_multi_steiner_tree(
    G: BaseGraph,
    terminal_per_condition,
    edge_weights_per_condition=None,
    root_vertices=None,
    tolerance=1e-3,
    strict_acyclic=False,
    backend: Backend = DEFAULT_BACKEND,
):
    if backend is None:
        raise ValueError("Invalid backend")
    # Detect the number of conditions
    if isinstance(terminal_per_condition, list):
        # Get all the internal lists (conditions)
        conditions = [l for l in terminal_per_condition if isinstance(l, list)]
    elif isinstance(terminal_per_condition, dict):
        conditions = [d for d in terminal_per_condition.values() if isinstance(d, dict)]
    else:
        raise ValueError("Invalid terminals format")

    Gc = G.copy()
    for i in range(len(conditions)):
        prized_nodes, prizes = [], []
        terminals = conditions[i]
        if isinstance(terminals, dict):
            prized = {k: v for k, v in terminals.items() if v != 0}
            if len(prized) > 0:
                prized_nodes, prizes = zip(*prized.items())
            terminals = list(terminals.keys())

        dummy_edges = dict()

        if root_vertices is None:
            root = terminals[0]
        else:
            root = root_vertices[i]
        # TODO: Check if this root does not have a flow edge
        eidx = Gc.add_edge((), root, type=EdgeType.UNDIRECTED)  # () -> root (input flow, source flow node)
        dummy_edges[root] = eidx
        ids = []
        for v in terminals:
            if v != root:
                idx = Gc.add_edge(v, (), type=EdgeType.UNDIRECTED)
                ids.append(idx)  # terminal -> () (sink node, remove flow)
                dummy_edges[v] = idx
        ids = np.array(ids)
        lb = np.array([0 if prop.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED) else -10 for prop in Gc.get_attr_edges()])

    if strict_acyclic:
        raise NotImplementedError("Acyclic Flow for multi flows required, or Acyclic component (WIP)")
    else:
        P = backend.Flow(Gc, lb=lb, ub=10, n_flows=len(conditions))

    for i in range(len(conditions)):
        F = P.expr.flow[:, i]
        # Here there are differences, depending on if terminals have prizes or not.
        # If they have prizes, terminals are optional. In order to be optimal, in-out flow
        # through terminals have to be optional.
        ids_e = list(set(range(Gc.ne)) - set([*ids, eidx]))
        # Indicators for the edges (1=unconstrained, 0=blocked flow)
        P += backend.Indicator(F, indexes=ids_e)
        Fi = P.symbols["_flow_i"]
        if strict_acyclic:
            P += P.symbols["_flow_ipos"] + P.symbols["_flow_ineg"] <= Fi
        edge_weights = None
        if edge_weights_per_condition is not None:
            edge_weights = edge_weights_per_condition[i]

        if edge_weights is None:
            edge_weights = np.array([prop.get("weight", 0) for prop in Gc.get_attr_edges()])
        elif isinstance(edge_weights, (list, tuple)):
            edge_weights = np.array(edge_weights)
        else:
            raise ValueError("Unknown type for edge_weights (list or tuple)")

        P.add_objectives(edge_weights @ Fi)  # sum the total cost of selected edges

        if len(prized_nodes) == 0:
            # If not prized, force all terminals to be present
            # NOTE: This can lead to infeasibilities if a terminal cannot be connected
            P += F[eidx] == 10  # inject non-zero flow
            P += F[ids] >= 10 / (len(terminals) + 1)
        else:
            id_edge_prized = np.array([dummy_edges[v] for v in prized_nodes])
            P += backend.NonZeroIndicator(F, indexes=id_edge_prized, tolerance=tolerance)
            I_prized_selected = P.symbols["_flow_ipos"] + P.symbols["_flow_ineg"]
            # Maximize the selection of nodes with prizes
            P.add_objectives(np.array(prizes) @ I_prized_selected, weights=-1)
