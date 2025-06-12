from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from corneto import DEFAULT_BACKEND
from corneto._constants import *

# from corneto._core import Graph
from corneto._graph import BaseGraph
from corneto._settings import sparsify
from corneto.backend import Backend
from corneto.backend._base import Indicators, ProblemDef


def create_flow_graph(
    g: BaseGraph,
    conditions: Dict[str, Dict[str, Tuple[str, float]]],
    pert_id: str = "P",
    meas_id: str = "M",
    longitudinal_samples=False,
) -> BaseGraph:
    gc = g.copy()
    ns = "_s"
    nt = "_t"
    if longitudinal_samples:
        gc.add_edge(ns, "_pert_CTP", interaction=1)  # _pert_CTP
    for c, v in conditions.items():
        if longitudinal_samples:
            dummy_cond_pert = f"_tp_pert_{c}"  # _tp_pert_{c} -> _t_x.{c}
            dummy_cond_meas = f"_tp_meas_{c}"  # _tp_meas_{c} -> _t_y.{c}
            # gc.add_edge(dummy_cond_pert, dummy_cond_meas, interaction=1)
            gc.add_edge("_pert_CTP", dummy_cond_pert, interaction=1)
        else:
            dummy_cond_pert = f"_pert_{c}"  # _x.{c}
            dummy_cond_meas = f"_meas_{c}"  # _y.{c}
            gc.add_edge(ns, dummy_cond_pert, interaction=1)
        for species, (type, value) in v.items():
            direction = 1 if value >= 0 else -1
            # Perturbations
            if type.casefold() == pert_id.casefold():
                gc.add_edge(dummy_cond_pert, species, interaction=direction, value=value)
                # If measurement is 0, add also an inhibitory edge
                if value == 0:
                    gc.add_edge(dummy_cond_pert, species, interaction=-1, value=value)
            # Measurements
            elif type.casefold() == meas_id.casefold():
                gc.add_edge(species, dummy_cond_meas, interaction=direction, value=value)
                if value == 0:
                    gc.add_edge(species, dummy_cond_meas, interaction=direction, value=value)
        if longitudinal_samples:
            gc.add_edge(dummy_cond_meas, "_meas_CTP", interaction=1)
        else:
            gc.add_edge(dummy_cond_meas, nt, interaction=1)
    if longitudinal_samples:
        gc.add_edge("_meas_CTP", nt, interaction=1)
    gc.add_edge((), ns, id="_inflow")  # () -> ns
    gc.add_edge(nt, (), id="_outflow")  # nt -> ()
    return gc


def signflow_constraints(
    g: BaseGraph,
    backend: Backend = DEFAULT_BACKEND,
    signal_implies_flow: bool = True,
    flow_implies_signal: bool = False,
    dag: bool = True,
    use_flow_indicators: bool = True,
    eps: float = 1e-3,
) -> ProblemDef:
    edges = g.E
    vertices = g.vertices
    A = g.vertex_incidence_matrix()
    if "_s" not in vertices:
        raise ValueError("The provided network does not have the `_s` and `_t` dummy nodes.")
    perturbations = g.successors("_s")
    conditions = []
    for pert in perturbations:
        if not pert.startswith("_pert_"):
            raise ValueError(
                "The provided network does not contain the `_pert_` dummy nodes (perturbations per condition)."
            )
        conditions.append(pert.split("_pert_")[1])
    n_conditions = len(conditions)

    if n_conditions > 1 and flow_implies_signal:
        raise ValueError("flow_implies_signal is not supported in multi-conditions")
    use_flow = use_flow_indicators or flow_implies_signal or signal_implies_flow
    if use_flow:
        p = backend.Flow(g, ub=10)
        p._graph = g
        F, Fi = p.get_symbol(VAR_FLOW), None
        idx = -1
        # TODO: find outflow edges
        for i in range(g.num_edges):
            s, t = g.get_edge(i)
            if len(t) == 0 and len(s) > 0:
                p += F[i] >= 1.01 * eps
        """
        for i, props in enumerate(g.get_attr_edges()):
            if props.get("id", None) == "_outflow":
                idx = i
                break
        p += F[idx] >= 1.01 * eps"""
        if use_flow_indicators:
            p += Indicators()
            Fi = p.get_symbol(VAR_FLOW + "_ipos")
    else:
        F, Fi = None, None
        p = backend.Problem()
        p._graph = g
    dist = dict()
    if dag:
        dist = g.bfs("_s")
        node_maxdist = g.num_vertices - 1
        dist_lbound = np.array([dist.get(v, 0) for v in g.vertices])
        dist_ubound = node_maxdist

    vidx = {v: i for i, v in enumerate(vertices)}
    eidx = {e: i for i, e in enumerate(edges)}

    for c in conditions:
        # reachable = list(g.vertices)
        non_reachable = []
        # If there is more than one condition, just check which nodes are reachable
        # from this condition (forward pass from perturbation to measurements and
        # backward pass from measurements to perturbations). Nodes and reactions
        # that cannot be reached should have a value of 0.
        fwd: Set[Any] = set(g.bfs([f"_pert_{c}"]).keys())
        bck: Set[Any] = set(g.bfs([f"_meas_{c}"], reverse=True).keys())
        reachable = list(fwd.intersection(bck) | {"_s", "_t"})
        non_reachable = list(set(g.vertices) - set(reachable))
        non_reachable = [vidx[v] for v in non_reachable]
        reachable = [vidx[v] for v in reachable]

        N_act = backend.Variable(f"species_activated_{c}", (g.num_vertices,), vartype=VarType.BINARY)
        N_inh = backend.Variable(f"species_inhibited_{c}", (g.num_vertices,), vartype=VarType.BINARY)
        R_act = backend.Variable(
            f"reaction_sends_activation_{c}",
            (g.num_edges,),
            vartype=VarType.BINARY,
        )
        R_inh = backend.Variable(
            f"reaction_sends_inhibition_{c}",
            (g.num_edges,),
            vartype=VarType.BINARY,
        )

        p.register(f"edge_values_{c}", R_act - R_inh)
        p.register(f"vertex_values_{c}", N_act - N_inh)

        if len(non_reachable) > 0:
            # TODO: Do the same for non reachable reactions
            p += N_act[non_reachable] == 0
            p += N_inh[non_reachable] == 0

        # Also, if the measurements are selected, their edges from
        # measurement to the dummy _meas_ node have to be selected.
        # Not required, but forces to have a connected graph.
        # meas_rxns = list(rn.get_reactions_with_product(rn.get_species_id(f"_meas_{c}")))

        # meas_rxns = [eidx[e] for e in g.get_edges_with_target_vertex(f"_meas_{c}")]
        meas_rxns = [i for i, _ in g.in_edges(f"_meas_{c}")]
        # meas_species = [rn.get_reactants_of_reaction(r).pop() for r in meas_rxns]
        meas_species = [vidx[list(g.get_edge(i)[0])[0]] for i in meas_rxns]  # assumes a single vertex
        p += R_act[meas_rxns] + R_inh[meas_rxns] == N_act[meas_species] + N_inh[meas_species]

        # Just for convenience, force the dummy measurement node to be active. Not required
        # p += N_act[vidx[f"_meas_{c}"]] == 1
        # Same for source and target nodes. Assume they are active as a convention. Note that
        # dummy source _s connects to perturbations with activatory edges if perturbation is up
        # and inhibitory edges if perturbation is down. Dummy _s could be also down and propagate
        # the inverse signal instead. Same for dummy target _t. As a convention, they are forced
        # to be always activated instead to avoid these alternative options.
        p += N_act[vidx["_s"]] == 1
        p += N_act[vidx["_t"]] >= 0
        # Dont define activation/inhibition for reactions with no reactants or no products
        # rids = np.flatnonzero(np.logical_and(rn.has_reactant(), rn.has_product()))
        # TODO: Simplify this by adding methods to ReNet
        # TODO: Filter out reactions that has reactant or product in the non reachable set
        valid = np.zeros(g.num_edges, dtype=bool)
        valid[reachable] = True
        has_reactant = (np.sum(A < 0, axis=0) > 0).astype(int)
        has_product = (np.sum(A > 0, axis=0) > 0).astype(int)
        # has_reactant = np.sum(np.logical_and(rn.stoichiometry < 0, valid), axis=0) > 0
        # has_product = np.sum(np.logical_and(rn.stoichiometry > 0, valid), axis=0) > 0
        rids = np.flatnonzero(np.logical_and(has_reactant, has_product))
        # TODO: Throw error if reactions have more than one reactant or product
        # Get reactants and products: note that reactions should have
        # only 1 reactant 1 product at most for CARNIVAL
        ix_react = [vidx[list(edges[i][0])[0]] for i in rids]
        ix_prod = [vidx[list(edges[i][1])[0]] for i in rids]
        signs = np.array([p.get("interaction", 0) for p in g.get_attr_edges()])[rids]
        Ra = R_act[rids]
        Ri = R_inh[rids]
        p += R_act + R_inh <= 1
        p += N_act + N_inh <= 1
        D_ai = N_act[ix_react] - N_inh[ix_react]
        D_ia = N_inh[ix_react] - N_act[ix_react]
        S_ai = (N_act[ix_react] + N_inh[ix_react]).multiply(signs)
        p += Ra <= (D_ai + S_ai).multiply(signs)
        p += Ri <= (D_ia + S_ai).multiply(signs)
        if dag:
            L = backend.Variable(
                f"dag_layer_position_{c}",
                (g.num_vertices,),
                vartype=VarType.CONTINUOUS,
                lb=dist_lbound,
                ub=dist_ubound,
            )
            p += L[ix_prod] - L[ix_react] >= Ra + Ri + (1 - g.num_vertices) * (1 - (Ra + Ri))
            p += L[ix_prod] - L[ix_react] <= g.num_vertices - 1

        # Link flow with signal
        if signal_implies_flow and flow_implies_signal:
            # Bi-directional implication
            if Fi is None:
                raise NotImplementedError("flow <=> signal implication supported only with flow indicators")
            else:
                p += R_act + R_inh == Fi
        elif signal_implies_flow:
            # If signal then flow (if no flow then no signal)
            # but a reaction with non-zero flow may not carry any signal
            if Fi is None:
                # If reaction has signal (r_act+r_inh == 1) then the flow on
                # that reaction has to be >= eps value (active)
                # If reaction has no signal (r_act+r_inh == 0) then the flow
                # can have any value
                p += F >= eps * (R_act + R_inh)
            else:
                p += R_act + R_inh <= Fi
        elif flow_implies_signal:
            if Fi is None:
                # If reaction has a non-zero flow (f >= eps)
                # then the reaction has to transmit signal.
                # If reaction has no flow, the signal can have
                # any value. Note that this option by itself
                # does not make sense for carnival, use only
                # for experimentation.
                p += eps * (R_act + R_inh) <= F
            else:
                p += Fi <= R_act + R_inh
        # Constrain the product species of the reactions. They can be only up or
        # down if at least one of the reactions that have the node as product
        # carry some signal. Clip neg. values since we only look at the positive
        # ones in the incidence matrix (the targets of each edge)
        incidence_matrix = sparsify(A.clip(0, 1))
        p += N_act <= incidence_matrix @ R_act
        p += N_inh <= incidence_matrix @ R_inh
    return p


# TODO: Create building block so problem is passed
# through composition
def default_sign_loss(
    conditions: Dict,
    problem: ProblemDef,
    l0_edges: float = 0.0,
    l0_vertices: float = 0.0,
    l1_flow: float = 0.0,
    ub_loss: Optional[Union[float, List[float]]] = None,
    lb_loss: Optional[Union[float, List[float]]] = None,
) -> ProblemDef:
    losses = []
    p = ProblemDef()
    g = problem._graph
    if g is None:
        raise ValueError("Graph not available in the given problem")
    F, Fi = None, None
    if VAR_FLOW in problem.symbols.keys():
        F = problem.get_symbol(VAR_FLOW)
    if VAR_FLOW + "_ipos" in problem.symbols.keys():
        Fi = problem.get_symbol(VAR_FLOW + "_ipos")
    for i, c in enumerate(conditions.keys()):
        N_act, N_inh = problem.get_symbols(f"species_activated_{c}", f"species_inhibited_{c}")
        # Get the values of the species for the given condition
        species_values = np.array([conditions[c][s][1] if s in conditions[c] else 0 for s in g.vertices])
        pos = species_values.clip(0, np.inf).reshape(1, -1) @ (1 - (N_act - N_inh))
        neg = np.abs(species_values.clip(-np.inf, 0)).reshape(1, -1) @ ((N_act - N_inh) + 1)
        loss = pos + neg
        losses.append(loss)
        if ub_loss is not None:
            if isinstance(ub_loss, list):
                p += loss <= ub_loss[i]
            else:
                p += loss <= ub_loss
        if lb_loss is not None:
            if isinstance(lb_loss, list):
                p += loss >= lb_loss[i]
            else:
                p += loss >= lb_loss

    # Add regularization
    weights = [1.0] * len(losses)
    if l0_edges > 0:
        if Fi is None:
            raise ValueError("L0 regularization on flow cannot be used if signal is not connected to flow.")
        # TODO: Issues with sum and PICOS (https://gitlab.com/picos-api/picos/-/issues/330)
        # override sum with picos.sum method
        losses.append(np.ones(Fi.shape) @ Fi)
        weights.append(l0_edges)
    if l1_flow != 0:
        if F is None:
            raise NotImplementedError("Use R_act and R_inh for regularization")
        # TODO: This is valid only for positive fluxes
        losses.append(np.ones(F.shape) @ F)
        weights.append(l1_flow)
    if l0_vertices != 0:
        # TODO: This is applied only to the last condition, fix this
        # Compute the mean of used nodes (nact+ninh) across conditions
        # Minimize the avg sum = l1, minimize the number of values != 0 -> l0 reg
        losses.append(np.ones(N_act.shape) @ (N_act + N_inh))
        weights.append(l0_vertices)
    # Add objective and weights to p
    p.add_objectives(losses, weights, inplace=True)
    return p


def signflow(
    g: BaseGraph,
    conditions: Dict,
    signal_implies_flow: bool = True,
    flow_implies_signal: bool = False,  # not supported in multi-conditions
    dag: bool = True,
    l0_penalty_edges: float = 0.0,
    l1_penalty_flow: float = 0.0,
    l0_penalty_vertices: float = 0.0,
    ub_loss: Optional[Union[float, List[float]]] = None,
    lb_loss: Optional[Union[float, List[float]]] = None,
    use_flow_indicators: bool = True,
    eps: float = 1e-3,
    backend: Backend = DEFAULT_BACKEND,
):
    p = signflow_constraints(
        g,
        backend=backend,
        signal_implies_flow=signal_implies_flow,
        flow_implies_signal=flow_implies_signal,
        dag=dag,
        use_flow_indicators=use_flow_indicators,
        eps=eps,
    )
    return p + default_sign_loss(
        conditions,
        p,
        l0_edges=l0_penalty_edges,
        l0_vertices=l0_penalty_vertices,
        l1_flow=l1_penalty_flow,
        ub_loss=ub_loss,
        lb_loss=lb_loss,
    )


def expand_graph_for_flows(G: BaseGraph, exp_list):
    """Add edges to the perturbed and measured nodes in graph G to make flow possible."""
    G1 = G.copy()
    output_names = list({key for exp in exp_list.values() for key in exp["output"].keys()})
    input_names = list({key for exp in exp_list.values() for key in exp["input"].keys()})

    output_names = list(set(output_names))
    input_names = list(set(input_names))

    for node in output_names:
        G1.add_edge(node, ())
    for node in input_names:
        G1.add_edge((), node)

    return G1
