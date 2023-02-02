import numpy as np
from typing import Callable, Union, Dict, List, Optional, Tuple, Set, Any
from corneto import DEFAULT_BACKEND
from corneto._legacy import ReNet
from corneto.backend import Backend
from corneto.backend._base import ProblemDef, Indicators
from corneto._constants import *
from corneto._settings import LOGGER
from corneto._core import Graph


# flowgraph(g, X, Y)

def flowgraph(
    g: Graph, 
    conditions: Dict[str, Dict[str, Tuple[str, float]]],
    pert_id: str = "P",
    meas_id: str = "M",
    longitudinal_samples=False
) -> Graph:
    gc = g.copy()
    #vertices = gc.vertices
    #edges = gc.edges
    ns = '_s'
    nt = '_t'
    if longitudinal_samples:
        gc.add_edge(ns, '_pert_CTP', interaction=1)
    for c, v in conditions.items():
        if longitudinal_samples:
            dummy_cond_pert = f"_tp_pert_{c}"
            dummy_cond_meas = f"_tp_meas_{c}"
            gc.add_edge(dummy_cond_pert, dummy_cond_meas, interaction=1)
        else:
            dummy_cond_pert = f"_pert_{c}"
            dummy_cond_meas = f"_meas_{c}"
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
            gc.add_edge(dummy_cond_meas, '_meas_CTP', interaction=1)
        else:
            gc.add_edge(dummy_cond_meas, nt, interaction=1)
    if longitudinal_samples:
        gc.add_edge('_meas_CTP', nt, interaction=1)
    gc.add_edge((), ns, id='_inflow') # () -> ns
    gc.add_edge(nt, (), id='_outflow') # nt -> ()
    return gc


def sg_constraints(
    g: Graph,
    backend: Backend = DEFAULT_BACKEND,
    signal_implies_flow: bool = True,
    flow_implies_signal: bool = False,
    dag: bool = True,
    use_flow_indicators: bool = True,
    eps: float = 1e-3,
) -> ProblemDef:
    if "_s" not in g.vertices:
        raise ValueError(
            "The provided network does not have the `_s` and `_t` dummy nodes. Please call `carnival_network` before this function."
        )
    perturbations = g.successors('_s')
    conditions = []
    for pert in perturbations:
        if not pert.startswith("_pert_"):
            raise ValueError(
                "The provided network does not contain the `_pert_` dummy nodes (perturbations per condition). Please call `carinval_network` before this function."
            )
        conditions.append(pert.split("_pert_")[1])
    n_conditions = len(conditions)
    
    if n_conditions > 1 and flow_implies_signal:
        raise ValueError("flow_implies_signal is not supported in multi-conditions")
    use_flow = use_flow_indicators or flow_implies_signal or signal_implies_flow
    if use_flow:
        p = backend.Flow(g)
        F, Fi = p.get_symbol(VAR_FLOW), None
        idx = -1
        for i, props in enumerate(g.edge_properties):
            if props.get('id', None) == '_outflow':
                idx = i
                break
        p += F[idx] >= 1.01 * eps
        if use_flow_indicators:
            p += Indicators()
            Fi = p.get_symbol(VAR_FLOW + "_ipos")
    else:
        F, Fi = None, None
        p = backend.Problem()
    
    dist = dict()
    if dag:
        dist = g.bfs('_s')
        node_maxdist = g.num_vertices - 1
        dist_lbound = np.array([dist.get(v, 0) for v in g.vertices])
        dist_ubound = node_maxdist

    vidx = {v: i for i, v in enumerate(g.vertices)}
    eidx = {e: i for i, e in enumerate(g.edges)}

    for c in conditions:
        #reachable = list(g.vertices)
        non_reachable = []
        # If there is more than one condition, just check which nodes are reachable
        # from this condition (forward pass from perturbation to measurements and
        # backward pass from measurements to perturbations). Nodes and reactions
        # that cannot be reached should have a value of 0.
        fwd: Set[Any] = set(g.bfs([f"_pert_{c}"]).keys())
        bck: Set[Any] = set(g.bfs([f"_meas_{c}"], rev=True).keys())
        reachable = list(fwd.intersection(bck) | {"_s", "_t"})
        non_reachable = list(set(g.vertices) - set(reachable))
        non_reachable = [vidx[v] for v in non_reachable]
        reachable = [vidx[v] for v in reachable]

        N_act = backend.Variable(
            f"species_activated_{c}", (g.num_vertices,), vartype=VarType.BINARY
        )
        N_inh = backend.Variable(
            f"species_inhibited_{c}", (g.num_vertices,), vartype=VarType.BINARY
        )
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

        if len(non_reachable) > 0:
            # TODO: Do the same for non reachable reactions
            p += N_act[non_reachable] == 0
            p += N_inh[non_reachable] == 0

        # Also, if the measurements are selected, their edges from
        # measurement to the dummy _meas_ node have to be selected.
        # Not required, but forces to have a connected graph.
        #meas_rxns = list(rn.get_reactions_with_product(rn.get_species_id(f"_meas_{c}")))
        meas_rxns = [eidx[e] for e in g.get_edges_with_target_vertex(f"_meas_{c}")]
        #meas_species = [rn.get_reactants_of_reaction(r).pop() for r in meas_rxns]
        meas_species = [vidx[list(g.edges[i][0])[0]] for i in meas_rxns]
        p += R_act[meas_rxns] + R_inh[meas_rxns] == N_act[meas_species] + N_inh[meas_species]

        
        # Just for convenience, force the dummy measurement node to be active. Not required
        p += N_act[vidx[f"_meas_{c}"]] == 1
        # Same for source and target nodes. Assume they are active as a convention. Note that
        # dummy source _s connects to perturbations with activatory edges if perturbation is up
        # and inhibitory edges if perturbation is down. Dummy _s could be also down and propagate
        # the inverse signal instead. Same for dummy target _t. As a convention, they are forced
        # to be always activated instead to avoid these alternative options.
        p += N_act[vidx["_s"]] == 1
        p += N_act[vidx["_t"]] == 1
        # Dont define activation/inhibition for reactions with no reactants or no products
        # rids = np.flatnonzero(np.logical_and(rn.has_reactant(), rn.has_product()))
        # TODO: Simplify this by adding methods to ReNet
        # TODO: Filter out reactions that has reactant or product in the non reachable set
        valid = np.zeros(g.num_edges, dtype=bool)
        valid[reachable] = True
        has_reactant = np.sum(g.vertex_incidence_matrix() < 0, axis=0) > 0
        has_product = np.sum(g.vertex_incidence_matrix() > 0, axis=0) > 0
        # has_reactant = np.sum(np.logical_and(rn.stoichiometry < 0, valid), axis=0) > 0
        # has_product = np.sum(np.logical_and(rn.stoichiometry > 0, valid), axis=0) > 0
        rids = np.flatnonzero(np.logical_and(has_reactant, has_product))
        # TODO: Throw error if reactions have more than one reactant or product
        # Get reactants and products: note that reactions should have
        # only 1 reactant 1 product at most for CARNIVAL
        # ix_react = np.array(list({rn.get_reactants([i]).pop() for i in rids}.intersection(reachable)))
        # ix_prod = np.array(list({rn.get_products([i]).pop() for i in rids}.intersection(reachable)))
        ix_react = [vidx[list(g.edges[i][0])[0]] for i in rids]
        ix_prod = [vidx[list(g.edges[i][1])[0]] for i in rids]
        #ix_react = np.array([rn.get_reactants([i]).pop() for i in rids])
        #ix_prod = np.array([rn.get_products([i]).pop() for i in rids])
        signs = np.array([p.get('interaction', 0) for p in g.edge_properties])[rids]
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
            p += L
            p += L[ix_prod] - L[ix_react] >= Ra + Ri + (1 - g.num_vertices) * (
                1 - (Ra + Ri)
            )
            p += L[ix_prod] - L[ix_react] <= g.num_vertices - 1

        # Link flow with signal
        if signal_implies_flow and flow_implies_signal:
            # Bi-directional implication
            if Fi is None:
                raise NotImplementedError(
                    "flow <=> signal implication supported only with flow indicators"
                )
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
        # carry some signal.
        incidence_matrix = g.vertex_incidence_matrix(sparse=False).clip(0, 1)
        p += N_act <= incidence_matrix @ R_act
        p += N_inh <= incidence_matrix @ R_inh
    return p



def create_flow_graph(
    rn: ReNet,
    conditions: Dict[str, Dict[str, Tuple[str, float]]],
    pert_id: str = "P",
    meas_id: str = "M",
    conditions_as_timepoints=False,
) -> ReNet:
    # Create dummy nodes from _s (source node) to dummy condition nodes
    rnc = rn.copy()
    ns = "_s"
    nt = "_t"
    if conditions_as_timepoints:
        # Add only one perturbation _pert_ that connects to the _tp_pert_
        # _pert_ nodes are used to discover the conditions in carnival and
        # we dont have duplication of states per condition in case timepoints
        # are used
        rnc.add_reaction(f"_s-(1)-_pert_CTP", {"_s": -1, "_pert_CTP": 1}, value=1)
    for c, v in conditions.items():
        if conditions_as_timepoints:
            dummy_cond_pert = f"_tp_pert_{c}"
            dummy_cond_meas = f"_tp_meas_{c}"
            rnc.add_reaction(
                f"_pert_CTP-(1)-{dummy_cond_pert}",
                {"_pert_CTP": -1, dummy_cond_pert: 1},
                value=1,
            )
        else:
            dummy_cond_pert = f"_pert_{c}"
            dummy_cond_meas = f"_meas_{c}"
            rnc.add_reaction(
                f"{ns}-(1)-{dummy_cond_pert}", {ns: -1, dummy_cond_pert: 1}, value=1
            )
        for species, (type, value) in v.items():
            direction = 1 if value >= 0 else -1
            # Perturbations
            if type.casefold() == pert_id.casefold():
                rnc.add_reaction(
                    f"{dummy_cond_pert}--({direction})--{species}",
                    {dummy_cond_pert: -1, species: 1},
                    value=direction,
                )
                # If measurement is 0, add also an inhibitory edge
                if value == 0:
                    rnc.add_reaction(
                        f"{dummy_cond_pert}--(-1)--{species}",
                        {dummy_cond_pert: -1, species: 1},
                        value=-1,
                    )
            # Measurements
            elif type.casefold() == meas_id.casefold():
                rnc.add_reaction(
                    f"{species}--({direction})--{dummy_cond_meas}",
                    {species: -1, dummy_cond_meas: 1},
                    value=direction,
                )
                if value == 0:
                    rnc.add_reaction(
                        f"{species}--(-1)--{dummy_cond_meas}",
                        {species: -1, dummy_cond_meas: 1},
                        value=-1,
                    )    
        if conditions_as_timepoints:
            rnc.add_reaction(
                f"{dummy_cond_pert}--(1)--_meas_CTP",
                {dummy_cond_meas: -1, "_meas_CTP": 1},
                value=1,
            )
        else:
            rnc.add_reaction(
                f"{dummy_cond_pert}-(1)-{nt}", {dummy_cond_meas: -1, nt: 1}, value=1
            )
    if conditions_as_timepoints:
        rnc.add_reaction(f"_meas_CTP-(1)-{nt}", {"_meas_CTP": -1, nt: 1}, value=1)
    rnc.add_reaction("_inflow", {"_s": 1})
    rnc.add_reaction("_outflow", {"_t": -1})
    return rnc

def colorize(rne: ReNet, edges: Dict[str, int], nodes: Dict[str, int]):
    edge_props, node_props = {}, {}
    reactant_ids = [rne.get_reactants_of_reaction(i) for i in range(rne.num_reactions)]
    #r = [rne.species[i.pop()] if len(i) > 0 else None for i in reactant_ids]
    product_ids = [rne.get_products_of_reaction(i) for i in range(rne.num_reactions)]
    #p = [rne.species[i.pop()] if len(i) > 0 else None for i in product_ids]
    #reactions = list(zip(r, p))
    up_edges = [r for r in rne.reactions if edges[r] > 0.5]
    down_edges = [r for r in rne.reactions if edges[r] < -0.5]
    up_nodes = [s for s in rne.species if nodes[s] > 0.5]
    down_nodes = [s for s in rne.species if nodes[s] < -0.5]
    for e in up_edges:
        edge_props[e] = {"edge_color": "tab:red", "width": 2.0}
    for e in down_edges:
        edge_props[e] = {"edge_color": "tab:blue", "width": 2.0}
    for n in up_nodes:
        node_props[n] = {"color": "tab:red"}
    for n in down_nodes:
        node_props[n] = {"color": "tab:blue"}
    return {"nodes": node_props, "edges": edge_props}

def nx_style(rne: ReNet, carnival_problem: ProblemDef, condition=None) -> Dict:
    pc = carnival_problem
    edge_props, node_props = {}, {}
    reactant_ids = [rne.get_reactants_of_reaction(i) for i in range(rne.num_reactions)]
    r = [rne.species[i.pop()] if len(i) > 0 else None for i in reactant_ids]
    product_ids = [rne.get_products_of_reaction(i) for i in range(rne.num_reactions)]
    p = [rne.species[i.pop()] if len(i) > 0 else None for i in product_ids]
    reactions = list(zip(r, p))
    r_a = "reaction_sends_activation"
    r_i = "reaction_sends_inhibition"
    s_a = "species_activated"
    s_i = "species_inhibited"
    if condition is not None:
        r_a = f"{r_a}_{condition}"
        r_i = f"{r_i}_{condition}"
        s_a = f"{s_a}_{condition}"
        s_i = f"{s_i}_{condition}"
    up_edges = [r for r, n in zip(reactions, pc.get_symbol(r_a).value) if n > 0.5]
    down_edges = [r for r, n in zip(reactions, pc.get_symbol(r_i).value) if n > 0.5]
    up_nodes = [s for s, n in zip(rne.species, pc.get_symbol(s_a).value) if n > 0.5]
    down_nodes = [s for s, n in zip(rne.species, pc.get_symbol(s_i).value) if n > 0.5]
    for e in up_edges:
        edge_props[e] = {"edge_color": "tab:red", "width": 2.0}
    for e in down_edges:
        edge_props[e] = {"edge_color": "tab:blue", "width": 2.0}
    for n in up_nodes:
        node_props[n] = {"color": "tab:red"}
    for n in down_nodes:
        node_props[n] = {"color": "tab:blue"}
    return {"nodes": node_props, "edges": edge_props}


def signflow_constraints(
    rn: ReNet,
    backend: Backend = DEFAULT_BACKEND,
    signal_implies_flow: bool = True,
    flow_implies_signal: bool = False,  # not supported in multi-conditions
    dag: bool = True,
    dag_flexibility: float = 1.0,
    use_flow_indicators: bool = True,
    eps: float = 1e-3,
) -> ProblemDef:
    if "_s" not in rn.species:
        raise ValueError(
            "The provided network does not have the `_s` and `_t` dummy nodes. Please call `carnival_network` before this function."
        )
    perturbations = rn.species_names(
        rn.successors(rn.get_species_id("_s"), id_type=IdType.SPECIES)
    )
    conditions = []
    for pert in perturbations:
        if not pert.startswith("_pert_"):
            raise ValueError(
                "The provided network does not contain the `_pert_` dummy nodes (perturbations per condition). Please call `carinval_network` before this function."
            )
        conditions.append(pert.split("_pert_")[1])
    n_conditions = len(conditions)
    LOGGER.debug(f"Creating CARNIVAL definition with {n_conditions} conditions")
    if n_conditions > 1 and flow_implies_signal:
        raise ValueError("flow_implies_signal is not supported in multi-conditions")
    use_flow = use_flow_indicators or flow_implies_signal or signal_implies_flow
    if use_flow:
        p = backend.Flow(rn)
        F, Fi = p.get_symbol(VAR_FLOW), None
        p += F[rn.get_reaction_id("_outflow")] >= 1.01 * eps
        if use_flow_indicators:
            p += Indicators()
            # TODO: make this simpler!!
            Fi = p.get_symbol(VAR_FLOW + "_ipos")
    else:
        F, Fi = None, None
        p = backend.Problem()
    
    
    # If a DAG is required, compute first bounds on distance
    # to have a tighter problem. Dag_flexibility param can
    # be used to obtain suboptimal solutions by allowing
    # less flexibility in the way nodes can be placed.
    dist = dict()
    if dag:
        dist = rn.bfs(["_s"])
        node_maxdist = rn.num_species - 1
        dist_lbound = np.array([dist.get(i, 0) for i in range(rn.num_species)])
        # compute upper bounds (max allowable distance in DAG for nodes)
        dist_ubound = (
            np.floor(
                dag_flexibility
                * (np.full(dist_lbound.shape, node_maxdist) - dist_lbound)
            )
            + dist_lbound
        ).astype(int)


    for c in conditions:
        reachable = list(range(rn.num_species))
        non_reachable = []
        # If there is more than one condition, just check which nodes are reachable
        # from this condition (forward pass from perturbation to measurements and
        # backward pass from measurements to perturbations). Nodes and reactions
        # that cannot be reached should have a value of 0.
        fwd = set(rn.bfs([f"_pert_{c}"]).keys())
        bck = set(rn.bfs([f"_meas_{c}"], rev=True).keys())
        reachable = list(
            fwd.intersection(bck) | {rn.species.index("_s"), rn.species.index("_t")}
        )
        non_reachable = list(set(range(rn.num_species)) - set(reachable))

        N_act = backend.Variable(
            f"species_activated_{c}", (rn.num_species,), vartype=VarType.BINARY
        )
        N_inh = backend.Variable(
            f"species_inhibited_{c}", (rn.num_species,), vartype=VarType.BINARY
        )
        R_act = backend.Variable(
            f"reaction_sends_activation_{c}",
            (rn.num_reactions,),
            vartype=VarType.BINARY,
        )
        R_inh = backend.Variable(
            f"reaction_sends_inhibition_{c}",
            (rn.num_reactions,),
            vartype=VarType.BINARY,
        )
        p += [N_act, N_inh, R_act, R_inh]

        if len(non_reachable) > 0:
            # TODO: Do the same for non reachable reactions
            p += N_act[non_reachable] == 0
            p += N_inh[non_reachable] == 0

        # Also, if the measurements are selected, their edges from
        # measurement to the dummy _meas_ node have to be selected.
        # Not required, but forces to have a connected graph.
        meas_rxns = list(rn.get_reactions_with_product(rn.get_species_id(f"_meas_{c}")))
        meas_species = [rn.get_reactants_of_reaction(r).pop() for r in meas_rxns]
        p += R_act[meas_rxns] + R_inh[meas_rxns] == N_act[meas_species] + N_inh[meas_species]


        # Just for convenience, force the dummy measurement node to be active. Not required
        p += N_act[rn.species.index(f"_meas_{c}")] == 1
        # Same for source and target nodes. Assume they are active as a convention. Note that
        # dummy source _s connects to perturbations with activatory edges if perturbation is up
        # and inhibitory edges if perturbation is down. Dummy _s could be also down and propagate
        # the inverse signal instead. Same for dummy target _t. As a convention, they are forced
        # to be always activated instead to avoid these alternative options.
        p += N_act[rn.species.index("_s")] == 1
        p += N_act[rn.species.index("_t")] == 1
        # Dont define activation/inhibition for reactions with no reactants or no products
        # rids = np.flatnonzero(np.logical_and(rn.has_reactant(), rn.has_product()))
        # TODO: Simplify this by adding methods to ReNet
        # TODO: Filter out reactions that has reactant or product in the non reachable set
        valid = np.zeros(rn.num_reactions, dtype=bool)
        valid[reachable] = True
        has_reactant = np.sum(rn.stoichiometry < 0, axis=0) > 0
        has_product = np.sum(rn.stoichiometry > 0, axis=0) > 0
        # has_reactant = np.sum(np.logical_and(rn.stoichiometry < 0, valid), axis=0) > 0
        # has_product = np.sum(np.logical_and(rn.stoichiometry > 0, valid), axis=0) > 0
        rids = np.flatnonzero(np.logical_and(has_reactant, has_product))
        # TODO: Throw error if reactions have more than one reactant or product
        # Get reactants and products: note that reactions should have
        # only 1 reactant 1 product at most for CARNIVAL
        # ix_react = np.array(list({rn.get_reactants([i]).pop() for i in rids}.intersection(reachable)))
        # ix_prod = np.array(list({rn.get_products([i]).pop() for i in rids}.intersection(reachable)))
        ix_react = np.array([rn.get_reactants([i]).pop() for i in rids])
        ix_prod = np.array([rn.get_products([i]).pop() for i in rids])
        signs = np.array(rn.properties.reaction_values())[rids]
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
                (rn.num_species,),
                vartype=VarType.CONTINUOUS,
                lb=dist_lbound,
                ub=dist_ubound,
            )
            p += L
            p += L[ix_prod] - L[ix_react] >= Ra + Ri + (1 - rn.num_species) * (
                1 - (Ra + Ri)
            )
            p += L[ix_prod] - L[ix_react] <= rn.num_species - 1

        # Link flow with signal
        if signal_implies_flow and flow_implies_signal:
            # Bi-directional implication
            if Fi is None:
                raise NotImplementedError(
                    "flow <=> signal implication supported only with flow indicators"
                )
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
        # carry some signal.
        incidence_matrix = rn.stoichiometry.clip(0, 1)
        p += N_act <= incidence_matrix @ R_act
        p += N_inh <= incidence_matrix @ R_inh
    return p


def default_sign_loss(
    g: Graph,
    conditions: Dict,
    problem: ProblemDef,
    l0_penalty_reaction: float = 0.0,
    l1_penalty_reaction: float = 0.0,
    l0_penalty_species: float = 0.0,
    ub_loss: Optional[Union[float, List[float]]] = None,
    lb_loss: Optional[Union[float, List[float]]] = None,
) -> ProblemDef:
    losses = []
    p = ProblemDef()
    F, Fi = None, None
    if VAR_FLOW in problem.symbols.keys():
        F = problem.get_symbol(VAR_FLOW)
    if VAR_FLOW + "_ipos" in problem.symbols.keys():
        Fi = problem.get_symbol(VAR_FLOW + "_ipos")
    for i, c in enumerate(conditions.keys()):
        N_act, N_inh = problem.get_symbols(
            f"species_activated_{c}", f"species_inhibited_{c}"
        )
        # Get the values of the species for the given condition
        species_values = np.array(
            [conditions[c][s][1] if s in conditions[c] else 0 for s in g.vertices]
        )
        pos = species_values.clip(0, np.inf).reshape(1, -1) @ (1 - (N_act - N_inh))
        neg = np.abs(species_values.clip(-np.inf, 0)).reshape(1, -1) @ (
            (N_act - N_inh) + 1
        )
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
    if l0_penalty_reaction > 0:
        if Fi is None:
            raise ValueError(
                "L0 regularization on flow cannot be used if signal is not connected to flow."
            )
        # TODO: Issues with sum and PICOS (https://gitlab.com/picos-api/picos/-/issues/330)
        # override sum with picos.sum method
        losses.append(np.ones(Fi.shape) @ Fi)
        weights.append(l0_penalty_reaction)
    if l1_penalty_reaction != 0:
        if F is None:
            raise NotImplementedError("Use R_act and R_inh for regularization")
        # TODO: This is valid only for positive fluxes
        losses.append(np.ones(F.shape) @ F)
        weights.append(l1_penalty_reaction)
    if l0_penalty_species != 0:
        losses.append(np.ones(N_act.shape) @ (N_act + N_inh))
        weights.append(l0_penalty_species)
    # Add objective and weights to p
    p.add_objectives(losses, weights, inplace=True)
    return p

def signflow(
    rn: ReNet,
    conditions: Dict,
    signal_implies_flow: bool = True,
    flow_implies_signal: bool = False,  # not supported in multi-conditions
    dag: bool = True,
    dag_flexibility: float = 1.0,
    l0_penalty_reaction: float = 0.0,
    l1_penalty_reaction: float = 0.0,
    l0_penalty_species: float = 0.0,
    ub_loss: Optional[Union[float, List[float]]] = None,
    lb_loss: Optional[Union[float, List[float]]] = None,
    use_flow_indicators: bool = True,
    eps: float = 1e-3,
    backend: Backend = DEFAULT_BACKEND,
):
    p = signflow_constraints(
        rn,
        backend=backend,
        signal_implies_flow=signal_implies_flow,
        flow_implies_signal=flow_implies_signal,
        dag=dag,
        dag_flexibility=dag_flexibility,
        use_flow_indicators=use_flow_indicators,
        eps=eps,
    )
    return p + default_sign_loss(
        rn,
        conditions,
        p,
        l0_penalty_reaction=l0_penalty_reaction,
        l0_penalty_species=l0_penalty_species,
        l1_penalty_reaction=l1_penalty_reaction,
        ub_loss=ub_loss,
        lb_loss=lb_loss,
    )
