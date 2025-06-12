import time
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np

import corneto as cn
from corneto._graph import BaseGraph, Graph
from corneto._settings import LOGGER, sparsify
from corneto.methods.signal._util import (
    get_incidence_matrices_of_edges,
    get_interactions,
)
from corneto.methods.signaling import create_flow_graph, signflow


def _info(s, show=True):
    if show:
        LOGGER.info(s)


def read_dataset(zip_path):
    """Extracts and processes a graph and its vertex attributes from a zipped dataset.

    The function reads two CSV files from a specified zipfile: 'pkn.csv' and 'data.csv'.
    The 'pkn.csv' contains graph edges with three columns: 'source', 'interaction',
    and 'target'. The 'interaction' column uses integers to denote the type of
    interaction (1 for activation, -1 for inhibition). The 'data.csv' file contains
    vertex attributes with three columns: 'vertex', 'value', and 'type', where 'value'
    can be a continuous measure such as from a t-statistic in differential expression,
    and 'type' categorizes vertices as either inputs ('P' for perturbation) or outputs
    ('M' for measurement).

    Args:
        zip_path (str): The file path to the zip file containing the dataset.

    Returns:
        tuple: A tuple containing:
            - Graph: A graph object initialized with edges from 'pkn.csv'. Each edge is
              defined by a source, a target, and an interaction type.
            - dict: A dictionary mapping each protein (vertex) to a tuple of ('type',
              'value'), where 'type' is either 'P' or 'M' and 'value' represents the
              continuous state of the protein.

    Raises:
        FileNotFoundError: If the zip file cannot be found at the provided path.
        KeyError: If expected columns are missing in the CSV files, indicating incorrect
                  or incomplete data.

    Example:
        >>> graph, vertex_attrs = read_dataset('path/to/dataset.zip')
        >>> print(graph.shape)  # Shape of the imported graph ([vertices, edges])
        >>> print(vertex_attrs) # Displays protein attributes
    """
    import zipfile

    import pandas as pd

    from corneto import Graph

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("pkn.csv") as pkn, z.open("data.csv") as data:
            df_pkn = pd.read_csv(pkn)
            df_data = pd.read_csv(data)

    # Convert pkn.csv rows to tuples and create a graph from them
    tpl = [tuple(x) for x in df_pkn.itertuples(index=False)]
    G = Graph.from_sif_tuples(tpl)

    # Process the 'data.csv' for vertex attributes
    df_data["type"] = df_data["type"].replace({"input": "P", "output": "M"})
    data_dict = dict(zip(df_data["vertex"], zip(df_data["type"], df_data["value"])))

    return G, data_dict


def preprocess_graph(priorKnowledgeNetwork: BaseGraph, perturbations: Dict, measurements: Dict):
    V = set(priorKnowledgeNetwork.vertices)
    inputs = set(perturbations.keys())
    outputs = set(measurements.keys())
    c_inputs = V.intersection(inputs)
    c_outputs = V.intersection(outputs)
    Gp = priorKnowledgeNetwork.prune(list(c_inputs), list(c_outputs))
    V = set(Gp.vertices)
    cp_inputs = {input: v for input, v in perturbations.items() if input in V}
    cp_outputs = {output: v for output, v in measurements.items() if output in V}
    return Gp, cp_inputs, cp_outputs


# TODO: Return a problem, which is associated to the carnival graph
# think about connecting edge/nodes with variables!
def runVanillaCarnival(
    perturbations: Dict,
    measurements: Dict,
    priorKnowledgeNetwork: Union[List[Tuple], Graph],
    betaWeight: float = 0.2,
    solver=None,
    backend_options=dict(),
    solve=True,
    verbose=True,
    **kwargs,
):
    if backend_options is None:
        backend_options = dict()
    backend_options["verbosity"] = verbose
    start = time.time()
    data = dict()
    for k, v in perturbations.items():
        data[k] = ("P", v)
    for k, v in measurements.items():
        data[k] = ("M", v)
    conditions = {"c0": data}
    if isinstance(priorKnowledgeNetwork, List):
        G = Graph.from_sif_tuples(priorKnowledgeNetwork)
    elif isinstance(priorKnowledgeNetwork, Graph):
        G = priorKnowledgeNetwork
    else:
        raise ValueError("Provide a list of sif tuples or a graph")
    # Prune the graph
    V = set(G.vertices)
    inputs = set(perturbations.keys())
    outputs = set(measurements.keys())
    c_inputs = V.intersection(inputs)
    c_outputs = V.intersection(outputs)
    _info(f"{len(c_inputs)}/{len(inputs)} inputs mapped to the graph", show=verbose)
    _info(f"{len(c_outputs)}/{len(outputs)} outputs mapped to the graph", show=verbose)
    _info(f"Pruning the graph with size: V x E = {G.shape}...", show=verbose)
    Gp = G.prune(list(c_inputs), list(c_outputs))
    _info(f"Finished. Final size: V x E = {Gp.shape}.", show=verbose)
    V = set(Gp.vertices)
    cp_inputs = {input: v for input, v in perturbations.items() if input in V}
    cp_outputs = {output: v for output, v in measurements.items() if output in V}
    _info(f"{len(cp_inputs)}/{len(c_inputs)} inputs after pruning.", show=verbose)
    _info(f"{len(cp_outputs)}/{len(c_outputs)} outputs after pruning.", show=verbose)
    _info("Converting into a flow graph...", show=verbose)
    Gf = create_flow_graph(Gp, conditions)
    _info("Creating a network flow problem...", show=verbose)
    P = signflow(Gf, conditions, l0_penalty_vertices=betaWeight, **kwargs)
    _info("Preprocess completed.", show=verbose)
    if solve:
        P.solve(solver=solver, **backend_options)
    end = time.time() - start
    _info(f"Finished in {end:.2f} s.", show=verbose)
    return P, Gf


def runInverseCarnival(
    measurements: Dict,
    priorKnowledgeNetwork: Union[List[Tuple], Graph],
    betaWeight: float = 0.2,
    solver=None,
    solve=True,
    **kwargs,
):
    raise NotImplementedError()
    if isinstance(priorKnowledgeNetwork, List):
        G = Graph.from_sif_tuples(priorKnowledgeNetwork)
    elif isinstance(priorKnowledgeNetwork, Graph):
        G = priorKnowledgeNetwork
    else:
        raise ValueError("Provide a list of sif tuples or a graph")
    perturbations = {v: 0.0 for v in G.get_source_vertices()}
    return runVanillaCarnival(perturbations, measurements, G, betaWeight=betaWeight, solver=solver, **kwargs)


def heuristic_carnival(
    priorKnowledgeNetwork: Union[List[Tuple], Graph],
    perturbations: Dict,
    measurements: Dict,
    restricted_search: bool = False,
    prune: bool = True,
    verbose=True,
    max_time=None,
    max_edges=None,
):
    if isinstance(priorKnowledgeNetwork, List):
        G = Graph.from_sif_tuples(priorKnowledgeNetwork)
    elif isinstance(priorKnowledgeNetwork, Graph):
        G = priorKnowledgeNetwork
    else:
        raise ValueError("Provide a list of sif tuples or a graph")
    Gp = G
    perts = set(perturbations.keys())
    meas = set(measurements.keys())
    V = set(G.vertices)
    inputs = V.intersection(perts)
    outputs = V.intersection(meas)
    if verbose:
        print(f"{len(inputs)}/{len(perts)} inputs mapped to the graph")
        print(f"{len(outputs)}/{len(meas)} outputs mapped to the graph")
    if prune:
        if verbose:
            print(f"Pruning the graph with size: V x E = {G.shape}...")
        Gp = G.prune(list(inputs), list(outputs))
        if verbose:
            print(f"Finished. Final size: V x E = {Gp.shape}.")
    V = set(Gp.V)
    inputs = V.intersection(perts)
    outputs = V.intersection(meas)
    # Clean unreachable inputs/outputs
    inputs_p = {k: perturbations[k] for k in inputs}
    outputs_p = {k: measurements[k] for k in outputs}
    selected_edges = None
    if restricted_search:
        selected_edges = reachability_graph(Gp, inputs_p, outputs_p, verbose=verbose)
    selected_edges, paths, stats = bfs_search(
        Gp,
        inputs_p,
        outputs_p,
        subset_edges=selected_edges,
        max_time=max_time,
        max_edges=max_edges,
        verbose=verbose,
    )
    # Estimate error, using inputs and outputs and comparing to what was selected
    predicted_values = {p[0]: p[1][p[0]][1] for p in paths}
    errors = dict()
    for k, v in measurements.items():
        error = abs(v - predicted_values.get(k, 0))
        errors[k] = error
    total_error = sum(errors.values())
    if verbose:
        print(f"Total error: {total_error}")
        print(f"Number of selected edges: {len(selected_edges)}")
    return Gp, selected_edges, paths, stats, errors


def get_result(P, G, condition="c0", exclude_dummies=True):
    V = P.expr["vertex_values_" + condition].value
    E = P.expr["edge_values_" + condition].value
    d_vertices = {"V": G.V, "value": V}
    d_edges = {"E": G.E, "value": E}
    return d_vertices, d_edges


def get_selected_edges(P, G, condition="c0", exclude_dummies=True):
    # Get the indexes of the edges whose value is not zero
    E = P.expr["edge_values_" + condition].value
    selected_edges = []
    for i, v in enumerate(E):
        if v != 0:
            # Check if the edge contains a
            # vertex starting with "_"
            if exclude_dummies:
                s, t = G.get_edge(i)
                s = list(s)
                t = list(t)
                if len(s) > 0 and s[0].startswith("_"):
                    continue
                if len(t) > 0 and t[0].startswith("_"):
                    continue
            selected_edges.append(i)
    return selected_edges


def _str_state(state, max_steps=3):
    v, path = state
    nodes = []
    n_steps = len(path) - 1
    skip = False
    for i, (k, v) in enumerate(path.items()):
        if i < n_steps and skip:
            continue
        pos, val, edge = v
        if max_steps is not None and i >= max_steps and i < n_steps:
            nodes.append("...")
            skip = True
        else:
            if val > 0:
                nodes.append(f"+{k}")
            else:
                nodes.append(f"-{k}")
    return " -> ".join(nodes)


def _extract_nodes(path, values=False, subset=None):
    d = []
    v, pd = path
    for k, v in pd.items():
        e = k
        if values:
            e = (k, v[1])
        if subset is not None:
            if k not in subset:
                continue
        d.append(e)
    return tuple(d)


def reachability_graph(
    G,
    input_nodes,
    output_nodes,
    subset_edges=None,
    verbose=True,
    early_stop=False,
    expand_outputs=True,
    max_printed_outputs=10,
):
    visited = set(input_nodes)
    current = set(input_nodes)
    unreached_outputs = set(output_nodes)
    outs = set(output_nodes)
    selected_edges = set()
    layer = 0
    if verbose:
        print("Starting reachability analysis...")
        print(f"L{layer:<3}: {len(input_nodes):<4} > input(s)")
    while current is not None and len(current) > 0:
        layer += 1
        new = set()
        for v in current:
            for i, (s, t) in G.out_edges(v):
                if subset_edges is not None and i not in subset_edges:
                    continue
                # Add only if t is a new node
                nt = list(t)
                if len(nt) == 0:
                    continue
                vt = nt[0]
                if vt not in visited:
                    new |= {vt}
                    selected_edges.add(i)
        # How many are output nodes?
        reached_outputs = outs.intersection(new)
        unreached_outputs -= reached_outputs
        if verbose:
            print(f"L{layer:<3}: {len(new):<4}", end="")
            if len(reached_outputs) > 0:
                if len(reached_outputs) <= max_printed_outputs:
                    str_reached = "/".join(reached_outputs)
                else:
                    # Get only the first max_printed_outputs items
                    str_reached = "/".join(list(reached_outputs)[:max_printed_outputs]) + "..."
                print(f" > {len(reached_outputs):<4} output(s): {str_reached}")
            else:
                print("")
        visited |= new
        current = set(new)
        if not expand_outputs:
            current -= reached_outputs
        if early_stop and len(unreached_outputs) == 0:
            break
    if verbose:
        print(f"Finished reachability analysis ({len(selected_edges)} selected edges).")
    return selected_edges


def _path_conflict(p, paths):
    nodes_in_path = _extract_nodes(p)
    valid = True
    p_a, p_b = None, None
    for path in paths:
        common = set(nodes_in_path).intersection(_extract_nodes(path))
        p_a = _extract_nodes(p, values=True, subset=common)
        p_b = _extract_nodes(path, values=True, subset=common)
        if p_a != p_b:
            valid = False
            break
    return valid, p_a, p_b


def _str_path_nodes(a):
    nodes = []
    for k, v in a:
        if v > 0:
            nodes.append(f"+{k}")
        else:
            nodes.append(f"-{k}")
    return "/".join(nodes)


def bfs_search(
    G,
    initial_dict,
    final_dict,
    max_time=None,
    queue_max_size=None,
    subset_edges=None,
    max_edges=None,
    verbose=True,
):
    Q = []
    reached = set()
    stats = dict(loops=0, iters=0, conflicts=0)
    paths = []
    exit = False
    maxq = 0
    last_level = 0
    selected_edges = set()
    first_level = G.bfs(list(initial_dict.keys()))
    outs = []
    for k in final_dict.keys():
        outs.append(str(k) + f" (L{first_level[k]})")
    if verbose:
        print(", ".join(outs))
    for k, w in initial_dict.items():
        Q.append((k, {k: (0, w, None)}))
    start = time.time()
    while len(Q) > 0 and not exit:
        if max_time is not None and time.time() - start > max_time:
            if verbose:
                print("Timeout reached.")
            break
        current = Q.pop(0)
        n, v = current
        if v[n][0] > last_level:
            last_level = v[n][0]
            if verbose:
                elapsed = time.time() - start
                print(f"L{last_level:<3}: {stats['iters']:>6} iters, {elapsed:.2f} s.")
        for i, (s, t) in G.out_edges(n):
            if subset_edges is not None and i not in subset_edges:
                continue
            val = int(G.get_attr_edge(i).get("interaction", 0))
            nt = list(t)
            if len(nt) == 0:
                continue
            nt = nt[0]
            if nt == n or nt in v:
                stats["loops"] += 1
                continue
            nv = dict(v)
            pos, w, _ = nv[n]
            value = w * val
            nv[nt] = (pos + 1, value, i)
            # State = (vertex, (dist. from source, value=+1/-1, edge index))
            new_state = (nt, nv)
            # Check if the vertex is in the goal set
            if nt not in reached:
                vf = final_dict.get(nt, None)
                if vf is not None and vf == value:
                    valid, p_a, p_b = _path_conflict(new_state, paths)
                    if verbose:
                        print(" >", _str_state(new_state))
                    if not valid:
                        print("   ! conflict: {} != {}".format(_str_path_nodes(p_a), _str_path_nodes(p_b)))
                        stats["conflicts"] += 1
                        continue
                    reached |= {nt}
                    paths.append(new_state)
                    # Add edges
                    selected_edges |= set(edge_idx for (_, _, edge_idx) in nv.values() if edge_idx is not None)
                    if max_edges is not None and len(selected_edges) >= max_edges:
                        if verbose:
                            print("Max edges reached.")
                        exit = True
                        break

            if len(reached) >= len(final_dict):
                exit = True
                break
            # No loop, add new state
            Q.append(new_state)
            if len(Q) > maxq:
                maxq = len(Q)
            if queue_max_size is not None and queue_max_size > 0 and len(Q) > queue_max_size:
                break
        stats["iters"] += 1
    if verbose:
        print(f"Finished ({time.time() - start:.2f} s)")
        print(f" > Number of selected edges: {len(selected_edges)}")
        print(f" > Total iterations: {stats['iters']}")
        print(f" > Detected loops: {stats['loops']}")
        print(f" > Conflicts: {stats['conflicts']}")
    return selected_edges, paths, stats


def create_flow_carnival_v4(
    G,
    exp_list,
    lambd=0.2,
    exclusive_vertex_values=True,
    upper_bound_flow=1000,
    penalty_on="signal",  # or "flow"
    slack_reg=False,
    set_perturbation_values=True,
    fix_input_values=True,
    backend=cn.DEFAULT_BACKEND,
):
    At, Ah = get_incidence_matrices_of_edges(G)
    interaction = get_interactions(G)

    # Create a single unique flow. The flow selects the
    # subset of the PKN that will contain all signal propagations.

    # NOTE: increased UB flow since we dont have indicator, fractional positive flows <1
    # will block signal in this case. To verify if this is a problem. The UB corresponds
    # to the value of the Big-M in the constraints.
    P = backend.Flow(G, ub=upper_bound_flow)
    if penalty_on == "flow":
        P += cn.Indicator()

    Eact = backend.Variable("edge_activates", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Einh = backend.Variable("edge_inhibits", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)

    # Edge cannot activate and inhibit at the same time
    P += Eact + Einh <= 1

    # The value of a vertex is the difference of the positive and negative incoming edges
    Va = At @ Eact
    Vi = At @ Einh
    V = Va - Vi

    if exclusive_vertex_values:
        # otherwise a vertex can be both active and inactive through diff. paths
        # NOTE: Seems to increase the gap of the relaxated problem. Slower than
        # the formulations using indicator variables.
        P += Va + Vi <= 1
    P.register("vertex_value", V)
    P.register("vertex_inhibited", Vi)
    P.register("vertex_activated", Va)
    P.register("edge_value", Eact - Einh)
    P.register("edge_has_signal", Eact + Einh)

    # Add acyclic constraints on the edge_has_signal (signal)
    P = backend.Acyclic(G, P, indicator_positive_var_name="edge_has_signal")

    edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

    # Extend flows (M,) to (M, N) where N is the num of experiments
    F = P.expr.flow.reshape((Eact.shape[0], 1)) @ np.ones((1, len(exp_list)))
    # Fi = P.expr._flow_i.reshape((Eact.shape[0], 1)) @ np.ones((1, len(exp_list)))

    # If no flow, no signal (signal cannot circulate in a non-selected subgraph)
    P += Eact + Einh <= F

    Int = sparsify(np.reshape(interaction, (interaction.shape[0], 1)) @ np.ones((1, len(exp_list))))

    # Sum incoming signals for edges with head (for all samples)
    # An edge can only be active in two situations:
    # - the head vertex is active and the edge activatory
    # - or the head vertex is inactive and the edge inhibitory
    sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(Int[edges_with_head, :] > 0)
    sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(Int[edges_with_head, :] < 0)
    P += Eact[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh
    # The opposite for inhibition
    sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(Int[edges_with_head, :] < 0)
    sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(Int[edges_with_head, :] > 0)
    P += Einh[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh

    vertex_indexes = np.array([G.V.index(v) for exp in exp_list for v in exp_list[exp]["input"]])
    perturbation_values = np.array([int(np.sign(val)) for exp in exp_list for val in exp_list[exp]["input"].values()])

    # Set the perturbations to the given values
    if set_perturbation_values:
        warnings.warn("Using set_perturbation_values, please disable since behavior differs from original carnival")
        nonzero_mask = perturbation_values != 0
        nonzero_vertex_indexes = vertex_indexes[nonzero_mask]
        nonzero_perturbation_values = perturbation_values[nonzero_mask]
        # Assign the perturbations only to the nonzero ones
        P += V[nonzero_vertex_indexes, :] == nonzero_perturbation_values[:, None]

    all_vertices = G.V
    all_inputs = [k for k in all_vertices if len(list(G.predecessors(k))) == 0]

    for i, exp in enumerate(exp_list):
        # Any input not indicated in the condition must be blocked
        if not set_perturbation_values:
            # Block flow from any input not in the set of valid inputs
            # for the given condition
            m_inputs = list(exp_list[exp]["input"].keys())
            for v_input in all_inputs:
                if v_input not in m_inputs:
                    P += V[all_vertices.index(v_input), i] == 0
                else:
                    input_value = int(exp_list[exp]["input"][v_input])
                    if input_value != 0:
                        if fix_input_values:
                            if input_value == -1 or input_value == 1:
                                P += V[all_vertices.index(v_input), i] == input_value
                            else:
                                raise ValueError(
                                    f"Invalid value for input vertex {v_input}: {input_value} (only -1, 0 or 1)"
                                )

        m_nodes = list(exp_list[exp]["output"].keys())
        m_values = np.array(list(exp_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]

        # Count the negative and the positive parts of the measurements
        # Compute the error and add it to the objective function
        error = (1 - V[m_nodes_positions, i].multiply(np.sign(m_values))).multiply(abs(m_values))
        P.add_objectives(sum(error))

    if lambd > 0:
        obj = None
        if penalty_on == "signal":
            P += cn.opt.linear_or(Eact + Einh, axis=1, varname="Y")
            obj = sum(P.expr.Y)
        elif penalty_on == "flow":
            obj = sum(P.expr._flow_i)
        else:
            raise ValueError(f"Invalid penalty_on={penalty_on} not valid. Only signal or flow are supported.")
        if slack_reg:
            s = backend.Variable(lb=0, ub=G.num_edges)
            max_edges = G.num_edges - s
            P += obj <= max_edges
            P.add_objectives(max_edges, weights=lambd)
        else:
            P.add_objectives(obj, weights=lambd)
    else:
        P.add_objectives(0)
    return P


def create_flow_carnival_v3(
    G,
    exp_list,
    lambd=0.2,
    exclusive_vertex_values=True,
    upper_bound_flow=1000,
    backend=cn.DEFAULT_BACKEND,
):
    At, Ah = get_incidence_matrices_of_edges(G)
    interaction = get_interactions(G)

    # Create a single unique flow. The flow selects the
    # subset of the PKN that will contain all signal propagations.

    # NOTE: increased UB flow since we dont have indicator, fractional positive flows <1
    # will block signal in this case. To verify if this is a problem. The UB corresponds
    # to the value of the Big-M in the constraints.
    P = backend.Flow(G, ub=upper_bound_flow)

    Eact = backend.Variable("edge_activates", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Einh = backend.Variable("edge_inhibits", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    # This var should be removed
    Z = backend.Variable("dummy", (G.num_vertices, len(exp_list)), vartype=cn.VarType.CONTINUOUS)
    P += Z >= 0

    # Edge cannot activate and inhibit at the same time
    P += Eact + Einh <= 1

    # The value of a vertex is the difference of the positive and negative incoming edges
    Va = At @ Eact
    Vi = At @ Einh
    V = Va - Vi

    if exclusive_vertex_values:
        # otherwise a vertex can be both active and inactive through diff. paths
        # NOTE: Seems to increase the gap of the relaxated problem. Slower than
        # the formulations using indicator variables.
        P += Va + Vi <= 1
    P.register("vertex_value", V)
    P.register("vertex_inhibited", Vi)
    P.register("vertex_activated", Va)
    P.register("edge_value", Eact - Einh)
    P.register("edge_has_signal", Eact + Einh)

    # Add acyclic constraints on the edge_has_signal (signal)
    P = backend.Acyclic(G, P, indicator_positive_var_name="edge_has_signal")

    edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

    for exp, iexp in zip(exp_list, range(len(exp_list))):
        # Edge cannot activate or inhibit downstream vertices if it is not carrying flow.
        # Note: since Eact and Einh are binary variables, the threshold activation for signal
        # is at least 1 unit of flow. That means that positive, non-zero flows below 1
        # are not enough for activating signal.
        P += Eact[:, iexp] + Einh[:, iexp] <= P.expr.flow

        P += Eact[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] > 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] < 0)  # constrain 1B
        P += Einh[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] < 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] > 0)  # constrain 2B

        # perturbation:
        p_nodes = list(exp_list[exp]["input"].keys())
        p_values = list(exp_list[exp]["input"].values())
        p_nodes_positions = [G.V.index(key) for key in p_nodes]

        P += V[p_nodes_positions, iexp] == p_values

        # measuremenents:
        m_nodes = list(exp_list[exp]["output"].keys())
        m_values = np.array(list(exp_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]

        # Count the negative and the positive parts of the measurements

        # linearization of the ABS function: https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
        P += V[m_nodes_positions, iexp] - np.sign(m_values) <= Z[m_nodes_positions, iexp]
        P += -V[m_nodes_positions, iexp] + np.sign(m_values) <= Z[m_nodes_positions, iexp]

        # Compute the error and add it to the objective function
        # error = abs(m_values) * (1 - np.sign(m_values) * V[m_nodes_positions, iexp])

        P.add_objectives(sum(Z[m_nodes_positions, iexp].multiply(abs(m_values))))
    if lambd > 0:
        P += cn.opt.linear_or(Eact + Einh, axis=1, varname="edge_has_signal_in_any_sample")
        P.add_objectives(sum(P.expr.edge_has_signal_in_any_sample), weights=lambd)
    else:
        P.add_objectives(0)
    return P


def create_flow_carnival_v2(G, exp_list, lambd=0.2):
    # This is the flow acyclic signal
    At, Ah = get_incidence_matrices_of_edges(G)
    interaction = get_interactions(G)

    VAR_FLOW = "with_flow"
    P = cn.opt.Flow(G, varname=VAR_FLOW)

    # TODO: check input graph, experiment list and their compatibility

    Eact = cn.opt.Variable("edge_activates", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Einh = cn.opt.Variable("edge_inhibits", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Z = cn.opt.Variable("dummy", (G.num_vertices, len(exp_list)), vartype=cn.VarType.CONTINUOUS)
    P += Z >= 0

    # Edge cannot activate and inhibit at the same time
    P += Eact + Einh <= 1

    # The value of a vertex is the difference of the positive and negative incoming edges
    Va = At @ Eact
    Vi = At @ Einh
    V = Va - Vi
    P.register("vertex_value", V)
    P.register("vertex_inhibited", Vi)
    P.register("vertex_activated", Va)
    P.register("edge_value", Eact - Einh)
    P.register("edge_has_signal", Eact + Einh)

    # Add acyclic constraints on the edge_has_signal (signal)
    P = cn.opt.Acyclic(G, P, indicator_positive_var_name="edge_has_signal")

    edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

    for exp, iexp in zip(exp_list, range(len(exp_list))):
        # Edge cannot activate or inhibit downstream vertices if it is not carrying flow
        P += Eact[:, iexp] + Einh[:, iexp] <= P.expr.with_flow

        P += Eact[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] > 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] < 0)  # constrain 1B
        P += Einh[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] < 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] > 0)  # constrain 2B

        # perturbation:
        p_nodes = list(exp_list[exp]["input"].keys())
        p_values = list(exp_list[exp]["input"].values())
        p_nodes_positions = [G.V.index(key) for key in p_nodes]

        P += V[p_nodes_positions, iexp] == p_values

        # measuremenents:
        m_nodes = list(exp_list[exp]["output"].keys())
        m_values = np.array(list(exp_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]

        # linearization of the ABS function: https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
        P += V[m_nodes_positions, iexp] - np.sign(m_values) <= Z[m_nodes_positions, iexp]
        P += -V[m_nodes_positions, iexp] + np.sign(m_values) <= Z[m_nodes_positions, iexp]

        P.add_objectives(sum(Z[m_nodes_positions, iexp].multiply(abs(m_values))))
    if lambd > 0:
        P.add_objectives(lambd * sum(sum(Eact + Einh)))
    else:
        P.add_objectives(0)
    return P


def create_flow_carnival(G, exp_list, lambd=0.2):
    At, Ah = get_incidence_matrices_of_edges(G)
    interaction = get_interactions(G)
    P = cn.opt.AcyclicFlow(G)
    Eact = cn.opt.Variable("edge_activates", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Einh = cn.opt.Variable("edge_inhibits", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    # TODO: Remove dummy
    Z = cn.opt.Variable("dummy", (G.num_vertices, len(exp_list)), vartype=cn.VarType.CONTINUOUS)
    P += Z >= 0

    # Edge cannot activate and inhibit at the same time
    P += Eact + Einh <= 1

    # The value of a vertex is the sign average of the incoming edges
    N_parents = At @ np.ones(len(G.E))
    Va = At @ Eact
    Vi = At @ Einh
    V = Va - Vi  # / N_parents
    P.register("vertex_value", V)
    P.register("vertex_inhibited", Vi)
    P.register("vertex_activated", Va)
    P.register("edge_value", Eact - Einh)

    edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

    for exp, iexp in zip(exp_list, range(len(exp_list))):
        # Edge cannot activate or inhibit downstream vertices if it is not carrying flow
        P += Eact[:, iexp] + Einh[:, iexp] <= P.expr.with_flow

        P += Eact[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] > 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] < 0)  # constrain 1B
        P += Einh[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] < 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] > 0)  # constrain 2B

        # perturbation:
        p_nodes = list(exp_list[exp]["input"].keys())
        p_values = list(exp_list[exp]["input"].values())
        p_nodes_positions = [G.V.index(key) for key in p_nodes]

        P += V[p_nodes_positions, iexp] == p_values

        # measuremenents:
        m_nodes = list(exp_list[exp]["output"].keys())
        m_values = np.array(list(exp_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]

        # linearization of the ABS function: https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
        P += V[m_nodes_positions, iexp] - np.sign(m_values) <= Z[m_nodes_positions, iexp]
        P += -V[m_nodes_positions, iexp] + np.sign(m_values) <= Z[m_nodes_positions, iexp]

        P.add_objectives(sum(Z[m_nodes_positions, iexp].multiply(abs(m_values))))
    if lambd > 0:
        P.add_objectives(lambd * sum(sum(Eact + Einh)))
    else:
        P.add_objectives(0)
    return P


# CARNIVAL with flow (single flow)
def runCARNIVAL_AcyclicFlow(G, exp_list, betaWeight: float = 0.2, solver=None, verbosity=False):
    At, Ah = get_incidence_matrices_of_edges(G)
    interaction = get_interactions(G)
    P = cn.K.AcyclicFlow(G)

    # TODO: check input grahp, experiment list and their compatibility

    Eact = cn.K.Variable("edge_activates", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Einh = cn.K.Variable("edge_inhibits", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Z = cn.K.Variable("dummy", (G.num_vertices, len(exp_list)), vartype=cn.VarType.CONTINUOUS)
    P += Z >= 0

    # Edge cannot activate and inhibit at the same time
    P += Eact + Einh <= 1

    # The value of a vertex is the sign average of the incoming edges
    N_parents = At @ np.ones(len(G.E))
    Va = At @ Eact
    Vi = At @ Einh
    V = Va - Vi  # / N_parents
    P.register("vertex_value", V)
    P.register("vertex_inhibited", Vi)
    P.register("vertex_activated", Va)
    P.register("edge_value", Eact - Einh)

    edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

    for exp, iexp in zip(exp_list, range(len(exp_list))):
        # Edge cannot activate or inhibit downstream vertices if it is not carrying flow
        P += Eact[:, iexp] + Einh[:, iexp] <= P.expr.with_flow

        P += Eact[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] > 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] < 0)  # constrain 1B
        P += Einh[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] < 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] > 0)  # constrain 2B

        # perturbation:
        p_nodes = list(exp_list[exp]["input"].keys())
        p_values = list(exp_list[exp]["input"].values())
        p_nodes_positions = [G.V.index(key) for key in p_nodes]

        P += V[p_nodes_positions, iexp] == p_values

        # measuremenents:
        m_nodes = list(exp_list[exp]["output"].keys())
        m_values = np.array(list(exp_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]

        # linearization of the ABS function: https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
        P += V[m_nodes_positions, iexp] - np.sign(m_values) <= Z[m_nodes_positions, iexp]
        P += -V[m_nodes_positions, iexp] + np.sign(m_values) <= Z[m_nodes_positions, iexp]

        P.add_objectives(sum(Z[m_nodes_positions, iexp].multiply(abs(m_values))))
    P.add_objectives(betaWeight * sum(sum(Eact + Einh)))

    P.solve(solver=solver, verbosity=verbosity)
    return P


# CARNIVAL with acyclic flow (single flow)
def runCARNIVAL_Flow_Acyclic(G, exp_list, betaWeight: float = 0.2, solver=None, verbosity=False):
    At, Ah = get_incidence_matrices_of_edges(G)
    interaction = get_interactions(G)

    VAR_FLOW = "with_flow"
    P = cn.opt.Flow(
        G,
        varname=VAR_FLOW,
        alias_flow_ipos="positive_flow",
        alias_flow_ineg="negative_flow",
        create_nonzero_indicators=True,
    )
    P = cn.opt.Acyclic(
        G,
        P,
        indicator_positive_var_name="positive_flow",
        indicator_negative_var_name="negative_flow",
    )

    # TODO: check input graph, experiment list and their compatibility

    Eact = cn.opt.Variable("edge_activates", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Einh = cn.opt.Variable("edge_inhibits", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Z = cn.opt.Variable("dummy", (G.num_vertices, len(exp_list)), vartype=cn.VarType.CONTINUOUS)
    P += Z >= 0

    # Edge cannot activate and inhibit at the same time
    P += Eact + Einh <= 1

    # The value of a vertex is the difference of the incoming edges
    Va = At @ Eact
    Vi = At @ Einh
    V = Va - Vi
    P.register("vertex_value", V)
    P.register("vertex_inhibited", Vi)
    P.register("vertex_activated", Va)
    P.register("edge_value", Eact - Einh)

    edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

    for exp, iexp in zip(exp_list, range(len(exp_list))):
        # Edge cannot activate or inhibit downstream vertices if it is not carrying flow
        P += Eact[:, iexp] + Einh[:, iexp] <= P.expr.with_flow

        P += Eact[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] > 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] < 0)  # constrain 1B
        P += Einh[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] < 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] > 0)  # constrain 2B

        # perturbation:
        p_nodes = list(exp_list[exp]["input"].keys())
        p_values = list(exp_list[exp]["input"].values())
        p_nodes_positions = [G.V.index(key) for key in p_nodes]

        P += V[p_nodes_positions, iexp] == p_values

        # measuremenents:
        m_nodes = list(exp_list[exp]["output"].keys())
        m_values = np.array(list(exp_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]

        # linearization of the ABS function: https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
        P += V[m_nodes_positions, iexp] - np.sign(m_values) <= Z[m_nodes_positions, iexp]
        P += -V[m_nodes_positions, iexp] + np.sign(m_values) <= Z[m_nodes_positions, iexp]

        P.add_objectives(sum(Z[m_nodes_positions, iexp].multiply(abs(m_values))))
    P.add_objectives(betaWeight * sum(sum(Eact + Einh)))

    P.solve(solver=solver, verbosity=verbosity)
    return P


# CARNIVAL with flow and acyclic signaling
def runCARNIVAL_Flow_Acyclic_Signal(G, exp_list, betaWeight: float = 0.2, solver=None, verbosity=False):
    At, Ah = get_incidence_matrices_of_edges(G)
    interaction = get_interactions(G)

    VAR_FLOW = "with_flow"
    P = cn.opt.Flow(G, varname=VAR_FLOW)

    # TODO: check input graph, experiment list and their compatibility

    Eact = cn.opt.Variable("edge_activates", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Einh = cn.opt.Variable("edge_inhibits", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY)
    Z = cn.opt.Variable("dummy", (G.num_vertices, len(exp_list)), vartype=cn.VarType.CONTINUOUS)
    P += Z >= 0

    # Edge cannot activate and inhibit at the same time
    P += Eact + Einh <= 1

    # The value of a vertex is the difference of the positive and negative incoming edges
    Va = At @ Eact
    Vi = At @ Einh
    V = Va - Vi
    P.register("vertex_value", V)
    P.register("vertex_inhibited", Vi)
    P.register("vertex_activated", Va)
    P.register("edge_value", Eact - Einh)
    P.register("edge_has_signal", Eact + Einh)

    # Add acyclic constraints on the edge_has_signal (signal)
    P = cn.opt.Acyclic(G, P, indicator_positive_var_name="edge_has_signal")

    edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

    for exp, iexp in zip(exp_list, range(len(exp_list))):
        # Edge cannot activate or inhibit downstream vertices if it is not carrying flow
        P += Eact[:, iexp] + Einh[:, iexp] <= P.expr.with_flow

        P += Eact[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] > 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] < 0)  # constrain 1B
        P += Einh[edges_with_head, iexp] <= (Ah.T @ Va)[edges_with_head, iexp].multiply(
            interaction[edges_with_head] < 0
        ) + (Ah.T @ Vi)[edges_with_head, iexp].multiply(interaction[edges_with_head] > 0)  # constrain 2B

        # perturbation:
        p_nodes = list(exp_list[exp]["input"].keys())
        p_values = list(exp_list[exp]["input"].values())
        p_nodes_positions = [G.V.index(key) for key in p_nodes]

        P += V[p_nodes_positions, iexp] == p_values

        # measuremenents:
        m_nodes = list(exp_list[exp]["output"].keys())
        m_values = np.array(list(exp_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]

        # linearization of the ABS function: https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
        P += V[m_nodes_positions, iexp] - np.sign(m_values) <= Z[m_nodes_positions, iexp]
        P += -V[m_nodes_positions, iexp] + np.sign(m_values) <= Z[m_nodes_positions, iexp]

        P.add_objectives(sum(Z[m_nodes_positions, iexp].multiply(abs(m_values))))
    P.add_objectives(betaWeight * sum(sum(Eact + Einh)))

    P.solve(solver=solver, verbosity=verbosity)
    return P


def milp_carnival(
    G,
    perturbations,
    measurements,
    beta_weight: float = 0.2,
    max_dist=None,
    penalize="edges",  # nodes, edges, both
    use_perturbation_weights=False,
    interaction_graph_attribute="interaction",
    disable_acyclicity=False,
    backend=cn.DEFAULT_BACKEND,
):
    """Improved port of the original Carnival R method.

    This implementation uses the CORNETO backend capabilities to create a ILP problem.
    However, it does not use the flow formulation and multi-sample capabilities of the
    novel method implemented in CORNETO. This method is kept for compatibility with the
    original Carnival R method and for comparison purposes.

    NOTE: Since the method is decoupled from specific solvers, the default pool of
    solutions generated using CPLEX is not available.

    Args:
        G: The graph object representing the network.
        perturbations: A dictionary of perturbations applied to specific vertices in the graph.
        measurements: A dictionary of measured values for specific vertices in the graph.
        beta_weight: The weight for the regularization term in the objective function.
        max_dist: The maximum distance allowed for vertex positions in the graph.
        penalize: The type of regularization to apply ('nodes', 'edges', or 'both').
        use_perturbation_weights: Whether to use perturbation weights in the objective function.
        interaction_graph_attribute: The attribute name for the interaction type in the graph.
        disable_acyclicity: Whether to disable the acyclicity constraint in the optimization.
        backend: The backend engine to use for the optimization.

    Returns:
        The optimization problem object.
    """
    max_dist = G.num_vertices if max_dist is None else max_dist

    # The problem uses 2*|V| + 2*|E| binary variables + |V| continuous variables
    V_act = backend.Variable("vertex_activated", shape=(len(G.V),), vartype=cn.VarType.BINARY)
    V_inh = backend.Variable("vertex_inhibited", shape=(len(G.V),), vartype=cn.VarType.BINARY)
    E_act = backend.Variable("edge_activating", shape=(len(G.E),), vartype=cn.VarType.BINARY)
    E_inh = backend.Variable("edge_inhibiting", shape=(len(G.E),), vartype=cn.VarType.BINARY)
    V_pos = backend.Variable(
        "vertex_position",
        shape=(len(G.V),),
        lb=0,
        ub=max_dist,
        vartype=cn.VarType.CONTINUOUS,
    )

    V_index = {v: i for i, v in enumerate(G.V)}

    P = backend.Problem()

    # A vertex can be activated or inhibited, but not both
    P += V_act + V_inh <= 1
    # An edge can activate or inhibit, but not both
    P += E_act + E_inh <= 1

    for i, (s, t) in enumerate(G.E):
        s = list(s)
        t = list(t)
        if len(s) == 0:
            continue
        if len(s) > 1:
            raise ValueError("Only one source vertex allowed")
        if len(t) > 1:
            raise ValueError("Only one target vertex allowed")
        s = s[0]
        t = t[0]
        # An edge can activate its downstream (target vertex) (E_act=1, E_inh=0),
        # inhibit it (E_act=0, E_inh=1), or do nothing (E_act=0, E_inh=0)
        si = V_index[s]
        ti = V_index[t]
        interaction = int(G.get_attr_edge(i).get(interaction_graph_attribute))
        # If edge interaction type is activatory, it can only activate the downstream
        # vertex if the source vertex is activated
        # NOTE: The 4 constraints can be merged by 2, but kept like this for clarity
        # This implements the basics of the sign consistency rules
        if interaction == 1:
            # Edge is activatory: E_act can only be 1 if V_act[source] is 1
            # edge (s->t) can activate t only if s is activated
            P += E_act[i] <= V_act[si]
            # edge (s->t) can inhibit t only if s is inhibited
            P += E_inh[i] <= V_inh[si]
        elif interaction == -1:
            # edge (s-|t) can activate t only if s is inhibited
            P += E_act[i] <= V_inh[si]
            # edge (s-|t) can inhibit t only if s is activated
            P += E_inh[i] <= V_act[si]
        else:
            raise ValueError(f"Invalid interaction value for edge {i}: {interaction}")

        # If the edge is selected, then we must respect the order of the vertices:
        # V_pos[target] - V_pos[source] >= 1
        # E.g., if a partial solution is A -> B -> C, and the ordering assigned is
        # A(0) -> B(1) -> C(2), we cannot select an edge C -> A since 2 > 0
        # The maximum numbering possible, starting with 0, cannot exceed the
        # number of vertices of the graph.
        # Note that there are two possible orderings: or target vertex is greater
        # than source (then edge can be selected), or less or equal to 0
        # (in which case the edge cannot be selected).
        # The acyclicity constraint is reduced to this simple constraint:
        # - if edge selected, then target vertex must be greater than source (diff >= 1)
        # - if edge not selected, then the diff. does not matter (we can assign any value)
        # IMPORTANT: acyclicity can be disabled, but then self activatory loops that are
        # not downstream the perturbations can appear in the solution.
        if not disable_acyclicity:
            edge_selected = E_act[i] + E_inh[i]
            P += V_pos[ti] - V_pos[si] >= 1 - max_dist * (1 - edge_selected)

    # Now compute the value of each vertex, based on the incoming selected edges
    # NOTE: Here we force that a vertex can have at most 1 incoming edge but this
    # could be relaxed (e.g. allow many inputs and integrate signal).
    for v in G.V:
        in_edges_idx = [i for i, _ in G.in_edges(v)]
        i = V_index[v]
        perturbed_value = 0
        perturbed = v in perturbations
        if perturbed:
            perturbed_value = np.sign(perturbations[v])
        in_edges_selected = [E_act[i] + E_inh[i] for i in in_edges_idx]
        if len(in_edges_idx) > 0:
            P += sum(in_edges_selected) <= 1
        # And the value of the target vertex equals the value of the selected edge
        # If no edge is selected, then the value is 0]
        incoming_activating = sum(E_act[j] for j in in_edges_idx) if len(in_edges_idx) > 0 else 0
        incoming_inhibiting = sum(E_inh[j] for j in in_edges_idx) if len(in_edges_idx) > 0 else 0
        P += V_act[i] <= int(perturbed) + incoming_activating
        P += V_inh[i] <= int(perturbed) + incoming_inhibiting
        # If perturbed but value is 0, then perturbation can take any value,
        # otherwise it must be the same as the perturbation
        if perturbed_value > 0:
            P += V_act[i] == 1
            P += V_inh[i] == 0
        elif perturbed_value < 0:
            P += V_act[i] == 0
            P += V_inh[i] == 1

    # TODO: Remove, this is not required, as L1309-L1310 already exclude these vertices
    other_potential_inputs = set(v for v in G.V if len(set(G.in_edges(v))) == 0)
    other_potential_inputs = other_potential_inputs.difference(perturbations.keys())
    for v in other_potential_inputs:
        i = V_index[v]
        P += V_act[i] == 0
        P += V_inh[i] == 0

    data = measurements.copy()
    if use_perturbation_weights:
        data.update(perturbations)

    error_terms = []
    for k, v in data.items():
        i = V_index[k]
        prediction = V_act[i] - V_inh[i]  # -1, 0, 1
        sign = np.sign(v)
        if sign > 0:
            error_terms.append(np.abs(v) * (sign - prediction))
        elif sign < 0:
            error_terms.append(np.abs(v) * (prediction - sign))
    obj = sum(error_terms)
    reg = 0
    P.add_objectives(obj)
    if beta_weight > 0:
        if penalize == "nodes":
            reg = sum(V_act) + sum(V_inh)
        elif penalize == "edges":
            reg = sum(E_act) + sum(E_inh)
        elif penalize == "both":
            # This is the default implemented in CarnivalR,
            # although regularization by edges should be preferred
            reg = sum(V_act) + sum(V_inh) + sum(E_act) + sum(E_inh)
        P.add_objectives(reg, weights=beta_weight)

    # Finally, register some aliases for convenience
    P.register("vertex_values", V_act - V_inh)
    P.register("edge_values", E_act - E_inh)
    return P
