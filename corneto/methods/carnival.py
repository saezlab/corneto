from corneto.methods.signaling import signflow, create_flow_graph
from corneto._graph import Graph
from typing import Dict, List, Tuple, Union
from corneto._settings import LOGGER
import time


def info(s, show=True):
    if show:
        LOGGER.info(s)


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
    info(f"{len(c_inputs)}/{len(inputs)} inputs mapped to the graph", show=verbose)
    info(f"{len(c_outputs)}/{len(outputs)} outputs mapped to the graph", show=verbose)
    info(f"Pruning the graph with size: V x E = {G.shape}...", show=verbose)
    Gp = G.prune(list(c_inputs), list(c_outputs))
    info(f"Finished. Final size: V x E = {Gp.shape}.", show=verbose)
    V = set(Gp.vertices)
    cp_inputs = {input: v for input, v in perturbations.items() if input in V}
    cp_outputs = {output: v for output, v in measurements.items() if output in V}
    info(f"{len(cp_inputs)}/{len(c_inputs)} inputs after pruning.", show=verbose)
    info(f"{len(cp_outputs)}/{len(c_outputs)} outputs after pruning.", show=verbose)
    info("Converting into a flow graph...", show=verbose)
    Gf = create_flow_graph(Gp, conditions)
    info("Creating a network flow problem...", show=verbose)
    P = signflow(Gf, conditions, l0_penalty_vertices=betaWeight, **kwargs)
    info("Preprocess completed.", show=verbose)
    if solve:
        P.solve(solver=solver, **backend_options)
    end = time.time() - start
    info(f"Finished in {end:.2f} s.", show=verbose)
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
    return runVanillaCarnival(
        perturbations, measurements, G, betaWeight=betaWeight, solver=solver, **kwargs
    )


def heuristic_carnival(
    priorKnowledgeNetwork: Union[List[Tuple], Graph],
    perturbations: Dict,
    measurements: Dict,
    full_bfs: bool = False,
    prune: bool = True,
    verbose=True,
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
    V = set(Gp.V)
    inputs = V.intersection(perts)
    outputs = V.intersection(meas)
    # Clean unreachable inputs/outputs
    inputs_p = {k: perturbations[k] for k in inputs}
    outputs_p = {k: measurements[k] for k in outputs}
    selected_edges = None
    if not full_bfs:
        selected_edges = reachability_graph(Gp, inputs_p, outputs_p, verbose=verbose)
    selected_edges, paths, _ = bfs_search(
        Gp, inputs_p, outputs_p, subset_edges=selected_edges
    )
    return Gp, selected_edges, paths


def get_result(P, G, condition="c0"):
    V = P.expr["vertex_values_" + condition].value
    E = P.expr["edge_values_" + condition].value
    return {'V': G.V, 'value': V}, {'E': G.E, 'value': E}


def get_selected_edges(P, G, condition="c0"):
    # Get the indexes of the edges whose value is not zero
    E = P.expr["edge_values_" + condition].value
    selected_edges = []
    for i, v in enumerate(E):
        if v != 0:
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
                    str_reached = (
                        "/".join(list(reached_outputs)[:max_printed_outputs]) + "..."
                    )
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
        print(f"Finished ({len(selected_edges)} selected edges).")
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
    G, initial_dict, final_dict, queue_max_size=None, subset_edges=None, verbose=True
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
            new_state = (nt, nv)
            # Check if the vertex is in the goal set
            if nt not in reached:
                vf = final_dict.get(nt, None)
                if vf is not None and vf == value:
                    valid, p_a, p_b = _path_conflict(new_state, paths)
                    if verbose:
                        print(" >", _str_state(new_state))
                    if not valid:
                        print(
                            "   ! conflict: {} != {}".format(
                                _str_path_nodes(p_a), _str_path_nodes(p_b)
                            )
                        )
                        stats["conflicts"] += 1
                        continue
                    reached |= {nt}
                    paths.append(new_state)
                    # Add edges
                    selected_edges |= set(
                        edge_idx
                        for (_, _, edge_idx) in nv.values()
                        if edge_idx is not None
                    )

            if len(reached) >= len(final_dict):
                exit = True
                break
            # No loop, add new state
            Q.append(new_state)
            if len(Q) > maxq:
                maxq = len(Q)
            if (
                queue_max_size is not None
                and queue_max_size > 0
                and len(Q) > queue_max_size
            ):
                break
        stats["iters"] += 1
    if verbose:
        print(f"Finished ({time.time() - start:.2f} s)")
        print(f" > Total iterations: {stats['iters']}")
        print(f" > Detected loops: {stats['loops']}")
        print(f" > Conflicts: {stats['conflicts']}")
    return selected_edges, paths, stats
