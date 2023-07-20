from corneto.methods.signflow import signflow, create_flow_graph, Graph
from typing import Dict, List, Tuple, Union
from corneto import K
import numpy as np
import warnings

def _check_graphviz():
    try:
        import graphviz
    except Exception as e:
        return ImportError("Graphviz not installed, but required for plotting. Please install it using conda: 'conda install python-graphviz.'", str(e))   
    return graphviz

def runVanillaCarnival(
    perturbations: Dict,
    measurements: Dict,
    priorKnowledgeNetwork: Union[List[Tuple], Graph],
    betaWeight: float = 0.2,
    solver=None,
    **kwargs
):
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
    Gf = create_flow_graph(G, conditions)
    P = signflow(Gf, conditions, l0_penalty_vertices=betaWeight)
    P.solve(solver=solver, **kwargs)
    return P, Gf


def runInverseCarnival(
    measurements: Dict,
    priorKnowledgeNetwork: Union[List[Tuple], Graph],
    betaWeight: float = 0.2,
    solver=None,
    **kwargs
):
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


def visualize_network(df, clean=True, node_attr=None):
    graphviz = _check_graphviz()
    if node_attr is None:
        node_attr = dict(fixedsize="true")
    g = graphviz.Digraph(node_attr=node_attr)
    for i, row in df.iterrows():
        # Create node source and target. If clean, ignore dummy nodes
        if clean:
            if len(row.source) == 0 or row.source.startswith('_'):
                continue
            if len(row.target) == 0 or row.target.startswith('_'):
                continue
            if row.source_type != 'input_ligand' and (row.source_pred_val == 0 or row.target_pred_val == 0):
                continue
        edge_props = dict(arrowhead='normal')
        edge_interaction = int(row.edge_type)
        if edge_interaction != 0:
            edge_props['penwidth'] = '2'
        if edge_interaction < 0:
            edge_props['arrowhead'] = 'tee'
        if int(row.edge_pred_val) < 0:
            edge_props['color'] = 'blue'
        if int(row.edge_pred_val) > 0:
            edge_props['color'] = 'red'
        g.node(row.source, **_get_node_props('source', row))
        g.node(row.target, **_get_node_props('target', row))
        g.edge(row.source, row.target, **edge_props)
    return g



def _get_node_props(prefix, row):
    props = dict(shape='circle')
    name = prefix
    pred_val = prefix + '_pred_val'
    node_type = prefix + '_type'

    if len(row[name]) == 0:
        props['shape'] = 'point'
    if row[pred_val] != 0:
        props['penwidth'] = '2'
        props['style'] = 'filled'
    if row[pred_val] > 0:
        props['color'] = 'red'
        props['fillcolor'] = 'lightcoral'
    if row[pred_val] < 0:
        props['color'] = 'blue'
        props['fillcolor'] = 'azure2'
    if row[node_type] == 'input':
        props['shape'] = 'invtriangle'
    if row[node_type] == 'input_ligand':
        props['shape'] = 'plaintext'
    if row[node_type] == 'output':
        props['shape'] = 'square'
    return props


def _select_solver():
    priority = ['gurobi', 'cplex', 'copt', 'mosek', 'scipy', 'scip', 'cbc', 'glpk_mi', None]
    solvers = [s.lower() for s in K.available_solvers()]
    solver = None
    for i, solver in enumerate(priority):
        if solver is None:
            raise ValueError(f"No valid MIP solver installed, solvers detected: {solvers}")
        if solver in solvers:
            solver = solver.upper()
            if i > 3:
                warnings.warn(f"Note: {solver} has been selected as the default solver. However, for optimal performance, it's recommended to use one of the following solvers: GUROBI, CPLEX, COPT, or MOSEK.")
            break
    return solver


def _print(*args, **kwargs):
    # TODO: To be replaced by a logger
    v = kwargs.get('v', None)
    if v is not None:
        kwargs.pop('v')
        print(*args, **kwargs)

def _get_scores(d):
    return (
       [v for v in d.values() if v < 0],
       [v for v in d.values() if v > 0]
    )

def _extended_carnival_problem(G,
                            input_node_scores,
                            output_node_scores,
                            node_penalties=None,
                            edge_penalty=1e-2):
    data = dict()
    V = set(G.vertices)
    for k, v in input_node_scores.items():
        if k in V:
            data[k] = ("P", v)
    for k, v in output_node_scores.items():
        if k in V:
            data[k] = ("M", v)
    conditions = {"c0": data}
    Gf = create_flow_graph(G, conditions)
    Pc = signflow(Gf, conditions,
                  l0_penalty_edges=edge_penalty,
                  flow_implies_signal=False)
   
    if node_penalties is not None:
        selected_nodes = Pc.symbols['species_inhibited_c0'] + Pc.symbols['species_activated_c0']
        node_penalty = np.array([node_penalties.get(n, 0.0) for n in Gf.vertices])
        Pc.add_objectives(node_penalty @ selected_nodes)
    
    return Pc, Gf


def search_causalnet(
        prior_graph,
        input_node_scores,
        output_node_scores,
        node_weights=None,
        node_cutoff=0.1,
        min_penalty=0.01, 
        max_penalty=1.0,
        missing_penalty=10,
        edge_penalty=1e-2,
        solver=None,
        verbose=True,
        show_solver_output=False,
        max_seconds=None,
        **kwargs
    ):
    v = verbose

    if solver is None:
        solver = _select_solver()

    measured_nodes = set(input_node_scores.keys()) | set(output_node_scores.keys())
 
    _print("Total positive/negative scores of the inputs and outputs:", v=verbose)
    w_neg_in, w_pos_in = _get_scores(input_node_scores)
    w_neg_out, w_pos_out = _get_scores(output_node_scores)
    _print(f" - (-) input nodes: {sum(w_neg_in)}", v=v)
    _print(f" - (+) input nodes: {sum(w_pos_in)}", v=v)
    _print(f" - (-) output nodes: {sum(w_neg_out)}", v=v)
    _print(f" - (+) output nodes: {sum(w_pos_out)}", v=v)
    
    # Total weights
    total = abs(sum(w_neg_in)) + abs(sum(w_neg_out)) + sum(w_pos_in) + sum(w_pos_out)
    _print(f" - abs total (inputs + outputs): {total}", v=v)
    
    if node_weights is None:
        node_weights = {}
    else:
        node_penalties = _weights_to_penalties(node_weights, cutoff=node_cutoff,
                                               max_penalty=max_penalty,
                                               min_penalty=min_penalty)

    # assign 0 penalties to input/output nodes, missing_penalty to missing nodes
    c_node_penalties = {k: node_penalties.get(k, missing_penalty) if k not in measured_nodes else 0.0 for k in prior_graph.vertices}

    _print("Building CORNETO problem...", v=v)
    P, G = _extended_carnival_problem(
        prior_graph,
        input_node_scores,
        output_node_scores,
        node_penalties=c_node_penalties,
        edge_penalty=edge_penalty
    )

    _print(f"Solving with {solver}...", v=v)
    ps = P.solve(
        solver=solver, 
        max_seconds=max_seconds, 
        verbosity=int(show_solver_output),
        scipy_options=dict(disp='true'), 
        **kwargs)
    
    _print("Done.", v=verbose)
    
    obj_names = ["Loss (unfitted inputs/output)", "Edge penalty error", "Node penalty error"]
    _print("Solution summary:", v=v)
    for s, o in zip(obj_names, P.objectives):
        _print(f" - {s}: {o.value}", v=v)
    df,cols = _export_results(P, G, input_node_scores, output_node_scores)
    return df, cols, P


def _weights_to_penalties(props, 
                          cutoff,
                          min_penalty, 
                          max_penalty):
    if any(p < 0 or p > 1 for p in props.values()):
        raise ValueError("Node weights were not between 0 and 1. Consider minmax or another normalization.")
    
    return {k: max_penalty if v < cutoff else min_penalty for k, v in props.items()}


def _export_results(P, G, input_dict, output_dict):
    # Get results
    nodes = P.symbols['species_activated_c0'].value - P.symbols['species_inhibited_c0'].value
    edges = P.symbols['reaction_sends_activation_c0'].value - P.symbols['reaction_sends_inhibition_c0'].value

    # For all edges
    E = G.edges
    V = {v: i for i, v in enumerate(G.vertices)}

    df_rows = []
    for i, e in enumerate(E):
        s, t = e
        if len(s) > 0:
            s = list(s)[0]
        else:
            s = ''
        if len(t) > 0:
            t = list(t)[0]
        else:
            t = ''
        if abs(edges[i]) > 0.5:
            # Get value of source/target
            edge_type = G.edge_properties[i].get('interaction', 0)
            s_val = nodes[V[s]] if s in V else 0
            t_val = nodes[V[t]] if t in V else 0
            s_type = 'unmeasured'
            s_weight = 0
            t_type = 'unmeasured'
            t_weight = 0
            if s in input_dict:
                s_type = 'input'
                s_weight = input_dict.get(s)
            if t in output_dict:
                t_type = 'output'
                t_weight = output_dict.get(t)

            df_rows.append([s, s_type, s_weight, s_val, t, t_type, t_weight, t_val, edge_type, edges[i]])
    
    columns = ['source', 'source_type',
               'source_weight', 'source_pred_val',
               'target', 'target_type',
               'target_weight', 'target_pred_val',
               'edge_type', 'edge_pred_val']
    
    return df_rows, columns