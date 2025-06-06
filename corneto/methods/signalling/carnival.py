import time
from typing import Any, Dict, Tuple

import numpy as np

import corneto as cn
from corneto._graph import BaseGraph
from corneto._settings import sparsify
from corneto.methods import expand_graph_for_flows
from corneto.methods.signal._util import (
    get_incidence_matrices_of_edges,
    get_interactions,
)


def preprocess(pkn: BaseGraph, inputs: set, outputs: set) -> Tuple[BaseGraph, set, set, Dict[str, Any]]:
    """Preprocesses the given graph by pruning it based on inputs (perturbations) and outputs (measurements).

    Then extracts the reachable inputs and outputs from the pruned graph, while collecting various statistics.

    Args:
        pkn (BaseGraph): The original PKN to be pruned.
        inputs (set): A set with the input nodes.
        outputs (set): A set with output nodes (measurements).

    Returns:
        Tuple[BaseGraph, Dict, Dict, Dict[str, Any]]:
            - The pruned graph based on the reachable inputs and outputs.
            - A set of reachable inputs after pruning.
            - A set of reachable outputs after pruning.
            - A dictionary of various statistics collected during the preprocessing.
              Includes timing, graph shape (before/after), and removed inputs/outputs.
    """
    # Start timing
    start_time = time.time()

    # Graph shape before pruning
    initial_shape = pkn.shape

    # Compute initial inputs and outputs
    vertices = set(pkn.V)
    pkn_inputs = vertices.intersection(inputs)
    pkn_outputs = vertices.intersection(outputs)

    inputs_not_in_pkn = inputs - pkn_inputs
    outputs_not_in_pkn = outputs - pkn_outputs

    # Prune the graph
    pruned_graph = pkn.prune(list(pkn_inputs), list(pkn_outputs))

    # Graph shape after pruning
    pruned_shape = pruned_graph.shape

    pruned_pkn_vertices = set(pruned_graph.V)

    # Reachable inputs/outputs after pruning
    reachable_inputs = pkn_inputs.intersection(pruned_pkn_vertices)
    reachable_outputs = pkn_outputs.intersection(pruned_pkn_vertices)

    # Identify removed inputs/outputs
    non_contributing_inputs = pkn_inputs - reachable_inputs
    non_reachable_outputs = pkn_outputs - reachable_outputs

    # End timing
    end_time = time.time()

    # Collect statistics
    stats = {
        "initial_shape": initial_shape,
        "pruned_shape": pruned_shape,
        "inputs_not_in_pkn": inputs_not_in_pkn,
        "outputs_not_in_pkn": outputs_not_in_pkn,
        "non_contributing_inputs": non_contributing_inputs,
        "non_reachable_outputs": non_reachable_outputs,
        "preprocess_time": end_time - start_time,
    }

    return pruned_graph, reachable_inputs, reachable_outputs, stats


def create_signed_error_expression(P, values, index_of_vertices=None, condition_index=None, vertex_variable=None):
    # If variable not provided, assumes we have the expected variables in the problem
    if vertex_variable is None:
        if "vertex_value" not in P.expr:
            raise ValueError("vertex_variable must be provided if not in the problem")
        vertex_variable = P.expr.vertex_value
    if index_of_vertices is None:
        index_of_vertices = range(vertex_variable.shape[0])
    if len(index_of_vertices) != len(values):
        raise ValueError("index_of_vertices must be the same length as values")
    if len(vertex_variable.shape) > 2:
        raise ValueError("vertex_variable must be 1D or 2D")
    if len(vertex_variable.shape) == 2:
        if condition_index is None:
            raise ValueError("condition_index must be provided if there are more than one sample")
        return (1 - vertex_variable[index_of_vertices, condition_index].multiply(np.sign(values))).multiply(abs(values))
    else:
        if condition_index is not None and condition_index > 0:
            raise ValueError("condition_index must be None or 0 if there is only one single sample")
        return (1 - vertex_variable[index_of_vertices].multiply(np.sign(values))).multiply(abs(values))


def create_carnival_problem(
    G,
    experiment_list,
    lambd=0.01,
    exclusive_vertex_values=True,
    upper_bound_flow=1000,
    penalty_on="signal",  # or "flow"
    slack_regularization=False,
    backend=cn.DEFAULT_BACKEND,
):
    """Create a CARNIVAL optimization problem.

    Args:
        G (Graph): The input Prior Knowledge Network.
        experiment_list (list): List of dictionaries with input-output mappings.
        lambd (float, optional): Regularization weight for flow or signal penalty. Defaults to 0.2.
        exclusive_vertex_values (bool, optional): If false, a vertex can be both active
            and inactive through different signalling cascades. It can be used to find
            parts of network where ther is a conflict in fitting a set of TFs.
            Defaults to True.
        upper_bound_flow (int, optional): Upper bound for the flow on edges. In
            general this does not need to be changed. In case of very large networks,
            it might be necessary to increase this value. If there are numerical issues
            in smaller networks, it can be decreased e.g. to 10. This only affects at the
            scale of the optimisation variables and the solver, but it does not have
            a specific meaning in the context of the problem. Defaults to 1000.
        penalty_on (str, optional): Whether the penalty is applied on "signal" or "flow". Defaults to "signal".
        slack_regularization (bool, optional): If True, applies slack regularization on the flow. Defaults to False.
        backend (Backend, optional): Optimization backend to use. Defaults to cn.DEFAULT_BACKEND.

    Returns:
        Problem: An optimization problem that models the flow for the input graph.

    Raises:
        ValueError: If the `penalty_on` parameter is not "signal" or "flow".
    """
    # Get incidence matrices and interactions for the graph
    At, Ah = get_incidence_matrices_of_edges(G)
    interaction = get_interactions(G)

    # Initialize flow problem with upper bound constraints
    P = backend.Flow(G, ub=upper_bound_flow)
    if penalty_on == "flow":
        P += cn.Indicator()

    # Create binary variables for edge activations and inhibitions
    Eact = backend.Variable("edge_activates", (G.num_edges, len(experiment_list)), vartype=cn.VarType.BINARY)
    Einh = backend.Variable("edge_inhibits", (G.num_edges, len(experiment_list)), vartype=cn.VarType.BINARY)

    # Ensure edges cannot activate and inhibit at the same time
    P += Eact + Einh <= 1

    # Calculate vertex values based on incoming activations and inhibitions
    Va = At @ Eact
    Vi = At @ Einh
    V = Va - Vi

    if exclusive_vertex_values:
        # Ensure vertices are either active or inactive through different paths
        P += Va + Vi <= 1

    # Register variables for use in constraints and objectives
    P.register("vertex_value", V)
    P.register("vertex_inhibited", Vi)
    P.register("vertex_activated", Va)
    P.register("edge_value", Eact - Einh)
    P.register("edge_has_signal", Eact + Einh)

    # Add acyclic constraints to ensure signal does not propagate in cycles
    P = backend.Acyclic(G, P, indicator_positive_var_name="edge_has_signal")

    # Identify edges with outgoing connections (heads)
    edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

    # Extend flows across all experiments
    F = P.expr.flow.reshape((Eact.shape[0], 1)) @ np.ones((1, len(experiment_list)))

    # Ensure signal propagates only where flow exists
    P += Eact + Einh <= F

    # Sparsify the interaction matrix for computational efficiency
    Int = sparsify(np.reshape(interaction, (interaction.shape[0], 1)) @ np.ones((1, len(experiment_list))))

    # Add constraints on activations and inhibitions based on upstream signals
    sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(Int[edges_with_head, :] > 0)
    sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(Int[edges_with_head, :] < 0)
    P += Eact[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh

    sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(Int[edges_with_head, :] < 0)
    sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(Int[edges_with_head, :] > 0)
    P += Einh[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh

    # Set the perturbations for each vertex based on experimental inputs
    """
    vertex_indexes = np.array(
        [G.V.index(v) for exp in experiment_list for v in experiment_list[exp]["input"]]
    )
    perturbation_values = np.array(
        [
            val
            for exp in experiment_list
            for val in experiment_list[exp]["input"].values()
        ]
    )
    P += V[vertex_indexes, :] == perturbation_values[:, None]
    """

    all_inputs = set(v for exp in experiment_list for v in experiment_list[exp]["input"])

    # For each experiment, set constraints and add error to the objective function
    for i, exp in enumerate(experiment_list):
        m_nodes = list(experiment_list[exp]["output"].keys())
        m_values = np.array(list(experiment_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]
        p_nodes = list(experiment_list[exp]["input"].keys())
        p_values = list(experiment_list[exp]["input"].values())
        p_nodes_positions = [G.V.index(key) for key in p_nodes]

        # Get the p_nodes_positiosn for which their values are non-zero
        p_nodes_positions_nz = [p_nodes_positions[i] for i in range(len(p_nodes_positions)) if p_values[i] != 0]
        # Get also the values of the non zero
        p_values_nz = [p_values[i] for i in range(len(p_values)) if p_values[i] != 0]

        # We dont want to force if v = 0 (in this case is free to be -1, 0 or 1)
        # P += V[p_nodes_positions, i] == p_values
        if len(p_nodes_positions_nz) > 0:
            P += V[p_nodes_positions_nz, i] == p_values_nz

        # Make sure that in each condition, only the given perturbation can be active
        # We need to take the incoming flow edges that are not part of the perturbation and block them
        if len(experiment_list) > 1:
            other_inputs = all_inputs - set(p_nodes)
            other_input_edges = [
                idx for v in other_inputs for (idx, _) in G.in_edges(v) if len(G.get_edge(idx)[0]) == 0
            ]
            if len(other_input_edges) > 0:
                P += Eact[other_input_edges, i] == 0
                P += Einh[other_input_edges, i] == 0

        error = create_signed_error_expression(
            P,
            m_values,
            index_of_vertices=m_nodes_positions,
            condition_index=i,
            vertex_variable=V,
        )
        # Compute the error and add to objective
        # error = (1 - V[m_nodes_positions, i].multiply(np.sign(m_values))).multiply(
        #    abs(m_values)
        # )
        P.add_objectives(sum(error))

    # Add penalty to the objective based on the chosen type (signal or flow)
    if lambd > 0:
        if penalty_on == "signal":
            P += cn.opt.linear_or(Eact + Einh, axis=1, varname="Y")
            y = sum(P.expr.Y)
        elif penalty_on == "flow":
            y = sum(P.expr._flow_i)
        else:
            raise ValueError(f"Invalid penalty_on={penalty_on}. Only 'signal' or 'flow' are supported.")

        # Apply slack regularization if specified
        if slack_regularization:
            s = backend.Variable(lb=0, ub=G.num_edges)
            max_edges = G.num_edges - s
            P += y <= max_edges
            P.add_objectives(max_edges, weights=lambd)
        else:
            P.add_objectives(y, weights=lambd)
    else:
        P.add_objectives(0)

    return P


def get_unique(dataset, key="input"):
    values = set()
    for data in dataset.values():
        values.update(set(data[key].keys()))
    return values


def multi_carnival(
    G,
    dataset: dict,
    lambd=0.01,
    exclusive_vertex_values=True,
    upper_bound_flow=1000,
    penalty_on="signal",  # or "flow"
    slack_regularization=False,
    backend=cn.DEFAULT_BACKEND,
):
    inputs = get_unique(dataset, key="input")
    outputs = get_unique(dataset, key="output")
    G_multi, input_multi, output_multi, stats = preprocess(G, inputs, outputs)
    all_v = input_multi.union(output_multi)
    exp_list = dict()
    for k, v in dataset.items():
        filtered_in = {key: value for key, value in v["input"].items() if key in all_v}
        filtered_out = {key: value for key, value in v["output"].items() if key in all_v}
        exp_list[k] = {"input": filtered_in, "output": filtered_out}
    G_exp_e = expand_graph_for_flows(G_multi, exp_list)

    P = create_carnival_problem(
        G_exp_e,
        exp_list,
        lambd=lambd,
        slack_regularization=slack_regularization,
        upper_bound_flow=upper_bound_flow,
        exclusive_vertex_values=exclusive_vertex_values,
        penalty_on=penalty_on,
        backend=backend,
    )
    return P, G_exp_e, stats
