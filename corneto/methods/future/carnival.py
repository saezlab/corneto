from typing import Any, Dict, Set, Tuple

import numpy as np

from corneto import DEFAULT_BACKEND
from corneto._constants import VarType
from corneto._graph import BaseGraph
from corneto._settings import sparsify
from corneto.methods.signal._util import (
    get_incidence_matrices_of_edges,
    get_interactions,
)
from corneto.methods.signalling.carnival import create_carnival_problem

# Define a type alias to make the annotations cleaner
# This represents the structure of your `conditions` parameter:
#
# {
#   "condition_name": {
#       "input": {
#           "vertexA": Any,
#           "vertexB": Any,
#           ...
#       },
#       "output": {
#           "vertexX": Any,
#           "vertexY": Any,
#           ...
#       }
#   },
#   ...
# }
#
ConditionsDict = Dict[str, Dict[str, Dict[str, Any]]]


def prune_graph(
        G: BaseGraph,
        conditions: ConditionsDict,
        inputs_dict_key: str = "input",
        outputs_dict_key: str = "output"
) -> Tuple[ConditionsDict, BaseGraph]:
    """Prune the given BaseGraph ``G`` according to specified conditions.

    This function performs the following steps:

    1. For each condition in ``conditions``:
       - Identifies vertices relevant to the condition (intersection of the graph's vertices
         and the condition's inputs/outputs).
       - Prunes a subgraph from ``G`` based on these relevant vertices.
       - Collects only the relevant input/output keys that remain within the pruned subgraph.
    2. Collects all pruned input/output vertices across all conditions.
    3. Finally, prunes the original graph once using all those collected vertices.

    Args:
        G (BaseGraph):
            A graph-like object with:
            - An attribute ``V`` (list or set of vertices).
            - A method ``prune(inputs, outputs)`` returning a subgraph (another ``BaseGraph``).
        conditions (ConditionsDict):
            A dictionary where each key is a condition name (e.g. "condition1"), and
            each value is a dict containing dictionaries for inputs/outputs. For example:

                {
                    "condition1": {
                        "input": {"vertexA": 1.0, "vertexB": 2.0, ...},
                        "output": {"vertexX": 3.0, "vertexY": 4.0, ...}
                    },
                    ...
                }

            The types inside the nested dicts can be anything (float, int, etc.).
        inputs_dict_key (str, optional):
            The dictionary key in each condition representing inputs.
            Defaults to ``"input"``.
        outputs_dict_key (str, optional):
            The dictionary key in each condition representing outputs.
            Defaults to ``"output"``.

    Returns:
        Tuple[ConditionsDict, BaseGraph]:
            A tuple ``(conditions_pruned, G_pruned)`` where:

            - ``conditions_pruned`` is a dictionary mirroring the structure of ``conditions``
              but containing only the vertices that remain in each pruned subgraph.
            - ``G_pruned`` is a new ``BaseGraph`` object pruned from the original ``G`` using
              all relevant input and output vertices identified across all conditions.
    """
    graph_vertices: Set[Any] = set(G.V)
    pruned_conditions: ConditionsDict = {}
    all_input_vertices: Set[Any] = set()
    all_output_vertices: Set[Any] = set()

    # Process each condition
    for cond_name, cond_data in conditions.items():
        # Convert dict keys to sets for easier intersection
        condition_inputs = set(cond_data[inputs_dict_key])
        condition_outputs = set(cond_data[outputs_dict_key])

        # Intersect with the current graph's vertices
        relevant_inputs = graph_vertices & condition_inputs
        relevant_outputs = graph_vertices & condition_outputs

        # Prune the graph based on relevant inputs and outputs
        sub_graph = G.prune(list(relevant_inputs), list(relevant_outputs))
        sub_vertices = set(sub_graph.V)

        # Gather only input/output items that remain in the pruned subgraph
        pruned_inputs = {
            i: cond_data[inputs_dict_key].get(i, 0)
            for i in sub_vertices & condition_inputs
        }
        pruned_outputs = {
            o: cond_data[outputs_dict_key].get(o, 0)
            for o in sub_vertices & condition_outputs
        }

        # Store the pruned condition
        pruned_conditions[cond_name] = {
            inputs_dict_key: pruned_inputs,
            outputs_dict_key: pruned_outputs,
        }

        # Collect all inputs/outputs for final pruning of the original graph
        all_input_vertices.update(pruned_inputs)
        all_output_vertices.update(pruned_outputs)

    # Prune the original graph with all collected inputs/outputs
    pruned_graph = G.prune(list(all_input_vertices), list(all_output_vertices))

    return pruned_conditions, pruned_graph


def create_signed_error_expression(
        P, values, index_of_vertices=None, condition_index=None, vertex_variable=None
):
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
            raise ValueError(
                "condition_index must be provided if there are more than one sample"
            )
        return (
                1
                - vertex_variable[index_of_vertices, condition_index].multiply(
            np.sign(values)
        )
        ).multiply(abs(values))
    else:
        if condition_index is not None and condition_index > 0:
            raise ValueError(
                "condition_index must be None or 0 if there is only one single sample"
            )
        return (
                1 - vertex_variable[index_of_vertices].multiply(np.sign(values))
        ).multiply(abs(values))


def create_carnival_problem(
        G,
        experiment_list,
        lambd=0.01,
        exclusive_vertex_values=True,
        upper_bound_flow=1000,
        penalty_on="signal",  # or "flow"
        backend=DEFAULT_BACKEND,
):
    """Create a CARNIVAL multi-condition optimization problem.

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
        P += backend.Indicator()

    # Create binary variables for edge activations and inhibitions
    Eact = backend.Variable(
        "edge_activates", (G.num_edges, len(experiment_list)), vartype=VarType.BINARY
    )
    Einh = backend.Variable(
        "edge_inhibits", (G.num_edges, len(experiment_list)), vartype=VarType.BINARY
    )

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
    Int = sparsify(
        np.reshape(interaction, (interaction.shape[0], 1))
        @ np.ones((1, len(experiment_list)))
    )

    # Add constraints on activations and inhibitions based on upstream signals
    sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(
        Int[edges_with_head, :] > 0
    )
    sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(
        Int[edges_with_head, :] < 0
    )
    P += Eact[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh

    sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(
        Int[edges_with_head, :] < 0
    )
    sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(
        Int[edges_with_head, :] > 0
    )
    P += Einh[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh

    all_inputs = set(
        v for exp in experiment_list for v in experiment_list[exp]["input"]
    )

    # Exclude

    # For each experiment, set constraints and add error to the objective function
    for i, exp in enumerate(experiment_list):
        m_nodes = list(experiment_list[exp]["output"].keys())
        m_values = np.array(list(experiment_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]
        p_nodes = list(experiment_list[exp]["input"].keys())
        p_values = list(experiment_list[exp]["input"].values())
        p_nodes_positions = [G.V.index(key) for key in p_nodes]
        # TODO: Don't constrain if value = 0
        P += V[p_nodes_positions, i] == p_values
        # Make sure that in each condition, only the given perturbation can be active
        # We need to take the incoming flow edges that are not part of the perturbation and block them
        if len(experiment_list) > 1:
            other_inputs = all_inputs - set(p_nodes)
            other_input_edges = [
                idx
                for v in other_inputs
                for (idx, _) in G.in_edges(v)
                if len(G.get_edge(idx)[0]) == 0
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
        P.add_objectives(sum(error))

    # Add penalty to the objective based on the chosen type (signal or flow)
    if lambd > 0:
        if penalty_on == "signal":
            P += backend.linear_or(Eact + Einh, axis=1, varname="Y")
            y = sum(P.expr.Y)
        elif penalty_on == "flow":
            y = sum(P.expr._flow_i)
        else:
            raise ValueError(
                f"Invalid penalty_on={penalty_on}. Only 'signal' or 'flow' are supported."
            )

        P.add_objectives(y, weights=lambd)
    else:
        P.add_objectives(0)

    return P
