from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from corneto._constants import VarType
from corneto._graph import BaseGraph
from corneto._settings import sparsify
from corneto.backend._base import Backend
from corneto.methods import expand_graph_for_flows
from corneto.methods.future.method import Dataset, FlowMethod
from corneto.methods.signal._util import (
    get_incidence_matrices_of_edges,
    get_interactions,
)

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

def pivoted_to_standard(pivoted_dict, metadata_key):
    """Converts a 'pivoted' dict of the form:
        {
          condition: {
            meta_value: {
               feature_name: feature_value
            },
            ...
          },
          ...
        }
    into a 'standard' dict of the form:
        {
          condition: {
            feature_name: {
              "value": feature_value,
              <metadata_key>: meta_value
            },
            ...
          },
          ...
        }

    Args:
        pivoted_dict (dict): The pivoted dictionary.
        metadata_key (str): The name of the metadata field to inject (e.g. "type").

    Returns:
        dict: The converted dictionary in standard format.
    """
    standard_dict = {}

    for condition, meta_groups in pivoted_dict.items():
        standard_dict[condition] = {}
        for meta_val, features in meta_groups.items():
            for feature_name, feature_value in features.items():
                standard_dict[condition][feature_name] = {
                    "value": feature_value,
                    metadata_key: meta_val
                }
    return standard_dict


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


class Carnival(FlowMethod):
    def __init__(
        self,
        exclusive_vertex_values=True,
        lambd=0.0,
        backend: Optional[Backend] = None,
    ):
        super().__init__(backend=backend, lambd=lambd, reg_varname="edge_has_signal")
        self.exclusive_vertex_values = exclusive_vertex_values

    def preprocess(self, graph: BaseGraph, data: Dataset) -> Tuple[BaseGraph, Dataset]:
        data, graph = prune_graph(graph, data.to_dict(key="type", return_value_only=True))
        graph = expand_graph_for_flows(graph, data)
        return graph, Dataset.from_dict(pivoted_to_standard(data, "type"))

    def create_flow_based_problem(self, flow_problem, graph: BaseGraph, data: Dataset):
        data = data.to_dict(key="type", return_value_only=True)
        P = flow_problem
        At, Ah = get_incidence_matrices_of_edges(graph)
        interaction = get_interactions(graph)

        #if self.penalty_on == "flow":
        #    P += self.backend.Indicator()

        # Create binary variables for edge activations and inhibitions
        Eact = self.backend.Variable(
            "edge_activates", (graph.num_edges, len(data)), vartype=VarType.BINARY
        )
        Einh = self.backend.Variable(
            "edge_inhibits", (graph.num_edges, len(data)), vartype=VarType.BINARY
        )

        # Ensure edges cannot activate and inhibit at the same time
        P += Eact + Einh <= 1

        # Calculate vertex values based on incoming activations and inhibitions
        Va = At @ Eact
        Vi = At @ Einh
        V = Va - Vi

        if self.exclusive_vertex_values:
            # Ensure vertices are either active or inactive through different paths
            P += Va + Vi <= 1

        # Register variables for use in constraints and objectives
        P.register("vertex_value", V)
        P.register("vertex_inhibited", Vi)
        P.register("vertex_activated", Va)
        P.register("edge_value", Eact - Einh)
        P.register("edge_has_signal", Eact + Einh)

        # Add acyclic constraints to ensure signal does not propagate in cycles
        P = self.backend.Acyclic(graph, P, indicator_positive_var_name="edge_has_signal")

        # Identify edges with outgoing connections (heads)
        edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

        # Extend flows across all experiments
        F = P.expr.flow.reshape((Eact.shape[0], 1)) @ np.ones((1, len(data)))

        # Ensure signal propagates only where flow exists
        P += Eact + Einh <= F

        # Sparsify the interaction matrix for computational efficiency
        Int = sparsify(
            np.reshape(interaction, (interaction.shape[0], 1))
            @ np.ones((1, len(data)))
        )
        # Add constraints on activations and inhibitions based on upstream signals
        sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(
            (Int[edges_with_head, :] > 0).astype(int)
        )
        sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(
            (Int[edges_with_head, :] < 0).astype(int)
        )
        P += Eact[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh

        sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(
            (Int[edges_with_head, :] < 0).astype(int)
        )
        sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(
            (Int[edges_with_head, :] > 0).astype(int)
        )
        P += Einh[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh

        all_inputs = set(
            v for exp in data for v in data[exp]["input"]
        )

        # Restrict which inputs can be used in different conditions.
        # Make sure that in each condition, only the given perturbation can be active
        # We need to take the incoming flow edges that are not part of the
        # perturbation and block them
        if len(data) > 1:
            for i, exp in enumerate(data):
                p_nodes = list(data[exp]["input"].keys())
                other_inputs = all_inputs - set(p_nodes)
                other_input_edges = [
                    idx
                    for v in other_inputs
                    for (idx, _) in graph.in_edges(v)
                    if len(graph.get_edge(idx)[0]) == 0
                ]
                if len(other_input_edges) > 0:
                    P += Eact[other_input_edges, i] == 0
                    P += Einh[other_input_edges, i] == 0

        # For each experiment, set constraints and add error to the objective function
        for i, exp in enumerate(data):
            p_nodes = list(data[exp]["input"].keys())
            p_values = list(data[exp]["input"].values())
            p_nodes_positions = [graph.V.index(key) for key in p_nodes]

            p_nodes_nz_idx = []
            p_nodes_values = []
            for p_node_idx, p_node_val in zip(p_nodes_positions, p_values):
                if p_node_val != 0:
                    p_nodes_nz_idx.append(p_node_idx)
                    p_nodes_values.append(np.sign(p_node_val))
            if len(p_nodes_nz_idx) > 0:
                P += V[np.array(p_nodes_nz_idx), i] == np.array(p_nodes_values)
        # Add error terms
        for i, exp in enumerate(data):
            m_nodes = list(data[exp]["output"].keys())
            m_values = np.array(list(data[exp]["output"].values()))
            m_nodes_positions = [graph.V.index(key) for key in m_nodes]
            # Inconsistency between PICOS norm and CVXPY, also CVXPY supports axis, whereas PICOS does not.
            # We need to compute norm per sample.
            if len(data) > 1:
                val = V[m_nodes_positions, i]
            else:
                val = V[m_nodes_positions]
            err =  (val - np.sign(m_values)).multiply(np.abs(m_values)).norm(p=1)
            P.add_objectives(err)

        return P


