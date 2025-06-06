from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import corneto as cn
from corneto import DEFAULT_BACKEND
from corneto._constants import DEFAULT_LB, DEFAULT_UB
from corneto._graph import BaseGraph
from corneto._settings import sparsify
from corneto.backend._base import Backend
from corneto.methods.signal._util import (
    get_incidence_matrices_of_edges,
    get_interactions,
)


class GraphData:
    def __init__(
        self,
        vertex_data: Optional[Dict[str, Any]] = None,
        edge_data: Optional[Dict[int, Any]] = None,
        **kwargs,
    ):
        # Only copy input if it's a dictionary; otherwise, use an empty dictionary
        self.vertex_data = vertex_data if vertex_data is not None else {}
        self.edge_data = edge_data if edge_data is not None else {}

        # Store additional attributes from kwargs
        self.attributes = {key: value for key, value in kwargs.items()}


class FlowMethod(ABC):
    def __init__(
        self,
        graph,
        flow_lower_bound: float = DEFAULT_LB,
        flow_upper_bound: float = DEFAULT_UB,
        num_flows: int = 1,
        shared_flow_bounds: bool = False,
        backend: Optional[Backend] = None,
    ):
        if backend is None:
            backend = DEFAULT_BACKEND
        self._backend = backend
        self._base_graph = graph
        self._flow_lb = flow_lower_bound
        self._flow_ub = flow_upper_bound
        self._num_flows = num_flows
        self._shared_flow_bounds = shared_flow_bounds

    @abstractmethod
    def preprocess(self, data: List[GraphData]) -> Tuple[BaseGraph, List[GraphData]]:
        pass

    @abstractmethod
    def transform_graph(self, graph: BaseGraph, data: List[GraphData]) -> BaseGraph:
        pass

    @abstractmethod
    def elementwise_error(self, predicted, expected):
        raise NotImplementedError

    @abstractmethod
    def create_flow_based_problem(self, flow_var, graph: BaseGraph, data: List[GraphData]):
        pass

    def build(self, data: List[GraphData]):
        graph, data = self.preprocess(data)
        graph = self.transform_graph(self.graph, data)
        flow_problem = self.backend.Flow(
            graph,
            lb=self._flow_lb,
            ub=self._flow_ub,
            n_flows=self._num_flows,
            shared_bounds=self._shared_flow_bounds,
        )
        flow_problem += self.create_flow_based_problem(flow_problem, graph, data)
        # Add the error for each sample
        for i, d in enumerate(data):
            output_vertices = d.attributes.get("outputs", set())
            output_values = list(d.vertex_data.get(v, 0) for v in output_vertices)
            vertex_indexes = [graph.V.index(key) for key in output_vertices]
            # error = self.elementwise_error()

    # Getter for the graph
    @property
    def graph(self):
        return self._base_graph

    # getter for backend
    @property
    def backend(self):
        return self._backend


class Carnival(FlowMethod):
    def __init__(
        self,
        graph,
        data,
        exclusive_vertex_values=True,
        backend: Optional[Backend] = None,
    ):
        super().__init__(graph, backend=backend)
        self.data = data
        self.exclusive_vertex_values = exclusive_vertex_values

    def preprocess(self, data: List[GraphData]) -> Tuple[BaseGraph, List[GraphData]]:
        inputs = set()
        outputs = set()
        # Get all inputs and outputs for all samples
        for sample_data in data:
            inputs.update(sample_data.attributes.get("inputs", set()))
            outputs.update(sample_data.attributes.get("outputs", set()))

        if len(inputs) == 0:
            raise ValueError("No inputs provided. Add `inputs` attribute to GraphData.")
        if len(outputs) == 0:
            raise ValueError("No outputs provided. Add `outputs` attribute to GraphData.")

        # Graph shape before pruning
        initial_shape = self.graph.shape

        # Compute initial inputs and outputs
        vertices = set(self.graph.V)
        pkn_inputs = vertices.intersection(inputs)
        pkn_outputs = vertices.intersection(outputs)

        inputs_not_in_pkn = inputs - pkn_inputs
        outputs_not_in_pkn = outputs - pkn_outputs

        # Prune the graph
        pruned_graph = self.graph.prune(list(pkn_inputs), list(pkn_outputs))

        # Graph shape after pruning
        pruned_shape = pruned_graph.shape

        pruned_pkn_vertices = set(pruned_graph.V)

        # Reachable inputs/outputs after pruning
        reachable_inputs = pkn_inputs.intersection(pruned_pkn_vertices)
        reachable_outputs = pkn_outputs.intersection(pruned_pkn_vertices)

        # Identify removed inputs/outputs
        non_contributing_inputs = pkn_inputs - reachable_inputs
        non_reachable_outputs = pkn_outputs - reachable_outputs

        out_data = []

        for sample_data in data:
            out_vertex_data = {k: v for k, v in sample_data.vertex_data.items() if k in pruned_pkn_vertices}
            out_edge_data = {k: v for k, v in sample_data.edge_data.items() if k in pruned_graph.E}
            out_data.append(GraphData(out_vertex_data, out_edge_data))

        # Collect statistics
        stats = {
            "initial_shape": initial_shape,
            "pruned_shape": pruned_shape,
            "inputs_not_in_pkn": inputs_not_in_pkn,
            "outputs_not_in_pkn": outputs_not_in_pkn,
            "non_contributing_inputs": non_contributing_inputs,
            "non_reachable_outputs": non_reachable_outputs,
        }

        return pruned_graph, out_data

    def elementwise_error(self, predicted, expected):
        return (1 - predicted.multiply(np.sign(expected))).multiply(abs(expected))

    def transform_graph(self, graph: BaseGraph, data: List[GraphData]) -> BaseGraph:
        g = graph.copy()
        inputs = set()
        outputs = set()
        # Get all inputs and outputs for all samples
        for sample_data in data:
            inputs.update(sample_data.attributes.get("inputs", set()))
            outputs.update(sample_data.attributes.get("outputs", set()))

        for v in inputs:
            g.add_edge(v, ())
        for v in outputs:
            g.add_edge((), v)

        return g

    def create_flow_based_problem(self, flow_var, graph: BaseGraph, data: List[GraphData]):
        # Get incidence matrices and interactions for the graph
        P = self.backend.Problem()
        At, Ah = get_incidence_matrices_of_edges(self.graph)
        interaction = get_interactions(self.graph)

        # Create binary variables for edge activations and inhibitions
        Eact = self.backend.Variable(
            "edge_activates",
            (self.graph.num_edges, len(self.data)),
            vartype=cn.VarType.BINARY,
        )
        Einh = self.backend.Variable(
            "edge_inhibits",
            (self.graph.num_edges, len(self.data)),
            vartype=cn.VarType.BINARY,
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
        P = self.backend.Acyclic(self.graph, P, indicator_positive_var_name="edge_has_signal")

        # Identify edges with outgoing connections (heads)
        edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

        # Extend flows across all experiments
        F = flow_var.reshape((Eact.shape[0], 1)) @ np.ones((1, len(self.data)))

        # Ensure signal propagates only where flow exists
        P += Eact + Einh <= F

        # Sparsify the interaction matrix for computational efficiency
        Int = sparsify(np.reshape(interaction, (interaction.shape[0], 1)) @ np.ones((1, len(self.data))))

        # Add constraints on activations and inhibitions based on upstream signals
        sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(Int[edges_with_head, :] > 0)
        sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(Int[edges_with_head, :] < 0)
        P += Eact[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh

        sum_upstream_act = (Ah.T @ Va)[edges_with_head, :].multiply(Int[edges_with_head, :] < 0)
        sum_upstream_inh = (Ah.T @ Vi)[edges_with_head, :].multiply(Int[edges_with_head, :] > 0)
        P += Einh[edges_with_head, :] <= sum_upstream_act + sum_upstream_inh

        all_inputs = set()
        for sample_data in data:
            all_inputs.update(sample_data.attributes.get("inputs", set()))

        for i, d in enumerate(data):
            input_vertices = d.attributes.get("inputs", set())
            input_values = list(d.vertex_data.get(v, 0) for v in input_vertices)
            vertex_indexes = [graph.V.index(key) for key in input_vertices]
            for value in input_values:
                if value not in [0, 1, -1]:
                    raise ValueError("Values for inputs must be 0, 1, or -1")

            P += V[vertex_indexes, i] == input_values
            # Make sure that in each condition, only the given perturbation can be active
            # We need to take the incoming flow edges that are not part of the perturbation and block them
            if len(data) > 1:
                other_inputs = all_inputs - set(input_vertices)
                other_input_edges = [
                    idx for v in other_inputs for (idx, _) in graph.in_edges(v) if len(graph.get_edge(idx)[0]) == 0
                ]
                if len(other_input_edges) > 0:
                    P += Eact[other_input_edges, i] == 0
                    P += Einh[other_input_edges, i] == 0
        return P
