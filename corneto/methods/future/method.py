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
    def create_flow_based_problem(
        self, flow_var, graph: BaseGraph, data: List[GraphData]
    ):
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

