from typing import Any, Dict, Iterable, Optional, Set, Tuple

import numpy as np

from corneto._constants import VarType
from corneto._graph import BaseGraph
from corneto._settings import sparsify
from corneto._util import uiter
from corneto.backend._base import Backend
from corneto.data._base import Data

# from corneto.methods import expand_graph_for_flows
from corneto.methods.future.method import FlowMethod, Method
from corneto.methods.signal._util import (
    get_incidence_matrices_of_edges,
    get_interactions,
)
import corneto as cn

class MultiSampleFBA(FlowMethod):
    def __init__(
        self,
        lambda_reg=0.0,
        sparse=False,
        backend: Optional[Backend] = None,
    ):
        super().__init__(
            backend=backend, lambda_reg=lambda_reg, reg_varname="edge_has_flux"
        )
        self.sparse = sparse

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        return graph, data

    def get_flow_bounds(self, graph: BaseGraph, data: Data) -> Dict[str, Any]:
        """Get dynamic flow bounds from graph edge attributes.

        Args:
            graph: The preprocessed network graph.
            data: The preprocessed dataset.

        Returns:
            A dictionary containing flow configuration parameters.
        """
        return {
            'lb': np.array(graph.get_attr_from_edges("default_lb")),
            'ub': np.array(graph.get_attr_from_edges("default_ub")),
            'n_flows': len(data),
            'shared_bounds': False,
        }

    def create_flow_based_problem(self, flow_problem, graph: BaseGraph, data: Data):
        # Implementation goes here
        pass