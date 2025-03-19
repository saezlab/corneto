from typing import Any, Dict, Optional, Tuple

import numpy as np

import corneto as cn
from corneto._graph import BaseGraph
from corneto.backend._base import Backend
from corneto.data._base import Data

# from corneto.methods import expand_graph_for_flows
from corneto.methods.future.method import FlowMethod


class MultiSampleFBA(FlowMethod):
    """Flux Balance Analysis (FBA) method for multiple samples.

    This class implements Flux Balance Analysis for metabolic networks across multiple samples.
    It allows for regularization to minimize the number of active reactions and
    optimization based on reaction objectives.

    Attributes:
        lambda_reg (float): Regularization parameter to minimize the number of active reactions.
        rxn_obj (Optional[Dict[str, Optional[float]]]): Dictionary mapping reaction IDs
            to their minimum flux values or None.
        backend (Optional[Backend]): The optimization backend to use.
    """

    def __init__(
        self,
        lambda_reg=0.0,
        rxn_obj: Optional[Dict[str, Optional[float]]] = None,
        backend: Optional[Backend] = None,
    ):
        """Initialize a MultiSampleFBA instance.

        Args:
            lambda_reg (float, optional): Regularization parameter. Defaults to 0.0.
            rxn_obj (Optional[Dict[str, Optional[float]]], optional): Reaction objectives as
                a dictionary mapping reaction IDs to minimum flux values. Defaults to None.
            backend (Optional[Backend], optional): The optimization backend. Defaults to None.
        """
        super().__init__(
            backend=backend, lambda_reg=lambda_reg, reg_varname="edge_has_flux"
        )
        self.rxn_obj = rxn_obj

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data before solving.

        This method can be extended to implement flux-consistent preprocessing techniques.

        Args:
            graph (BaseGraph): The metabolic network graph.
            data (Data): The experimental data.

        Returns:
            Tuple[BaseGraph, Data]: The preprocessed graph and data.
        """
        # No preprocess needed, although flux-consistent techniques could be implemented
        return graph, data

    def get_flow_bounds(self, graph: BaseGraph, data: Data) -> Dict[str, Any]:
        """Get the flow bounds for the optimization problem.

        Args:
            graph (BaseGraph): The metabolic network graph.
            data (Data): The experimental data.

        Returns:
            Dict[str, Any]: Dictionary containing flow bounds information including
                lower bounds, upper bounds, number of flows, and whether bounds are shared.
        """
        return {
            "lb": np.array(graph.get_attr_from_edges("default_lb")),
            "ub": np.array(graph.get_attr_from_edges("default_ub")),
            "n_flows": len(data),
            "shared_bounds": False,
        }

    def create_flow_based_problem(self, flow_problem, graph: BaseGraph, data: Data):
        """Create the flow-based optimization problem.

        This method sets up the constraints and objectives for the FBA optimization problem.

        Args:
            flow_problem: The optimization problem object.
            graph (BaseGraph): The metabolic network graph.
            data (Data): The experimental data.

        Returns:
            The configured optimization problem.
        """
        # The flow_problem is already created in the parent class
        F = flow_problem.expr.flow
        if self.lambda_reg_param.value > 0:
            # Indicator creates a new binary variable vector (_flow_i)
            # where each position of the vector indicates if the flux
            # of the reaction is unblocked or not. Minimising this
            # vector is eq. to blocking (removing) as many reactions as
            # possiblE
            flow_problem += cn.opt.Indicator(F)
            flow_problem.add_objectives(sum(flow_problem.expr._flow_i))
        if self.rxn_obj is not None:
            for rxn_id, min_flux in self.rxn_obj.items():
                rxn_obj = graph.get_edges_by_attr("id", rxn_id)
            if min_flux is not None:
                flow_problem += F[rxn_obj] >= min_flux
            flow_problem.add_objectives(F[rxn_obj], weights=-1)
        return flow_problem
