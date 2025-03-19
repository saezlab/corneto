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
        beta_reg=0.0,
        flux_indicator_name: str = "edge_has_flux",
        disable_structured_sparsity: bool = False,
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
            backend=backend,
            lambda_reg=lambda_reg,
            reg_varname=flux_indicator_name,
            disable_structured_sparsity=disable_structured_sparsity,
            use_flow_coefficients=True,
        )
        self.flux_indicator_name = flux_indicator_name
        self.beta_reg = self.backend.Parameter(name="beta_reg_param", value=beta_reg)

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
        flow_problem += self.backend.Indicator(F, name=self.flux_indicator_name)
        for i, (sample_id, sample_data) in enumerate(data.items()):
            rxn_objectives = []
            rxn_weights = []
            lb_rxn = []
            ub_rxn = []
            sample_flux = F[:, i] if len(F.shape) > 1 else F
            objs = sample_data.filter_by("type", "objective")
            for rxn_id, metadata in objs.items():
                rxn_obj = next(iter(graph.get_edges_by_attr("id", rxn_id)))
                weight = metadata.get("weight", -1.0)
                rxn_objectives.append(rxn_obj)
                rxn_weights.append(weight)

            for rxn_id, metadata in sample_data.features.items():
                rid = next(iter(graph.get_edges_by_attr("id", rxn_id)))
                if metadata.get("lower_bound", None) is not None:
                    lb_rxn.append((rid, float(metadata["lower_bound"])))
                if metadata.get("upper_bound", None) is not None:
                    ub_rxn.append((rid, float(metadata["upper_bound"])))
            if lb_rxn:
                lb_rxn_id, lb_vals = map(list, zip(*lb_rxn))
                #print(lb_rxn_id, lb_vals)
                flow_problem += sample_flux[lb_rxn_id] >= np.array(lb_vals)
            if ub_rxn:
                ub_rxn_id, ub_vals = map(list, zip(*ub_rxn))
                #print(ub_rxn_id, ub_vals)
                flow_problem += sample_flux[ub_rxn_id] <= np.array(ub_vals)
            if rxn_objectives:
                #print(rxn_objectives, rxn_weights)
                # Add the objective for this sample
                flow_problem.add_objectives(
                    sample_flux[rxn_objectives].multiply(np.array(rxn_weights)).sum()
                )
            if self.beta_reg.value > 0:
                # Add the regularization term
                flow_problem.add_objectives(
                    flow_problem.expr[self.flux_indicator_name].sum(), weights=self.beta_reg
                )

        return flow_problem
