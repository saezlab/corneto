"""Multi-sample flux balance analysis."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from corneto.backend._base import Backend
from corneto.data import Data
from corneto.graph import BaseGraph

# from corneto.methods import expand_graph_for_flows
from corneto.methods._base import FlowMethod


class MultiSampleFBA(FlowMethod):
    """Flux Balance Analysis (FBA) method for multiple samples.

    This class implements Flux Balance Analysis for metabolic networks across
    multiple samples. It supports sparsity regularization and reaction objectives.

    Note:
        The graph is expected to be a genome scale metabolic network.
        It can also be imported in SBML format (XML) using the
        :func:`corneto.io.import_cobra_model` function (requires cobrapy).

    Flux Balance Analysis calculates metabolic fluxes by optimizing an objective
    such as biomass production subject to stoichiometric constraints.

    MultiSampleFBA extends traditional FBA to analyze multiple samples or
    conditions simultaneously.

    Attributes:
        lambda_reg (float): Regularization parameter to minimize the number of
            active reactions across samples (only when samples > 1).
        beta_reg (float): Additional regularization parameter for controlling sparsity
            individually for each sample.
        flux_indicator_name (str): Name of the variable used to indicate active fluxes.
        disable_structured_sparsity (bool): Whether to disable structured
            sparsity optimization.
        backend (Backend): The optimization backend to use.

    Examples:
        Basic usage with a single sample:

        >>> from corneto.io import import_miom_model
        >>> from corneto.data import Data
        >>> from corneto.methods.fba import MultiSampleFBA
        >>> # Load a metabolic model
        >>> model = import_miom_model("path/to/metabolic_model.miom")
        >>> # Create data with objective function (typically biomass)
        >>> data = Data.from_dict({
        ...     "sample1": {
        ...         "EX_biomass_e": {
        ...             "role": "objective",
        ...         },
        ...     }
        ... })
        >>> # Initialize FBA and solve
        >>> fba = MultiSampleFBA()
        >>> P = fba.build(model, data)
        >>> P.solve()
        >>> # Access the flux values
        >>> biomass_rid = next(iter(model.get_edges_by_attr("id", "EX_biomass_e")))
        >>> biomass_flux = P.expr.flow[biomass_rid].value
        >>> print(f"Biomass flux: {biomass_flux}")

        Multi-sample analysis with gene knockouts:

        >>> # Create data with two samples - control and knockout
        >>> data = Data.from_cdict({
        ...     "control": {
        ...         "EX_biomass_e": {
        ...             "role": "objective",
        ...         },
        ...     },
        ...     "knockout": {
        ...         "EX_biomass_e": {
        ...             "role": "objective",
        ...         },
        ...         "MDHm": {  # Malate dehydrogenase knockout
        ...             "lower_bound": 0,
        ...             "upper_bound": 0,
        ...         },
        ...     }
        ... })
        >>> # Initialize FBA and solve
        >>> fba = MultiSampleFBA()
        >>> P = fba.build(model, data)
        >>> P.solve()
        >>> # Compare biomass production between conditions
        >>> rid = next(iter(model.get_edges_by_attr("id", "EX_biomass_e")))
        >>> control_flux = P.expr.flow[rid, 0].value
        >>> knockout_flux = P.expr.flow[rid, 1].value
        >>> print(f"Control biomass: {control_flux}")
        >>> print(f"Knockout biomass: {knockout_flux}")
        >>> reduction = (control_flux - knockout_flux) / control_flux * 100
        >>> print(f"Growth reduction: {reduction:.2f}%")

        Sparse FBA to minimize the number of active reactions:

        >>> # Create data with biomass lower bound constraint
        >>> data = Data.from_dict({
        ...     "sample1": {
        ...         "EX_biomass_e": {
        ...             "type": "objective",
        ...             "lower_bound": 100.80,  # Enforce minimum biomass production
        ...         },
        ...     }
        ... })
        >>> # Initialize FBA with regularization for sparsity
        >>> fba = MultiSampleFBA(beta_reg=1)
        >>> P = fba.build(model, data)
        >>> P.solve()
        >>> # Count number of active reactions
        >>> n_active_reactions = np.sum(np.round(P.expr.edge_has_flux.value))
        >>> print(f"Number of active reactions: {n_active_reactions}")
    """

    def __init__(
        self,
        lambda_reg=0.0,
        beta_reg=0.0,
        flux_indicator_name: str = "edge_has_flux",
        disable_structured_sparsity: bool = False,
        default_flow_upper_bound: Optional[float] = None,
        default_flow_lower_bound: Optional[float] = None,
        backend: Optional[Backend] = None,
    ):
        """Initialize a MultiSampleFBA instance.

        Args:
            lambda_reg (float, optional): Primary sparsity regularization.
                Higher values encourage fewer active reactions. Defaults to 0.0.
            beta_reg (float, optional): Secondary regularization parameter for sparsity.
                Used when both types of regularization are needed. Defaults to 0.0.
            flux_indicator_name (str, optional): Name for the flux indicator variables.
                These track whether a reaction is active. Defaults to
                "edge_has_flux".
            disable_structured_sparsity (bool, optional): Disable structured
                sparsity optimization. Defaults to False.
            default_flow_lower_bound (float, optional): Default reaction lower
                bound when the graph does not provide one.
            default_flow_upper_bound (float, optional): Default reaction upper
                bound when the graph does not provide one.
            backend (Optional[Backend], optional): The optimization backend to use.
                If None, the default backend is used. Defaults to None.
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
        self.default_flow_upper_bound = default_flow_upper_bound
        self.default_flow_lower_bound = default_flow_lower_bound

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data before solving.

        This can be extended with flux-consistent preprocessing such as removing
        blocked reactions or dead-end metabolites.

        Args:
            graph (BaseGraph): The metabolic network graph to be analyzed.
            data (Data): The experimental data containing sample information.

        Returns:
            Tuple[BaseGraph, Data]: The preprocessed graph and data.
        """
        return graph, data

    def get_flow_bounds(self, graph: BaseGraph, data: Data) -> Dict[str, Any]:
        """Return sample-specific flux bounds for the flow formulation."""
        n_edges = graph.num_edges
        n_samples = len(data.samples)

        lb = (
            np.full(n_edges, self.default_flow_lower_bound, dtype=float)
            if self.default_flow_lower_bound is not None
            else np.asarray(graph.get_attr_from_edges("default_lb"), dtype=float)
        )

        ub = (
            np.full(n_edges, self.default_flow_upper_bound, dtype=float)
            if self.default_flow_upper_bound is not None
            else np.asarray(graph.get_attr_from_edges("default_ub"), dtype=float)
        )

        # broadcast to (n_edges, n_samples)
        lb = np.tile(lb[:, None], (1, n_samples))
        ub = np.tile(ub[:, None], (1, n_samples))

        # edge-id -> row mapping
        edge_ids = graph.get_attr_from_edges("id")
        row_of = {eid: idx for idx, eid in enumerate(edge_ids)}

        # TODO: Seems that different bounds for a Variable (cvxpy, picos)
        # is problematic? not supported?
        for col, sample in enumerate(data.samples.values()):
            for feature in sample.features:
                row = row_of.get(feature.id)
                if row is None:
                    continue
                if (lb_val := feature.data.get("lower_bound")) is not None:
                    lb[row, col] = float(lb_val)
                if (ub_val := feature.data.get("upper_bound")) is not None:
                    ub[row, col] = float(ub_val)

        # squeeze back to 1-D for single-sample cases
        if n_samples == 1:
            lb, ub = lb.squeeze(1), ub.squeeze(1)

        return {
            "lb": lb,
            "ub": ub,
            "n_flows": n_samples,
            "shared_bounds": False,
        }

    def create_flow_based_problem(self, flow_problem, graph: BaseGraph, data: Data):
        """Create the flow-based optimization problem.

        Args:
            flow_problem: The optimization problem object.
            graph (BaseGraph): The metabolic network graph.
            data (Data): The experimental data containing sample information.

        Returns:
            The configured optimization problem ready to be solved.
        """
        # The flow_problem is already created in the parent class
        F = flow_problem.expr.flow
        flow_problem += self.backend.Indicator(F, name=self.flux_indicator_name)

        for i, (sample_name, sample_data) in enumerate(data.samples.items()):
            rxn_objectives = []
            rxn_objective_ids = []
            rxn_weights = []
            sample_flux = F[:, i] if len(F.shape) > 1 else F

            # Process objective reactions
            # objs = sample_data.filter_by("type", "objective")
            objs = dict(
                sample_data.query.filter(
                    lambda f: f.data.get("role", None) == "objective"
                ).pluck(lambda f: (f.id, f.value))
            )
            for rxn_id, value in objs.items():
                rxn_obj = next(iter(graph.get_edges_by_attr("id", rxn_id)))
                value = float(value) if value is not None else -1.0
                # weight = metadata.get("weight", -1.0)
                rxn_objectives.append(rxn_obj)
                rxn_objective_ids.append(str(rxn_id))
                rxn_weights.append(value)

            # Add objectives for this sample
            if rxn_objectives:
                ids_str = "_".join(rxn_objective_ids)
                flow_problem.add_objective(
                    sample_flux[rxn_objectives].multiply(np.array(rxn_weights)).sum(),
                    name=f"objective_{sample_name}__{ids_str}",
                )

            # Add regularization term for sparsity if requested
            if self.beta_reg.value > 0:
                flow_problem.add_objective(
                    flow_problem.expr[self.flux_indicator_name].sum(),
                    weight=self.beta_reg,
                    name=f"{self.flux_indicator_name}_beta_reg_{i}",
                )

        return flow_problem

    @staticmethod
    def references():
        """Return citation keys for the method."""
        return ["savinell1992network", "rodriguez2024unified"]
