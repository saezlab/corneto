"""Implementation of the integrative Metabolic Analysis Tool (iMAT).

This module provides the iMAT class which extends MultiSampleFBA to incorporate
gene expression data into metabolic network analysis by maximizing the agreement between
flux distributions and gene expression measurements across samples.
"""

from typing import Optional, Tuple

import numpy as np

from corneto._data import Data, Feature
from corneto._graph import BaseGraph
from corneto.backend._base import Backend
from corneto.methods.future.fba import MultiSampleFBA
from corneto.methods.metabolism import evaluate_gpr_expression, get_genes_from_gpr


class MultiSampleIMAT(MultiSampleFBA):
    """Integrative Metabolic Analysis Tool (iMAT) implementation for multiple samples.

    This implementation extends the original iMAT method for multi-sample analysis.

    iMAT integrates gene expression data with metabolic network analysis by selecting
    flux distributions that maximize the number of highly expressed reactions carrying
    flux while minimizing the number of lowly expressed reactions carrying flux.

    Args:
        lambda_reg (float, optional): Network size regularization parameter.
            Higher values encourage fewer active reactions. Defaults to 1e-3.
        beta_reg (float, optional): Secondary regularization parameter for sparsity.
            Used when both types of regularization are needed. Defaults to 0.0.
        eps (float, optional): Tolerance for considering a flux as non-zero.
            Defaults to 1e-2.
        scale (bool, optional): Whether to scale the weights. Defaults to False.
        gpr_field (str, optional): Name of the attribute field containing GPR rules.
            Defaults to "GPR".
        high_expression_threshold (Optional[float], optional): Threshold above which genes are
            considered highly expressed. If None, no threshold is applied. Defaults to 1.0.
        low_expression_threshold (Optional[float], optional): Threshold below which genes are
            considered lowly expressed. If None, no threshold is applied. Defaults to -1.0.
        use_mean_for_missing_reactions (bool, optional): When True, use the mean score
            for reactions with no gene mappings. Defaults to False.
        use_bigm_constraints (bool, optional): When True, uses binary indicator variables with
            big-M constraints to strictly force flux to zero for lowly expressed reactions,
            instead of just constraining it to be below epsilon. Defaults to True.
        backend (Backend, optional): The optimization backend to use. Defaults to K.
    """

    def __init__(
        self,
        lambda_reg: float = 0.0,
        beta_reg: float = 0.0,
        eps: float = 1e-2,
        scale: bool = False,
        use_bigm_constraints: bool = True,
        gpr_field: str = "GPR",
        default_flux_lower_bound: Optional[float] = None,
        default_flux_upper_bound: Optional[float] = None,
        high_expression_threshold: Optional[float] = None,
        low_expression_threshold: Optional[float] = None,
        use_mean_for_missing_reactions: bool = False,
        backend: Optional[Backend] = None,
    ):
        # Use proper inheritance with MultiSampleFBA
        super().__init__(
            lambda_reg=lambda_reg,  # Use lambda_reg for structured sparsity
            beta_reg=beta_reg,
            flux_indicator_name="edge_has_flux",
            default_flow_lower_bound=default_flux_lower_bound,
            default_flow_upper_bound=default_flux_upper_bound,
            backend=backend,
        )
        self.eps = eps
        self.scale = scale
        self.gpr_field = gpr_field
        self.high_expression_threshold = high_expression_threshold
        self.low_expression_threshold = low_expression_threshold
        self.use_mean_for_missing_reactions = use_mean_for_missing_reactions
        self.use_bigm_constraints = use_bigm_constraints

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data before solving.

        This method handles:
        1. Checking if there are features with mapping="edge" (reaction features)
        2. If not, applying GPR rules to calculate reaction features from gene features (mapping="none")

        Args:
            graph (BaseGraph): The metabolic network graph to be analyzed.
            data (Data): The experimental data containing gene/reaction scores.

        Returns:
            Tuple[BaseGraph, Data]: The preprocessed graph and data.
        """
        # First apply any preprocessing from the parent class
        graph, data = super().preprocess(graph, data)

        # Check if we have reaction features (mapping="edge")
        has_reaction_features = False

        for sample_name, sample in data.samples.items():
            edge_features_count = 0

            # Count features with mapping="edge"
            for feature in sample.features:
                if feature.mapping == "edge":
                    edge_features_count += 1

            # Check if we have a significant number of edge features
            if edge_features_count > 0:
                has_reaction_features = True
                break

        # If we already have edge-mapped features, no need for further processing
        if has_reaction_features:
            return graph, data

        # Otherwise, apply GPR rules to calculate reaction features from gene features
        processed_data = self._apply_gpr_rules(graph, data)
        return graph, processed_data

    def _apply_gpr_rules(self, graph: BaseGraph, data: Data) -> Data:
        """Apply GPR rules to calculate reaction scores from gene scores.

        Args:
            graph (BaseGraph): The metabolic network graph.
            data (Data): The data containing gene expression values (mapping="none").

        Returns:
            Data: Data object with added reaction features (mapping="edge").
        """
        # Create a new data object to hold the result
        result_data = data.copy()

        # Process each sample
        for sample_name, sample in data.samples.items():
            # Extract gene scores
            gene_scores = {}
            for feature in sample.features:
                # Check if this is a gene feature (mapping="none")
                if feature.mapping == "none":
                    gene_scores[feature.id] = float(feature.value) if feature.value is not None else 0.0

            if not gene_scores:
                continue

            # Check if thresholds are provided (non-None values)
            use_thresholds = self.high_expression_threshold is not None or self.low_expression_threshold is not None

            # Process gene scores - either apply thresholds or use directly
            processed_gene_scores = {}
            if use_thresholds:
                # Apply thresholds to gene scores before GPR evaluation
                for gene, score in gene_scores.items():
                    if self.high_expression_threshold is not None and score >= self.high_expression_threshold:
                        processed_gene_scores[gene] = 1.0
                    elif self.low_expression_threshold is not None and score <= self.low_expression_threshold:
                        processed_gene_scores[gene] = -1.0
                    # Genes with expressions between thresholds are not included
            else:
                # Use gene scores directly if no thresholds are provided
                processed_gene_scores = gene_scores

            if not processed_gene_scores:
                continue

            # Process each reaction to calculate scores
            rxn_scores = {}
            for i in range(graph.ne):
                rxn_attr = graph.get_attr_edge(i)
                rxn_id = rxn_attr.get("id")

                if not rxn_id:
                    continue

                # Get GPR rule for this reaction
                gpr_rule = rxn_attr.get(self.gpr_field, "")

                if not gpr_rule:
                    continue

                # Get all genes in the GPR rule
                rule_genes = get_genes_from_gpr(gpr_rule)

                # Filter for genes we have scores for
                relevant_genes = {
                    g: processed_gene_scores.get(g, 0.0) for g in rule_genes if g in processed_gene_scores
                }

                if not relevant_genes:
                    continue

                # Calculate reaction score using GPR rule
                rxn_score = evaluate_gpr_expression([gpr_rule], relevant_genes)[0]

                rxn_scores[rxn_id] = rxn_score

            # Add reaction features to the result data
            if rxn_scores:
                for rxn_id, score in rxn_scores.items():
                    # TODO: Before adding, check if there is already
                    # a feature for that edge, if so, update the
                    # value and warn.
                    result_data.samples[sample_name].add(
                        Feature(
                            id=rxn_id,
                            value=score,
                            mapping="edge",
                        )
                    )

        return result_data

    def create_flow_based_problem(self, flow_problem, graph: BaseGraph, data: Data):
        """Create the iMAT optimization problem.

        This method sets up the iMAT-specific constraints and objectives by:
        1. First letting the parent class set up basic FBA constraints and objectives
        2. Adding iMAT-specific indicators for flux activity with custom tolerance
        3. Setting up weight-based optimization for highly/lowly expressed reactions

        Args:
            flow_problem: The optimization problem object from parent class.
            graph (BaseGraph): The metabolic network graph.
            data (Data): The experimental data containing sample information.

        Returns:
            The configured optimization problem ready to be solved.
        """
        # First let the parent class set up the basic FBA problem
        # This sets up the flow variables, flux bounds, objectives, etc.
        flow_problem = super().create_flow_based_problem(flow_problem, graph, data)

        # Now add iMAT-specific components
        # Get the flow variables created by parent class
        F = flow_problem.expr.flow

        # Add non-zero indicators for the iMAT-specific logic with custom tolerance
        # NOTE: This adds a new variable "_flow_ineg" and "_flow_ipos" to the problem
        # for all reactions, including those that don't have a score.
        flow_problem += self.backend.NonZeroIndicator(F, tolerance=self.eps)

        active = flow_problem.expr["_flow_ineg"] + flow_problem.expr["_flow_ipos"]
        # flow_problem.register(active, "reaction_active")

        if self.use_bigm_constraints:
            # flow_problem += self.backend.Indicator(F)  # I = 0 <=> F = 0
            unblocked = flow_problem.expr.edge_has_flux
            flow_problem += active <= unblocked
        else:
            unblocked = active

        # Process weights for each sample
        n_samples = len(data.samples)
        for i, sample_data in enumerate(data.samples.values()):
            weights = []
            rxn_ids = []

            # Get reaction values from the data (features with mapping="edge")
            for feature in sample_data.features:
                if feature.mapping == "edge" and feature.value is not None:
                    rxn_ids.append(feature.id)
                    weights.append(float(feature.value))

            if not rxn_ids:
                continue

            # Convert reaction IDs to indices
            rxn_indices = [next(iter(graph.get_edges_by_attr("id", rxn_id))) for rxn_id in rxn_ids]
            weights = np.array(weights)

            # Scale weights if requested
            if self.scale:
                weights = (weights / np.abs(weights).sum()) * 100

            # Split into highly and lowly expressed reactions
            idx_pos = weights > 0
            idx_neg = weights < 0

            # Get the correct active indicators for this sample
            # If we have multiple samples, extract the column for this sample
            if n_samples > 1 and len(active.shape) > 1:
                sample_active = active[:, i]
                unblocked_sample = unblocked[:, i]
            else:
                sample_active = active
                unblocked_sample = unblocked

            # Add objectives for highly expressed reactions
            if np.any(idx_pos):
                pos_weights = weights[idx_pos]
                pos_indices = np.array(rxn_indices)[idx_pos]
                flow_problem.add_objectives(pos_weights @ (1 - sample_active[pos_indices]))

            # Add objectives for lowly expressed reactions
            if np.any(idx_neg):
                neg_weights = weights[idx_neg]
                neg_indices = np.array(rxn_indices)[idx_neg]
                if self.use_bigm_constraints:
                    # 1 if the reactions is unblocked (can have positive/negative flux)
                    flow_problem.add_objectives(np.abs(neg_weights) @ unblocked_sample[neg_indices])
                else:
                    flow_problem.add_objectives(np.abs(neg_weights) @ sample_active[neg_indices])

        return flow_problem

    @staticmethod
    def references():
        return ["shlomi2008network", "rodriguez2024unified"]
