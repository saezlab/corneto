"""Implementation of the integrative Metabolic Analysis Tool (iMAT).

This module provides the iMAT class which extends MultiSampleFBA to incorporate
gene expression data into metabolic network analysis by maximizing the agreement between
flux distributions and gene expression measurements across samples.
"""

from typing import Optional, Tuple

import numpy as np

from corneto.backend._base import Backend
from corneto.data import Data, Feature
from corneto.graph import BaseGraph
from corneto.methods._input_utils import (
    DEFAULT_CONDITION,
    legacy_data,
    require_mapping,
    validate_condition_keys,
    validate_numeric,
    validate_reaction_values,
)
from corneto.methods.fba import MultiSampleFBA, _fba_data
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
        scale (bool, optional): If True, normalize the nonzero iMAT score
            weights independently for each condition so their absolute values
            sum to 100. Defaults to False.
        gpr_field (str, optional): Name of the attribute field containing GPR rules.
            Defaults to "GPR".
        high_expression_threshold (Optional[float], optional): Threshold above
            which genes are highly expressed. Defaults to 1.0.
        low_expression_threshold (Optional[float], optional): Threshold below
            which genes are lowly expressed. Defaults to -1.0.
        use_mean_for_missing_reactions (bool, optional): When True, use the mean score
            for reactions with no gene mappings. Defaults to False.
        use_bigm_constraints (bool, optional): Use binary indicators and big-M
            constraints to force zero flux for lowly expressed reactions instead
            of only constraining it below epsilon. Defaults to True.
        backend (Backend, optional): The optimization backend to use.
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

    def build(
        self,
        model: BaseGraph,
        data: Optional[Data] = None,
        *,
        gene_expression=None,
        reaction_scores=None,
        objectives=None,
        reaction_bounds=None,
    ):
        """Build a single-condition iMAT problem from explicit inputs."""
        old_data = legacy_data(data, method=self.__class__.__name__)
        if old_data is not None:
            if any(value is not None for value in (gene_expression, reaction_scores, objectives, reaction_bounds)):
                raise TypeError("Do not combine a Data object with explicit scientific inputs.")
            return self.build_from_data(model, old_data)
        return self.build_many(
            model,
            gene_expression=None if gene_expression is None else {DEFAULT_CONDITION: gene_expression},
            reaction_scores=None if reaction_scores is None else {DEFAULT_CONDITION: reaction_scores},
            objectives={DEFAULT_CONDITION: objectives or {}},
            reaction_bounds={DEFAULT_CONDITION: reaction_bounds or {}},
        )

    def build_many(
        self,
        model: BaseGraph,
        *,
        gene_expression=None,
        reaction_scores=None,
        objectives=None,
        reaction_bounds=None,
    ):
        """Build a multi-condition iMAT problem from named condition mappings."""
        if (gene_expression is None) == (reaction_scores is None):
            raise ValueError("Provide exactly one of gene_expression or reaction_scores.")
        conditions = validate_condition_keys(
            gene_expression=gene_expression,
            reaction_scores=reaction_scores,
            objectives=objectives,
            reaction_bounds=reaction_bounds,
        )
        data = _fba_data(model, conditions, objectives, reaction_bounds)

        if gene_expression is not None:
            known_genes = set()
            for attributes in model.get_attr_edges():
                rule = attributes.get(self.gpr_field, "")
                if rule:
                    known_genes.update(get_genes_from_gpr(rule))
            for condition in conditions:
                values = require_mapping(gene_expression[condition], argument="gene_expression", condition=condition)
                for identifier, value in values.items():
                    if identifier not in known_genes:
                        raise ValueError(f"Unknown gene {identifier!r} in gene_expression for condition {condition!r}.")
                    score = validate_numeric(
                        value,
                        argument="gene_expression",
                        identifier=identifier,
                        condition=condition,
                    )
                    data.samples[condition].add(Feature(id=identifier, value=score, mapping="none", role="expression"))
        else:
            for condition in conditions:
                values = require_mapping(reaction_scores[condition], argument="reaction_scores", condition=condition)
                scores = validate_reaction_values(
                    model,
                    values,
                    argument="reaction_scores",
                    condition=condition,
                )
                existing = {feature.id: feature for feature in data.samples[condition].features}
                for identifier, score in scores.items():
                    if identifier in existing:
                        feature = existing[identifier]
                        if feature.data.get("role") == "objective":
                            feature.data["imat_score"] = score
                        else:
                            feature.data["value"] = score
                            feature.data["role"] = "expression"
                    else:
                        data.samples[condition].add(
                            Feature(id=identifier, value=score, mapping="edge", role="expression")
                        )
        return self.build_from_data(model, data)

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data before solving.

        This method checks whether reaction features use ``mapping="edge"``.
        If none are present, it applies GPR rules to derive reaction features
        from gene features using ``mapping="none"``.

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
                if (
                    feature.mapping == "edge"
                    and (feature.value is not None or "imat_score" in feature.data)
                    and (feature.data.get("role") != "objective" or "imat_score" in feature.data)
                ):
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
                if feature.mapping == "none" and feature.data.get("role") in {None, "expression"}:
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
                existing = {feature.id: feature for feature in result_data.samples[sample_name].features}
                for rxn_id, score in rxn_scores.items():
                    if rxn_id in existing:
                        feature = existing[rxn_id]
                        if feature.data.get("role") == "objective":
                            feature.data["imat_score"] = score
                        else:
                            feature.data["value"] = score
                            feature.data["mapping"] = "edge"
                            feature.data["role"] = "expression"
                    else:
                        result_data.samples[sample_name].add(
                            Feature(
                                id=rxn_id,
                                value=score,
                                mapping="edge",
                                role="expression",
                            )
                        )

        return result_data

    def create_flow_based_problem(self, flow_problem, graph: BaseGraph, data: Data):
        """Create the iMAT optimization problem.

        The parent class first creates the FBA constraints and objectives. This
        method then adds iMAT-specific flux-activity indicators and weight-based
        optimization for highly and lowly expressed reactions.

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

        # Keep the full-size edge_has_flux indicator from the parent class for
        # structured regularization across samples. iMAT-specific nonzero
        # support vars are added only for scored reactions per sample below.
        unblocked = flow_problem.expr.edge_has_flux if self.use_bigm_constraints else None

        # Process weights for each sample
        n_samples = len(data.samples)
        for i, (sample_name, sample_data) in enumerate(data.samples.items()):
            weights = []
            rxn_ids = []

            # Get reaction values from the data (features with mapping="edge")
            for feature in sample_data.features:
                if (
                    feature.mapping == "edge"
                    and (feature.value is not None or "imat_score" in feature.data)
                    and (feature.data.get("role") != "objective" or "imat_score" in feature.data)
                ):
                    rxn_ids.append(feature.id)
                    weights.append(float(feature.data.get("imat_score", feature.value)))

            if not rxn_ids:
                continue

            # Convert reaction IDs to indices
            rxn_indices = np.array([next(iter(graph.get_edges_by_attr("id", rxn_id))) for rxn_id in rxn_ids])
            weights = np.array(weights, dtype=float)

            # Scale weights if requested
            if self.scale:
                denom = np.abs(weights).sum()
                if denom > 0:
                    weights = (weights / denom) * 100

            # Only non-zero scored reactions contribute to iMAT support vars/objectives.
            nonzero_mask = ~np.isclose(weights, 0.0)
            if not np.any(nonzero_mask):
                continue

            scored_indices = rxn_indices[nonzero_mask]
            scored_weights = weights[nonzero_mask]

            # Add sparse support vars only for this sample's scored reactions.
            suffix_pos = f"_ipos_s{i}"
            suffix_neg = f"_ineg_s{i}"
            if n_samples > 1 and len(F.shape) > 1:
                indicator_indexes = (scored_indices, i)
            else:
                indicator_indexes = scored_indices

            flow_problem += self.backend.NonZeroIndicator(
                F,
                indexes=indicator_indexes,
                tolerance=self.eps,
                suffix_pos=suffix_pos,
                suffix_neg=suffix_neg,
            )

            sample_active = flow_problem.expr[f"{F.name}{suffix_neg}"] + flow_problem.expr[f"{F.name}{suffix_pos}"]

            # Split into highly and lowly expressed reactions
            idx_pos = np.where(scored_weights > 0)[0]
            idx_neg = np.where(scored_weights < 0)[0]
            if self.use_bigm_constraints:
                if n_samples > 1 and len(unblocked.shape) > 1:
                    unblocked_sample = unblocked[scored_indices, i]
                else:
                    unblocked_sample = unblocked[scored_indices]
                flow_problem += sample_active <= unblocked_sample
            else:
                unblocked_sample = sample_active

            # Add objectives for highly expressed reactions
            sample_name_str = str(sample_name).replace(" ", "_")

            if len(idx_pos) > 0:
                pos_weights = scored_weights[idx_pos]
                flow_problem.add_objective(
                    pos_weights @ (1 - sample_active[idx_pos]),
                    name=f"imat_fit_pos_{sample_name_str}_{i}",
                )

            # Add objectives for lowly expressed reactions
            if len(idx_neg) > 0:
                neg_weights = scored_weights[idx_neg]
                if self.use_bigm_constraints:
                    # 1 if the reactions is unblocked (can have positive/negative flux)
                    flow_problem.add_objective(
                        np.abs(neg_weights) @ unblocked_sample[idx_neg],
                        name=f"imat_fit_neg_{sample_name_str}_{i}",
                    )
                else:
                    flow_problem.add_objective(
                        np.abs(neg_weights) @ sample_active[idx_neg],
                        name=f"imat_fit_neg_{sample_name_str}_{i}",
                    )

        return flow_problem

    @staticmethod
    def references():
        """Return citation keys for the method."""
        return ["shlomi2008network", "rodriguez2025unifying"]
