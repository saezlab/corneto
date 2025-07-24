from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np

from corneto._constants import VAR_FLOW
from corneto._data import Data
from corneto._graph import BaseGraph, EdgeType
from corneto.backend._base import Backend, ProblemDef
from corneto.methods.future.steiner import SteinerTreeFlow


class PrizeCollectingSteinerTree(SteinerTreeFlow):
    """Prize-Collecting Steiner Tree optimization method as a flow-based problem.

    This class extends the basic Steiner Tree algorithm with prize-collecting functionality.
    In a prize-collecting Steiner tree problem, terminals can have prizes (values > 0),
    making them optional terminals that provide a benefit if included in the solution.
    The class now supports providing multiple max_flow and root_vertex values per sample.
    """

    def __init__(
        self,
        include_all_terminals: bool = True,
        max_flow: Optional[Union[float, List[float]]] = None,
        default_edge_cost: float = 1.0,
        flow_name: str = VAR_FLOW,
        root_vertex: Optional[Union[Any, List[Any]]] = None,
        root_selection_strategy: Literal["first", "best"] = "first",
        epsilon: float = 1,
        strict_acyclic: bool = True,
        disable_structured_sparsity: bool = False,
        in_flow_edge_type: EdgeType = EdgeType.DIRECTED,
        out_flow_edge_type: EdgeType = EdgeType.DIRECTED,
        lambda_reg: float = 0.0,
        force_flow_through_root: bool = False,
        backend: Optional[Backend] = None,
    ):
        super().__init__(
            max_flow=max_flow,
            default_edge_cost=default_edge_cost,
            flow_name=flow_name,
            root_vertex=root_vertex,
            root_selection_strategy=root_selection_strategy,
            epsilon=epsilon,
            strict_acyclic=strict_acyclic,
            disable_structured_sparsity=disable_structured_sparsity,
            in_flow_edge_type=in_flow_edge_type,
            out_flow_edge_type=out_flow_edge_type,
            force_flow_through_root=force_flow_through_root,
            lambda_reg=lambda_reg,
            backend=backend,
        )
        self.include_all_terminals = include_all_terminals
        self.prized_flow_edges = dict()

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data for the Prize-Collecting Steiner tree optimization.

        Extends the base class preprocessing to handle prized nodes.
        For each sample, we retrieve the sample-specific selected root from the base class
        and add an out-edge for a prized node only if it is not that sample's root and
        it does not already have a flow edge.
        """
        # Process the graph normally (this sets up terminals and sample-specific roots)
        flow_graph, processed_data = super().preprocess(graph, data)

        # For each sample, add extra edges for prized nodes
        for i, sample_data in enumerate(data.samples.values()):
            sample_root = self._selected_roots[i]
            prized_nodes = sample_data.query.select(lambda f: f.mapping == "vertex" and f.value).pluck()
            for prized in prized_nodes:
                # Only add an edge if the prized node is not the sample's root
                # and it has not been already assigned a flow edge
                if prized != sample_root and prized not in self.flow_edges:
                    # TODO: If node not in the graph, raise an error, otherwise
                    # this creates an infeasible problem.
                    idx = flow_graph.add_edge(prized, (), type=self.out_flow_edge_type)
                    self.flow_edges[prized] = idx
                    self.prized_flow_edges[prized] = idx

        return flow_graph, processed_data

    def create_flow_based_problem(self, flow_problem: ProblemDef, graph: BaseGraph, data: Data):
        """Create the flow-based Prize-Collecting Steiner tree optimization problem.

        Builds on the base Steiner tree problem and adds prize-collecting functionality.
        For each sample, after setting up the base flow problem (with sample-specific max_flow
        and selected roots), we add objective terms to maximize the collected prizes.
        """
        # Create base Steiner tree problem (this uses per-sample max_flow and root values)
        flow_problem = super().create_flow_based_problem(flow_problem, graph, data)

        # Add prize-collecting functionality for each sample.
        for i, sample_data in enumerate(data.samples.values()):
            F = flow_problem.expr.flow
            F = F if len(F.shape) == 1 else F[:, i]

            # Retrieve prized terminals from the sample.
            prized_terminals = dict(
                sample_data.query.select(lambda f: f.mapping == "vertex" and f.value).pluck(lambda f: (f.id, f.value))
            )

            if prized_terminals:
                prized_idx = [
                    self.flow_edges[prized] for prized in prized_terminals.keys() if prized in self.flow_edges
                ]
                if prized_idx:
                    # Build a prize vector (order must match the prized_idx)
                    prizes = np.array(
                        [prized_terminals[prized] for prized in prized_terminals.keys() if prized in self.flow_edges]
                    )
                    if self.strict_acyclic:
                        selected = flow_problem.expr._flow_ipos + flow_problem.expr._flow_ineg
                        selected = selected if len(selected.shape) == 1 else selected[:, i]
                        selected_prized_flow_edges = selected[prized_idx]
                    else:
                        indicator_terminal_pos = self.flow_name + f"_terminal_pos_{i}"
                        indicator_terminal_neg = self.flow_name + f"_terminal_neg_{i}"
                        if (
                            indicator_terminal_pos not in flow_problem.expr
                            or indicator_terminal_neg not in flow_problem.expr
                        ):
                            flow_problem += self.backend.NonZeroIndicator(
                                flow_problem.expr.flow,
                                prized_idx,
                                i,
                                tolerance=self.epsilon,
                                suffix_pos=f"_terminal_pos_{i}",
                                suffix_neg=f"_terminal_neg_{i}",
                            )
                            selected_prized_flow_edges = (
                                flow_problem.expr[indicator_terminal_pos] + flow_problem.expr[indicator_terminal_neg]
                            )
                        else:
                            flow_edges_idxs = self._terminal_edgeflow_idx[i]
                            prized_idx_in_list = [flow_edges_idxs.index(idx) for idx in prized_idx]
                            selected = (
                                flow_problem.expr[indicator_terminal_pos] + flow_problem.expr[indicator_terminal_neg]
                            )
                            selected = selected if len(selected.shape) == 1 else selected[:, i]
                            selected_prized_flow_edges = selected[prized_idx_in_list]

                    flow_problem.register(f"selected_prized_flow_edges_{i}", selected_prized_flow_edges)

                    # Add an objective term (with negative weight) to maximize prizes.
                    flow_problem.add_objective(prizes @ selected_prized_flow_edges, weight=-1, name="prizes")

        return flow_problem
