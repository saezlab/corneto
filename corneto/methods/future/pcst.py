from typing import Any, Literal, Optional, Tuple

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

    The algorithm balances the cost of edges against the benefit of including prize nodes.

    In a Prize-Collecting Steiner Tree problem, given:
    - A graph G with edge weights (costs)
    - A subset of vertices called terminals (required)
    - A subset of vertices with prizes (optional)
    - Optionally, a root node

    The goal is to find a minimum-weight connected subgraph that contains all required terminals
    and maximizes the collection of prizes from optional terminals.
    """

    def __init__(
        self,
        include_all_terminals: bool = True,
        max_flow: Optional[float] = None,
        default_edge_cost: float = 1.0,
        flow_name: str = VAR_FLOW,
        root_vertex: Optional[Any] = None,
        root_selection_strategy: Literal["first", "best"] = "first",
        epsilon: float = 1,
        strict_acyclic: bool = True,
        disable_structured_sparsity: bool = False,
        in_flow_edge_type: EdgeType = EdgeType.DIRECTED,
        out_flow_edge_type: EdgeType = EdgeType.DIRECTED,
        lambda_reg: float = 0.0,
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
            lambda_reg=lambda_reg,
            backend=backend,
        )
        self.include_all_terminals = include_all_terminals
        self.prized_flow_edges = dict()

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data for the Prize-Collecting Steiner tree optimization.

        Extends the base class preprocessing to handle both terminals and prized nodes.

        Args:
            graph (BaseGraph): The input graph.
            data (Data): The data containing terminal and prized nodes.

        Returns:
            Tuple[BaseGraph, Data]: Preprocessed graph and data.
        """
        # First process the graph normally to set up the terminals
        flow_graph, processed_data = super().preprocess(graph, data)

        # Then add edges for prized terminals if they exist
        for sample_data in data.samples.values():
            prized_nodes = sample_data.query.select(
                lambda f: f.mapping == "vertex" and f.value
            ).pluck()

            for prized in prized_nodes:
                if prized != self.root_vertex and prized not in self.flow_edges:
                    idx = flow_graph.add_edge(prized, (), type=self.out_flow_edge_type)
                    self.flow_edges[prized] = idx
                    self.prized_flow_edges[prized] = idx

        return flow_graph, processed_data

    def create_flow_based_problem(
        self, flow_problem: ProblemDef, graph: BaseGraph, data: Data
    ):
        """Create the flow-based Prize-Collecting Steiner tree optimization problem.

        Builds on the base Steiner tree problem and adds prize-collecting functionality.

        Args:
            flow_problem: The base flow problem.
            graph (BaseGraph): The input graph.
            data (Data): The data containing terminal and prize nodes.

        Returns:
            The configured optimization problem ready to be solved.
        """
        # Create base Steiner tree problem
        flow_problem = super().create_flow_based_problem(flow_problem, graph, data)

        # Add prize-collecting functionality
        for i, sample_data in enumerate(data.samples.values()):
            F = flow_problem.expr.flow
            F = F if len(F.shape) == 1 else F[:, i]

            # Get prized terminals
            prized_terminals = dict(
                sample_data.query.select(
                    lambda f: f.mapping == "vertex" and f.value
                ).pluck(lambda f: (f.id, f.value))
            )

            # If there are prized terminals, add them to the objective
            if len(prized_terminals) > 0:
                prized_idx = [
                    self.flow_edges[prized]
                    for prized in prized_terminals.keys()
                    if prized in self.flow_edges
                ]

                if prized_idx:
                    # Vector with the prizes (same order as the flow edges)
                    prizes = np.array(
                        [
                            prized_terminals[prized]
                            for prized in prized_terminals.keys()
                            if prized in self.flow_edges
                        ]
                    )

                    if self.strict_acyclic:
                        selected_prized_flow_edges = (
                            flow_problem.expr._flow_ipos[prized_idx]
                            + flow_problem.expr._flow_ineg[prized_idx]
                        )
                    else:
                        indicator_terminal_pos = self.flow_name + f"_terminal_pos_{i}"
                        indicator_terminal_neg = self.flow_name + f"_terminal_neg_{i}"
                        if (
                            indicator_terminal_pos not in flow_problem.expr
                            or indicator_terminal_neg not in flow_problem.expr
                        ):
                            flow_problem += self.backend.NonZeroIndicator(
                                F,
                                indexes=prized_idx,
                                tolerance=self.epsilon,
                                suffix_pos=f"_terminal_pos_{i}",
                                suffix_neg=f"_terminal_neg_{i}",
                            )
                            selected_prized_flow_edges = (
                                flow_problem.expr[indicator_terminal_pos]
                                + flow_problem.expr[indicator_terminal_neg]
                            )
                        else:
                            flow_edges_idxs = self._terminal_edgeflow_idx[i]
                            # Find the indexes of the prized edges
                            prized_idx_in_list = [
                                flow_edges_idxs.index(idx) for idx in prized_idx
                            ]
                            selected_prized_flow_edges = (
                                flow_problem.expr[indicator_terminal_pos][
                                    prized_idx_in_list
                                ]
                                + flow_problem.expr[indicator_terminal_neg][
                                    prized_idx_in_list
                                ]
                            )

                    flow_problem.register(
                        f"selected_prized_flow_edges_{i}", selected_prized_flow_edges
                    )

                    # Maximize the collection of prizes
                    flow_problem.add_objectives(
                        prizes @ selected_prized_flow_edges,
                        weights=-1,  # maximize the prizes
                    )

        return flow_problem
