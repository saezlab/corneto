from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from corneto._constants import VAR_FLOW
from corneto._data import Data
from corneto._graph import Attr, BaseGraph, EdgeType
from corneto.backend._base import Backend, ProblemDef
from corneto.methods.future.method import FlowMethod


class SteinerTreeFlow(FlowMethod):
    """Exact Steiner Tree optimization method as a flow-based problem.

    This class implements the exact Steiner tree algorithm where given a graph and a set of
    terminal nodes, it finds a minimal-weight connected subgraph (tree) that spans all terminals.
    The implementation extends FlowMethod.

    In a Steiner tree problem, given:
    - A graph G with edge weights
    - A subset of vertices called terminals
    - Optionally, a root node

    The goal is to find a minimum-weight connected subgraph that contains all terminals.
    If terminals have prizes (values > 0), these are treated as optional terminals that
    provide a benefit if included in the solution.
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
            lambda_reg=lambda_reg,
            disable_structured_sparsity=disable_structured_sparsity,
            backend=backend,
        )
        self.max_flow = max_flow
        self.flow_name = flow_name
        self.epsilon = epsilon
        self.strict_acyclic = strict_acyclic
        self.in_flow_edge_type = in_flow_edge_type
        self.out_flow_edge_type = out_flow_edge_type
        self.include_all_terminals = include_all_terminals
        self.root_vertex = root_vertex
        self.default_edge_cost = default_edge_cost
        self.root_selection_strategy = root_selection_strategy
        self.flow_edges = dict()

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data for the Steiner tree optimization.

        This method prepares the graph by adding flow edges for root and terminal
        vertices. Different flavors of the Steiner tree problem can be handled by
        modifying the graph structure and/or the direction of flows.

        Args:
            graph (BaseGraph): The input graph.
            data (Data): The data containing terminal nodes.

        Returns:
            Tuple[BaseGraph, Data]: Preprocessed graph and data.
        """
        flow_graph = graph.copy()
        terminal_vertices = data.query.filter_features(
            lambda f: f.mapping == "vertex",
        ).pluck_features()
        # for v in data.filter_by("type", "terminal").values():
        #    # Collect all steiner vertices across samples
        #   terminal_vertices.update(v.features.keys())

        if self.max_flow is None:
            self.max_flow = len(terminal_vertices)

        in_type = self.in_flow_edge_type
        out_type = self.out_flow_edge_type

        if self.root_vertex is None:
            if self.root_selection_strategy == "first":
                # Takes the first terminal vertex as root.
                # This does not matter if the graph is undirected
                self.root_vertex = next(iter(terminal_vertices))
                idx_root = flow_graph.add_edge((), self.root_vertex, type=in_type)
                self.flow_edges[self.root_vertex] = idx_root
            elif self.root_selection_strategy == "best":
                # All in/out edges can inject/extract flow
                # so no distinction. Root is selected through
                # optimization.
                out_type = EdgeType.UNDIRECTED
            else:
                raise ValueError(f"Unknown root selection strategy: {self.root_selection_strategy}")
        for v in terminal_vertices:
            if v != self.root_vertex:
                idx = flow_graph.add_edge(v, (), type=out_type)
                self.flow_edges[v] = idx
        return flow_graph, data

    def get_flow_bounds(self, graph: BaseGraph, data: Data) -> Dict[str, Any]:
        """Get the flow bounds for the optimization problem.

        Determines the lower and upper bounds for flows based on graph edge types.

        Args:
            graph (BaseGraph): The input graph.
            data (Data): The data containing terminal nodes.

        Returns:
            Dict[str, Any]: Dictionary with flow bounds configuration.
        """
        # Create lower bounds based on edge types
        lb = np.array(
            [
                0 if prop.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED) else -self.max_flow
                for prop in graph.get_attr_edges()
            ]
        )

        return {
            "lb": lb,
            "ub": self.max_flow,
            "n_flows": len(data.samples),
            "shared_bounds": False,
        }

    def create_flow_based_problem(self, flow_problem: ProblemDef, graph: BaseGraph, data: Data):
        """Create the flow-based Steiner tree optimization problem.

        Sets up the constraints and objectives for the Steiner tree problem.

        Args:
            flow_problem: The base flow problem.
            graph (BaseGraph): The input graph.
            data (Data): The data containing terminal nodes.

        Returns:
            The configured optimization problem ready to be solved.
        """
        # V_idx = {v: i for i, v in enumerate(graph.V)}
        flow_edge_ids = list(self.flow_edges.values())
        edge_ids = list(set(range(graph.num_edges)) - set(flow_edge_ids))
        if self.strict_acyclic:
            # We need to create indicator variables for the flow
            flow_problem += self.backend.NonZeroIndicator(flow_problem.expr._flow, tolerance=self.epsilon)
            flow_problem += self.backend.Acyclic(
                graph,
                flow_problem,
                indicator_negative_var_name="_flow_ineg",
                indicator_positive_var_name="_flow_ipos",
            )
            with_flow = flow_problem.expr._flow_ipos + flow_problem.expr._flow_ineg
        else:
            # We create indicator variables for the edges except the in/out flow edges
            flow_problem += self.backend.Indicator(flow_problem.expr._flow, indexes=edge_ids)
            with_flow = flow_problem.expr._flow_i

        flow_problem.register("with_flow", with_flow)
        # Force flow through the root vertex to the terminals
        for i, sample_data in enumerate(data.samples.values()):
            F = flow_problem.expr.flow
            F = F if len(F.shape) == 1 else F[:, i]
            terminal_edgeflow_idx = []
            prized_edgeflow_idx = []
            terminals = sample_data.query.select(lambda f: f.mapping == "vertex" and not f.value).pluck()
            prized_terminals = dict(
                sample_data.query.select(lambda f: f.mapping == "vertex" and f.value).pluck(lambda f: (f.id, f.value))
            )
            # all_terminals = set(terminals).union(prized_terminals.keys())
            for terminal in terminals:
                if terminal != self.root_vertex:
                    idx = self.flow_edges[terminal]
                    terminal_edgeflow_idx.append(idx)
            for prized in prized_terminals.keys():
                if prized != self.root_vertex:
                    idx = self.flow_edges[prized]
                    prized_edgeflow_idx.append(idx)
            # All other flow edges not in this sample have to be blocked
            sample_flow_edges = set(terminal_edgeflow_idx) | set(prized_edgeflow_idx)
            if self.root_vertex is not None:
                sample_flow_edges.add(self.flow_edges[self.root_vertex])
            sample_flow_edges = list(sample_flow_edges)

            other_flow_edges = list(set(self.flow_edges.values()) - set(sample_flow_edges))
            if len(other_flow_edges) > 0:
                flow_problem += F[other_flow_edges] == 0

            if self.root_vertex is not None:
                # Injected flow through root
                flow_problem += F[self.flow_edges[self.root_vertex]] == self.max_flow
                # Extracted flow through terminals
                if self.include_all_terminals and len(terminals) > 0:
                    flow_problem += F[terminal_edgeflow_idx] >= 1
            else:
                # Note: non zero indicators might exist for strict acyclicity
                if len(terminals) > 0:
                    flow_problem += self.backend.NonZeroIndicator(
                        F,
                        indexes=terminal_edgeflow_idx,
                        tolerance=self.epsilon,
                        suffix_pos="_terminal_pos",
                        suffix_neg="_terminal_neg",
                    )
                    # We need to force the selection of terminals
                    flow_problem += flow_problem.expr._flow_terminal_pos.sum() == len(terminals) - 1
                    flow_problem += flow_problem.expr._flow_terminal_neg.sum() == 1

        for i, sample_data in enumerate(data.samples.values()):
            edge_costs = np.ones((len(edge_ids))) * self.default_edge_cost
            selected = with_flow if len(with_flow.shape) == 1 else with_flow[:, i]
            F = flow_problem.expr.flow
            F = F if len(F.shape) == 1 else F[:, i]
            edge_data = sample_data.query.select(lambda f: f.mapping == "edge").to_list()
            # Note: ids of edges correspond to the original graph, not the preprocessed graph
            for edata in edge_data:
                edge_costs[edata.id] = edata.value
            flow_problem.add_objectives(edge_costs[edge_ids] @ selected[edge_ids])
            # If prized terminals are present, we need to:
            # 1. Add non-zero binary vars for the in/out flow edges of prized vertices
            # 2. Collect prizes only if the in/out flow of prized edges are non-zero
            if len(prized_terminals) > 0:
                prized_idx = [self.flow_edges[prized] for prized in prized_terminals.keys()]
                # Vector with the prizes (same order as the flow edges)
                prizes = np.array([prized_terminals[prized] for prized in prized_terminals.keys()])
                flow_problem += self.backend.NonZeroIndicator(
                    F,
                    indexes=prized_idx,
                    tolerance=self.epsilon,
                    suffix_pos="_prized_pos",
                    suffix_neg="_prized_neg",
                )
                if self.root_selection_strategy == "best":
                    # Avoid "forest" effect, only 1 of the bidirectional edges can
                    # inject flow (since all flow edges are pointing out, the flow that goes
                    # in has a negative sign)
                    flow_problem += flow_problem.expr._flow_prized_pos.sum() == len(prized_terminals) - 1
                    flow_problem += flow_problem.expr._flow_prized_neg.sum() == 1
                # Binary vector with the selected prized vertices
                selected_prized = flow_problem.expr._flow_prized_pos + flow_problem.expr._flow_prized_neg
                flow_problem.add_objectives(
                    prizes @ selected_prized,
                    weights=-1,  # maximize the prizes
                )

        return flow_problem
