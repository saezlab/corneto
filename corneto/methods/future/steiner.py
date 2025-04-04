from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from corneto._constants import VAR_FLOW
from corneto._data import Data
from corneto._graph import Attr, BaseGraph, EdgeType
from corneto.backend._base import Backend, ProblemDef
from corneto.methods.future.method import FlowMethod


class SteinerTreeFlow(FlowMethod):
    """Basic Steiner Tree optimization method as a flow-based problem.

    This class implements the exact Steiner tree algorithm where given a graph and a set of
    terminal nodes, it finds a minimal-weight connected subgraph (tree) that spans all terminals.
    The implementation extends FlowMethod.

    In a Steiner tree problem, given:
    - A graph G with edge weights
    - A subset of vertices called terminals
    - Optionally, a root node

    The goal is to find a minimum-weight connected subgraph that contains all terminals.
    """

    def __init__(
        self,
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
        self.root_vertex = root_vertex
        self.default_edge_cost = default_edge_cost
        self.root_selection_strategy = root_selection_strategy
        self._terminal_edgeflow_idx = []
        self.flow_edges = dict()
        self.terminal_flow_edges = dict()
        self._selected_root_vertex = root_vertex
        self._max_flow = max_flow

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data for the Steiner tree optimization.

        This method prepares the graph by adding flow edges for root and terminal
        vertices.

        Args:
            graph (BaseGraph): The input graph.
            data (Data): The data containing terminal nodes.

        Returns:
            Tuple[BaseGraph, Data]: Preprocessed graph and data.
        """
        self._selected_root_vertex = None
        self._terminal_edgeflow_idx = []
        self.flow_edges = dict()
        self.terminal_flow_edges = dict()
        flow_graph = graph.copy()
        all_vertices = data.query.filter_features(
            lambda f: f.mapping == "vertex",
        ).pluck_features()
        # Vertices which have a value are considered prized vertices,
        # otherwise they are terminals (and forced to be included in the solution)
        terminals = data.query.filter_features(
            lambda f: f.mapping == "vertex" and not f.value,
        ).pluck_features()
        #other = set(all_vertices) - set(terminals)

        if self.max_flow is None:
            self._max_flow = len(all_vertices)

        in_type = self.in_flow_edge_type
        out_type = self.out_flow_edge_type

        self._selected_root_vertex = self.root_vertex

        if self._selected_root_vertex is None:
            if self.root_selection_strategy == "first":
                # Takes the first terminal vertex as root.
                # This does not matter if the graph is undirected
                self._selected_root_vertex = next(iter(terminals)) if len(terminals) > 0 else next(
                    iter(all_vertices)
                )
                idx_root = flow_graph.add_edge((),self._selected_root_vertex, type=in_type)
                self.flow_edges[self._selected_root_vertex] = idx_root
            elif self.root_selection_strategy == "best":
                # All in/out edges can inject/extract flow
                # so no distinction. Root is selected through
                # optimization.
                out_type = EdgeType.UNDIRECTED
            else:
                raise ValueError(
                    f"Unknown root selection strategy: {self.root_selection_strategy}"
                )
        else:
            idx = flow_graph.add_edge((), self._selected_root_vertex, type=in_type)
            self.flow_edges[self._selected_root_vertex] = idx
        for v in all_vertices:
            if v != self._selected_root_vertex:
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
                0
                if prop.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED)
                else -self._max_flow
                for prop in graph.get_attr_edges()
            ]
        )

        return {
            "lb": lb,
            "ub": self._max_flow,
            "n_flows": len(data.samples),
            "shared_bounds": False,
        }

    def create_flow_based_problem(
        self, flow_problem: ProblemDef, graph: BaseGraph, data: Data
    ):
        """Create the flow-based Steiner tree optimization problem.

        Sets up the constraints and objectives for the Steiner tree problem.

        Args:
            flow_problem: The base flow problem.
            graph (BaseGraph): The input graph.
            data (Data): The data containing terminal nodes.

        Returns:
            The configured optimization problem ready to be solved.
        """
        flow_edge_ids = list(self.flow_edges.values())
        edge_ids = list(set(range(graph.num_edges)) - set(flow_edge_ids))

        if self.strict_acyclic:
            # We need to create indicator variables for the flow
            flow_problem += self.backend.NonZeroIndicator(
                flow_problem.expr._flow, tolerance=self.epsilon
            )
            # TODO: Names should be provided
            flow_problem += self.backend.Acyclic(
                graph,
                flow_problem,
                indicator_negative_var_name="_flow_ineg",
                indicator_positive_var_name="_flow_ipos",
            )
            with_flow = flow_problem.expr._flow_ipos + flow_problem.expr._flow_ineg
        else:
            # We create indicator variables for the edges except the in/out flow edges
            flow_problem += self.backend.Indicator(
                flow_problem.expr._flow, indexes=edge_ids
            )
            with_flow = flow_problem.expr._flow_i

        flow_problem.register("with_flow", with_flow)
        self._reg_varname = "with_flow"

        # Configure objectives and constraints for each sample
        for i, sample_data in enumerate(data.samples.values()):
            F = flow_problem.expr.flow
            F = F if len(F.shape) == 1 else F[:, i]

            vertices_edgeflow_idx = []
            all_vertices_with_data = sample_data.query.select(
                lambda f: f.mapping == "vertex"
            ).pluck()
            # Get the vertices which are considered terminals
            # and thus must be included in the solution
            terminals_edgeflow_idx = []
            terminals = sample_data.query.select(
                lambda f: f.mapping == "vertex" and not f.value
            ).pluck()
            for terminal in terminals:
                if terminal != self._selected_root_vertex:
                    idx = self.flow_edges[terminal]
                    terminals_edgeflow_idx.append(idx)

            # Collect terminal flow edges
            for vertex in all_vertices_with_data:
                if vertex != self._selected_root_vertex:
                    idx = self.flow_edges[vertex]
                    vertices_edgeflow_idx.append(idx)

            # Sample flow edges
            sample_flow_edges = set(vertices_edgeflow_idx)
            if self._selected_root_vertex is not None:
                sample_flow_edges.add(self.flow_edges[self._selected_root_vertex])
            sample_flow_edges = list(sample_flow_edges)
            # TODO: move to preprocessing, this can be used by
            # extension (e.g. PCST) to quickly get the flow edges/
            # This is for both terminals and prizes
            self._terminal_edgeflow_idx.append(sample_flow_edges)

            # Block flow for edges not related to this sample
            other_flow_edges = list(
                set(self.flow_edges.values()) - set(sample_flow_edges)
            )
            if len(other_flow_edges) > 0:
                flow_problem += F[other_flow_edges] == 0

            if self._selected_root_vertex is not None:
                # Injected flow through root
                flow_problem += F[self.flow_edges[self._selected_root_vertex]] == self._max_flow
                # Extracted flow through terminals (only for terminals!)
                #if len(all_vertices_with_data) > 0:
                #    flow_problem += F[vertices_edgeflow_idx] >= 1
                # Only force flow for terminals, not for prized vertices
                if len(terminals_edgeflow_idx) > 0:
                    flow_problem += F[terminals_edgeflow_idx] >= 1
            else:
                # Root selection through optimization
                if len(all_vertices_with_data) > 0:
                    flow_problem += self.backend.NonZeroIndicator(
                        F,
                        indexes=vertices_edgeflow_idx,
                        tolerance=self.epsilon,
                        suffix_pos=f"_terminal_pos_{i}",
                        suffix_neg=f"_terminal_neg_{i}",
                    )
                    terminal_pos = self.flow_name + f"_terminal_pos_{i}"
                    terminal_neg = self.flow_name + f"_terminal_neg_{i}"
                    # Only one terminal can be selected as root. Since all edges
                    # when no root is provided are pointing outwards, the injection
                    # of flow has negative sign. We only allow one of the edges to
                    # inject or extract flow to take the inflow direction
                    flow_problem += flow_problem.expr[terminal_neg].sum() == 1
                    # All others except the selected root, have to extract the flow.
                    # This only applies to terminal vertices, not the root neither
                    # other vertices which are optional.
                    t_idx = [vertices_edgeflow_idx.index(idx) for idx in terminals_edgeflow_idx]
                    if len(t_idx) > 0:
                        flow_problem += (
                            flow_problem.expr[terminal_pos][t_idx].sum() == len(t_idx) - 1
                        )

            # Add edge costs to the objective
            edge_costs = np.ones((len(edge_ids))) * self.default_edge_cost
            selected = with_flow if len(with_flow.shape) == 1 else with_flow[:, i]

            # Process any edge-specific costs from the data
            edge_data = sample_data.query.select(
                lambda f: f.mapping == "edge"
            ).to_list()
            for edata in edge_data:
                edge_costs[edata.id] = float(edata.value)

            flow_problem.add_objectives(edge_costs[edge_ids] @ selected[edge_ids])

        return flow_problem
