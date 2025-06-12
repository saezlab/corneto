from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from corneto._constants import VAR_FLOW
from corneto._data import Data
from corneto._graph import Attr, BaseGraph, EdgeType
from corneto.backend._base import Backend, ProblemDef
from corneto.methods.future.method import FlowMethod


class SteinerTreeFlow(FlowMethod):
    """Basic Steiner Tree optimization method as a flow-based problem.

    This class implements the exact Steiner tree algorithm where, given a graph and a set of
    terminal nodes, it finds a minimal-weight connected subgraph (tree) that spans all terminals.
    It now accepts a single value or a list for both max_flow and root_vertex.
    If a list is provided, its length must equal the number of samples;
    if a single value is provided, it is used for all samples.
    """

    def __init__(
        self,
        # Accept a single value or list of values
        max_flow: Optional[Union[float, List[float]]] = None,
        default_edge_cost: float = 1.0,
        flow_name: str = VAR_FLOW,
        # Accept a single value or list of values
        root_vertex: Optional[Union[Any, List[Any]]] = None,
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
        # Store the raw values
        self.max_flow = max_flow
        self.root_vertex = root_vertex
        self.default_edge_cost = default_edge_cost
        self.flow_name = flow_name
        self.epsilon = epsilon
        self.strict_acyclic = strict_acyclic
        self.in_flow_edge_type = in_flow_edge_type
        self.out_flow_edge_type = out_flow_edge_type
        self.root_selection_strategy = root_selection_strategy

        # Initialize containers and placeholders
        self._terminal_edgeflow_idx = []
        self.flow_edges = dict()
        self.terminal_flow_edges = dict()

        # Internal storage for per-sample values.
        # If a list is provided, we store it separately; if a single value is provided,
        # we “broadcast” it later.
        self._max_flow: Optional[float] = None
        self._max_flow_list: Optional[List[float]] = None
        self._root_vertex: Optional[Any] = None
        self._root_vertex_list: Optional[List[Any]] = None

        if isinstance(max_flow, list):
            self._max_flow_list = max_flow
        else:
            self._max_flow = max_flow

        if isinstance(root_vertex, list):
            self._root_vertex_list = root_vertex
        else:
            self._root_vertex = root_vertex

        # This attribute will later hold the selected root for each sample.
        self._selected_roots: List[Any] = []

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data for the Steiner tree optimization.

        This method prepares the graph by adding flow edges for the chosen root vertices
        (if provided) and for all other vertices.

        Args:
            graph (BaseGraph): The input graph.
            data (Data): The data containing terminal nodes.

        Returns:
            Tuple[BaseGraph, Data]: Preprocessed graph and data.
        """
        # Reset per-run attributes
        self._terminal_edgeflow_idx = []
        self.flow_edges = dict()
        self.terminal_flow_edges = dict()

        flow_graph = graph.copy()
        all_vertices = data.query.filter_features(
            lambda f: f.mapping == "vertex",
        ).pluck_features()
        terminals = data.query.filter_features(
            lambda f: f.mapping == "vertex" and not f.value,
        ).pluck_features()

        num_samples = len(data.samples)
        # Set up max_flow values per sample
        if self._max_flow_list is not None:
            if len(self._max_flow_list) != num_samples:
                raise ValueError("Length of max_flow list must equal number of samples")
        else:
            # For single value case, use default if None
            if self._max_flow is None:
                self._max_flow = len(all_vertices)

        # Determine the root vertex to use for each sample.
        selected_roots: List[Any] = []
        if self._root_vertex_list is not None:
            if len(self._root_vertex_list) != num_samples:
                raise ValueError("Length of root_vertex list must equal number of samples")
            for i in range(num_samples):
                rv = self._root_vertex_list[i]
                if rv is None:
                    # Use selection strategy if not provided.
                    if self.root_selection_strategy == "first":
                        chosen = next(iter(terminals)) if terminals else next(iter(all_vertices))
                    elif self.root_selection_strategy == "best":
                        chosen = None  # Will be determined by optimization.
                    else:
                        raise ValueError(f"Unknown root selection strategy: {self.root_selection_strategy}")
                else:
                    chosen = rv
                selected_roots.append(chosen)
        else:
            # Single root vertex case.
            if self._root_vertex is None:
                # Use selection strategy for all samples.
                if self.root_selection_strategy == "first":
                    chosen = next(iter(terminals)) if terminals else next(iter(all_vertices))
                elif self.root_selection_strategy == "best":
                    chosen = None
                else:
                    raise ValueError(f"Unknown root selection strategy: {self.root_selection_strategy}")
                selected_roots = [chosen] * num_samples
            else:
                selected_roots = [self._root_vertex] * num_samples

        self._selected_roots = selected_roots

        # Determine edge types based on the selected roots.
        in_type = self.in_flow_edge_type
        out_type = self.out_flow_edge_type
        # If using "best" strategy for all samples (i.e. no root is provided), use undirected edges.
        if all(r is None for r in selected_roots):
            out_type = EdgeType.UNDIRECTED

        # Create an in-edge for every unique root that is not None.
        unique_roots = set(r for r in selected_roots if r is not None)
        for r in unique_roots:
            idx_root = flow_graph.add_edge((), r, type=in_type)
            self.flow_edges[r] = idx_root

        # For all other vertices, add an out-edge.
        for v in all_vertices:
            if v not in unique_roots:
                idx = flow_graph.add_edge(v, (), type=out_type)
                self.flow_edges[v] = idx

        return flow_graph, data

    def get_flow_bounds(self, graph: BaseGraph, data: Data) -> Dict[str, Any]:
        """Get the flow bounds for the optimization problem.

        Returns per-edge lower bounds and per-sample upper bounds. If a list of max_flow
        values was provided, then the bounds are computed for each sample separately;
        otherwise, a single value is used.

        Args:
            graph (BaseGraph): The input graph.
            data (Data): The data containing terminal nodes.

        Returns:
            Dict[str, Any]: Dictionary with flow bounds configuration.
        """
        if self._max_flow_list is not None:
            # Create a list of lower-bound arrays (one per sample)
            lb = [
                np.array(
                    [0 if prop.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED) else -mf for prop in graph.get_attr_edges()]
                )
                for mf in self._max_flow_list
            ]
            ub = self._max_flow_list
        else:
            lb = np.array(
                [
                    0 if prop.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED) else -self._max_flow
                    for prop in graph.get_attr_edges()
                ]
            )
            ub = self._max_flow

        return {
            "lb": lb,
            "ub": ub,
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
        flow_edge_ids = list(self.flow_edges.values())
        edge_ids = list(set(range(graph.num_edges)) - set(flow_edge_ids))

        if self.strict_acyclic:
            flow_problem += self.backend.NonZeroIndicator(flow_problem.expr._flow, tolerance=self.epsilon)
            flow_problem += self.backend.Acyclic(
                graph,
                flow_problem,
                indicator_negative_var_name="_flow_ineg",
                indicator_positive_var_name="_flow_ipos",
            )
            with_flow = flow_problem.expr._flow_ipos + flow_problem.expr._flow_ineg
        else:
            flow_problem += self.backend.Indicator(flow_problem.expr._flow, indexes=edge_ids)
            with_flow = flow_problem.expr._flow_i

        flow_problem.register("with_flow", with_flow)
        self._reg_varname = "with_flow"

        # Process each sample individually.
        for i, sample_data in enumerate(data.samples.values()):
            # Determine the sample-specific max_flow
            sample_max_flow = self._max_flow_list[i] if self._max_flow_list is not None else self._max_flow
            # Determine the sample-specific root vertex
            sample_selected_root = self._selected_roots[i]

            F = flow_problem.expr.flow
            F = F if len(F.shape) == 1 else F[:, i]

            vertices_edgeflow_idx = []
            all_vertices_with_data = sample_data.query.select(lambda f: f.mapping == "vertex").pluck()
            terminals_edgeflow_idx = []
            terminals = sample_data.query.select(lambda f: f.mapping == "vertex" and not f.value).pluck()
            # For terminals, if a root is chosen, skip it
            for terminal in terminals:
                if sample_selected_root is None or terminal != sample_selected_root:
                    idx = self.flow_edges[terminal]
                    terminals_edgeflow_idx.append(idx)

            # Collect flow edge indices for all vertices with data
            for vertex in all_vertices_with_data:
                if sample_selected_root is None or vertex != sample_selected_root:
                    idx = self.flow_edges[vertex]
                    vertices_edgeflow_idx.append(idx)

            sample_flow_edges = set(vertices_edgeflow_idx)
            if sample_selected_root is not None:
                sample_flow_edges.add(self.flow_edges[sample_selected_root])
            sample_flow_edges = list(sample_flow_edges)
            self._terminal_edgeflow_idx.append(sample_flow_edges)

            # Block flow for edges not related to this sample
            other_flow_edges = list(set(self.flow_edges.values()) - set(sample_flow_edges))
            if other_flow_edges:
                flow_problem += F[other_flow_edges] == 0

            if sample_selected_root is not None:
                # For samples with a designated root, force injected flow.
                flow_problem += F[self.flow_edges[sample_selected_root]] == sample_max_flow
                if terminals_edgeflow_idx:
                    flow_problem += F[terminals_edgeflow_idx] >= 1
            else:
                # For "best" strategy: let optimization select the root.
                if all_vertices_with_data:
                    flow_problem += self.backend.NonZeroIndicator(
                        flow_problem.expr.flow,
                        vertices_edgeflow_idx,
                        i,
                        tolerance=self.epsilon,
                        suffix_pos=f"_terminal_pos_{i}",
                        suffix_neg=f"_terminal_neg_{i}",
                    )
                    terminal_pos = self.flow_name + f"_terminal_pos_{i}"
                    terminal_neg = self.flow_name + f"_terminal_neg_{i}"
                    flow_problem += flow_problem.expr[terminal_neg].sum() == 1
                    t_idx = [vertices_edgeflow_idx.index(idx) for idx in terminals_edgeflow_idx]
                    if t_idx:
                        flow_problem += flow_problem.expr[terminal_pos][t_idx].sum() == len(t_idx) - 1

            # Add edge cost to the objective.
            edge_costs = np.ones((len(edge_ids))) * self.default_edge_cost
            selected = with_flow if len(with_flow.shape) == 1 else with_flow[:, i]

            # Incorporate edge-specific costs from the sample data.
            edge_data = sample_data.query.select(lambda f: f.mapping == "edge").to_list()
            for edata in edge_data:
                edge_costs[edata.id] = float(edata.value)

            flow_problem.add_objectives(edge_costs[edge_ids] @ selected[edge_ids])

        return flow_problem
