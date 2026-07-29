"""Prize-collecting Steiner tree optimization."""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from corneto._constants import VAR_FLOW
from corneto.backend._base import Backend, ProblemDef
from corneto.data import Data
from corneto.graph import Attr, BaseGraph, EdgeType
from corneto.methods._base import FlowMethod
from corneto.methods._flow_utils import add_selected_flow
from corneto.methods._input_utils import (
    DEFAULT_CONDITION,
    data_from_features,
    legacy_data,
    require_mapping,
    validate_condition_keys,
    validate_edge_costs,
    validate_vertex_collection,
    validate_vertices,
)
from corneto.methods._network_utils import augment_with_boundaries


class PrizeCollectingSteinerTree(FlowMethod):
    """Prize-Collecting Steiner Tree optimization method built on top of FlowMethod.

    In a prize-collecting Steiner tree problem, terminals can have prizes (values > 0),
    making them optional terminals that provide a benefit if included in the solution.
    The class supports providing multiple max_flow and root_vertex values per sample.
    """

    def __init__(
        self,
        include_all_terminals: bool = False,
        max_flow: Optional[Union[float, List[float]]] = None,
        default_edge_cost: float = 1.0,
        flow_name: str = VAR_FLOW,
        root_vertex: Optional[Union[Any, List[Any]]] = None,
        root_selection_strategy: Literal["first", "best"] = "first",
        best_root_candidates: Literal["data", "graph"] = "data",
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
            lambda_reg=lambda_reg,
            disable_structured_sparsity=disable_structured_sparsity,
            backend=backend,
        )
        self.include_all_terminals = include_all_terminals
        self.max_flow = max_flow
        self.root_vertex = root_vertex
        self.default_edge_cost = default_edge_cost
        self.flow_name = flow_name
        self.epsilon = epsilon
        self.strict_acyclic = strict_acyclic
        self.in_flow_edge_type = in_flow_edge_type
        self.out_flow_edge_type = out_flow_edge_type
        self.root_selection_strategy = root_selection_strategy
        self.best_root_candidates = best_root_candidates
        self.force_flow_through_root = force_flow_through_root

        # Initialize containers and placeholders
        self._terminal_edgeflow_idx = []
        self.flow_edges = dict()
        self.flow_edges_in = dict()
        self.flow_edges_out = dict()
        self.prized_flow_edges = dict()
        self._candidate_vertices_per_sample: List[Tuple[Any, ...]] = []

        # Internal storage for per-sample values.
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

        self._selected_roots: List[Any] = []

    def build(
        self,
        graph: BaseGraph,
        data: Optional[Data] = None,
        *,
        prizes=None,
        terminals=None,
        edge_costs=None,
    ):
        """Build a single-condition PCST problem from explicit inputs."""
        old_data = legacy_data(data, method=self.__class__.__name__)
        if old_data is not None:
            if any(value is not None for value in (prizes, terminals, edge_costs)):
                raise TypeError("Do not combine a Data object with explicit scientific inputs.")
            return self.build_from_data(graph, old_data)
        if prizes is None:
            raise TypeError("build() requires prizes=.")
        return self.build_many(
            graph,
            prizes={DEFAULT_CONDITION: prizes},
            terminals=None if terminals is None else {DEFAULT_CONDITION: terminals},
            edge_costs=None if edge_costs is None else {DEFAULT_CONDITION: edge_costs},
        )

    def build_many(self, graph: BaseGraph, *, prizes, terminals=None, edge_costs=None):
        """Build a multi-condition PCST problem from named condition mappings."""
        conditions = validate_condition_keys(prizes=prizes, terminals=terminals, edge_costs=edge_costs)
        features_by_condition = {}
        for condition in conditions:
            prize_values = validate_vertices(
                graph,
                require_mapping(prizes[condition], argument="prizes", condition=condition),
                argument="prizes",
                condition=condition,
            )
            for identifier, value in prize_values.items():
                if value <= 0:
                    raise ValueError(
                        f"Prize for vertex {identifier!r} in condition {condition!r} "
                        f"must be greater than zero, got {value}."
                    )
            terminal_values = validate_vertex_collection(
                graph,
                [] if terminals is None else terminals[condition],
                argument="terminals",
                condition=condition,
            )
            cost_values = validate_edge_costs(
                graph,
                {}
                if edge_costs is None
                else require_mapping(edge_costs[condition], argument="edge_costs", condition=condition),
                condition=condition,
            )
            feature_data = {
                identifier: {"mapping": "vertex", "value": value, "role": "prize"}
                for identifier, value in prize_values.items()
            }
            for identifier in terminal_values:
                feature_data.setdefault(identifier, {"mapping": "vertex"})["role"] = "terminal"
            features = [{"id": identifier, **attributes} for identifier, attributes in feature_data.items()]
            features.extend(
                {"id": identifier, "mapping": "edge", "value": value} for identifier, value in cost_values.items()
            )
            features_by_condition[condition] = features
        return self.build_from_data(graph, data_from_features(features_by_condition))

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the graph and data."""
        # Reset per-run attributes
        self._terminal_edgeflow_idx = []
        self.flow_edges = dict()
        self.flow_edges_in = dict()
        self.flow_edges_out = dict()
        self.prized_flow_edges = dict()
        self._candidate_vertices_per_sample = []

        graph_vertices = tuple(graph.V)
        all_vertices = tuple(
            data.query.filter_features(
                lambda f: f.mapping == "vertex",
            ).pluck_features()
        )
        terminals = data.query.filter_features(
            lambda f: f.mapping == "vertex" and not f.value,
        ).pluck_features()

        num_samples = len(data.samples)
        # Set up max_flow values per sample
        if self._max_flow_list is not None:
            if len(self._max_flow_list) != num_samples:
                raise ValueError("Length of max_flow list must equal number of samples")
        else:
            if self._max_flow is None:
                self._max_flow = len(all_vertices)

        # Determine the root vertex to use for each sample.
        selected_roots: List[Any] = []

        def _first_vertex_choice():
            if terminals:
                return next(iter(terminals))
            if all_vertices:
                return next(iter(all_vertices))
            if graph_vertices:
                return next(iter(graph_vertices))
            raise ValueError("Cannot select a root from an empty graph")

        if self._root_vertex_list is not None:
            if len(self._root_vertex_list) != num_samples:
                raise ValueError("Length of root_vertex list must equal number of samples")
            for i in range(num_samples):
                rv = self._root_vertex_list[i]
                if rv is None:
                    if self.root_selection_strategy == "first":
                        chosen = _first_vertex_choice()
                    elif self.root_selection_strategy == "best":
                        chosen = None
                    else:
                        raise ValueError(f"Unknown root selection strategy: {self.root_selection_strategy}")
                else:
                    chosen = rv
                selected_roots.append(chosen)
        else:
            if self._root_vertex is None:
                if self.root_selection_strategy == "first":
                    chosen = _first_vertex_choice()
                elif self.root_selection_strategy == "best":
                    chosen = None
                else:
                    raise ValueError(f"Unknown root selection strategy: {self.root_selection_strategy}")
                selected_roots = [chosen] * num_samples
            else:
                selected_roots = [self._root_vertex] * num_samples

        self._selected_roots = selected_roots

        graph_vertex_set = set(graph_vertices)
        for root in selected_roots:
            if root is not None and root not in graph_vertex_set:
                raise ValueError(f"Root vertex {root!r} is not present in the graph vertices")

        if self.best_root_candidates not in {"data", "graph"}:
            raise ValueError(
                f"best_root_candidates must be either 'data' or 'graph', got {self.best_root_candidates!r}"
            )

        # Compute per-sample candidate vertices, then create only the union
        # of actually-needed auxiliary edges across samples.
        out_vertices_union: List[Any] = []
        in_vertices_union: List[Any] = []
        sample_data_list = list(data.samples.values())
        for i, sample_data in enumerate(sample_data_list):
            sample_selected_root = selected_roots[i]
            sample_vertices = tuple(sample_data.query.select(lambda f: f.mapping == "vertex").pluck())

            if sample_selected_root is None:
                if self.best_root_candidates == "graph":
                    candidate_vertices = graph_vertices
                else:
                    candidate_vertices = sample_vertices
            else:
                candidate_vertices = tuple(v for v in sample_vertices if v != sample_selected_root)
                in_vertices_union.append(sample_selected_root)

            candidate_vertices = tuple(dict.fromkeys(candidate_vertices))
            self._candidate_vertices_per_sample.append(candidate_vertices)
            out_vertices_union.extend(candidate_vertices)

        out_vertices = tuple(dict.fromkeys(out_vertices_union))
        in_vertices = tuple(dict.fromkeys(in_vertices_union))

        # Determine edge types based on the selected roots.
        in_type = self.in_flow_edge_type
        out_type = self.out_flow_edge_type
        # Mixed root strategies across samples require undirected out-flow edges
        # so "best" samples can still use negative flow to select a root.
        if any(r is None for r in selected_roots):
            out_type = EdgeType.UNDIRECTED

        layout = augment_with_boundaries(
            graph,
            inflow_vertices=in_vertices,
            outflow_vertices=out_vertices,
            inflow_type=in_type,
            outflow_type=out_type,
        )
        flow_graph = layout.graph
        self.flow_edges_in = layout.inflow_edges
        self.flow_edges_out = layout.outflow_edges

        # Backward-compatible alias used by some callers; map to out-edges.
        self.flow_edges = self.flow_edges_out.copy()

        return flow_graph, data

    def get_flow_bounds(self, graph: BaseGraph, data: Data) -> Dict[str, Any]:
        """Get the flow bounds for the optimization problem."""
        if self._max_flow_list is not None:
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
        """Create the flow-based optimization problem."""
        flow_edge_ids = list(set(self.flow_edges_in.values()) | set(self.flow_edges_out.values()))
        edge_ids = list(set(range(graph.num_edges)) - set(flow_edge_ids))

        selected_flow = add_selected_flow(
            self.backend,
            flow_problem,
            graph,
            biological_edge_indices=edge_ids,
            epsilon=self.epsilon,
            acyclic=self.strict_acyclic,
        )
        with_flow = selected_flow.all_edges

        flow_problem.register("with_flow", with_flow)
        self._reg_varname = "with_flow"

        for i, sample_data in enumerate(data.samples.values()):
            sample_max_flow = self._max_flow_list[i] if self._max_flow_list is not None else self._max_flow
            sample_selected_root = self._selected_roots[i]

            F = flow_problem.expr.flow
            F = F if len(F.shape) == 1 else F[:, i]

            vertices_edgeflow_idx = []
            candidate_vertices = self._candidate_vertices_per_sample[i]

            terminals_edgeflow_idx = []
            terminals = sample_data.query.select(
                lambda f: f.mapping == "vertex" and f.data.get("role", None) == "terminal"
            ).pluck()

            prized_terminals = dict(
                sample_data.query.select(lambda f: f.mapping == "vertex" and f.value).pluck(lambda f: (f.id, f.value))
            )

            for terminal in terminals:
                if sample_selected_root is None or terminal != sample_selected_root:
                    if terminal in self.flow_edges_out:
                        idx = self.flow_edges_out[terminal]
                        terminals_edgeflow_idx.append(idx)

            for vertex in candidate_vertices:
                if sample_selected_root is None or vertex != sample_selected_root:
                    if vertex in self.flow_edges_out:
                        idx = self.flow_edges_out[vertex]
                        vertices_edgeflow_idx.append(idx)

            sample_flow_edges = set(vertices_edgeflow_idx)
            if sample_selected_root is not None:
                sample_flow_edges.add(self.flow_edges_in[sample_selected_root])
            sample_flow_edges = list(sample_flow_edges)
            self._terminal_edgeflow_idx.append(sample_flow_edges)

            all_flow_edges = set(self.flow_edges_in.values()) | set(self.flow_edges_out.values())
            other_flow_edges = list(all_flow_edges - set(sample_flow_edges))
            if other_flow_edges:
                flow_problem += F[other_flow_edges] == 0

            # Root Flow Constraints
            if sample_selected_root is not None:
                if self.force_flow_through_root:
                    flow_problem += F[self.flow_edges_in[sample_selected_root]] == sample_max_flow
                else:
                    flow_problem += F[self.flow_edges_in[sample_selected_root]] >= 0

                if self.include_all_terminals and terminals_edgeflow_idx:
                    flow_problem += F[terminals_edgeflow_idx] >= 1
            else:
                if candidate_vertices:
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

                    if self.include_all_terminals:
                        t_idx = [vertices_edgeflow_idx.index(idx) for idx in terminals_edgeflow_idx]
                        if t_idx:
                            # Enforce all terminals selected, whether root is a terminal
                            # (negative sign) or not (all terminals positive).
                            flow_problem += flow_problem.expr[terminal_pos][t_idx].sum() + flow_problem.expr[
                                terminal_neg
                            ][t_idx].sum() == len(t_idx)

            # Costs Objective
            edge_costs = np.ones((graph.num_edges)) * self.default_edge_cost
            selected = with_flow if len(with_flow.shape) == 1 else with_flow[:, i]

            edge_data = sample_data.query.select(lambda f: f.mapping == "edge").to_list()
            for edata in edge_data:
                if not isinstance(edata.id, int) or edata.id < 0 or edata.id >= graph.num_edges:
                    raise ValueError(
                        f"Invalid edge feature id={edata.id!r}. Expected an integer in [0, {graph.num_edges - 1}]"
                    )
                edge_costs[edata.id] = float(edata.value)

            flow_problem.add_objective(edge_costs[edge_ids] @ selected[edge_ids], name="edge_cost")

            # Prizes Objective (PCST)
            if prized_terminals:
                prized_idx = []
                prized_vertices = []
                for prized in prized_terminals.keys():
                    if sample_selected_root is not None and prized == sample_selected_root:
                        if prized in self.flow_edges_in:
                            prized_idx.append(self.flow_edges_in[prized])
                            prized_vertices.append(prized)
                    else:
                        if prized in self.flow_edges_out:
                            prized_idx.append(self.flow_edges_out[prized])
                            prized_vertices.append(prized)
                if prized_idx:
                    prizes = np.array([prized_terminals[prized] for prized in prized_vertices])
                    if self.strict_acyclic:
                        selected_for_prizes = with_flow if len(with_flow.shape) == 1 else with_flow[:, i]
                        selected_prized_flow_edges = selected_for_prizes[prized_idx]
                    else:
                        flow_variable = flow_problem.expr._flow
                        indicator_args = (prized_idx,) if len(flow_variable.shape) == 1 else (prized_idx, i)
                        flow_problem += self.backend.NonZeroIndicator(
                            flow_variable,
                            *indicator_args,
                            tolerance=self.epsilon,
                            suffix_pos=f"_prize_pos_{i}",
                            suffix_neg=f"_prize_neg_{i}",
                        )
                        selected_prized_flow_edges = (
                            flow_problem.expr[f"{self.flow_name}_prize_pos_{i}"]
                            + flow_problem.expr[f"{self.flow_name}_prize_neg_{i}"]
                        )

                    flow_problem.register(f"selected_prized_flow_edges_{i}", selected_prized_flow_edges)
                    flow_problem.add_objective(prizes @ selected_prized_flow_edges, weight=-1, name="prizes")

        return flow_problem
