"""Globally optimized upstream/downstream PHONEMeS formulation."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np

from corneto._constants import VarType
from corneto.backend._base import Backend
from corneto.data import Data
from corneto.graph import BaseGraph, EdgeType
from corneto.methods._flow_utils import add_vertex_selection
from corneto.methods._input_utils import (
    DEFAULT_CONDITION,
    data_from_features,
    validate_edge_costs,
    validate_numeric,
)
from corneto.methods._network_utils import prune_to_paths
from corneto.methods._optimization_utils import add_condition_union
from corneto.methods._phonemes_preprocessing import (
    normalize_phonemes_score_mapping,
)
from corneto.methods.phonemes import PHONEMeS, _data_from_phonemes_inputs

_ROLE_REGULATED_KINASE = "regulated_kinase"
_ROLE_PHOSPHOSITE = "phosphosite"
_ROLE_BOTH = "regulated_kinase_phosphosite"
_ANCHOR_POLICIES = {"either", "both", "downstream"}


def _pairwise_or(backend, problem, left, right, *, name: str):
    union = backend.Variable(name, left.shape, vartype=VarType.BINARY)
    problem += union >= left
    problem += union >= right
    problem += union <= left + right
    return union


class BidirectionalPHONEMeS(PHONEMeS):
    """Infer one globally optimized network around regulated kinase anchors.

    The method searches both upstream and downstream of each regulated kinase.
    Phosphosite scores and interaction costs are applied to the combined
    biological network rather than to two independently inferred networks.

    Args:
        default_edge_cost: Cost assigned to unspecified PKN edges.
        max_flow: Upper bound for every directional flow.
        epsilon: Smallest positive flow associated with a selected edge.
        anchor_policy: Whether every anchor must participate on either side,
            both sides, or the downstream side.
        backend: Optimization backend.
    """

    def __init__(
        self,
        default_edge_cost: float = 1e-5,
        max_flow: Optional[float] = None,
        epsilon: float = 1,
        anchor_policy: Literal["either", "both", "downstream"] = "either",
        backend: Optional[Backend] = None,
    ):
        super().__init__(
            default_edge_cost=default_edge_cost,
            max_flow=max_flow,
            epsilon=epsilon,
            backend=backend,
        )
        if anchor_policy not in _ANCHOR_POLICIES:
            allowed = ", ".join(repr(value) for value in sorted(_ANCHOR_POLICIES))
            raise ValueError(f"anchor_policy must be one of {allowed}, got {anchor_policy!r}.")
        self.anchor_policy = anchor_policy
        self._biological_graph: BaseGraph | None = None
        self._reverse_edge_offset = 0
        self._anchor_mask = np.empty((0, 0), dtype=bool)
        self._anchor_downstream_eligible = np.empty((0, 0), dtype=bool)
        self._anchor_upstream_eligible = np.empty((0, 0), dtype=bool)
        self._measured_downstream_mask = np.empty((0, 0), dtype=bool)
        self._measured_upstream_mask = np.empty((0, 0), dtype=bool)
        self._edge_downstream_mask = np.empty((0, 0), dtype=bool)
        self._edge_upstream_mask = np.empty((0, 0), dtype=bool)
        self._anchor_inflow_edges: dict[Any, int] = {}
        self._measured_outflow_edges: dict[Any, int] = {}

    def build(
        self,
        pkn: BaseGraph,
        *,
        regulated_kinases,
        phosphosite_scores,
        edge_costs=None,
    ):
        """Build one bidirectional PHONEMeS condition."""
        phosphosite_scores = normalize_phonemes_score_mapping(
            phosphosite_scores,
            many=False,
        )
        return self.build_many(
            pkn,
            regulated_kinases={DEFAULT_CONDITION: regulated_kinases},
            phosphosite_scores={DEFAULT_CONDITION: phosphosite_scores},
            edge_costs=edge_costs,
        )

    def build_many(
        self,
        pkn: BaseGraph,
        *,
        regulated_kinases,
        phosphosite_scores,
        edge_costs=None,
    ):
        """Build named bidirectional PHONEMeS conditions."""
        phosphosite_scores = normalize_phonemes_score_mapping(
            phosphosite_scores,
            many=True,
        )
        data = _data_from_phonemes_inputs(
            pkn,
            primary_inputs=regulated_kinases,
            phosphosite_scores=phosphosite_scores,
            edge_costs=edge_costs,
            primary_argument="regulated_kinases",
            primary_role=_ROLE_REGULATED_KINASE,
            overlap_role=_ROLE_BOTH,
        )
        return self.build_from_data(pkn, data)

    def preprocess(self, graph: BaseGraph, data: Data):
        """Validate, prune exact directional paths, and build the flow layout."""
        self._validate_pkn(graph)
        if not data.samples:
            raise ValueError("BidirectionalPHONEMeS requires at least one condition.")

        condition_names = tuple(data.samples)
        if any(not isinstance(name, str) or not name for name in condition_names):
            raise ValueError("BidirectionalPHONEMeS condition names must be non-empty strings.")

        graph_vertex_index = {vertex: index for index, vertex in enumerate(graph.V)}
        anchors_by_condition: dict[str, set[Any]] = {}
        measured_by_condition: dict[str, dict[Any, float]] = {}
        condition_costs: list[dict[int, float]] = []

        for condition, sample in data.samples.items():
            anchors: set[Any] = set()
            measured: dict[Any, float] = {}
            costs: dict[int, float] = {}
            for feature in sample.features:
                if feature.mapping == "vertex":
                    if feature.id not in graph_vertex_index:
                        raise ValueError(
                            f"Unknown vertex {feature.id!r} in bidirectional PHONEMeS data for condition {condition!r}."
                        )
                    role = feature.data.get("role")
                    if role not in {
                        _ROLE_REGULATED_KINASE,
                        _ROLE_PHOSPHOSITE,
                        _ROLE_BOTH,
                    }:
                        raise ValueError(
                            f"Vertex feature {feature.id!r} for condition {condition!r} "
                            f"must have role {_ROLE_REGULATED_KINASE!r}, "
                            f"{_ROLE_PHOSPHOSITE!r}, or {_ROLE_BOTH!r}."
                        )
                    if role in {_ROLE_REGULATED_KINASE, _ROLE_BOTH}:
                        anchors.add(feature.id)
                    if role in {_ROLE_PHOSPHOSITE, _ROLE_BOTH}:
                        measured[feature.id] = validate_numeric(
                            feature.value,
                            argument="phosphosite_scores",
                            identifier=feature.id,
                            condition=condition,
                        )
                elif feature.mapping == "edge":
                    costs.update(
                        validate_edge_costs(
                            graph,
                            {feature.id: feature.value},
                            condition=condition,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported feature mapping {feature.mapping!r} in "
                        f"bidirectional PHONEMeS data for condition {condition!r}."
                    )
            if not anchors:
                raise ValueError(f"regulated_kinases for condition {condition!r} must not be empty.")
            if not measured:
                raise ValueError(f"phosphosite_scores for condition {condition!r} must not be empty.")
            anchors_by_condition[condition] = anchors
            measured_by_condition[condition] = measured
            condition_costs.append(costs)

        first_costs = condition_costs[0]
        for condition, costs in zip(condition_names[1:], condition_costs[1:]):
            if costs != first_costs:
                raise ValueError(
                    "BidirectionalPHONEMeS edge costs are global; edge features "
                    "must be identical in every condition "
                    f"(mismatch in condition {condition!r})."
                )

        downstream = prune_to_paths(
            graph,
            sources=anchors_by_condition,
            targets={condition: measured.keys() for condition, measured in measured_by_condition.items()},
        )
        upstream = prune_to_paths(
            graph,
            sources={condition: measured.keys() for condition, measured in measured_by_condition.items()},
            targets=anchors_by_condition,
        )

        retained_edge_indices = np.asarray(
            sorted(set(downstream.original_edge_indices.tolist()) | set(upstream.original_edge_indices.tolist())),
            dtype=int,
        )
        retained_vertices = set()
        for condition in condition_names:
            retained_vertices.update(downstream.vertices_by_condition[condition])
            retained_vertices.update(upstream.vertices_by_condition[condition])
        retained_vertex_indices = np.asarray(
            [index for index, vertex in enumerate(graph.V) if vertex in retained_vertices],
            dtype=int,
        )
        if retained_edge_indices.size == 0:
            raise ValueError("BidirectionalPHONEMeS connectivity pruning removed every PKN interaction.")

        ordered_vertices = [graph.V[index] for index in retained_vertex_indices]
        ordered_edges = retained_edge_indices.tolist()
        ordered_extract = getattr(graph, "_extract_subgraph_keep_order", None)
        if callable(ordered_extract):
            biological_graph = ordered_extract(
                vertices=ordered_vertices,
                edges=ordered_edges,
            )
        else:
            biological_graph = graph.extract_subgraph(
                vertices=ordered_vertices,
                edges=ordered_edges,
            )

        vertices = tuple(biological_graph.V)
        vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
        num_vertices = len(vertices)
        num_edges = biological_graph.num_edges
        num_conditions = len(condition_names)
        original_to_processed_edge = {
            int(original): processed for processed, original in enumerate(retained_edge_indices)
        }

        anchor_mask = np.zeros((num_vertices, num_conditions), dtype=bool)
        measured_mask = np.zeros((num_vertices, num_conditions), dtype=bool)
        measured_downstream = np.zeros_like(measured_mask)
        measured_upstream = np.zeros_like(measured_mask)
        node_scores = np.zeros((num_vertices, num_conditions), dtype=float)
        edge_downstream = np.zeros((num_edges, num_conditions), dtype=bool)
        edge_upstream = np.zeros((num_edges, num_conditions), dtype=bool)

        for condition_index, condition in enumerate(condition_names):
            down_vertices = downstream.vertices_by_condition[condition]
            up_vertices = upstream.vertices_by_condition[condition]
            for anchor in anchors_by_condition[condition]:
                if anchor in vertex_index:
                    anchor_mask[vertex_index[anchor], condition_index] = True
            for vertex, score in measured_by_condition[condition].items():
                if vertex not in vertex_index:
                    continue
                index = vertex_index[vertex]
                measured_mask[index, condition_index] = True
                measured_downstream[index, condition_index] = vertex in down_vertices
                measured_upstream[index, condition_index] = vertex in up_vertices
                node_scores[index, condition_index] = score
            for processed, original in enumerate(retained_edge_indices):
                source, target = graph.get_edge(int(original))
                source_vertex = next(iter(source))
                target_vertex = next(iter(target))
                edge_downstream[processed, condition_index] = (
                    source_vertex in down_vertices and target_vertex in down_vertices
                )
                edge_upstream[processed, condition_index] = (
                    source_vertex in up_vertices and target_vertex in up_vertices
                )

        source_index = np.asarray(
            [vertex_index[next(iter(source))] for source, _ in biological_graph.E],
            dtype=int,
        )
        target_index = np.asarray(
            [vertex_index[next(iter(target))] for _, target in biological_graph.E],
            dtype=int,
        )
        anchor_downstream = np.zeros_like(anchor_mask)
        anchor_upstream = np.zeros_like(anchor_mask)
        for condition_index in range(num_conditions):
            down_sources = np.zeros(num_vertices, dtype=bool)
            down_targets = np.zeros(num_vertices, dtype=bool)
            up_sources = np.zeros(num_vertices, dtype=bool)
            up_targets = np.zeros(num_vertices, dtype=bool)
            down_sources[source_index[edge_downstream[:, condition_index]]] = True
            down_targets[target_index[edge_downstream[:, condition_index]]] = True
            up_sources[source_index[edge_upstream[:, condition_index]]] = True
            up_targets[target_index[edge_upstream[:, condition_index]]] = True
            anchor_downstream[:, condition_index] = anchor_mask[:, condition_index] & down_sources
            anchor_upstream[:, condition_index] = anchor_mask[:, condition_index] & up_targets
            measured_downstream[:, condition_index] &= down_targets
            measured_upstream[:, condition_index] &= up_sources

        for condition_index, condition in enumerate(condition_names):
            for anchor in anchors_by_condition[condition]:
                index = vertex_index.get(anchor)
                has_downstream = index is not None and anchor_downstream[index, condition_index]
                has_upstream = index is not None and anchor_upstream[index, condition_index]
                if self.anchor_policy == "both" and not (has_downstream and has_upstream):
                    missing = []
                    if not has_upstream:
                        missing.append("upstream")
                    if not has_downstream:
                        missing.append("downstream")
                    raise ValueError(
                        f"Regulated kinase {anchor!r} in condition {condition!r} "
                        f"has no {' or '.join(missing)} path to a measured phosphosite."
                    )
                if self.anchor_policy == "downstream" and not has_downstream:
                    raise ValueError(
                        f"Regulated kinase {anchor!r} in condition {condition!r} "
                        "cannot reach any measured phosphosite downstream."
                    )
                if self.anchor_policy == "either" and not (has_downstream or has_upstream):
                    raise ValueError(
                        f"Regulated kinase {anchor!r} in condition {condition!r} "
                        "has no upstream or downstream path to a measured phosphosite."
                    )

        processed_costs = {
            original_to_processed_edge[edge]: cost
            for edge, cost in first_costs.items()
            if edge in original_to_processed_edge
        }
        processed_features = {}
        for condition_index, condition in enumerate(condition_names):
            features = []
            anchors = anchors_by_condition[condition]
            measured = measured_by_condition[condition]
            for vertex in vertices:
                is_anchor = vertex in anchors
                is_measured = vertex in measured and measured_mask[vertex_index[vertex], condition_index]
                if not is_anchor and not is_measured:
                    continue
                role = (
                    _ROLE_BOTH
                    if is_anchor and is_measured
                    else (_ROLE_REGULATED_KINASE if is_anchor else _ROLE_PHOSPHOSITE)
                )
                features.append(
                    {
                        "id": vertex,
                        "mapping": "vertex",
                        "role": role,
                        "value": measured[vertex] if is_measured else None,
                    }
                )
            features.extend({"id": edge, "mapping": "edge", "value": cost} for edge, cost in processed_costs.items())
            processed_features[condition] = features
        processed_data = data_from_features(processed_features)

        edge_cost_array = np.full(
            num_edges,
            self.default_edge_cost,
            dtype=float,
        )
        for edge, cost in processed_costs.items():
            edge_cost_array[edge] = cost

        flow_graph = biological_graph.copy()
        for source, target in biological_graph.E:
            flow_graph.add_edge(
                next(iter(target)),
                next(iter(source)),
                type=EdgeType.DIRECTED,
                auxiliary_direction="upstream",
            )
        anchor_union = anchor_mask.any(axis=1)
        measured_union = (measured_downstream | measured_upstream).any(axis=1)
        anchor_inflow_edges = {}
        measured_outflow_edges = {}
        for vertex, include in zip(vertices, anchor_union):
            if include:
                anchor_inflow_edges[vertex] = flow_graph.add_edge(
                    (),
                    vertex,
                    type=EdgeType.DIRECTED,
                    auxiliary_direction="anchor_inflow",
                )
        for vertex, include in zip(vertices, measured_union):
            if include:
                measured_outflow_edges[vertex] = flow_graph.add_edge(
                    vertex,
                    (),
                    type=EdgeType.DIRECTED,
                    auxiliary_direction="measurement_outflow",
                )

        flow_max = self.epsilon * max(1, num_edges) if self.max_flow is None else self.max_flow
        if flow_max < self.epsilon:
            raise ValueError(f"max_flow ({flow_max:g}) must be greater than or equal to epsilon ({self.epsilon:g}).")
        flow_lb = np.zeros((flow_graph.num_edges, 2 * num_conditions), dtype=float)
        flow_ub = np.zeros_like(flow_lb)
        flow_ub[:num_edges, :num_conditions] = edge_downstream.astype(float) * flow_max
        flow_ub[
            num_edges : 2 * num_edges,
            num_conditions:,
        ] = edge_upstream.astype(float) * flow_max
        for vertex, edge in anchor_inflow_edges.items():
            index = vertex_index[vertex]
            flow_ub[edge, :num_conditions] = anchor_downstream[index].astype(float) * flow_max
            flow_ub[edge, num_conditions:] = anchor_upstream[index].astype(float) * flow_max
        for vertex, edge in measured_outflow_edges.items():
            index = vertex_index[vertex]
            flow_ub[edge, :num_conditions] = measured_downstream[index].astype(float) * flow_max
            flow_ub[edge, num_conditions:] = measured_upstream[index].astype(float) * flow_max

        self._condition_names = condition_names
        self._biological_graph = biological_graph
        self._biological_num_edges = num_edges
        self._reverse_edge_offset = num_edges
        self._anchor_mask = anchor_mask
        self._anchor_downstream_eligible = anchor_downstream
        self._anchor_upstream_eligible = anchor_upstream
        self._measured_mask = measured_mask
        self._measured_downstream_mask = measured_downstream
        self._measured_upstream_mask = measured_upstream
        self._node_scores = node_scores
        self._edge_costs = edge_cost_array
        self._edge_downstream_mask = edge_downstream
        self._edge_upstream_mask = edge_upstream
        self._flow_max = flow_max
        self._flow_lb = flow_lb
        self._flow_ub = flow_ub
        self._anchor_inflow_edges = anchor_inflow_edges
        self._measured_outflow_edges = measured_outflow_edges
        return flow_graph, processed_data

    def get_flow_bounds(self, graph: BaseGraph, data: Data):
        """Return bounds for downstream and upstream flow columns."""
        return {
            "lb": self._flow_lb,
            "ub": self._flow_ub,
            "n_flows": 2 * len(self._condition_names),
            "shared_bounds": False,
        }

    def create_problem(self, graph: BaseGraph, data: Data):
        """Create the global vectorized bidirectional problem."""
        num_conditions = len(self._condition_names)
        num_edges = self._biological_num_edges
        flow_problem = self.backend.Flow(
            graph,
            lb=self._flow_lb,
            ub=self._flow_ub,
            n_flows=2 * num_conditions,
            shared_bounds=False,
            force_matrix=True,
        )
        flow = flow_problem.expr._flow
        flow_problem += self.backend.NonZeroIndicator(
            flow,
            tolerance=self.epsilon,
        )
        positive = flow_problem.expr._flow_ipos

        flow_downstream = flow[:num_edges, :num_conditions]
        flow_upstream = flow[
            self._reverse_edge_offset : self._reverse_edge_offset + num_edges,
            num_conditions:,
        ]
        edge_downstream = positive[:num_edges, :num_conditions]
        edge_upstream = positive[
            self._reverse_edge_offset : self._reverse_edge_offset + num_edges,
            num_conditions:,
        ]
        flow_problem.register("flow_downstream", flow_downstream)
        flow_problem.register("flow_upstream", flow_upstream)
        flow_problem.register("edge_selected_downstream", edge_downstream)
        flow_problem.register("edge_selected_upstream", edge_upstream)

        num_vertices = graph.num_vertices
        anchor_downstream = self.backend.Variable(
            "anchor_active_downstream",
            (num_vertices, num_conditions),
            vartype=VarType.BINARY,
        )
        anchor_upstream = self.backend.Variable(
            "anchor_active_upstream",
            (num_vertices, num_conditions),
            vartype=VarType.BINARY,
        )
        flow_problem.register("anchor_active_downstream", anchor_downstream)
        flow_problem.register("anchor_active_upstream", anchor_upstream)
        flow_problem += anchor_downstream <= self._anchor_downstream_eligible.astype(float)
        flow_problem += anchor_upstream <= self._anchor_upstream_eligible.astype(float)
        if self.anchor_policy == "both":
            flow_problem += anchor_downstream >= self._anchor_mask.astype(float)
            flow_problem += anchor_upstream >= self._anchor_mask.astype(float)
        elif self.anchor_policy == "downstream":
            flow_problem += anchor_downstream >= self._anchor_mask.astype(float)
        else:
            flow_problem += anchor_downstream + anchor_upstream >= self._anchor_mask.astype(float)

        vertices = tuple(graph.V)
        vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
        anchor_vertices = tuple(self._anchor_inflow_edges)
        anchor_rows = [self._anchor_inflow_edges[vertex] for vertex in anchor_vertices]
        anchor_indexes = [vertex_index[vertex] for vertex in anchor_vertices]
        downstream_inflow = flow[anchor_rows, :num_conditions]
        upstream_inflow = flow[anchor_rows, num_conditions:]
        downstream_active = anchor_downstream[anchor_indexes, :]
        upstream_active = anchor_upstream[anchor_indexes, :]
        flow_problem += downstream_inflow >= self.epsilon * downstream_active
        flow_problem += downstream_inflow <= self._flow_max * downstream_active
        flow_problem += upstream_inflow >= self.epsilon * upstream_active
        flow_problem += upstream_inflow <= self._flow_max * upstream_active

        internal_downstream = ~(self._anchor_mask | self._measured_downstream_mask)
        downstream_vertices = add_vertex_selection(
            self.backend,
            flow_problem,
            graph,
            edge_downstream,
            edge_indices=range(num_edges),
            force_selected=anchor_downstream,
            require_outgoing=internal_downstream | self._anchor_mask,
            require_incoming=internal_downstream | self._measured_downstream_mask,
            name="vertex_selected_downstream",
        )
        internal_upstream = ~(self._anchor_mask | self._measured_upstream_mask)
        upstream_vertices = add_vertex_selection(
            self.backend,
            flow_problem,
            graph,
            edge_upstream,
            edge_indices=range(num_edges),
            force_selected=anchor_upstream,
            require_outgoing=internal_upstream | self._anchor_mask,
            require_incoming=internal_upstream | self._measured_upstream_mask,
            name="vertex_selected_upstream",
            reverse=True,
        )
        edge_selected = _pairwise_or(
            self.backend,
            flow_problem,
            edge_downstream,
            edge_upstream,
            name="edge_selected",
        )
        vertex_selected = _pairwise_or(
            self.backend,
            flow_problem,
            downstream_vertices.selected,
            upstream_vertices.selected,
            name="vertex_selected",
        )
        edge_selected_any = add_condition_union(
            self.backend,
            flow_problem,
            edge_selected,
            name="edge_selected_any",
        )

        assert self._biological_graph is not None
        self.backend.Acyclic(
            self._biological_graph,
            flow_problem,
            indicator_positive_var_name="edge_selected",
            acyclic_var_name="_dag_layer",
        )
        flow_problem.register("dag_layer", flow_problem.expr._dag_layer)
        flow_problem.add_objective(
            vertex_selected.multiply(self._node_scores).sum(),
            name="phosphosite_scores",
        )
        flow_problem.add_objective(
            self._edge_costs @ edge_selected_any,
            name="edge_costs",
        )
        return flow_problem

    @staticmethod
    def name() -> str:
        """Return the method name."""
        return "Bidirectional PHONEMeS"

    @staticmethod
    def description() -> str:
        """Return a short method description."""
        return "Global upstream/downstream signaling-network inference around regulated kinase anchors"
