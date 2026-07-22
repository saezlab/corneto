"""Vectorized PHONEMeS network-inference method."""

from __future__ import annotations

from numbers import Real
from typing import Any, Optional

import numpy as np

from corneto._constants import VarType
from corneto.backend._base import Backend, ProblemDef
from corneto.data import Data
from corneto.graph import Attr, BaseGraph, EdgeType
from corneto.methods._base import FlowMethod
from corneto.methods._flow_utils import add_acyclic_flow_selection
from corneto.methods._input_utils import (
    DEFAULT_CONDITION,
    data_from_features,
    require_mapping,
    validate_condition_keys,
    validate_edge_costs,
    validate_numeric,
    validate_vertex_collection,
    validate_vertices,
)

_ROLE_PERTURBATION = "perturbation"
_ROLE_PHOSPHOSITE = "phosphosite"
_ROLE_BOTH = "perturbation_phosphosite"


class PHONEMeS(FlowMethod):
    """Infer acyclic signaling networks from phosphoproteomic scores.

    PHONEMeS uses one directed flow per condition. Perturbations inject flow,
    measured phosphosites can extract it, and explicit binary vertex variables
    attach signed phosphosite scores to every selected biological node.

    Args:
        default_edge_cost: Cost assigned to unspecified PKN edges.
        max_flow: Upper bound for every flow. Defaults to the number of PKN edges.
        epsilon: Smallest positive flow associated with a selected edge.
        backend: Optimization backend.
    """

    def __init__(
        self,
        default_edge_cost: float = 1e-5,
        max_flow: Optional[float] = None,
        epsilon: float = 1,
        backend: Optional[Backend] = None,
    ):
        super().__init__(disable_structured_sparsity=True, backend=backend)
        self.default_edge_cost = self._finite_number(default_edge_cost, argument="default_edge_cost")
        self.max_flow = None if max_flow is None else self._positive_number(max_flow, argument="max_flow")
        self.epsilon = self._positive_number(epsilon, argument="epsilon")
        if self.max_flow is not None and self.max_flow < self.epsilon:
            raise ValueError("max_flow must be greater than or equal to epsilon.")

        self._condition_names: tuple[str, ...] = ()
        self._original_num_edges = 0
        self._flow_max = 0.0
        self._target_inflow_edges: dict[Any, int] = {}
        self._phosphosite_outflow_edges: dict[Any, int] = {}
        self._target_mask = np.empty((0, 0), dtype=bool)
        self._measured_mask = np.empty((0, 0), dtype=bool)
        self._node_scores = np.empty((0, 0), dtype=float)
        self._edge_costs = np.empty((0,), dtype=float)
        self._flow_lb = np.empty((0, 0), dtype=float)
        self._flow_ub = np.empty((0, 0), dtype=float)

    @staticmethod
    def _finite_number(value: Any, *, argument: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{argument} must be a finite number, got {value!r}.")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{argument} must be finite, got {value!r}.")
        return number

    @classmethod
    def _positive_number(cls, value: Any, *, argument: str) -> float:
        number = cls._finite_number(value, argument=argument)
        if number <= 0:
            raise ValueError(f"{argument} must be greater than zero, got {value!r}.")
        return number

    def build(
        self,
        pkn: BaseGraph,
        *,
        perturbations,
        phosphosite_scores,
        edge_costs=None,
    ):
        """Build a single-condition PHONEMeS problem."""
        return self.build_many(
            pkn,
            perturbations={DEFAULT_CONDITION: perturbations},
            phosphosite_scores={DEFAULT_CONDITION: phosphosite_scores},
            edge_costs=edge_costs,
        )

    def build_many(
        self,
        pkn: BaseGraph,
        *,
        perturbations,
        phosphosite_scores,
        edge_costs=None,
    ):
        """Build a PHONEMeS problem for multiple named conditions."""
        conditions = validate_condition_keys(
            perturbations=perturbations,
            phosphosite_scores=phosphosite_scores,
        )
        cost_values = validate_edge_costs(
            pkn,
            {} if edge_costs is None else require_mapping(edge_costs, argument="edge_costs"),
            condition="all conditions",
        )

        features_by_condition = {}
        for condition in conditions:
            targets = validate_vertex_collection(
                pkn,
                perturbations[condition],
                argument="perturbations",
                condition=condition,
                required=True,
            )
            scores = validate_vertices(
                pkn,
                require_mapping(
                    phosphosite_scores[condition],
                    argument="phosphosite_scores",
                    condition=condition,
                ),
                argument="phosphosite_scores",
                condition=condition,
            )
            if not scores:
                raise ValueError(f"phosphosite_scores for condition {condition!r} must not be empty.")

            target_set = set(targets)
            measured_set = set(scores)
            vertex_order = list(dict.fromkeys([*targets, *scores]))
            features = []
            for vertex in vertex_order:
                is_target = vertex in target_set
                is_measured = vertex in measured_set
                role = (
                    _ROLE_BOTH
                    if is_target and is_measured
                    else (_ROLE_PERTURBATION if is_target else _ROLE_PHOSPHOSITE)
                )
                features.append(
                    {
                        "id": vertex,
                        "mapping": "vertex",
                        "role": role,
                        "value": scores[vertex] if is_measured else None,
                    }
                )
            features.extend({"id": edge, "mapping": "edge", "value": cost} for edge, cost in cost_values.items())
            features_by_condition[condition] = features

        return self.build_from_data(pkn, data_from_features(features_by_condition))

    def preprocess(self, graph: BaseGraph, data: Data):
        """Validate inputs and add condition-independent boundary edges."""
        self._validate_pkn(graph)
        if not data.samples:
            raise ValueError("PHONEMeS requires at least one condition.")

        condition_names = tuple(data.samples)
        if any(not isinstance(name, str) or not name for name in condition_names):
            raise ValueError("PHONEMeS condition names must be non-empty strings.")

        vertices = tuple(graph.V)
        vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
        num_vertices = len(vertices)
        num_conditions = len(condition_names)
        target_mask = np.zeros((num_vertices, num_conditions), dtype=bool)
        measured_mask = np.zeros((num_vertices, num_conditions), dtype=bool)
        node_scores = np.zeros((num_vertices, num_conditions), dtype=float)
        condition_costs = []

        for condition_index, (condition, sample) in enumerate(data.samples.items()):
            targets = set()
            measured = {}
            costs = {}
            for feature in sample.features:
                if feature.mapping == "vertex":
                    if feature.id not in vertex_index:
                        raise ValueError(f"Unknown vertex {feature.id!r} in PHONEMeS data for condition {condition!r}.")
                    role = feature.data.get("role")
                    if role not in {_ROLE_PERTURBATION, _ROLE_PHOSPHOSITE, _ROLE_BOTH}:
                        raise ValueError(
                            f"Vertex feature {feature.id!r} for condition {condition!r} must have role "
                            f"{_ROLE_PERTURBATION!r}, {_ROLE_PHOSPHOSITE!r}, or {_ROLE_BOTH!r}."
                        )
                    if role in {_ROLE_PERTURBATION, _ROLE_BOTH}:
                        targets.add(feature.id)
                    if role in {_ROLE_PHOSPHOSITE, _ROLE_BOTH}:
                        measured[feature.id] = validate_numeric(
                            feature.value,
                            argument="phosphosite_scores",
                            identifier=feature.id,
                            condition=condition,
                        )
                elif feature.mapping == "edge":
                    costs.update(validate_edge_costs(graph, {feature.id: feature.value}, condition=condition))
                else:
                    raise ValueError(
                        f"Unsupported feature mapping {feature.mapping!r} in PHONEMeS data for condition {condition!r}."
                    )

            if not targets:
                raise ValueError(f"PHONEMeS perturbations for condition {condition!r} must not be empty.")
            if not measured:
                raise ValueError(f"PHONEMeS phosphosite_scores for condition {condition!r} must not be empty.")
            self._validate_target_outgoing_edges(graph, targets, condition=condition)

            target_indexes = [vertex_index[vertex] for vertex in targets]
            measured_indexes = [vertex_index[vertex] for vertex in measured]
            target_mask[target_indexes, condition_index] = True
            measured_mask[measured_indexes, condition_index] = True
            node_scores[measured_indexes, condition_index] = [measured[vertices[index]] for index in measured_indexes]
            condition_costs.append(costs)

        first_costs = condition_costs[0]
        for condition, costs in zip(condition_names[1:], condition_costs[1:]):
            if costs != first_costs:
                raise ValueError(
                    "PHONEMeS edge costs are global; edge features must be identical in every condition "
                    f"(mismatch in condition {condition!r})."
                )

        self._condition_names = condition_names
        self._original_num_edges = graph.num_edges
        self._target_mask = target_mask
        self._measured_mask = measured_mask
        self._node_scores = node_scores
        self._edge_costs = np.full(graph.num_edges, self.default_edge_cost, dtype=float)
        for edge, cost in first_costs.items():
            self._edge_costs[edge] = cost

        default_max_flow = float(graph.num_edges)
        self._flow_max = default_max_flow if self.max_flow is None else self.max_flow
        if self._flow_max < self.epsilon:
            raise ValueError(
                f"max_flow ({self._flow_max:g}) must be greater than or equal to epsilon ({self.epsilon:g})."
            )

        flow_graph = graph.copy()
        target_union = target_mask.any(axis=1)
        measured_union = measured_mask.any(axis=1)
        self._target_inflow_edges = {}
        self._phosphosite_outflow_edges = {}
        for vertex, include in zip(vertices, target_union):
            if include:
                self._target_inflow_edges[vertex] = flow_graph.add_edge((), vertex, type=EdgeType.DIRECTED)
        for vertex, include in zip(vertices, measured_union):
            if include:
                self._phosphosite_outflow_edges[vertex] = flow_graph.add_edge(vertex, (), type=EdgeType.DIRECTED)

        flow_lb = np.zeros((flow_graph.num_edges, num_conditions), dtype=float)
        flow_ub = np.full((flow_graph.num_edges, num_conditions), self._flow_max, dtype=float)
        auxiliary_edges = [*self._target_inflow_edges.values(), *self._phosphosite_outflow_edges.values()]
        flow_ub[auxiliary_edges, :] = 0
        for vertex, edge in self._target_inflow_edges.items():
            active = target_mask[vertex_index[vertex], :]
            flow_lb[edge, active] = self.epsilon
            flow_ub[edge, active] = self._flow_max
        for vertex, edge in self._phosphosite_outflow_edges.items():
            active = measured_mask[vertex_index[vertex], :]
            flow_ub[edge, active] = self._flow_max
        self._flow_lb = flow_lb
        self._flow_ub = flow_ub
        return flow_graph, data

    @staticmethod
    def _validate_pkn(graph: BaseGraph) -> None:
        if graph.num_vertices == 0 or graph.num_edges == 0:
            raise ValueError("PHONEMeS requires a non-empty directed PKN.")
        for edge_index, ((source, target), attributes) in enumerate(zip(graph.E, graph.get_attr_edges())):
            if len(source) != 1 or len(target) != 1:
                raise ValueError(f"PHONEMeS does not support boundary edges or hyperedges; invalid edge {edge_index}.")
            if not attributes.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED):
                raise ValueError(f"PHONEMeS requires directed edges; edge {edge_index} is not directed.")

    @staticmethod
    def _validate_target_outgoing_edges(graph: BaseGraph, targets: set[Any], *, condition: str) -> None:
        sources_with_outgoing = {next(iter(source)) for source, target in graph.E if source and target}
        missing = [target for target in targets if target not in sources_with_outgoing]
        if missing:
            raise ValueError(
                f"Perturbation target {missing[0]!r} in condition {condition!r} has no outgoing PKN interaction."
            )

    def get_flow_bounds(self, graph: BaseGraph, data: Data):
        """Return condition-specific bounds for internal and boundary flows."""
        return {
            "lb": self._flow_lb,
            "ub": self._flow_ub,
            "n_flows": len(self._condition_names),
            "shared_bounds": False,
        }

    def create_problem(self, graph: BaseGraph, data: Data):
        """Create a consistently two-dimensional flow problem."""
        flow_params = self.get_flow_bounds(graph, data)
        flow_problem = self.backend.Flow(
            graph,
            lb=flow_params["lb"],
            ub=flow_params["ub"],
            n_flows=flow_params["n_flows"],
            shared_bounds=False,
            force_matrix=True,
        )
        return self.create_flow_based_problem(flow_problem, graph, data)

    def create_flow_based_problem(self, flow_problem: ProblemDef, graph: BaseGraph, data: Data):
        """Add vectorized PHONEMeS selection, role, and objective terms."""
        selected_augmented = add_acyclic_flow_selection(
            self.backend,
            flow_problem,
            graph,
            epsilon=self.epsilon,
        )
        edge_selected = selected_augmented[: self._original_num_edges, :]
        flow_problem.register("edge_selected", edge_selected)
        flow_problem.register("dag_layer", flow_problem.expr._dag_layer)

        num_vertices = graph.num_vertices
        num_conditions = len(self._condition_names)
        vertex_selected = self.backend.Variable(
            "vertex_selected",
            (num_vertices, num_conditions),
            vartype=VarType.BINARY,
        )

        incidence = graph.vertex_incidence_matrix(sparse=True)[:, : self._original_num_edges]
        outgoing_incidence = (incidence < 0).astype(float)
        incoming_incidence = (incidence > 0).astype(float)
        outgoing = self.backend.Constant(outgoing_incidence) @ edge_selected
        incoming = self.backend.Constant(incoming_incidence) @ edge_selected

        out_degree = np.asarray(outgoing_incidence.sum(axis=1)).reshape(-1, 1)
        in_degree = np.asarray(incoming_incidence.sum(axis=1)).reshape(-1, 1)
        out_degree = np.tile(out_degree, (1, num_conditions))
        in_degree = np.tile(in_degree, (1, num_conditions))
        internal_mask = ~(self._target_mask | self._measured_mask)
        require_outgoing = (internal_mask | self._target_mask).astype(float)
        require_incoming = (internal_mask | self._measured_mask).astype(float)

        flow_problem += outgoing <= vertex_selected.multiply(out_degree)
        flow_problem += incoming <= vertex_selected.multiply(in_degree)
        flow_problem += vertex_selected >= self._target_mask.astype(float)
        flow_problem += vertex_selected.multiply(require_outgoing) <= outgoing
        flow_problem += vertex_selected.multiply(require_incoming) <= incoming

        if num_conditions == 1:
            edge_selected_any = edge_selected[:, 0]
            flow_problem.register("edge_selected_any", edge_selected_any)
        else:
            edge_selected_any = self.backend.Variable(
                "edge_selected_any",
                (self._original_num_edges,),
                vartype=VarType.BINARY,
            )
            selected_count = edge_selected.sum(axis=1)
            flow_problem += edge_selected_any >= selected_count / num_conditions
            flow_problem += edge_selected_any <= selected_count

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
        return "PHONEMeS"

    @staticmethod
    def description() -> str:
        """Return a short method description."""
        return "Acyclic signaling-network inference from phosphoproteomic scores"

    @staticmethod
    def references():
        """Return PHONEMeS publication citation keys."""
        return ["terfve2015large", "gjerga2021phonemes"]
