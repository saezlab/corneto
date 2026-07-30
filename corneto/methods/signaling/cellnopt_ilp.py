"""Flow-based Boolean network inference in the style of CellNOpt."""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Real
from typing import Any, Optional

import numpy as np
from scipy.sparse import csr_matrix

from corneto._constants import VarType
from corneto.backend._base import Backend, ProblemDef
from corneto.data import Data
from corneto.graph import Attr, BaseGraph, EdgeType, Graph
from corneto.methods._base import FlowMethod
from corneto.methods._input_utils import (
    DEFAULT_CONDITION,
    data_from_features,
    require_mapping,
    validate_condition_keys,
)
from corneto.methods._network_utils import augment_with_boundaries

__all__ = ["BooleanReaction", "CellNOptILP"]

_AND_PATTERN = re.compile(r"^and[0-9]+$", re.IGNORECASE)
_ROLE_INPUT = "input"
_ROLE_OUTPUT = "output"
_ROLE_INPUT_OUTPUT = "input_output"


@dataclass(frozen=True)
class BooleanReaction:
    """A normalized Boolean reaction with one product.

    Positive literals must be active and negative literals must be inactive
    for the reaction to propagate in a condition. ``source_edges`` records the
    PKN edges from which the reaction was compiled.
    """

    positive_literals: tuple[Any, ...]
    negative_literals: tuple[Any, ...]
    product: Any
    source_edges: tuple[int, ...]

    @property
    def literals(self) -> tuple[Any, ...]:
        """Return positive and negative literals in deterministic order."""
        return self.positive_literals + self.negative_literals


@dataclass(frozen=True)
class _ReactionNetwork:
    graph: Graph
    reactions: tuple[BooleanReaction, ...]
    dependency_reactions: np.ndarray
    positive_literals: csr_matrix
    negative_literals: csr_matrix
    products: csr_matrix


def _is_and_vertex(vertex: Any) -> bool:
    return isinstance(vertex, str) and _AND_PATTERN.fullmatch(vertex) is not None


def _interaction(graph: BaseGraph, edge_index: int) -> int:
    value = graph.get_attr_edge(edge_index).get("interaction")
    if isinstance(value, bool) or not isinstance(value, Real) or value not in (-1, 1):
        raise ValueError(f"CellNOptILP requires interaction +1 or -1 on edge {edge_index}; got {value!r}.")
    return int(value)


def _compile_reactions(graph: BaseGraph) -> _ReactionNetwork:
    """Compile SIF-style dummy AND vertices into reaction-level dependencies."""
    if graph.num_vertices == 0 or graph.num_edges == 0:
        raise ValueError("CellNOptILP requires a non-empty directed PKN.")

    and_vertices = {vertex for vertex in graph.V if _is_and_vertex(vertex)}
    vertex_order = {vertex: index for index, vertex in enumerate(graph.V)}
    incoming_by_and = {vertex: [] for vertex in and_vertices}
    outgoing_by_and = {vertex: [] for vertex in and_vertices}

    for edge_index, ((source, target), attributes) in enumerate(zip(graph.E, graph.get_attr_edges())):
        if len(source) != 1 or len(target) != 1:
            raise ValueError(
                "CellNOptILP accepts simple directed PKN edges only; "
                f"edge {edge_index} has {len(source)} sources and {len(target)} targets."
            )
        if not attributes.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED):
            raise ValueError(f"CellNOptILP requires directed edges; edge {edge_index} is not directed.")
        _interaction(graph, edge_index)
        source_vertex = next(iter(source))
        target_vertex = next(iter(target))
        if source_vertex in and_vertices and target_vertex in and_vertices:
            raise ValueError(f"Nested dummy AND gates are not supported (edge {edge_index}).")
        if target_vertex in and_vertices:
            incoming_by_and[target_vertex].append(edge_index)
        if source_vertex in and_vertices:
            outgoing_by_and[source_vertex].append(edge_index)

    reactions: list[BooleanReaction] = []
    reaction_keys: set[tuple[frozenset, frozenset, Any]] = set()

    def ordered(vertices):
        return tuple(sorted(vertices, key=vertex_order.__getitem__))

    def add_reaction(positive, negative, product, source_edges):
        positive = set(positive)
        negative = set(negative)
        contradictory = positive & negative
        if contradictory:
            literal = min(contradictory, key=vertex_order.__getitem__)
            raise ValueError(f"Reaction producing {product!r} contains both {literal!r} and !{literal!r}.")
        key = (frozenset(positive), frozenset(negative), product)
        if key in reaction_keys:
            return
        reaction_keys.add(key)
        reactions.append(
            BooleanReaction(
                positive_literals=ordered(positive),
                negative_literals=ordered(negative),
                product=product,
                source_edges=tuple(source_edges),
            )
        )

    for edge_index, (source, target) in enumerate(graph.E):
        source_vertex = next(iter(source))
        target_vertex = next(iter(target))
        if source_vertex in and_vertices or target_vertex in and_vertices:
            continue
        sign = _interaction(graph, edge_index)
        add_reaction(
            [source_vertex] if sign > 0 else [],
            [source_vertex] if sign < 0 else [],
            target_vertex,
            [edge_index],
        )

    for gate in (vertex for vertex in graph.V if vertex in and_vertices):
        incoming = incoming_by_and[gate]
        outgoing = outgoing_by_and[gate]
        if len(incoming) < 2:
            raise ValueError(f"Dummy AND gate {gate!r} must have at least two incoming edges.")
        if not outgoing:
            raise ValueError(f"Dummy AND gate {gate!r} must have at least one outgoing edge.")
        positive = []
        negative = []
        for edge_index in incoming:
            source_vertex = next(iter(graph.get_edge(edge_index)[0]))
            if _interaction(graph, edge_index) > 0:
                positive.append(source_vertex)
            else:
                negative.append(source_vertex)
        for edge_index in outgoing:
            if _interaction(graph, edge_index) < 0:
                raise ValueError(
                    f"Dummy AND gate {gate!r} must activate its product; edge {edge_index} has interaction -1."
                )
            product = next(iter(graph.get_edge(edge_index)[1]))
            add_reaction(
                positive,
                negative,
                product,
                [*incoming, edge_index],
            )

    if not reactions:
        raise ValueError("CellNOptILP preprocessing produced no Boolean reactions.")

    dependency_graph = Graph()
    species = [vertex for vertex in graph.V if vertex not in and_vertices]
    for vertex in species:
        dependency_graph.add_vertex(vertex)

    dependency_reactions = []
    for reaction_index, reaction in enumerate(reactions):
        for literal in reaction.positive_literals:
            dependency_graph.add_edge(
                literal,
                reaction.product,
                interaction=1,
                reaction=reaction_index,
            )
            dependency_reactions.append(reaction_index)
        for literal in reaction.negative_literals:
            dependency_graph.add_edge(
                literal,
                reaction.product,
                interaction=-1,
                reaction=reaction_index,
            )
            dependency_reactions.append(reaction_index)

    vertex_index = {vertex: index for index, vertex in enumerate(dependency_graph.V)}
    num_reactions = len(reactions)
    positive_rows = []
    positive_columns = []
    negative_rows = []
    negative_columns = []
    product_rows = []
    product_columns = []
    for reaction_index, reaction in enumerate(reactions):
        positive_rows.extend([reaction_index] * len(reaction.positive_literals))
        positive_columns.extend(vertex_index[vertex] for vertex in reaction.positive_literals)
        negative_rows.extend([reaction_index] * len(reaction.negative_literals))
        negative_columns.extend(vertex_index[vertex] for vertex in reaction.negative_literals)
        product_rows.append(vertex_index[reaction.product])
        product_columns.append(reaction_index)

    literal_shape = (num_reactions, dependency_graph.num_vertices)
    product_shape = (dependency_graph.num_vertices, num_reactions)
    positive_literals = csr_matrix(
        (
            np.ones(len(positive_rows)),
            (positive_rows, positive_columns),
        ),
        shape=literal_shape,
    )
    negative_literals = csr_matrix(
        (
            np.ones(len(negative_rows)),
            (negative_rows, negative_columns),
        ),
        shape=literal_shape,
    )
    products = csr_matrix(
        (
            np.ones(num_reactions),
            (product_rows, product_columns),
        ),
        shape=product_shape,
    )
    return _ReactionNetwork(
        graph=dependency_graph,
        reactions=tuple(reactions),
        dependency_reactions=np.asarray(dependency_reactions, dtype=int),
        positive_literals=positive_literals,
        negative_literals=negative_literals,
        products=products,
    )


def _bounded_number(
    value: Any,
    *,
    argument: str,
    identifier: Any,
    condition: str,
    binary: bool,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        number = float(value)
    elif isinstance(value, Real):
        number = float(value)
    else:
        raise TypeError(f"{argument}[{identifier!r}] for condition {condition!r} must be numeric, got {value!r}.")
    if not np.isfinite(number):
        raise ValueError(f"{argument}[{identifier!r}] for condition {condition!r} must be finite.")
    allowed = number in (0, 1) if binary else 0 <= number <= 1
    if not allowed:
        domain = "0 or 1" if binary else "between 0 and 1"
        raise ValueError(f"{argument}[{identifier!r}] for condition {condition!r} must be {domain}; got {value!r}.")
    return number


def _cellnopt_data(
    graph: BaseGraph,
    *,
    inputs,
    measurements,
    inhibitors=None,
) -> Data:
    condition_names = validate_condition_keys(
        inputs=inputs,
        measurements=measurements,
        inhibitors=inhibitors,
    )
    if not condition_names:
        raise ValueError("CellNOptILP requires at least one named condition.")
    if inhibitors is None:
        inhibitors = {condition: {} for condition in condition_names}

    graph_vertices = set(graph.V)
    features_by_condition = {}
    for condition in condition_names:
        condition_inputs = require_mapping(
            inputs[condition],
            argument="inputs",
            condition=condition,
        )
        condition_measurements = require_mapping(
            measurements[condition],
            argument="measurements",
            condition=condition,
        )
        condition_inhibitors = require_mapping(
            inhibitors[condition],
            argument="inhibitors",
            condition=condition,
        )
        if not condition_inputs and not condition_inhibitors:
            raise ValueError(f"Condition {condition!r} must contain an input or active inhibitor.")
        if not condition_measurements:
            raise ValueError(f"measurements for condition {condition!r} must not be empty.")

        input_values = {}
        measurement_values = {}
        active_inhibitors = set()
        for vertex, value in condition_inputs.items():
            if vertex not in graph_vertices:
                raise ValueError(f"Unknown vertex {vertex!r} in inputs for condition {condition!r}.")
            input_values[vertex] = _bounded_number(
                value,
                argument="inputs",
                identifier=vertex,
                condition=condition,
                binary=True,
            )
        for vertex, value in condition_measurements.items():
            if vertex not in graph_vertices:
                raise ValueError(f"Unknown vertex {vertex!r} in measurements for condition {condition!r}.")
            measurement_values[vertex] = _bounded_number(
                value,
                argument="measurements",
                identifier=vertex,
                condition=condition,
                binary=False,
            )
        for vertex, value in condition_inhibitors.items():
            if vertex not in graph_vertices:
                raise ValueError(f"Unknown vertex {vertex!r} in inhibitors for condition {condition!r}.")
            inhibitor = _bounded_number(
                value,
                argument="inhibitors",
                identifier=vertex,
                condition=condition,
                binary=True,
            )
            if inhibitor:
                active_inhibitors.add(vertex)

        overlap = active_inhibitors & set(input_values)
        if overlap:
            vertex = next(vertex for vertex in graph.V if vertex in overlap)
            raise ValueError(
                f"Vertex {vertex!r} cannot be both an input and an active inhibitor in condition {condition!r}."
            )
        for vertex in active_inhibitors:
            input_values[vertex] = 0.0

        features = []
        included = set(input_values) | set(measurement_values)
        for vertex in (vertex for vertex in graph.V if vertex in included):
            is_input = vertex in input_values
            is_output = vertex in measurement_values
            role = _ROLE_INPUT_OUTPUT if is_input and is_output else (_ROLE_INPUT if is_input else _ROLE_OUTPUT)
            feature = {
                "id": vertex,
                "mapping": "vertex",
                "role": role,
                "value": (measurement_values[vertex] if is_output else input_values[vertex]),
            }
            if is_input:
                feature["input_value"] = input_values[vertex]
            if vertex in active_inhibitors:
                feature["intervention"] = "inhibitor"
            features.append(feature)
        features_by_condition[condition] = features
    return data_from_features(features_by_condition)


class CellNOptILP(FlowMethod):
    """Infer a shared acyclic Boolean model from multiple conditions.

    The method selects reactions globally and evaluates their Boolean truth in
    every condition. A single nonnegative flow has exactly the selected
    reaction dependencies as its internal support. Conservation, positive
    support, and acyclicity therefore require every selected dependency to lie
    on a path from a controlled input or inhibitor to a measured output.

    Dummy vertices named ``AND<number>`` are compiled into one reaction per
    product. All operands of such a reaction share one selection variable and
    are evaluated conjunctively.

    Args:
        lambda_reg: Penalty for every selected reaction.
        max_flow: Upper bound for structural flow. By default, the number of
            compiled dependency edges is used.
        epsilon: Minimum flow on every selected dependency edge.
        backend: Optimization backend.
    """

    def __init__(
        self,
        lambda_reg: float = 1e-3,
        max_flow: Optional[float] = None,
        epsilon: float = 1.0,
        backend: Optional[Backend] = None,
    ):
        if isinstance(lambda_reg, bool) or not isinstance(lambda_reg, Real):
            raise TypeError("lambda_reg must be a finite nonnegative number.")
        if not np.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError("lambda_reg must be a finite nonnegative number.")
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise TypeError("epsilon must be a finite positive number.")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number.")
        if max_flow is not None:
            if isinstance(max_flow, bool) or not isinstance(max_flow, Real):
                raise TypeError("max_flow must be a finite positive number.")
            if not np.isfinite(max_flow) or max_flow <= 0:
                raise ValueError("max_flow must be a finite positive number.")
            if max_flow < epsilon:
                raise ValueError("max_flow must be greater than or equal to epsilon.")

        super().__init__(
            lambda_reg=lambda_reg,
            reg_varname="reaction_selected",
            flow_lower_bound=0,
            flow_upper_bound=1,
            backend=backend,
        )
        self.max_flow = None if max_flow is None else float(max_flow)
        self.epsilon = float(epsilon)
        self.reactions: tuple[BooleanReaction, ...] = ()
        self._condition_names: tuple[str, ...] = ()
        self._biological_num_edges = 0
        self._flow_max = 0.0
        self._dependency_to_reaction = csr_matrix((0, 0))
        self._positive_literals = csr_matrix((0, 0))
        self._negative_literals = csr_matrix((0, 0))
        self._products = csr_matrix((0, 0))
        self._forced_mask = np.empty((0, 0), dtype=bool)
        self._forced_values = np.empty((0, 0), dtype=float)
        self._measurement_mask = np.empty((0, 0), dtype=bool)
        self._measurements = np.empty((0, 0), dtype=float)

    def build(
        self,
        pkn: BaseGraph,
        *,
        inputs,
        measurements,
        inhibitors=None,
    ) -> ProblemDef:
        """Build a single-condition CellNOpt problem."""
        return self.build_many(
            pkn,
            inputs={DEFAULT_CONDITION: inputs},
            measurements={DEFAULT_CONDITION: measurements},
            inhibitors=(None if inhibitors is None else {DEFAULT_CONDITION: inhibitors}),
        )

    def build_many(
        self,
        pkn: BaseGraph,
        *,
        inputs,
        measurements,
        inhibitors=None,
    ) -> ProblemDef:
        """Build a problem for multiple named experimental conditions."""
        data = _cellnopt_data(
            pkn,
            inputs=inputs,
            measurements=measurements,
            inhibitors=inhibitors,
        )
        return self.build_from_data(pkn, data)

    def preprocess(self, graph: BaseGraph, data: Data):
        """Compile reactions, validate condition data, and add flow boundaries."""
        network = _compile_reactions(graph)
        if not data.samples:
            raise ValueError("CellNOptILP requires at least one condition.")

        condition_names = tuple(data.samples)
        if any(not isinstance(name, str) or not name for name in condition_names):
            raise ValueError("CellNOptILP condition names must be non-empty strings.")

        vertex_index = {vertex: index for index, vertex in enumerate(network.graph.V)}
        num_vertices = network.graph.num_vertices
        num_conditions = len(condition_names)
        forced_mask = np.zeros((num_vertices, num_conditions), dtype=bool)
        forced_values = np.zeros((num_vertices, num_conditions), dtype=float)
        measurement_mask = np.zeros((num_vertices, num_conditions), dtype=bool)
        measurements = np.zeros((num_vertices, num_conditions), dtype=float)

        for condition_index, (condition, sample) in enumerate(data.samples.items()):
            condition_inputs = 0
            condition_measurements = 0
            for feature in sample.features:
                if feature.mapping != "vertex":
                    raise ValueError(
                        f"CellNOptILP only accepts vertex features; got mapping "
                        f"{feature.mapping!r} in condition {condition!r}."
                    )
                if feature.id not in vertex_index:
                    raise ValueError(
                        f"Unknown species {feature.id!r} in CellNOptILP data for "
                        f"condition {condition!r}; dummy AND vertices cannot be measured "
                        "or perturbed."
                    )
                role = feature.data.get("role")
                if role not in {
                    _ROLE_INPUT,
                    _ROLE_OUTPUT,
                    _ROLE_INPUT_OUTPUT,
                }:
                    raise ValueError(
                        f"Vertex feature {feature.id!r} for condition {condition!r} "
                        "must have role 'input', 'output', or 'input_output'."
                    )
                index = vertex_index[feature.id]
                if role in {_ROLE_INPUT, _ROLE_INPUT_OUTPUT}:
                    input_value = feature.data.get(
                        "input_value",
                        feature.value if role == _ROLE_INPUT else None,
                    )
                    forced_mask[index, condition_index] = True
                    forced_values[index, condition_index] = _bounded_number(
                        input_value,
                        argument="input_value",
                        identifier=feature.id,
                        condition=condition,
                        binary=True,
                    )
                    condition_inputs += 1
                if role in {_ROLE_OUTPUT, _ROLE_INPUT_OUTPUT}:
                    measurement_mask[index, condition_index] = True
                    measurements[index, condition_index] = _bounded_number(
                        feature.value,
                        argument="measurement",
                        identifier=feature.id,
                        condition=condition,
                        binary=False,
                    )
                    condition_measurements += 1
            if condition_inputs == 0:
                raise ValueError(f"Condition {condition!r} must contain at least one input.")
            if condition_measurements == 0:
                raise ValueError(f"Condition {condition!r} must contain at least one output.")

        input_union = forced_mask.any(axis=1)
        output_union = measurement_mask.any(axis=1)
        layout = augment_with_boundaries(
            network.graph,
            inflow_vertices=(vertex for vertex, include in zip(network.graph.V, input_union) if include),
            outflow_vertices=(vertex for vertex, include in zip(network.graph.V, output_union) if include),
        )
        flow_graph = layout.graph
        num_dependencies = network.graph.num_edges
        num_reactions = len(network.reactions)
        dependency_rows = np.arange(num_dependencies)
        dependency_to_reaction = csr_matrix(
            (
                np.ones(num_dependencies),
                (dependency_rows, network.dependency_reactions),
            ),
            shape=(flow_graph.num_edges, num_reactions),
        )

        default_max_flow = float(max(num_dependencies, 1))
        flow_max = default_max_flow if self.max_flow is None else self.max_flow
        if flow_max < self.epsilon:
            raise ValueError(f"max_flow ({flow_max:g}) must be greater than or equal to epsilon ({self.epsilon:g}).")

        self.reactions = network.reactions
        self._condition_names = condition_names
        self._biological_num_edges = num_dependencies
        self._flow_max = float(flow_max)
        self._dependency_to_reaction = dependency_to_reaction
        self._positive_literals = network.positive_literals
        self._negative_literals = network.negative_literals
        self._products = network.products
        self._forced_mask = forced_mask
        self._forced_values = forced_values
        self._measurement_mask = measurement_mask
        self._measurements = measurements
        return flow_graph, data.copy()

    def get_flow_bounds(self, graph: BaseGraph, data: Data):
        """Return bounds for the single shared structural flow."""
        return {
            "lb": 0,
            "ub": self._flow_max,
            "n_flows": 1,
            "shared_bounds": False,
        }

    def create_flow_based_problem(
        self,
        flow_problem: ProblemDef,
        graph: BaseGraph,
        data: Data,
    ) -> ProblemDef:
        """Add vectorized Boolean propagation and shared-flow selection."""
        problem = flow_problem
        num_vertices = graph.num_vertices
        num_reactions = len(self.reactions)
        num_conditions = len(self._condition_names)
        condition_ones = np.ones((1, num_conditions))

        reaction_selected = self.backend.Variable(
            "reaction_selected",
            (num_reactions,),
            vartype=VarType.BINARY,
        )
        reaction_active = self.backend.Variable(
            "reaction_active",
            (num_reactions, num_conditions),
            vartype=VarType.BINARY,
        )
        vertex_value = self.backend.Variable(
            "vertex_value",
            (num_vertices, num_conditions),
            vartype=VarType.BINARY,
        )

        dependency_selected_all = self.backend.Constant(self._dependency_to_reaction) @ reaction_selected
        dependency_selected = dependency_selected_all[: self._biological_num_edges]
        flow = problem.expr.flow
        biological_flow = flow[: self._biological_num_edges]
        # The shared flow and shared reaction selection have exactly the same
        # support on biological dependencies. Boundary flow remains free to
        # choose which controlled species and measurements connect that support.
        problem += biological_flow >= self.epsilon * dependency_selected
        problem += biological_flow <= self._flow_max * dependency_selected

        problem.register("_dependency_selected_all", dependency_selected_all)
        self.backend.Acyclic(
            graph,
            problem,
            indicator_positive_var_name="_dependency_selected_all",
        )

        positive_literals = self.backend.Constant(self._positive_literals)
        negative_literals = self.backend.Constant(self._negative_literals)
        products = self.backend.Constant(self._products)
        negative_count = np.asarray(self._negative_literals.sum(axis=1)).reshape(-1, 1)
        literal_count = np.asarray((self._positive_literals + self._negative_literals).sum(axis=1)).reshape(-1, 1)
        negative_count = negative_count @ condition_ones
        literal_count = literal_count @ condition_ones
        # S[r, c] counts true literals of reaction r in condition c. With k
        # literals, the three inequalities below impose
        # Z[r, c] = Y[r] AND (S[r, c] == k[r]) without a big-M constant.
        literal_satisfaction = positive_literals @ vertex_value + negative_count - negative_literals @ vertex_value
        selected_by_condition = reaction_selected.reshape((num_reactions, 1)) @ condition_ones

        problem += reaction_active <= selected_by_condition
        problem += reaction_active.multiply(literal_count) <= literal_satisfaction
        problem += reaction_active >= selected_by_condition + literal_satisfaction - literal_count

        producing_reactions = products @ reaction_active
        producer_count = np.asarray(self._products.sum(axis=1)).reshape(-1, 1)
        producer_count = producer_count @ condition_ones
        free_mask = (~self._forced_mask).astype(float)
        # A non-intervened species is the OR of its active producing reactions.
        # Interventions mask only this product relation; upstream reaction truth
        # remains intact when a product is experimentally forced to zero.
        problem += vertex_value.multiply(free_mask) <= producing_reactions.multiply(free_mask)
        problem += producing_reactions.multiply(free_mask) <= vertex_value.multiply(producer_count * free_mask)
        problem += vertex_value.multiply(self._forced_mask.astype(float)) == self._forced_values

        error_coefficients = self._measurement_mask.astype(float) * (1 - 2 * self._measurements)
        error_constant = float(np.sum(self._measurement_mask * self._measurements))
        # For binary x and measurement m in [0, 1],
        # |x - m| = m + (1 - 2m)x exactly.
        measurement_error = vertex_value.multiply(error_coefficients).sum() + error_constant
        problem.add_objective(
            measurement_error,
            name="measurement_error",
        )

        problem.register("dependency_selected", dependency_selected)
        problem.register("literal_satisfaction", literal_satisfaction)
        problem.register("dag_layer", problem.expr._dag_layer)
        return problem

    @staticmethod
    def name() -> str:
        """Return the method name."""
        return "CellNOptILP"

    @staticmethod
    def description() -> str:
        """Return a short method description."""
        return "Shared acyclic Boolean-reaction inference with structural flow connectivity"
