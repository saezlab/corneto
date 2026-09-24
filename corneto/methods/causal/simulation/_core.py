"""Core execution and validation for structural causal models."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from corneto.graph import Attr, BaseGraph, EdgeType

from ._mechanisms import _Constant, _Shifted
from ._noise import Zero, _rng, _sample_count

if TYPE_CHECKING:
    from corneto.data import Data


@dataclass(frozen=True)
class Node:
    """One scalar structural equation in an ``SCM``.

    ``parents`` fixes the column order passed to ``mechanism``. A mechanism
    receives arrays with shapes ``(n, len(parents))`` and ``(n,)`` and must
    return one numeric value per sample. A noise callable receives a NumPy
    generator and the sample count and must return an array with shape
    ``(n,)``.
    """

    parents: tuple[Any, ...]
    mechanism: Callable[[np.ndarray, np.ndarray], Any]
    noise: Callable[[np.random.Generator, int], Any] = field(default_factory=Zero)

    def __post_init__(self) -> None:
        if self.parents is None:
            parents = ()
        elif isinstance(self.parents, (str, bytes)):
            parents = (self.parents,)
        else:
            try:
                parents = tuple(self.parents)
            except TypeError as error:
                raise TypeError("parents must be an iterable of variable identifiers.") from error
        try:
            if len(set(parents)) != len(parents):
                raise ValueError("parents must not contain duplicates.")
        except TypeError as error:
            raise TypeError("Parent identifiers must be hashable.") from error
        if not callable(self.mechanism):
            raise TypeError("mechanism must be callable.")
        if not callable(self.noise):
            raise TypeError("noise must be callable.")
        object.__setattr__(self, "parents", parents)


@dataclass(frozen=True)
class _Intervention:
    kind: str
    value: float | None = None


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _vector(value: Any, n: int, description: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{description} must be a numeric array with shape ({n},).") from error
    if array.shape != (n,):
        raise ValueError(f"{description} must have shape ({n},), got {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{description} must contain real numeric values.")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{description} must contain only finite values.")
    return array


def _topological_order(nodes: Mapping[Any, Node], variables: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return a stable topological order and validate all parent references."""
    children = {variable: [] for variable in variables}
    indegree = {variable: 0 for variable in variables}
    for variable, node in nodes.items():
        for parent in node.parents:
            if parent not in nodes:
                raise ValueError(f"Node {variable!r} has missing parent {parent!r}.")
            children[parent].append(variable)
            indegree[variable] += 1

    ready = deque(variable for variable in variables if indegree[variable] == 0)
    order = []
    while ready:
        variable = ready.popleft()
        order.append(variable)
        for child in children[variable]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(variables):
        cyclic = tuple(variable for variable in variables if indegree[variable] > 0)
        raise ValueError(f"SCM nodes must form a directed acyclic graph; cycle involves {cyclic!r}.")
    return tuple(order)


def _graph_structure(
    graph: BaseGraph,
) -> tuple[tuple[Any, ...], dict[Any, tuple[Any, ...]], tuple[tuple[Any, Any], ...]]:
    """Extract and validate simple directed graph edges in vertex order."""
    if not isinstance(graph, BaseGraph):
        raise TypeError("graph must be a corneto.graph.BaseGraph instance.")
    variables = tuple(graph.V)
    try:
        variable_index = {variable: index for index, variable in enumerate(variables)}
    except TypeError as error:
        raise TypeError("Graph vertex identifiers must be hashable.") from error
    if len(variable_index) != len(variables):
        raise ValueError("Graph vertex identifiers must be unique.")

    parents: dict[Any, list[Any]] = {variable: [] for variable in variables}
    edges = []
    seen_edges = set()
    for edge_index in range(graph.num_edges):
        source, target = graph.get_edge(edge_index)
        if not source or not target:
            raise ValueError(f"Graph boundary edge {edge_index} is not supported by SCM simulation.")
        if len(source) != 1 or len(target) != 1:
            raise ValueError(f"Graph hyperedge {edge_index} is not supported by SCM simulation.")
        attributes = graph.get_attr_edge(edge_index)
        edge_type = attributes.get(Attr.EDGE_TYPE.value)
        if edge_type not in (EdgeType.DIRECTED, EdgeType.DIRECTED.value):
            raise ValueError(f"Graph edge {edge_index} is not directed.")
        source_vertex, target_vertex = next(iter(source)), next(iter(target))
        if source_vertex == target_vertex:
            raise ValueError(f"Graph self-loop at {source_vertex!r} is not supported by SCM simulation.")
        edge = (source_vertex, target_vertex)
        if edge in seen_edges:
            raise ValueError(f"Graph contains parallel edge {edge!r}; SCM simulation requires a simple graph.")
        seen_edges.add(edge)
        if source_vertex not in variable_index or target_vertex not in variable_index:
            raise ValueError(f"Graph edge {edge_index} references a vertex missing from graph.V.")
        parents[target_vertex].append(source_vertex)
        edges.append(edge)

    ordered_parents = {
        variable: tuple(sorted(parent_list, key=variable_index.__getitem__))
        for variable, parent_list in parents.items()
    }
    return variables, ordered_parents, tuple(edges)


class SCM:
    """Execute immutable scalar structural causal models over NumPy arrays.

    The variable order follows the input mapping. Evaluation uses a stable
    topological order, while each node receives parents in its declared order.
    ``draw_noise`` returns one vector per variable in a read-only mapping;
    passing those same vectors to another SCM with the same variables enables
    paired simulations across interventions.
    """

    __slots__ = ("_interventions", "_nodes", "_topological_order", "_variables")

    def __init__(
        self,
        nodes: Mapping[Any, Node],
        *,
        _interventions: Mapping[Any, _Intervention] | None = None,
    ) -> None:
        """Build a validated SCM from an insertion-ordered variable mapping."""
        if not isinstance(nodes, Mapping):
            raise TypeError("nodes must be a mapping from variable identifiers to Node objects.")
        copied = dict(nodes)
        if not copied:
            raise ValueError("SCM requires at least one node.")
        for variable, node in copied.items():
            try:
                hash(variable)
            except TypeError as error:
                raise TypeError(f"Variable identifier {variable!r} must be hashable.") from error
            if not isinstance(node, Node):
                raise TypeError(f"Node definition for {variable!r} must be a Node instance.")
        variables = tuple(copied)
        order = _topological_order(copied, variables)
        interventions = dict(_interventions or {})
        unknown = tuple(variable for variable in interventions if variable not in copied)
        if unknown:
            raise ValueError(f"Intervention metadata references unknown nodes: {unknown!r}.")
        object.__setattr__(self, "_nodes", MappingProxyType(copied))
        object.__setattr__(self, "_variables", variables)
        object.__setattr__(self, "_topological_order", order)
        object.__setattr__(self, "_interventions", MappingProxyType(interventions))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SCM instances are immutable; use do, shift, or replace to create a new model.")

    @property
    def nodes(self) -> Mapping[Any, Node]:
        """Read-only node definitions in variable order."""
        return self._nodes

    @property
    def variables(self) -> tuple[Any, ...]:
        """Variable identifiers in output-column order."""
        return self._variables

    @property
    def topological_order(self) -> tuple[Any, ...]:
        """Stable node execution order."""
        return self._topological_order

    def parents(self, variable: Any) -> tuple[Any, ...]:
        """Return the declared parent order for a variable."""
        try:
            return self._nodes[variable].parents
        except KeyError as error:
            raise KeyError(f"Unknown SCM variable {variable!r}.") from error

    @classmethod
    def from_graph(
        cls,
        graph: BaseGraph,
        mechanisms: Mapping[Any, Callable[[np.ndarray, np.ndarray], Any]],
        noises: Mapping[Any, Callable[[np.random.Generator, int], Any]]
        | Callable[[np.random.Generator, int], Any]
        | None = None,
    ) -> SCM:
        """Build an SCM from a CORNETO graph and per-node mechanisms.

        Graph vertices define output order and each node's parent order. Every
        vertex needs a mechanism. ``noises`` may be one callable used by all
        nodes, a partial node-to-callable mapping, or ``None`` for zero noise.
        """
        variables, parents, _ = _graph_structure(graph)
        if not isinstance(mechanisms, Mapping):
            raise TypeError("mechanisms must be a mapping from every graph vertex to a callable.")
        unknown = tuple(variable for variable in mechanisms if variable not in parents)
        missing = tuple(variable for variable in variables if variable not in mechanisms)
        if unknown or missing:
            raise ValueError(f"Mechanism keys must match graph vertices; missing={missing!r}, unknown={unknown!r}.")
        if noises is None:
            noise_by_variable = {}
        elif callable(noises):
            noise_by_variable = {variable: noises for variable in variables}
        elif isinstance(noises, Mapping):
            unknown_noises = tuple(variable for variable in noises if variable not in parents)
            if unknown_noises:
                raise ValueError(f"Noise mapping contains unknown graph vertices: {unknown_noises!r}.")
            noise_by_variable = dict(noises)
        else:
            raise TypeError("noises must be a callable, a node-to-callable mapping, or None.")
        nodes = {
            variable: Node(parents[variable], mechanisms[variable], noise_by_variable.get(variable, Zero()))
            for variable in variables
        }
        return cls(nodes)

    def draw_noise(self, n: int, rng: int | np.random.Generator | None = None) -> Mapping[Any, np.ndarray]:
        """Draw each node's exogenous noise in stable variable order."""
        n = _sample_count(n)
        generator = _rng(rng)
        draws = {}
        for variable in self._variables:
            node = self._nodes[variable]
            try:
                value = node.noise(generator, n)
            except Exception as error:
                raise ValueError(f"Noise callable for node {variable!r} failed: {error}") from error
            draws[variable] = _vector(value, n, f"Noise for node {variable!r}")
        return MappingProxyType(draws)

    def evaluate(self, noise: Mapping[Any, Any]) -> np.ndarray:
        """Evaluate all node equations for supplied exogenous noise vectors."""
        if not isinstance(noise, Mapping):
            raise TypeError("noise must be a mapping from every variable to a one-dimensional array.")
        missing = tuple(variable for variable in self._variables if variable not in noise)
        extra = tuple(variable for variable in noise if variable not in self._nodes)
        if missing or extra:
            raise ValueError(f"Noise keys must match SCM variables; missing={missing!r}, unknown={extra!r}.")
        n = None
        noise_vectors = {}
        for variable in self._variables:
            value = np.asarray(noise[variable])
            if value.ndim != 1:
                raise ValueError(f"Noise for node {variable!r} must be one-dimensional, got {value.shape}.")
            if n is None:
                n = value.shape[0]
            elif value.shape[0] != n:
                raise ValueError("All exogenous noise vectors must have the same sample count.")
            noise_vectors[variable] = _vector(value, n, f"Noise for node {variable!r}")

        values = {}
        for variable in self._topological_order:
            node = self._nodes[variable]
            parent_values = (
                np.column_stack([values[parent] for parent in node.parents])
                if node.parents
                else np.empty((n, 0), dtype=float)
            )
            try:
                output = node.mechanism(parent_values, noise_vectors[variable])
            except Exception as error:
                raise ValueError(f"Mechanism for node {variable!r} failed: {error}") from error
            values[variable] = _vector(output, n, f"Mechanism output for node {variable!r}")
        return np.column_stack([values[variable] for variable in self._variables])

    def sample(self, n: int, rng: int | np.random.Generator | None = None) -> np.ndarray:
        """Draw exogenous noise and return an ``(n, number_of_variables)`` array."""
        return self.evaluate(self.draw_noise(n, rng))

    def do(self, values: Mapping[Any, Real]) -> SCM:
        """Return a model with each named node replaced by a hard clamp."""
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping from variables to finite numeric constants.")
        unknown = tuple(variable for variable in values if variable not in self._nodes)
        if unknown:
            raise ValueError(f"Cannot intervene on unknown SCM variables: {unknown!r}.")
        if not values:
            return self
        nodes = dict(self._nodes)
        interventions = dict(self._interventions)
        for variable, value in values.items():
            value = _real(value, f"Hard intervention on {variable!r}")
            nodes[variable] = Node((), _Constant(value), Zero())
            interventions[variable] = _Intervention("hard", value)
        return SCM(nodes, _interventions=interventions)

    def shift(self, values: Mapping[Any, Real]) -> SCM:
        """Return a model with additive output shifts and unchanged parents."""
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping from variables to finite numeric shifts.")
        unknown = tuple(variable for variable in values if variable not in self._nodes)
        if unknown:
            raise ValueError(f"Cannot shift unknown SCM variables: {unknown!r}.")
        if not values:
            return self
        nodes = dict(self._nodes)
        interventions = dict(self._interventions)
        for variable, value in values.items():
            value = _real(value, f"Shift on {variable!r}")
            node = nodes[variable]
            previous = interventions.get(variable)
            if previous is None:
                mechanism = node.mechanism
                total_shift = value
                intervention = _Intervention("shift", value)
            elif previous.kind == "shift":
                mechanism = node.mechanism.mechanism
                total_shift = previous.value + value
                intervention = _Intervention("shift", total_shift)
            else:
                mechanism = node.mechanism
                total_shift = value
                intervention = _Intervention("replace")
            nodes[variable] = Node(node.parents, _Shifted(mechanism, total_shift), node.noise)
            interventions[variable] = intervention
        return SCM(nodes, _interventions=interventions)

    def replace(self, nodes: Mapping[Any, Node]) -> SCM:
        """Return a model with named node definitions replaced immutably."""
        if not isinstance(nodes, Mapping):
            raise TypeError("nodes must be a mapping from variables to Node definitions.")
        unknown = tuple(variable for variable in nodes if variable not in self._nodes)
        if unknown:
            raise ValueError(f"Cannot replace unknown SCM variables: {unknown!r}.")
        if not nodes:
            return self
        replacements = dict(self._nodes)
        interventions = dict(self._interventions)
        for variable, node in nodes.items():
            if not isinstance(node, Node):
                raise TypeError(f"Replacement for {variable!r} must be a Node instance.")
            replacements[variable] = node
            interventions[variable] = _Intervention("replace")
        return SCM(replacements, _interventions=interventions)

    def to_data(
        self,
        values: Any,
        *,
        sample_ids: Iterable[Any] | None = None,
        intervention_group: Any = None,
    ) -> Data:
        """Convert an ``(n, d)`` simulation matrix to CORNETO ``Data``.

        Hard interventions and known additive shifts are annotated using the
        feature metadata consumed by ``LinearDAGDiscovery``. General node
        replacements have no supported discovery annotation and are rejected.
        """
        replacements = tuple(variable for variable, item in self._interventions.items() if item.kind == "replace")
        if replacements:
            raise ValueError(
                "Cannot convert arbitrary node replacements to LinearDAGDiscovery data; "
                f"replace metadata is unsupported for {replacements!r}."
            )
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[1] != len(self._variables):
            raise ValueError(f"values must have shape (n, {len(self._variables)}), got {matrix.shape}.")
        if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(matrix.dtype, np.complexfloating):
            raise TypeError("values must contain real numeric values.")
        matrix = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("values must contain only finite numbers.")
        n = matrix.shape[0]
        if sample_ids is None:
            sample_ids = tuple(range(n))
        elif isinstance(sample_ids, (str, bytes)):
            sample_ids = (sample_ids,)
        else:
            sample_ids = tuple(sample_ids)
        if len(sample_ids) != n:
            raise ValueError(f"sample_ids must contain exactly {n} identifiers.")
        try:
            if len(set(sample_ids)) != n:
                raise ValueError("sample_ids must be unique.")
        except TypeError as error:
            raise TypeError("sample_ids must be hashable.") from error
        if intervention_group is not None:
            try:
                hash(intervention_group)
            except TypeError as error:
                raise TypeError("intervention_group must be hashable.") from error
            if isinstance(intervention_group, Real) and not isinstance(intervention_group, bool):
                if not np.isfinite(float(intervention_group)):
                    raise ValueError("Numeric intervention_group values must be finite.")

        samples = {}
        for row_index, sample_id in enumerate(sample_ids):
            features = {}
            for column_index, variable in enumerate(self._variables):
                feature = {"mapping": "vertex", "value": float(matrix[row_index, column_index])}
                intervention = self._interventions.get(variable)
                if intervention is not None:
                    feature["intervention"] = intervention.kind
                    if intervention.kind == "shift":
                        feature["shift"] = intervention.value
                    if intervention_group is not None:
                        feature["intervention_group"] = intervention_group
                features[variable] = feature
            samples[sample_id] = features
        from corneto.data import Data

        return Data.from_cdict(samples)

    def sample_data(
        self,
        n: int,
        rng: int | np.random.Generator | None = None,
        *,
        sample_ids: Iterable[Any] | None = None,
        intervention_group: Any = None,
    ) -> Data:
        """Sample the model and return CORNETO ``Data`` with intervention metadata."""
        return self.to_data(self.sample(n, rng), sample_ids=sample_ids, intervention_group=intervention_group)


__all__ = ["SCM", "Node"]
