"""Flow-supported linear DAG discovery over a prior knowledge network."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np
from scipy.sparse import csr_matrix

from corneto._constants import VarType
from corneto.backend._base import (
    Backend,
    ProblemDef,
    _sparse_vector_entry_repetition,
    _sparse_vector_replication,
)
from corneto.data import Data
from corneto.graph import Attr, BaseGraph, EdgeType
from corneto.methods._base import FlowMethod
from corneto.methods._network_utils import BoundaryFlowLayout, augment_with_boundaries
from corneto.methods._predictive import (
    require_expression_value,
    validate_solve_result,
)
from corneto.methods._predictive import (
    solve_options as normalize_solve_options,
)

__all__ = ["LinearDAGDiscovery"]


_MISSING = object()


@dataclass(frozen=True)
class _InterventionDesign:
    """Validated intervention semantics aligned to vertex-by-sample arrays.

    ``mapping`` maps estimated shift parameters to a Fortran-order flattened
    vertex-by-sample matrix.  Its rows are therefore indexed as
    ``vertex_index + num_vertices * sample_index``.  Keeping this indexing in
    one object avoids duplicating the grouping and shape bookkeeping in the
    optimization model.
    """

    hard: np.ndarray
    shift: np.ndarray
    groups: np.ndarray
    known_shift: np.ndarray
    mapping: csr_matrix
    effect_keys: tuple[tuple[Any, Any], ...]

    @staticmethod
    def _validate_group(group: Any) -> None:
        if group is _MISSING or group is None:
            raise ValueError("Estimated shift interventions require a non-null intervention group.")
        if isinstance(group, Real) and not isinstance(group, bool) and not np.isfinite(float(group)):
            raise ValueError("Estimated shift intervention groups must be finite when numeric.")
        try:
            hash(group)
        except TypeError as error:
            raise TypeError("Intervention groups must be hashable.") from error

    @classmethod
    def build(
        cls,
        vertex_keys: tuple[Any, ...],
        num_vertices: int,
        num_samples: int,
        annotations: Mapping[tuple[int, int], tuple[str, Any, Any]],
    ) -> "_InterventionDesign":
        hard = np.zeros((num_vertices, num_samples), dtype=bool)
        shift = np.zeros((num_vertices, num_samples), dtype=bool)
        groups = np.full((num_vertices, num_samples), _MISSING, dtype=object)
        known_shift = np.zeros((num_vertices, num_samples), dtype=float)
        effect_indices: dict[tuple[int, Any], int] = {}
        effect_keys: list[tuple[Any, Any]] = []
        mapping_rows: list[int] = []
        mapping_columns: list[int] = []

        for (vertex_index, sample_index), (kind, shift_value, group) in annotations.items():
            if kind != "none" and group is not _MISSING:
                groups[vertex_index, sample_index] = group
            if kind == "hard":
                hard[vertex_index, sample_index] = True
                continue
            if kind != "shift":
                continue

            shift[vertex_index, sample_index] = True
            if shift_value is not None:
                known_shift[vertex_index, sample_index] = float(shift_value)
                continue
            cls._validate_group(group)
            effect_key = (vertex_index, group)
            effect_index = effect_indices.get(effect_key)
            if effect_index is None:
                effect_index = len(effect_keys)
                effect_indices[effect_key] = effect_index
                effect_keys.append((vertex_keys[vertex_index], group))
            mapping_rows.append(vertex_index + num_vertices * sample_index)
            mapping_columns.append(effect_index)

        mapping = csr_matrix(
            (
                np.ones(len(mapping_rows), dtype=float),
                (mapping_rows, mapping_columns),
            ),
            shape=(num_vertices * num_samples, len(effect_keys)),
        )
        return cls(
            hard=hard,
            shift=shift,
            groups=groups,
            known_shift=known_shift,
            mapping=mapping,
            effect_keys=tuple(effect_keys),
        )


@dataclass(frozen=True)
class _EvidenceUnit:
    """One distinct intervention unit contributing to a flow signature."""

    source: Any
    intervention_group: Any
    sample_indices: tuple[int, ...]
    explicit_group: bool


@dataclass(frozen=True)
class _Commodity:
    """Immutable structural flow column and its evidence units.

    Several intervention groups can share one structural column.  In that
    case ``evidence_units`` retains the distinct groups for coverage and
    unexplained-commodity weighting, while the flow itself is represented
    only once.
    """

    source: Any
    source_edge: int
    sink_edges: tuple[int, ...]
    hard_blocked_edges: tuple[int, ...]
    evidence_units: tuple[_EvidenceUnit, ...]

    @property
    def group_count(self) -> int:
        """Number of distinct intervention evidence units represented."""
        return len(self.evidence_units)


class LinearDAGDiscovery(FlowMethod):
    """Discover a sparse linear DAG inside a directed prior network.

    A linear structural equation is fitted for every vertex. One nonnegative
    flow commodity is created for each unique intervention-to-measurement
    structural signature. Repeated cells and explicitly grouped intervention
    evidence therefore share a flow column when their source, eligible sinks,
    and hard-intervention edge blocks agree. Every selected biological edge
    must be used by at least one commodity, so the learned DAG is structurally
    supported by intervention-to-measurement paths without requiring every
    intervention to be explained.

    Missing outcomes are excluded from the loss. If a candidate parent is
    missing in a sample, the corresponding child equation is also excluded;
    this conservative complete-equation rule prevents missing predictors from
    being silently treated as zero while preserving a linear formulation.

    Feature metadata may declare ``intervention="shift"`` for an additive
    soft intervention. Its structural equation remains in the loss and its
    measured value remains a flow source. Provide ``shift`` (or
    ``intervention_shift``) for a known effect, or provide
    ``intervention_group`` to estimate one bounded effect shared by its
    replicates. The legacy ``intervened=True`` metadata remains a hard
    intervention.

    Args:
        lambda_edges: Penalty for every selected prior edge.
        coefficient_bound: Absolute bound on raw linear edge coefficients.
        fit_intercept: Whether to fit one intercept per vertex.
        intercept_bound: Absolute bound on vertex intercepts.
        max_parents: Optional global or vertex-specific parent limit.
        loss: ``"absolute"`` for a MILP or ``"squared"`` for a MIQP.
        standardize_loss: Divide residuals by the observed response standard
            deviation of each vertex. Coefficients remain in the original
            measurement units.
        vertex_weights: Optional positive loss weight per PKN vertex.
        sample_weights: Optional positive loss weight per sample.
        enforce_signs: Constrain coefficients using numeric PKN interactions
            ``+1`` and ``-1`` when present.
        intervention_type_key: Feature metadata key containing ``"none"``,
            ``"hard"``, or ``"shift"``. A boolean ``intervened=True`` under
            ``intervention_key`` remains a hard intervention for compatibility.
        intervention_shift_key: Feature metadata key containing a known
            additive shift. If it is absent for a shift intervention, the
            shift is estimated and shared by ``intervention_group_key``.
        intervention_group_key: Feature metadata key identifying an explicit
            intervention evidence group. The target vertex is always part of
            the group key, so one group cannot couple different sources. For
            estimated shifts, the same key also shares one bounded effect
            parameter across its replicates.
        intervention_shift_bound: Absolute bound for each estimated shift.
        lambda_intervention_shifts: Optional L1 penalty on estimated shifts.
        coefficient_support: ``"exact"`` requires every selected edge to have
            a fitted coefficient with magnitude at least ``min_abs_coefficient``;
            ``"structural"`` permits selected connector edges with zero
            coefficients.
        min_abs_coefficient: Minimum absolute fitted coefficient for selected
            edges in ``"exact"`` mode, in normalized coefficient units. The
            normalized coefficient for ``u -> v`` is
            ``beta[u -> v] * scale[u] / scale[v]``.
            The default ``0.25`` is a modeling assumption about normalized
            direct effects, not a solver tolerance or a guarantee of a
            nonzero total intervention effect.
        interaction_attribute: Edge attribute containing PKN interaction signs.
        intervention_key: Feature metadata key marking legacy hard
            interventions.
        flow_capacity: Upper bound on every commodity flow. By default, the
            number of PKN edges multiplied by ``flow_epsilon`` is used.
        flow_epsilon: Minimum positive flow on every used biological edge.
        min_commodity_coverage: Minimum fraction of distinct intervention
            evidence units that must have a selected-edge path to a measured,
            non-intervened response. Repeated cells in one explicit group count
            once, while separate groups sharing one flow signature each count
            once. If the unique flows represent ``K`` evidence units in total,
            a value ``f`` requires at least ``ceil(f * K)`` of those units to
            be connected. This does not measure prediction accuracy. Zero does
            not force any intervention to be connected.
        lambda_unexplained: Optional objective penalty for every intervention
            evidence unit without such a structural path. Unlike
            ``min_commodity_coverage``, this is a soft preference rather than
            a minimum feasibility requirement.
        backend: Optimization backend.
    """

    def __init__(
        self,
        lambda_edges: float = 1e-2,
        coefficient_bound: float = 5.0,
        fit_intercept: bool = True,
        intercept_bound: float = 10.0,
        max_parents: Optional[int | dict[Any, int]] = None,
        loss: str = "absolute",
        standardize_loss: bool = True,
        vertex_weights: Optional[Mapping[Any, float]] = None,
        sample_weights: Optional[Mapping[Any, float]] = None,
        enforce_signs: bool = False,
        interaction_attribute: str = "interaction",
        intervention_key: str = "intervened",
        flow_capacity: Optional[float] = None,
        flow_epsilon: float = 1.0,
        min_commodity_coverage: float = 0.0,
        lambda_unexplained: float = 0.0,
        backend: Optional[Backend] = None,
        coefficient_support: str = "exact",
        min_abs_coefficient: float = 0.25,
        *,
        intervention_type_key: str = "intervention",
        intervention_shift_key: str = "shift",
        intervention_group_key: str = "intervention_group",
        intervention_shift_bound: float = 10.0,
        lambda_intervention_shifts: float = 0.0,
    ):
        self._validate_nonnegative(lambda_edges, "lambda_edges")
        self._validate_positive(coefficient_bound, "coefficient_bound")
        self._validate_positive(intercept_bound, "intercept_bound")
        self._validate_positive(intervention_shift_bound, "intervention_shift_bound")
        self._validate_positive(flow_epsilon, "flow_epsilon")
        if coefficient_support not in {"exact", "structural"}:
            raise ValueError("coefficient_support must be 'exact' or 'structural'.")
        self._validate_positive(min_abs_coefficient, "min_abs_coefficient")
        if flow_capacity is not None:
            self._validate_positive(flow_capacity, "flow_capacity")
            if float(flow_capacity) < float(flow_epsilon):
                raise ValueError("flow_capacity must be greater than or equal to flow_epsilon.")
        if loss not in {"absolute", "squared"}:
            raise ValueError("loss must be 'absolute' or 'squared'.")
        if not isinstance(standardize_loss, bool):
            raise TypeError("standardize_loss must be boolean.")
        if vertex_weights is not None and not isinstance(vertex_weights, Mapping):
            raise TypeError("vertex_weights must be a mapping or None.")
        if sample_weights is not None and not isinstance(sample_weights, Mapping):
            raise TypeError("sample_weights must be a mapping or None.")
        if isinstance(max_parents, int) and not isinstance(max_parents, bool):
            if max_parents < 0:
                raise ValueError("max_parents must be nonnegative.")
        elif max_parents is not None:
            if not isinstance(max_parents, dict):
                raise TypeError("max_parents must be an integer, mapping, or None.")
            for vertex, value in max_parents.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"max_parents[{vertex!r}] must be a nonnegative integer.")
        if isinstance(min_commodity_coverage, bool) or not isinstance(min_commodity_coverage, Real):
            raise TypeError("min_commodity_coverage must be a finite number in [0, 1].")
        if not np.isfinite(min_commodity_coverage) or not 0 <= min_commodity_coverage <= 1:
            raise ValueError("min_commodity_coverage must be in [0, 1].")
        self._validate_nonnegative(lambda_unexplained, "lambda_unexplained")
        self._validate_nonnegative(lambda_intervention_shifts, "lambda_intervention_shifts")

        super().__init__(
            flow_lower_bound=0,
            flow_upper_bound=1,
            num_flows=1,
            shared_flow_bounds=False,
            lambda_reg=0,
            disable_structured_sparsity=True,
            backend=backend,
        )
        self.lambda_edges = float(lambda_edges)
        self.coefficient_bound = float(coefficient_bound)
        self.fit_intercept = fit_intercept
        self.intercept_bound = float(intercept_bound)
        self.max_parents = max_parents
        self.loss = loss
        self.standardize_loss = standardize_loss
        self.vertex_weights = dict(vertex_weights or {})
        self.sample_weights = dict(sample_weights or {})
        self.enforce_signs = enforce_signs
        self.coefficient_support = coefficient_support
        self.min_abs_coefficient = float(min_abs_coefficient)
        self.interaction_attribute = interaction_attribute
        self.intervention_key = intervention_key
        self.intervention_type_key = intervention_type_key
        self.intervention_shift_key = intervention_shift_key
        self.intervention_group_key = intervention_group_key
        self.intervention_shift_bound = float(intervention_shift_bound)
        self.lambda_intervention_shifts = float(lambda_intervention_shifts)
        self.flow_capacity = None if flow_capacity is None else float(flow_capacity)
        self.flow_epsilon = float(flow_epsilon)
        self.min_commodity_coverage = float(min_commodity_coverage)
        self.lambda_unexplained = float(lambda_unexplained)

        self._original_graph: Optional[BaseGraph] = None
        self._layout: Optional[BoundaryFlowLayout] = None
        self._vertex_index: dict[Any, int] = {}
        self._sample_names: tuple[Any, ...] = ()
        self._values = np.empty((0, 0))
        self._observed = np.empty((0, 0), dtype=bool)
        self._valid_residual = np.empty((0, 0), dtype=bool)
        self._loss_scales = np.empty((0,))
        self._coefficient_scales = np.empty((0,))
        self._observation_weights = np.empty((0, 0))
        self._intervened = np.empty((0, 0), dtype=bool)
        self._hard_intervened = np.empty((0, 0), dtype=bool)
        self._soft_intervened = np.empty((0, 0), dtype=bool)
        self._intervention_design: Optional[_InterventionDesign] = None
        self._edge_sources = np.empty((0,), dtype=int)
        self._edge_targets = np.empty((0,), dtype=int)
        self._commodities: tuple[_Commodity, ...] = ()
        self._flow_lb = np.empty((0, 0))
        self._flow_ub = np.empty((0, 0))
        self._resolved_flow_capacity = 0.0
        # ``fit`` records the backend result for callers who use the
        # convenience API.  Manual ``build``/``problem.solve`` remains fully
        # supported; prediction checks expression values directly in that
        # case, because ProblemDef.solve cannot notify its owning method.
        self.solve_result = None

    @property
    def _commodity_samples(self) -> list[int]:
        """Compatibility view of the first sample supporting each flow."""
        return [unit.sample_indices[0] for commodity in self._commodities for unit in commodity.evidence_units[:1]]

    @property
    def _commodity_sources(self) -> list[Any]:
        """Compatibility view of structural flow sources."""
        return [commodity.source for commodity in self._commodities]

    @property
    def _commodity_source_edges(self) -> list[int]:
        """Compatibility view of structural flow source boundary edges."""
        return [commodity.source_edge for commodity in self._commodities]

    @property
    def _commodity_sink_edges(self) -> list[list[int]]:
        """Compatibility view of structural flow sink boundary edges."""
        return [list(commodity.sink_edges) for commodity in self._commodities]

    @staticmethod
    def _validate_positive(value: Any, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite positive number.")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number.")

    @staticmethod
    def _validate_nonnegative(value: Any, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite nonnegative number.")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number.")

    @classmethod
    def _resolve_positive_weights(
        cls,
        keys: tuple[Any, ...],
        supplied: Mapping[Any, float],
        name: str,
    ) -> np.ndarray:
        unknown = set(supplied).difference(keys)
        if unknown:
            raise ValueError(f"{name} contains unknown keys: {sorted(unknown, key=repr)!r}.")
        weights = np.ones((len(keys),), dtype=float)
        index = {key: position for position, key in enumerate(keys)}
        for key, value in supplied.items():
            cls._validate_positive(value, f"{name}[{key!r}]")
            weights[index[key]] = float(value)
        return weights

    def build(self, pkn: BaseGraph, data: Data) -> ProblemDef:
        """Build a discovery problem from vertex measurements and interventions."""
        if not isinstance(data, Data):
            raise TypeError("data must be a corneto.data.Data object.")
        self.solve_result = None
        return self.build_from_data(pkn, data)

    def fit(
        self,
        pkn: BaseGraph,
        data: Data,
        *,
        solve_options: Optional[Mapping[str, Any]] = None,
        **solver_options: Any,
    ) -> "LinearDAGDiscovery":
        """Build and solve the model, returning this reusable estimator.

        ``fit`` is intentionally a thin convenience wrapper.  Use
        ``solve_options`` (or solver keyword arguments) only for backend
        options; scientific inputs remain the ``pkn`` and ``data`` arguments.
        The constructed :attr:`problem` is retained, and the backend result is
        available as :attr:`solve_result`.  A time-limited incumbent is
        accepted when all expressions needed by prediction have finite values.
        """
        options = normalize_solve_options(solve_options, solver_options)
        problem = self.build(pkn, data)
        result = problem.solve(**options)
        required = ["edge_selected", "edge_coefficient"]
        if self._intervention_design is not None and self._intervention_design.effect_keys:
            required.append("intervention_shift_parameters")
        validate_solve_result(result, problem.expr, tuple(required), self.name())
        # ``intercept`` is a symbol when fitting is enabled and a registered
        # constant otherwise; both are checked before prediction is allowed.
        if self.fit_intercept:
            require_expression_value(problem, "intercept", self.name())
        self.solve_result = result
        return self

    def _validate_graph(self, graph: BaseGraph) -> tuple[np.ndarray, np.ndarray]:
        if graph.num_vertices == 0 or graph.num_edges == 0:
            raise ValueError("LinearDAGDiscovery requires a non-empty directed PKN.")
        vertex_index = {vertex: index for index, vertex in enumerate(graph.V)}
        sources = []
        targets = []
        for edge_index, (source, target) in enumerate(graph.E):
            if len(source) != 1 or len(target) != 1:
                raise ValueError(
                    "LinearDAGDiscovery accepts simple directed PKN edges only; "
                    f"edge {edge_index} has {len(source)} sources and {len(target)} targets."
                )
            attributes = graph.get_attr_edge(edge_index)
            if not attributes.has_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED):
                raise ValueError(f"LinearDAGDiscovery requires directed edges; edge {edge_index} is not directed.")
            sources.append(vertex_index[next(iter(source))])
            targets.append(vertex_index[next(iter(target))])
        return np.asarray(sources, dtype=int), np.asarray(targets, dtype=int)

    @staticmethod
    def _normalize_intervention_type(raw: Any, source: str) -> str:
        if raw is None:
            return "none"
        if isinstance(raw, (bool, np.bool_)):
            return "hard" if bool(raw) else "none"
        if not isinstance(raw, str):
            raise TypeError(f"{source} must be boolean or one of 'none', 'hard', and 'shift'.")
        normalized = raw.strip().lower()
        if normalized == "soft":
            normalized = "shift"
        if normalized not in {"none", "hard", "shift"}:
            raise ValueError(f"{source} must be one of 'none', 'hard', and 'shift'; got {raw!r}.")
        return normalized

    def _get_intervention_type(self, feature: Any, sample_name: Any) -> str:
        explicit_values = []
        for key in (self.intervention_type_key, "intervention_type"):
            if key not in explicit_values and key in feature.data:
                explicit_values.append(key)
        explicit = _MISSING
        if explicit_values:
            explicit = self._normalize_intervention_type(
                feature.data[explicit_values[0]],
                f"{explicit_values[0]!r} for vertex {feature.id!r} in sample {sample_name!r}",
            )
            for key in explicit_values[1:]:
                other = self._normalize_intervention_type(
                    feature.data[key],
                    f"{key!r} for vertex {feature.id!r} in sample {sample_name!r}",
                )
                if other != explicit:
                    raise ValueError(
                        f"Conflicting intervention types for vertex {feature.id!r} in sample {sample_name!r}."
                    )

        legacy = _MISSING
        if self.intervention_key in feature.data:
            legacy = self._normalize_intervention_type(
                feature.data[self.intervention_key],
                f"{self.intervention_key!r} for vertex {feature.id!r} in sample {sample_name!r}",
            )
        if explicit is _MISSING:
            return "none" if legacy is _MISSING else legacy
        if legacy is not _MISSING and legacy != "none" and legacy != explicit:
            raise ValueError(f"Conflicting intervention types for vertex {feature.id!r} in sample {sample_name!r}.")
        return explicit

    def _get_shift_value(self, feature: Any, sample_name: Any) -> Any:
        keys = tuple(dict.fromkeys((self.intervention_shift_key, "shift", "shift_value", "intervention_shift")))
        present = [key for key in keys if key in feature.data]
        if not present:
            return None
        values = [feature.data[key] for key in present]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Conflicting shift values for vertex {feature.id!r} in sample {sample_name!r}.")
        value = values[0]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"Intervention shift for vertex {feature.id!r} in sample {sample_name!r} "
                "must be numeric or omitted for estimation."
            )
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Intervention shift for vertex {feature.id!r} in sample {sample_name!r} must be finite.")
        return value

    def _get_intervention_group(self, feature: Any) -> Any:
        for key in (self.intervention_group_key, "shift_group", "intervention_id"):
            if key in feature.data:
                return feature.data[key]
        return _MISSING

    # Keep the old private helper name for downstream code that may have used
    # it while making the broader intervention-group semantics explicit.
    _get_shift_group = _get_intervention_group

    def _extract_data(
        self,
        graph: BaseGraph,
        data: Data,
    ) -> tuple[np.ndarray, np.ndarray, _InterventionDesign]:
        if not data.samples:
            raise ValueError("LinearDAGDiscovery requires at least one sample.")
        values = np.full((graph.num_vertices, len(data.samples)), np.nan, dtype=float)
        observed = np.zeros_like(values, dtype=bool)
        annotations: dict[tuple[int, int], tuple[str, Any, Any]] = {}
        graph_vertices = set(graph.V)

        for sample_index, (sample_name, sample) in enumerate(data.samples.items()):
            seen = set()
            for feature in sample.features:
                if feature.mapping != "vertex":
                    continue
                if feature.id not in graph_vertices:
                    raise ValueError(f"Unknown vertex {feature.id!r} in sample {sample_name!r}.")
                if feature.id in seen:
                    raise ValueError(f"Duplicate vertex {feature.id!r} in sample {sample_name!r}.")
                seen.add(feature.id)
                vertex_index = self._vertex_index[feature.id]
                intervention_type = self._get_intervention_type(feature, sample_name)
                shift_value = None
                shift_group = _MISSING
                if intervention_type != "none":
                    shift_group = self._get_intervention_group(feature)
                    if shift_group is None:
                        shift_group = _MISSING
                    if shift_group is not _MISSING:
                        _InterventionDesign._validate_group(shift_group)
                if intervention_type == "shift":
                    shift_value = self._get_shift_value(feature, sample_name)
                elif any(
                    key in feature.data
                    for key in dict.fromkeys(
                        (self.intervention_shift_key, "shift", "shift_value", "intervention_shift")
                    )
                ):
                    raise ValueError(
                        f"A shift value is only valid for a 'shift' intervention "
                        f"(vertex {feature.id!r}, sample {sample_name!r})."
                    )
                annotations[(vertex_index, sample_index)] = (
                    intervention_type,
                    shift_value,
                    shift_group,
                )
                value = feature.value
                if value is None or (
                    isinstance(value, Real) and not isinstance(value, bool) and np.isnan(float(value))
                ):
                    if intervention_type != "none":
                        raise ValueError(
                            f"Intervened vertex {feature.id!r} in sample {sample_name!r} requires an observed value."
                        )
                    continue
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(
                        f"Measurement for vertex {feature.id!r} in sample {sample_name!r} must be numeric or missing."
                    )
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(
                        f"Measurement for vertex {feature.id!r} in sample {sample_name!r} must be finite or missing."
                    )
                values[vertex_index, sample_index] = value
                observed[vertex_index, sample_index] = True

        design = _InterventionDesign.build(
            tuple(graph.V),
            graph.num_vertices,
            len(data.samples),
            annotations,
        )
        if not np.any(design.hard | design.shift):
            raise ValueError("LinearDAGDiscovery requires at least one intervened vertex.")
        never_observed = [graph.V[index] for index in np.flatnonzero(~observed.any(axis=1))]
        if never_observed:
            raise ValueError(f"PKN vertices {never_observed!r} are never observed; latent vertices are not supported.")
        return values, observed, design

    def _build_commodities(self, graph: BaseGraph) -> tuple[_Commodity, ...]:
        """Resolve intervention cells into unique structural flow signatures.

        Cells remain independent observations for regression, but flow
        evidence is deduplicated by explicit ``(source, group)`` units or by
        one implicit unit per structural signature.  A repeated explicit unit
        is only valid when all of its cells have the same signature.
        """
        if self._layout is None or self._intervention_design is None:
            raise ValueError("LinearDAGDiscovery intervention preprocessing has not been initialized.")

        vertices = tuple(graph.V)
        measured_responses = self._observed & ~self._intervened
        explicit_signatures: dict[tuple[int, Any], tuple[int, tuple[int, ...], tuple[int, ...]]] = {}
        explicit_samples: dict[tuple[int, Any], list[int]] = {}
        implicit_samples: dict[tuple[int, tuple[int, ...], tuple[int, ...]], list[int]] = {}
        signature_refs: dict[
            tuple[int, tuple[int, ...], tuple[int, ...]],
            list[tuple[str, Any]],
        ] = {}
        signature_order: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []

        for sample_index in range(len(self._sample_names)):
            sink_indices = np.flatnonzero(measured_responses[:, sample_index])
            sink_edges = tuple(self._layout.outflow_edges[vertices[index]] for index in sink_indices)
            blocked_edges = tuple(np.flatnonzero(self._hard_intervened[self._edge_targets, sample_index]).tolist())
            source_indices = np.flatnonzero(self._intervened[:, sample_index])
            for source_index in source_indices:
                source = vertices[source_index]
                signature = (int(source_index), sink_edges, blocked_edges)
                if signature not in signature_refs:
                    signature_refs[signature] = []
                    signature_order.append(signature)

                group = self._intervention_design.groups[source_index, sample_index]
                if group is _MISSING:
                    if signature not in implicit_samples:
                        implicit_samples[signature] = []
                        signature_refs[signature].append(("implicit", signature))
                    implicit_samples[signature].append(sample_index)
                    continue

                unit_key = (int(source_index), group)
                prior_signature = explicit_signatures.get(unit_key)
                if prior_signature is not None and prior_signature != signature:
                    raise ValueError(
                        "Intervention evidence unit for source "
                        f"{source!r} and intervention_group {group!r} has inconsistent "
                        "structural signatures across samples. Repeated explicit groups "
                        "must have the same eligible sinks and hard-intervention blocks."
                    )
                if prior_signature is None:
                    explicit_signatures[unit_key] = signature
                    explicit_samples[unit_key] = []
                    signature_refs[signature].append(("explicit", unit_key))
                explicit_samples[unit_key].append(sample_index)

        commodities: list[_Commodity] = []
        for signature in signature_order:
            source_index, sink_edges, blocked_edges = signature
            units = []
            for kind, reference in signature_refs[signature]:
                if kind == "explicit":
                    unit_source_index, group = reference
                    units.append(
                        _EvidenceUnit(
                            source=vertices[unit_source_index],
                            intervention_group=group,
                            sample_indices=tuple(explicit_samples[reference]),
                            explicit_group=True,
                        )
                    )
                else:
                    units.append(
                        _EvidenceUnit(
                            source=vertices[source_index],
                            intervention_group=None,
                            sample_indices=tuple(implicit_samples[reference]),
                            explicit_group=False,
                        )
                    )
            commodities.append(
                _Commodity(
                    source=vertices[source_index],
                    source_edge=self._layout.inflow_edges[vertices[source_index]],
                    sink_edges=sink_edges,
                    hard_blocked_edges=blocked_edges,
                    evidence_units=tuple(units),
                )
            )
        return tuple(commodities)

    def preprocess(self, graph: BaseGraph, data: Data):
        """Validate data and construct signature-specific flow boundaries."""
        self.solve_result = None
        self._original_graph = graph.copy()
        self._vertex_index = {vertex: index for index, vertex in enumerate(graph.V)}
        self._sample_names = tuple(data.samples)
        self._edge_sources, self._edge_targets = self._validate_graph(graph)
        self._values, self._observed, self._intervention_design = self._extract_data(graph, data)
        self._hard_intervened = self._intervention_design.hard
        self._soft_intervened = self._intervention_design.shift
        self._intervened = self._hard_intervened | self._soft_intervened
        vertices = tuple(graph.V)
        self._valid_residual = self._observed & ~self._hard_intervened
        for source, target in zip(self._edge_sources, self._edge_targets, strict=True):
            self._valid_residual[target, :] &= self._observed[source, :]
        if not np.any(self._valid_residual):
            raise ValueError("No complete, non-intervened structural equations are available.")
        self._values = np.where(self._observed, self._values, 0.0)

        vertex_weights = self._resolve_positive_weights(
            tuple(graph.V),
            self.vertex_weights,
            "vertex_weights",
        )
        sample_weights = self._resolve_positive_weights(
            self._sample_names,
            self.sample_weights,
            "sample_weights",
        )
        self._observation_weights = vertex_weights[:, None] * sample_weights[None, :]
        self._loss_scales = np.ones((graph.num_vertices,), dtype=float)
        if self.standardize_loss:
            for vertex_index in range(graph.num_vertices):
                fitted_values = self._values[vertex_index, self._valid_residual[vertex_index, :]]
                if fitted_values.size > 1:
                    scale = float(np.std(fitted_values))
                    if np.isfinite(scale) and scale > np.finfo(float).eps:
                        self._loss_scales[vertex_index] = scale
        predictor_scale_mask = np.zeros_like(self._observed)
        for source, target in zip(self._edge_sources, self._edge_targets, strict=True):
            predictor_scale_mask[source, :] |= self._valid_residual[target, :] & self._observed[source, :]

        # A variable can appear on either side of an equation. Use both roles
        # so that its coefficient scale transforms with its measurement units
        # even when it is a terminal response or has no usable child equation.
        coefficient_scale_mask = self._valid_residual | predictor_scale_mask
        self._coefficient_scales = np.ones((graph.num_vertices,), dtype=float)
        for vertex_index in range(graph.num_vertices):
            coefficient_values = self._values[vertex_index, coefficient_scale_mask[vertex_index, :]]
            if coefficient_values.size:
                scale = float(np.std(coefficient_values)) if coefficient_values.size > 1 else 0.0
                if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                    scale = float(np.max(np.abs(coefficient_values)))
                if np.isfinite(scale) and scale > np.finfo(float).eps:
                    self._coefficient_scales[vertex_index] = scale

        intervened_union = self._intervened.any(axis=1)
        measured_responses = self._observed & ~self._intervened
        sink_union = measured_responses.any(axis=1)
        self._layout = augment_with_boundaries(
            graph,
            inflow_vertices=(vertex for vertex, include in zip(vertices, intervened_union, strict=True) if include),
            outflow_vertices=(vertex for vertex, include in zip(vertices, sink_union, strict=True) if include),
        )

        self._commodities = self._build_commodities(graph)
        num_commodities = len(self._commodities)
        num_flow_edges = self._layout.graph.num_edges
        flow_capacity = self.flow_capacity
        if flow_capacity is None:
            flow_capacity = float(max(graph.num_edges, 1)) * self.flow_epsilon
        if flow_capacity < self.flow_epsilon:
            raise ValueError("Resolved flow capacity must be greater than or equal to flow_epsilon.")
        self._resolved_flow_capacity = float(flow_capacity)
        self._flow_lb = np.zeros((num_flow_edges, num_commodities), dtype=float)
        self._flow_ub = np.zeros((num_flow_edges, num_commodities), dtype=float)
        self._flow_ub[: graph.num_edges, :] = self._resolved_flow_capacity
        for commodity_index, commodity in enumerate(self._commodities):
            self._flow_ub[commodity.source_edge, commodity_index] = self._resolved_flow_capacity
            self._flow_ub[np.asarray(commodity.sink_edges, dtype=int), commodity_index] = self._resolved_flow_capacity
            self._flow_ub[np.asarray(commodity.hard_blocked_edges, dtype=int), commodity_index] = 0.0
        return self._layout.graph, data.copy()

    def get_flow_bounds(self, graph: BaseGraph, data: Data):
        """Return the precomputed per-edge, per-commodity bounds."""
        return {
            "lb": self._flow_lb,
            "ub": self._flow_ub,
            "n_flows": len(self._commodities),
            "shared_bounds": False,
        }

    def create_problem(self, graph: BaseGraph, data: Data):
        """Build an efficient matrix-valued commodity-flow formulation."""
        if self._original_graph is None or self._layout is None:
            raise ValueError("LinearDAGDiscovery preprocessing has not been initialized.")
        num_edges = self._original_graph.num_edges
        edge_selected = self.backend.Variable(
            "edge_selected",
            (num_edges,),
            vartype=VarType.BINARY,
        )
        problem = self.backend.SelectedFlow(
            graph,
            lb=self._flow_lb,
            ub=self._flow_ub,
            n_flows=len(self._commodities),
            edge_indices=range(num_edges),
            epsilon=self.flow_epsilon,
            selected=edge_selected,
            acyclic_graph=self._original_graph,
            max_parents=self.max_parents,
            selected_by_flow_name="edge_used_by_commodity",
            selected_any_name="edge_selected",
        )
        return self.create_flow_based_problem(problem, graph, data)

    def create_flow_based_problem(self, flow_problem: ProblemDef, graph: BaseGraph, data: Data) -> ProblemDef:
        """Add commodity coverage and linear structural-equation fitting."""
        if self._original_graph is None:
            raise ValueError("LinearDAGDiscovery preprocessing has not been initialized.")
        if self._intervention_design is None:
            raise ValueError("LinearDAGDiscovery intervention preprocessing has not been initialized.")
        problem = flow_problem
        num_vertices = self._original_graph.num_vertices
        num_edges = self._original_graph.num_edges
        num_samples = self._values.shape[1]
        num_commodities = len(self._commodities)

        flow = problem.expr.flow
        edge_selected = problem.expr.edge_selected
        edge_used = problem.expr.edge_used_by_commodity
        coverage_enabled = self.min_commodity_coverage > 0 or self.lambda_unexplained > 0
        if coverage_enabled:
            commodity_active = self.backend.Variable(
                "commodity_active",
                (num_commodities,),
                vartype=VarType.BINARY,
            )
            active_matrix = (
                self.backend.Constant(_sparse_vector_entry_repetition(num_commodities, num_edges)) @ commodity_active
            )
            active_matrix = active_matrix.reshape((num_edges, num_commodities))
            problem += edge_used <= active_matrix
            problem += commodity_active.reshape((num_commodities, 1)) <= edge_used.sum(axis=0).reshape(
                (num_commodities, 1)
            )

            num_flow_edges = graph.num_edges
            flow_flat = flow.reshape((num_flow_edges * num_commodities, 1))
            commodity_column = commodity_active.reshape((num_commodities, 1))
            commodity_indices = np.arange(num_commodities, dtype=int)
            source_edges = np.asarray([commodity.source_edge for commodity in self._commodities], dtype=int)
            source_positions = source_edges + num_flow_edges * commodity_indices
            source_selector = csr_matrix(
                (
                    np.ones(num_commodities, dtype=float),
                    (commodity_indices, source_positions),
                ),
                shape=(num_commodities, num_flow_edges * num_commodities),
            )
            source_flow = self.backend.Constant(source_selector) @ flow_flat
            problem += source_flow >= self.flow_epsilon * commodity_column
            problem += source_flow <= self._resolved_flow_capacity * commodity_column

            sink_commodities = np.asarray(
                [
                    commodity_index
                    for commodity_index, commodity in enumerate(self._commodities)
                    for _ in commodity.sink_edges
                ],
                dtype=int,
            )
            sink_edges = np.asarray(
                [edge for commodity in self._commodities for edge in commodity.sink_edges],
                dtype=int,
            )
            sink_positions = sink_edges + num_flow_edges * sink_commodities
            sink_selector = csr_matrix(
                (
                    np.ones(sink_positions.size, dtype=float),
                    (sink_commodities, sink_positions),
                ),
                shape=(num_commodities, num_flow_edges * num_commodities),
            )
            sink_flow = self.backend.Constant(sink_selector) @ flow_flat
            problem += sink_flow >= self.flow_epsilon * commodity_column

            if sink_positions.size:
                sink_entry_rows = np.arange(sink_positions.size, dtype=int)
                sink_entry_selector = csr_matrix(
                    (
                        np.ones(sink_positions.size, dtype=float),
                        (sink_entry_rows, sink_positions),
                    ),
                    shape=(sink_positions.size, num_flow_edges * num_commodities),
                )
                sink_entry_flow = self.backend.Constant(sink_entry_selector) @ flow_flat
                sink_active_selector = csr_matrix(
                    (
                        np.ones(sink_positions.size, dtype=float),
                        (sink_entry_rows, sink_commodities),
                    ),
                    shape=(sink_positions.size, num_commodities),
                )
                sink_entry_active = self.backend.Constant(sink_active_selector) @ commodity_column
                problem += sink_entry_flow <= self._resolved_flow_capacity * sink_entry_active

            group_counts = np.asarray([commodity.group_count for commodity in self._commodities], dtype=float)
            problem.register(
                "commodity_group_count",
                self.backend.Constant(group_counts, name="commodity_group_count"),
            )
            minimum_active = int(np.ceil(self.min_commodity_coverage * group_counts.sum()))
            if minimum_active:
                weighted_active = self.backend.Constant(csr_matrix(group_counts.reshape((1, num_commodities)))) @ (
                    commodity_active.reshape((num_commodities, 1))
                )
                problem += weighted_active >= minimum_active

        if self.lambda_unexplained:
            group_counts = np.asarray([commodity.group_count for commodity in self._commodities], dtype=float)
            unexplained = self.backend.Constant(group_counts).multiply(
                self.backend.Constant(np.ones((num_commodities,))) - commodity_active
            )
            problem.register("commodity_unexplained", unexplained)
            problem.add_objective(
                unexplained.sum(),
                weight=self.lambda_unexplained,
                name="commodity_coverage",
            )

        coefficient_normalization = (
            self._coefficient_scales[self._edge_sources] / self._coefficient_scales[self._edge_targets]
        )
        if self.coefficient_support == "exact":
            # Translate the raw coefficient bound to normalized units. This
            # preserves the raw estimator bound without imposing a second
            # normalized upper bound.
            normalized_bound = self.coefficient_bound * coefficient_normalization
            normalized_coefficient = self.backend.Variable(
                "edge_coefficient_normalized",
                (num_edges,),
                lb=-normalized_bound,
                ub=normalized_bound,
            )
            edge_coefficient = normalized_coefficient.multiply(1.0 / coefficient_normalization)
            problem.register("edge_coefficient", edge_coefficient)
            problem += self.backend.ExactSupport(
                normalized_coefficient,
                selected=edge_selected,
                epsilon=self.min_abs_coefficient,
                name="coefficient_support",
                positive_name="coefficient_support_positive",
                negative_name="coefficient_support_negative",
            )
        else:
            edge_coefficient = self.backend.Variable(
                "edge_coefficient",
                (num_edges,),
                lb=-self.coefficient_bound,
                ub=self.coefficient_bound,
            )
            normalized_coefficient = edge_coefficient.multiply(coefficient_normalization)
            problem.register("edge_coefficient_normalized", normalized_coefficient)
            problem += edge_coefficient <= self.coefficient_bound * edge_selected
            problem += edge_coefficient >= -self.coefficient_bound * edge_selected

        if self.enforce_signs:
            positive = []
            negative = []
            for edge_index in range(num_edges):
                interaction = self._original_graph.get_attr_edge(edge_index).get(
                    self.interaction_attribute,
                    None,
                )
                if interaction in {1, 1.0}:
                    positive.append(edge_index)
                elif interaction in {-1, -1.0}:
                    negative.append(edge_index)
            if positive:
                problem += edge_coefficient[np.asarray(positive, dtype=int)] >= 0
            if negative:
                problem += edge_coefficient[np.asarray(negative, dtype=int)] <= 0

        if self.fit_intercept:
            intercept = self.backend.Variable(
                "intercept",
                (num_vertices,),
                lb=-self.intercept_bound,
                ub=self.intercept_bound,
            )
        else:
            intercept = self.backend.Constant(np.zeros((num_vertices,)), name="intercept_zero")
        # The fitted case is already visible as the ``intercept`` symbol.  A
        # stable registered alias also makes the fixed zero-intercept case
        # inspectable through the normal CORNETO expression API.
        problem.register("intercept_value", intercept)
        if not self.fit_intercept:
            problem.register("intercept", intercept)

        parent_values = self._values[self._edge_sources, :]
        parent_rows = np.arange(num_edges * num_samples, dtype=int)
        parent_columns = np.tile(np.arange(num_edges, dtype=int), num_samples)
        weighted_parent_operator = csr_matrix(
            (
                parent_values.reshape(-1, order="F"),
                (parent_rows, parent_columns),
            ),
            shape=(num_edges * num_samples, num_edges),
        )
        weighted_parent_operator.eliminate_zeros()
        weighted_parents = self.backend.Constant(weighted_parent_operator) @ edge_coefficient
        weighted_parents = weighted_parents.reshape((num_edges, num_samples))
        target_incidence = csr_matrix(
            (
                np.ones(num_edges, dtype=float),
                (self._edge_targets, np.arange(num_edges, dtype=int)),
            ),
            shape=(num_vertices, num_edges),
        )
        prediction = self.backend.Constant(target_incidence) @ weighted_parents
        if self.fit_intercept:
            intercept_matrix = self.backend.Constant(_sparse_vector_replication(num_vertices, num_samples)) @ intercept
            prediction = prediction + intercept_matrix.reshape((num_vertices, num_samples))

        intervention_design = self._intervention_design
        known_shift = self.backend.Constant(
            csr_matrix(intervention_design.known_shift),
            name="known_intervention_shift",
        )
        shift_prediction = known_shift
        estimated_shift_parameters = None
        if intervention_design.effect_keys:
            estimated_shift_parameters = self.backend.Variable(
                "intervention_shift_parameters",
                (len(intervention_design.effect_keys), 1),
                lb=-self.intervention_shift_bound,
                ub=self.intervention_shift_bound,
            )
            shift_mapping = self.backend.Constant(
                intervention_design.mapping,
                name="intervention_shift_mapping",
            )
            estimated_shift = shift_mapping @ estimated_shift_parameters
            shift_prediction = shift_prediction + estimated_shift.reshape((num_vertices, num_samples))
        prediction = prediction + shift_prediction

        problem.register("known_intervention_shift", known_shift)
        problem.register("intervention_shift", shift_prediction)
        if estimated_shift_parameters is not None:
            problem.register("intervention_shift_mapping", shift_mapping)
            problem.register("intervention_shift_parameters", estimated_shift_parameters)
            if self.lambda_intervention_shifts:
                problem.add_objective(
                    estimated_shift_parameters.norm(1),
                    weight=self.lambda_intervention_shifts,
                    name="intervention_shift_sparsity",
                )

        residual = self._values - prediction
        problem.register("prediction", prediction)
        problem.register("residual", residual)
        problem.register(
            "observed_mask",
            self.backend.Constant(self._observed.astype(float), name="observed_mask"),
        )
        problem.register(
            "fitted_mask",
            self.backend.Constant(self._valid_residual.astype(float), name="fitted_mask"),
        )
        problem.register(
            "loss_scale",
            self.backend.Constant(self._loss_scales, name="loss_scale"),
        )
        problem.register(
            "observation_weight",
            self.backend.Constant(
                self._observation_weights,
                name="observation_weight",
            ),
        )
        residual_vector = residual.reshape((num_vertices * num_samples,))
        valid_indices = np.flatnonzero(self._valid_residual.reshape(-1, order="F"))
        fitted_residual = residual_vector[valid_indices]
        observation_weights = self._observation_weights.reshape(-1, order="F")[valid_indices]
        scales = np.broadcast_to(
            self._loss_scales[:, None],
            (num_vertices, num_samples),
        ).reshape(-1, order="F")[valid_indices]
        if self.loss == "absolute":
            absolute_residual = self.backend.Variable(
                "absolute_residual",
                (valid_indices.size,),
                lb=0,
            )
            problem += fitted_residual <= absolute_residual
            problem += -fitted_residual <= absolute_residual
            fit = absolute_residual.multiply(observation_weights / scales).sum()
            problem.add_objective(fit, name="absolute_fit")
        else:
            standardized_residual = fitted_residual.multiply(np.sqrt(observation_weights) / scales)
            fit = standardized_residual.norm(2) ** 2
            problem.add_objective(fit, name="squared_fit")

        if self.lambda_edges:
            problem.add_objective(
                edge_selected.sum(),
                weight=self.lambda_edges,
                name="edge_sparsity",
            )
        problem.register("dag_layer", problem.expr._dag_layer)
        return problem

    # ------------------------------------------------------------------
    # Fixed-model prediction and evaluation
    # ------------------------------------------------------------------
    @property
    def loss_scales(self) -> np.ndarray:
        """Training-derived response scales used by held-out losses."""
        if self._loss_scales.size == 0:
            raise ValueError("LinearDAGDiscovery has not been built.")
        return self._loss_scales.copy()

    @property
    def intercepts(self) -> np.ndarray:
        """Return fitted vertex intercepts in graph-vertex order."""
        if self._original_graph is None:
            raise ValueError("LinearDAGDiscovery has not been built.")
        require_expression_value(self.problem, "edge_selected", self.name())
        if not self.fit_intercept:
            return np.zeros((self._original_graph.num_vertices,), dtype=float)
        return require_expression_value(self.problem, "intercept", self.name()).reshape(-1).copy()

    @property
    def edge_coefficients(self) -> np.ndarray:
        """Return fitted coefficients in original PKN edge order."""
        return require_expression_value(self.problem, "edge_coefficient", self.name()).reshape(-1).copy()

    def _fitted_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return selected edges, coefficients, and intercepts after solving."""
        selected = require_expression_value(self.problem, "edge_selected", self.name()).reshape(-1) > 0.5
        coefficients = require_expression_value(self.problem, "edge_coefficient", self.name()).reshape(-1)
        intercept = (
            require_expression_value(self.problem, "intercept", self.name()).reshape(-1)
            if self.fit_intercept
            else np.zeros((self._original_graph.num_vertices,), dtype=float)
        )
        if self._original_graph is None or coefficients.size != self._original_graph.num_edges:
            raise ValueError("LinearDAGDiscovery has no usable fitted edge parameters.")
        return selected, coefficients, intercept

    def _prediction_input_arrays(
        self,
        data: Data,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Parse new observations without applying training-time fitting rules."""
        if not isinstance(data, Data):
            raise TypeError("data must be a corneto.data.Data object.")
        if self._original_graph is None or self._intervention_design is None:
            raise ValueError("LinearDAGDiscovery has not been built; call build or fit first.")
        if not data.samples:
            raise ValueError("Prediction data must contain at least one sample.")

        n_vertices = self._original_graph.num_vertices
        n_samples = len(data.samples)
        values = np.full((n_vertices, n_samples), np.nan, dtype=float)
        observed = np.zeros((n_vertices, n_samples), dtype=bool)
        hard = np.zeros((n_vertices, n_samples), dtype=bool)
        soft = np.zeros((n_vertices, n_samples), dtype=bool)
        shifts = np.zeros((n_vertices, n_samples), dtype=float)
        explicit_shift = np.zeros((n_vertices, n_samples), dtype=bool)
        groups = np.full((n_vertices, n_samples), _MISSING, dtype=object)
        vertices = set(self._original_graph.V)

        for sample_index, (sample_name, sample) in enumerate(data.samples.items()):
            seen = set()
            for feature in sample.features:
                if feature.mapping != "vertex":
                    continue
                if feature.id not in vertices:
                    raise ValueError(f"Unknown vertex {feature.id!r} in sample {sample_name!r}.")
                if feature.id in seen:
                    raise ValueError(f"Duplicate vertex {feature.id!r} in sample {sample_name!r}.")
                seen.add(feature.id)
                vertex_index = self._vertex_index[feature.id]
                intervention_type = self._get_intervention_type(feature, sample_name)
                if intervention_type == "hard":
                    hard[vertex_index, sample_index] = True
                elif intervention_type == "shift":
                    soft[vertex_index, sample_index] = True
                elif any(
                    key in feature.data
                    for key in dict.fromkeys(
                        (self.intervention_shift_key, "shift", "shift_value", "intervention_shift")
                    )
                ):
                    raise ValueError(
                        f"A shift value is only valid for a 'shift' intervention "
                        f"(vertex {feature.id!r}, sample {sample_name!r})."
                    )

                if intervention_type != "none":
                    group = self._get_intervention_group(feature)
                    if group is not _MISSING and group is not None:
                        _InterventionDesign._validate_group(group)
                        groups[vertex_index, sample_index] = group
                shift_value = None
                if intervention_type == "shift":
                    shift_value = self._get_shift_value(feature, sample_name)
                    if shift_value is not None:
                        shifts[vertex_index, sample_index] = shift_value
                        explicit_shift[vertex_index, sample_index] = True
                value = feature.value
                if value is None or (
                    isinstance(value, Real) and not isinstance(value, bool) and np.isnan(float(value))
                ):
                    if intervention_type == "hard":
                        raise ValueError(
                            f"Hard-intervened vertex {feature.id!r} in sample {sample_name!r} "
                            "requires an observed value."
                        )
                    continue
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(
                        f"Measurement for vertex {feature.id!r} in sample {sample_name!r} must be numeric or missing."
                    )
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(
                        f"Measurement for vertex {feature.id!r} in sample {sample_name!r} must be finite or missing."
                    )
                values[vertex_index, sample_index] = value
                observed[vertex_index, sample_index] = True

        # Fill omitted estimated shifts only from their fitted training group.
        fitted_shifts: dict[tuple[Any, Any], float] = {}
        if self._intervention_design.effect_keys:
            parameters = require_expression_value(
                self.problem,
                "intervention_shift_parameters",
                self.name(),
            ).reshape(-1)
            fitted_shifts = dict(zip(self._intervention_design.effect_keys, parameters, strict=True))
        for vertex_index, sample_index in zip(*np.where(soft), strict=True):
            # An explicit numeric shift wins and is already in ``shifts``.
            feature_group = groups[vertex_index, sample_index]
            if explicit_shift[vertex_index, sample_index]:
                continue
            if feature_group is _MISSING:
                raise ValueError(
                    "Soft-intervention prediction requires an explicit shift or a fitted intervention_group; "
                    f"none was supplied for vertex {self._original_graph.V[vertex_index]!r}."
                )
            key = (self._original_graph.V[vertex_index], feature_group)
            if key not in fitted_shifts:
                raise ValueError(
                    f"No fitted soft-intervention shift is available for vertex/group {key!r}; "
                    "test outcomes are never used to estimate it."
                )
            shifts[vertex_index, sample_index] = float(fitted_shifts[key])
        return values, observed, hard, soft, shifts, tuple(data.samples)

    def _selected_order(self, selected: np.ndarray) -> tuple[list[int], list[list[int]]]:
        """Topologically order the fitted selected DAG."""
        if self._original_graph is None:
            raise ValueError("LinearDAGDiscovery has not been built.")
        incoming = [[] for _ in range(self._original_graph.num_vertices)]
        outgoing = [[] for _ in range(self._original_graph.num_vertices)]
        indegree = np.zeros((self._original_graph.num_vertices,), dtype=int)
        for edge_index in np.flatnonzero(selected):
            source = int(self._edge_sources[edge_index])
            target = int(self._edge_targets[edge_index])
            incoming[target].append(int(edge_index))
            outgoing[source].append(target)
            indegree[target] += 1
        queue = [index for index, degree in enumerate(indegree) if degree == 0]
        order = []
        while queue:
            vertex = queue.pop(0)
            order.append(vertex)
            for target in outgoing[vertex]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(order) != self._original_graph.num_vertices:
            raise ValueError("The fitted LinearDAGDiscovery support is cyclic and cannot be predicted.")
        return order, incoming

    def _forward_prediction_arrays(
        self,
        data: Optional[Data] = None,
        parsed: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
        if parsed is None:
            if data is None:
                raise ValueError("Prediction data is required.")
            parsed = self._prediction_input_arrays(data)
        values, observed, hard, soft, shifts, sample_names = parsed
        selected, coefficients, intercept = self._fitted_parameters()
        order, incoming = self._selected_order(selected)
        predictions = np.zeros_like(values)
        n_samples = values.shape[1]
        for sample_index in range(n_samples):
            for vertex in order:
                if hard[vertex, sample_index]:
                    predictions[vertex, sample_index] = values[vertex, sample_index]
                    continue
                edge_indices = incoming[vertex]
                prediction = intercept[vertex]
                for edge_index in edge_indices:
                    source = self._edge_sources[edge_index]
                    prediction += coefficients[edge_index] * predictions[source, sample_index]
                if soft[vertex, sample_index]:
                    prediction += shifts[vertex, sample_index]
                predictions[vertex, sample_index] = prediction
        return predictions, observed, hard, sample_names

    @staticmethod
    def _data_from_arrays(
        vertices: tuple[Any, ...],
        sample_names: tuple[Any, ...],
        values: np.ndarray,
        *,
        observed: Optional[np.ndarray] = None,
        valid: Optional[np.ndarray] = None,
        predicted_mask: Optional[np.ndarray] = None,
        clamped: Optional[np.ndarray] = None,
        equation_prediction: Optional[np.ndarray] = None,
    ) -> Data:
        samples = {}
        for sample_index, sample_name in enumerate(sample_names):
            features = {}
            for vertex_index, vertex in enumerate(vertices):
                value = values[vertex_index, sample_index]
                features[vertex] = {
                    "mapping": "vertex",
                    "value": None if not np.isfinite(value) else float(value),
                    "predicted": bool(predicted_mask[vertex_index, sample_index])
                    if predicted_mask is not None
                    else False,
                }
                if observed is not None:
                    features[vertex]["observed"] = bool(observed[vertex_index, sample_index])
                if valid is not None:
                    features[vertex]["valid"] = bool(valid[vertex_index, sample_index])
                if clamped is not None:
                    features[vertex]["clamped"] = bool(clamped[vertex_index, sample_index])
                if equation_prediction is not None and np.isfinite(equation_prediction[vertex_index, sample_index]):
                    features[vertex]["equation_prediction"] = float(equation_prediction[vertex_index, sample_index])
            samples[sample_name] = features
        return Data.from_cdict(samples)

    def predict(self, data: Data) -> Data:
        """Predict held-out vertex values using the fixed fitted DAG.

        All non-hard measurements, including roots, are ignored as predictors.
        Roots use their fitted intercept baseline; hard interventions are
        clamped, soft shifts are applied to their equations, and descendants
        use previously predicted parents. This method never calls a solver.
        """
        predictions, observed, _hard, sample_names = self._forward_prediction_arrays(data)
        if self._original_graph is None:
            raise ValueError("LinearDAGDiscovery has not been built.")
        return self._data_from_arrays(
            tuple(self._original_graph.V),
            sample_names,
            predictions,
            observed=observed,
            predicted_mask=~_hard,
            clamped=_hard,
        )

    def _local_prediction_arrays(
        self,
        data: Optional[Data] = None,
        parsed: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
        if parsed is None:
            if data is None:
                raise ValueError("Prediction data is required.")
            parsed = self._prediction_input_arrays(data)
        values, observed, hard, soft, shifts, sample_names = parsed
        _selected, coefficients, intercept = self._fitted_parameters()
        # Training's complete-equation mask is based on every candidate PKN
        # parent, even when its fitted coefficient is zero.  Retain that
        # conservative mask here so local diagnostics remain comparable to
        # the optimization objective.
        all_incoming = [[] for _ in range(values.shape[0])]
        for edge_index, target in enumerate(self._edge_targets):
            all_incoming[int(target)].append(edge_index)
        local = np.full_like(values, np.nan)
        valid = observed & ~hard
        for sample_index in range(values.shape[1]):
            for vertex in range(values.shape[0]):
                if not valid[vertex, sample_index]:
                    continue
                parent_edges = all_incoming[vertex]
                if any(not observed[self._edge_sources[e], sample_index] for e in parent_edges):
                    valid[vertex, sample_index] = False
                    continue
                value = intercept[vertex]
                for edge_index in parent_edges:
                    value += coefficients[edge_index] * values[self._edge_sources[edge_index], sample_index]
                if soft[vertex, sample_index]:
                    value += shifts[vertex, sample_index]
                local[vertex, sample_index] = value
        return local, values, valid, sample_names

    def residuals(self, data: Data) -> Data:
        """Return observed-minus-local-equation residuals.

        Unlike :meth:`predict`, this diagnostic intentionally uses observed
        parent values. It mirrors the structural-equation training objective,
        including its complete candidate-parent missingness mask, and is useful
        for local fit auditing but is not a forward intervention forecast.
        """
        local, values, valid, sample_names = self._local_prediction_arrays(data)
        residual = np.full_like(values, np.nan)
        residual[valid] = values[valid] - local[valid]
        if self._original_graph is None:
            raise ValueError("LinearDAGDiscovery has not been built.")
        return self._data_from_arrays(
            tuple(self._original_graph.V),
            sample_names,
            residual,
            observed=np.isfinite(values),
            valid=valid,
            equation_prediction=local,
        )

    def evaluate(
        self,
        data: Data,
        *,
        sample_weights: Optional[Mapping[Any, float]] = None,
    ) -> dict[str, Any]:
        """Evaluate fixed forward and local losses on supplied measurements.

        Forward loss excludes hard-intervention targets (which are clamped) and
        never uses observed descendants as predictors. Local loss uses observed
        values for every candidate PKN parent (with zero coefficients for
        unselected edges), matching the training equation objective. Both use
        training-derived response scales and report valid counts plus
        per-vertex and per-sample normalized losses. Reported losses are
        weighted means (configured vertex weights and optional held-out
        ``sample_weights``), not the raw summed optimization objective; training
        sample weights are never reused for unrelated held-out names.
        """
        parsed = self._prediction_input_arrays(data)
        values, observed, hard, _soft, _shifts, sample_names = parsed
        predictions, observed, hard, sample_names = self._forward_prediction_arrays(parsed=parsed)
        local, values, local_valid, _ = self._local_prediction_arrays(parsed=parsed)
        forward_valid = observed & ~hard
        if self._original_graph is None:
            raise ValueError("LinearDAGDiscovery has not been built.")

        if sample_weights is not None and not isinstance(sample_weights, Mapping):
            raise TypeError("sample_weights must be a mapping or None.")
        evaluation_sample_weights = self._resolve_positive_weights(
            tuple(sample_names),
            dict(sample_weights or {}),
            "sample_weights",
        )
        evaluation_vertex_weights = self._resolve_positive_weights(
            tuple(self._original_graph.V),
            self.vertex_weights,
            "vertex_weights",
        )

        def metrics(
            errors: np.ndarray,
            valid: np.ndarray,
        ) -> tuple[float, int, dict[Any, float], dict[Any, float]]:
            normalized = errors / self._loss_scales[:, None]
            score = np.abs(normalized) if self.loss == "absolute" else normalized**2
            weights = evaluation_vertex_weights[:, None] * evaluation_sample_weights[None, :]
            count = int(np.count_nonzero(valid))
            denominator = float(np.sum(weights[valid])) if count else 0.0
            loss = float(np.sum(score[valid] * weights[valid]) / denominator) if denominator else float("nan")
            per_vertex = {
                vertex: (
                    float(
                        np.sum(score[index, valid[index]] * evaluation_sample_weights[valid[index]])
                        / np.sum(evaluation_sample_weights[valid[index]])
                    )
                    if np.any(valid[index])
                    else float("nan")
                )
                for index, vertex in enumerate(self._original_graph.V)
            }
            per_sample = {
                sample: (
                    float(
                        np.sum(score[valid[:, index], index] * evaluation_vertex_weights[valid[:, index]])
                        / np.sum(evaluation_vertex_weights[valid[:, index]])
                    )
                    if np.any(valid[:, index])
                    else float("nan")
                )
                for index, sample in enumerate(sample_names)
            }
            return loss, count, per_vertex, per_sample

        forward_loss, forward_count, forward_by_vertex, forward_by_sample = metrics(
            values - predictions,
            forward_valid,
        )
        local_loss, local_count, local_by_vertex, local_by_sample = metrics(values - local, local_valid)
        residual = np.full_like(values, np.nan)
        residual[local_valid] = values[local_valid] - local[local_valid]
        return {
            "forward_loss": forward_loss,
            "forward_count": forward_count,
            "forward_by_vertex": forward_by_vertex,
            "forward_by_sample": forward_by_sample,
            "forward_by_condition": forward_by_sample,
            "local_loss": local_loss,
            "local_count": local_count,
            "local_by_vertex": local_by_vertex,
            "local_by_sample": local_by_sample,
            "local_by_condition": local_by_sample,
            "loss_is_weighted_mean": True,
            "sample_weights": dict(zip(sample_names, evaluation_sample_weights, strict=True)),
            "predictions": self._data_from_arrays(
                tuple(self._original_graph.V),
                sample_names,
                predictions,
                observed=observed,
                predicted_mask=~hard,
                clamped=hard,
            ),
            "residuals": self._data_from_arrays(
                tuple(self._original_graph.V),
                sample_names,
                residual,
                observed=np.isfinite(values),
                valid=local_valid,
                equation_prediction=local,
            ),
        }

    def get_commodity_info(self) -> tuple[Mapping[str, Any], ...]:
        """Return read-only metadata for the resolved structural flows.

        The result is a tuple of mapping-like records, one per flow column.
        ``group_count`` is the number of distinct intervention evidence units
        represented by that column; repeated cells in one explicit group are
        listed in that unit's ``sample_names`` but do not increase the count.
        An absent group is reported as ``None`` and is represented by one
        implicit unit per structural signature.
        """
        if self._original_graph is None or self._layout is None:
            raise ValueError("The method has not been preprocessed.")
        vertex_by_outflow = {edge: vertex for vertex, edge in self._layout.outflow_edges.items()}
        records = []
        for index, commodity in enumerate(self._commodities):
            evidence_units = []
            for unit in commodity.evidence_units:
                evidence_units.append(
                    MappingProxyType(
                        {
                            "source": unit.source,
                            "intervention_group": unit.intervention_group,
                            "explicit_group": unit.explicit_group,
                            "sample_names": tuple(self._sample_names[i] for i in unit.sample_indices),
                        }
                    )
                )
            records.append(
                MappingProxyType(
                    {
                        "index": index,
                        "source": commodity.source,
                        "sink_vertices": tuple(vertex_by_outflow[edge] for edge in commodity.sink_edges),
                        "sink_edges": commodity.sink_edges,
                        "hard_blocked_edges": commodity.hard_blocked_edges,
                        "group_count": commodity.group_count,
                        "evidence_units": tuple(evidence_units),
                    }
                )
            )
        return tuple(records)

    def get_selected_edge_indices(self, threshold: float = 0.5) -> np.ndarray:
        """Return selected edge indices in the original PKN."""
        if self.problem is None:
            raise ValueError("The method has not been built.")
        self.validate_solution_support()
        values = self.problem.expr.edge_selected.value
        if values is None:
            raise ValueError("The problem has not been solved.")
        return np.flatnonzero(np.asarray(values).reshape(-1) > threshold)

    def validate_solution_support(self) -> None:
        """Reject solved exact-support models that violate coefficient support."""
        if self.problem is None:
            raise ValueError("The method has not been built.")
        selected_values = self.problem.expr.edge_selected.value
        coefficient_values = self.problem.expr.edge_coefficient.value
        if selected_values is None or coefficient_values is None:
            raise ValueError("The problem has not been solved.")
        if self.coefficient_support != "exact":
            return

        selected = np.asarray(selected_values).reshape(-1) > 0.5
        coefficients = np.asarray(coefficient_values).reshape(-1)
        normalized = coefficients * (
            self._coefficient_scales[self._edge_sources] / self._coefficient_scales[self._edge_targets]
        )
        tolerance = min(1e-9, self.min_abs_coefficient * 0.5)
        magnitude = np.abs(normalized)
        invalid_selected = selected & (magnitude < self.min_abs_coefficient - tolerance)
        invalid_unselected = ~selected & (magnitude > tolerance)
        invalid = invalid_selected | invalid_unselected
        if np.any(invalid):
            edges = np.flatnonzero(invalid).tolist()
            raise ValueError(
                "Solved exact coefficient support is invalid for edges "
                f"{edges!r}: selected edges must have normalized magnitude at least "
                f"min_abs_coefficient={self.min_abs_coefficient:g} and unselected "
                "edges must have zero normalized coefficients."
            )

    def get_edge_usage(self, threshold: float = 0.5) -> np.ndarray:
        """Return the edge-by-commodity support matrix."""
        if self.problem is None:
            raise ValueError("The method has not been built.")
        self.validate_solution_support()
        values = self.problem.expr.edge_used_by_commodity.value
        if values is None:
            raise ValueError("The problem has not been solved.")
        return np.asarray(values).reshape((self._original_graph.num_edges, len(self._commodities))) > threshold

    def get_solution_graph(self, threshold: float = 0.5):
        """Return the selected PKN subgraph with fitted edge metadata.

        Each solution edge receives ``coefficient`` and
        ``normalized_coefficient`` attributes. The configured interaction
        attribute is set to the sign of the fitted coefficient and, when the
        original graph provided that attribute, its prior value is preserved
        as ``prior_<interaction_attribute>``. Structural-support edges whose
        fitted coefficient is numerically zero receive ``connector=True``.
        """
        if self._original_graph is None:
            raise ValueError("The method has not been built.")
        selected = self.get_selected_edge_indices(threshold)
        extract_keep_order = getattr(self._original_graph, "_extract_subgraph_keep_order", None)
        if callable(extract_keep_order):
            solution_graph = extract_keep_order(edges=selected)
        else:
            solution_graph = self._original_graph.edge_subgraph(selected)

        coefficients = np.asarray(self.problem.expr.edge_coefficient.value).reshape(-1)
        normalized = coefficients * (
            self._coefficient_scales[self._edge_sources] / self._coefficient_scales[self._edge_targets]
        )
        for solution_index, edge_index in enumerate(selected):
            attributes = solution_graph.get_attr_edge(solution_index)
            if self.interaction_attribute in attributes:
                attributes[f"prior_{self.interaction_attribute}"] = attributes[self.interaction_attribute]
            inferred_interaction = int(np.sign(coefficients[edge_index]))
            attributes[self.interaction_attribute] = inferred_interaction
            attributes["coefficient"] = float(coefficients[edge_index])
            attributes["normalized_coefficient"] = float(normalized[edge_index])
            attributes["inferred_interaction"] = inferred_interaction
            attributes["connector"] = bool(
                self.coefficient_support == "structural"
                and np.isclose(coefficients[edge_index], 0.0, atol=1e-9, rtol=0.0)
            )
        return solution_graph

    @staticmethod
    def name() -> str:
        """Return the method name."""
        return "LinearDAGDiscovery"

    @staticmethod
    def description() -> str:
        """Return a short method description."""
        return "Commodity-flow-supported linear DAG discovery over a prior network"
