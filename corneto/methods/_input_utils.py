"""Validation helpers for user-facing method inputs."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np

from corneto.data import Data
from corneto.graph import BaseGraph

DEFAULT_CONDITION = "condition"


def legacy_data(data: Any, *, method: str) -> Data | None:
    """Return a legacy Data argument or reject an invalid positional value."""
    if data is None:
        return None
    if not isinstance(data, Data):
        raise TypeError(
            f"{method}.build() accepts scientific inputs as keyword arguments; "
            f"use {method}.build_from_data(graph, data) for a Data object."
        )
    warnings.warn(
        f"{method}.build(graph, data) is deprecated; use "
        f"{method}.build_from_data(graph, data) for the advanced Data interface.",
        DeprecationWarning,
        stacklevel=3,
    )
    return data


def require_mapping(value: Any, *, argument: str, condition: str | None = None) -> Mapping:
    if not isinstance(value, Mapping):
        suffix = f" for condition {condition!r}" if condition is not None else ""
        raise TypeError(f"{argument}{suffix} must be a mapping, got {type(value).__name__}.")
    return value


def validate_condition_maps(**arguments: Any) -> dict[str, Mapping]:
    """Validate named condition mappings and require matching condition keys."""
    provided = {name: value for name, value in arguments.items() if value is not None}
    if not provided:
        return {}

    normalized: dict[str, Mapping] = {}
    expected: tuple[str, ...] | None = None
    expected_argument = ""
    for argument, value in provided.items():
        outer = require_mapping(value, argument=argument)
        if not outer:
            raise ValueError(f"{argument} must contain at least one named condition.")
        keys = tuple(outer.keys())
        for condition in keys:
            if not isinstance(condition, str) or not condition:
                raise ValueError(f"Condition names in {argument} must be non-empty strings, got {condition!r}.")
            require_mapping(outer[condition], argument=argument, condition=condition)
        if expected is None:
            expected = keys
            expected_argument = argument
        elif set(keys) != set(expected):
            raise ValueError(
                f"Condition names in {argument} must match {expected_argument}: "
                f"expected {list(expected)!r}, got {list(keys)!r}."
            )
        normalized[argument] = outer
    return normalized


def validate_condition_keys(**arguments: Any) -> tuple[str, ...]:
    """Require all provided outer mappings to use the same named conditions."""
    provided = {name: value for name, value in arguments.items() if value is not None}
    expected: tuple[str, ...] | None = None
    expected_argument = ""
    for argument, value in provided.items():
        outer = require_mapping(value, argument=argument)
        if not outer:
            raise ValueError(f"{argument} must contain at least one named condition.")
        keys = tuple(outer.keys())
        for condition in keys:
            if not isinstance(condition, str) or not condition:
                raise ValueError(f"Condition names in {argument} must be non-empty strings, got {condition!r}.")
        if expected is None:
            expected = keys
            expected_argument = argument
        elif set(keys) != set(expected):
            raise ValueError(
                f"Condition names in {argument} must match {expected_argument}: "
                f"expected {list(expected)!r}, got {list(keys)!r}."
            )
    return expected or ()


def validate_numeric(value: Any, *, argument: str, identifier: Any, condition: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{argument}[{identifier!r}] for condition {condition!r} must be a finite number, "
            f"got {value!r}."
        )
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(
            f"{argument}[{identifier!r}] for condition {condition!r} must be finite, got {value!r}."
        )
    return number


def validate_vertices(
    graph: BaseGraph,
    values: Mapping,
    *,
    argument: str,
    condition: str,
) -> dict[Any, float]:
    vertices = set(graph.V)
    result = {}
    for identifier, value in values.items():
        if identifier not in vertices:
            raise ValueError(f"Unknown vertex {identifier!r} in {argument} for condition {condition!r}.")
        result[identifier] = validate_numeric(
            value, argument=argument, identifier=identifier, condition=condition
        )
    return result


def validate_vertex_collection(
    graph: BaseGraph,
    values: Collection,
    *,
    argument: str,
    condition: str,
    required: bool = False,
) -> list[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError(f"{argument} for condition {condition!r} must be a collection of vertices.")
    result = list(values)
    if required and not result:
        raise ValueError(f"{argument} for condition {condition!r} must not be empty.")
    vertices = set(graph.V)
    for identifier in result:
        if identifier not in vertices:
            raise ValueError(f"Unknown vertex {identifier!r} in {argument} for condition {condition!r}.")
    return list(dict.fromkeys(result))


def reaction_ids(model: BaseGraph) -> set[Any]:
    return set(model.get_attr_from_edges("id"))


def validate_reaction_values(
    model: BaseGraph,
    values: Mapping,
    *,
    argument: str,
    condition: str,
) -> dict[Any, float]:
    identifiers = reaction_ids(model)
    result = {}
    for identifier, value in values.items():
        if identifier not in identifiers:
            raise ValueError(f"Unknown reaction {identifier!r} in {argument} for condition {condition!r}.")
        result[identifier] = validate_numeric(
            value, argument=argument, identifier=identifier, condition=condition
        )
    return result


def validate_reaction_bounds(
    model: BaseGraph,
    values: Mapping,
    *,
    condition: str,
) -> dict[Any, tuple[float | None, float | None]]:
    identifiers = reaction_ids(model)
    result = {}
    for identifier, bounds in values.items():
        if identifier not in identifiers:
            raise ValueError(f"Unknown reaction {identifier!r} in reaction_bounds for condition {condition!r}.")
        if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Collection) or len(bounds) != 2:
            raise ValueError(
                f"reaction_bounds[{identifier!r}] for condition {condition!r} "
                "must be a (lower, upper) pair."
            )
        lower, upper = tuple(bounds)
        lower_value = (
            None
            if lower is None
            else validate_numeric(
                lower,
                argument="reaction_bounds lower bound",
                identifier=identifier,
                condition=condition,
            )
        )
        upper_value = (
            None
            if upper is None
            else validate_numeric(
                upper,
                argument="reaction_bounds upper bound",
                identifier=identifier,
                condition=condition,
            )
        )
        if lower_value is not None and upper_value is not None and lower_value > upper_value:
            raise ValueError(
                f"Lower bound {lower_value} exceeds upper bound {upper_value} "
                f"for reaction {identifier!r} in condition {condition!r}."
            )
        result[identifier] = (lower_value, upper_value)
    return result


def validate_edge_costs(
    graph: BaseGraph,
    values: Mapping,
    *,
    condition: str,
) -> dict[int, float]:
    result = {}
    for identifier, value in values.items():
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, Integral)
            or not 0 <= identifier < graph.num_edges
        ):
            raise ValueError(
                f"Invalid edge index {identifier!r} in edge_costs for condition {condition!r}; "
                f"expected an integer in [0, {graph.num_edges - 1}]."
            )
        result[int(identifier)] = validate_numeric(
            value, argument="edge_costs", identifier=identifier, condition=condition
        )
    return result


def data_from_features(features_by_condition: Mapping[str, list[dict[str, Any]]]) -> Data:
    return Data.from_dict(
        {condition: {"features": features} for condition, features in features_by_condition.items()}
    )
