"""Convenience constructors for common structural causal models."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any, Callable

import numpy as np

from corneto.graph import BaseGraph

from ._core import SCM, Node, _graph_structure
from ._mechanisms import Linear
from ._noise import Normal, _finite_real, _rng


def linear_scm(
    graph: BaseGraph,
    weights: Mapping[tuple[Any, Any], Real] | None = None,
    biases: Mapping[Any, Real] | Real | None = None,
    noise: Mapping[Any, Callable] | Callable | None = None,
    *,
    weight_attribute: str | None = None,
    rng: int | np.random.Generator | None = None,
    random_weight_range: tuple[float, float] = (0.5, 2.0),
) -> SCM:
    """Build a linear SCM from a simple directed CORNETO graph.

    Explicit ``weights`` map each ``(parent, child)`` edge to its coefficient.
    If omitted, signed magnitudes are sampled uniformly from
    ``random_weight_range`` so generated effects stay away from zero. Set
    ``weight_attribute`` to read coefficients from a graph edge attribute;
    edge interaction signs are never treated as coefficient magnitudes.

    ``rng`` is used only to generate model coefficients. Pass a separate seed
    or generator to ``SCM.sample`` for simulation noise.
    """
    variables, parents, edges = _graph_structure(graph)
    generator = _rng(rng)
    if weight_attribute is not None and not isinstance(weight_attribute, str):
        raise TypeError("weight_attribute must be a string or None.")
    if weight_attribute == "":
        raise ValueError("weight_attribute must be a nonempty string.")
    if weights is not None and weight_attribute is not None:
        raise ValueError("Provide either weights or weight_attribute, not both.")

    if weights is not None:
        if not isinstance(weights, Mapping):
            raise TypeError("weights must map each graph edge to a finite real coefficient.")
        edge_set = set(edges)
        unknown = tuple(edge for edge in weights if edge not in edge_set)
        missing = tuple(edge for edge in edges if edge not in weights)
        if unknown or missing:
            raise ValueError(f"Weight keys must match graph edges; missing={missing!r}, unknown={unknown!r}.")
        edge_weights = {edge: _finite_real(weights[edge], f"Weight for edge {edge!r}") for edge in edges}
    elif weight_attribute is not None:
        edge_weights = {}
        for edge_index, edge in enumerate(edges):
            attributes = graph.get_attr_edge(edge_index)
            if weight_attribute not in attributes:
                raise ValueError(f"Graph edge {edge!r} has no {weight_attribute!r} attribute.")
            edge_weights[edge] = _finite_real(attributes[weight_attribute], f"Weight for edge {edge!r}")
    else:
        try:
            lower, upper = random_weight_range
        except (TypeError, ValueError) as error:
            raise ValueError("random_weight_range must contain two finite positive bounds.") from error
        lower, upper = (
            _finite_real(lower, "random_weight_range lower bound"),
            _finite_real(upper, "random_weight_range upper bound"),
        )
        if lower <= 0 or upper < lower:
            raise ValueError("random_weight_range must satisfy 0 < lower <= upper.")
        magnitudes = generator.uniform(lower, upper, len(edges))
        signs = generator.choice(np.array((-1.0, 1.0)), size=len(edges))
        edge_weights = {
            edge: float(sign * magnitude) for edge, sign, magnitude in zip(edges, signs, magnitudes, strict=True)
        }

    if biases is None:
        bias_values = {variable: 0.0 for variable in variables}
    elif isinstance(biases, Real) and not isinstance(biases, bool):
        bias = _finite_real(biases, "biases")
        bias_values = {variable: bias for variable in variables}
    elif isinstance(biases, Mapping):
        unknown = tuple(variable for variable in biases if variable not in parents)
        if unknown:
            raise ValueError(f"Bias mapping contains unknown graph vertices: {unknown!r}.")
        bias_values = {
            variable: _finite_real(biases.get(variable, 0.0), f"Bias for {variable!r}") for variable in variables
        }
    else:
        raise TypeError("biases must be a finite real number, a vertex-to-real mapping, or None.")

    if noise is None:
        noise_values = {variable: Normal() for variable in variables}
    elif callable(noise):
        noise_values = {variable: noise for variable in variables}
    elif isinstance(noise, Mapping):
        unknown = tuple(variable for variable in noise if variable not in parents)
        if unknown:
            raise ValueError(f"Noise mapping contains unknown graph vertices: {unknown!r}.")
        noise_values = {variable: noise.get(variable, Normal()) for variable in variables}
    else:
        raise TypeError("noise must be a callable, a vertex-to-callable mapping, or None.")

    nodes = {
        variable: Node(
            parents[variable],
            Linear(tuple(edge_weights[(parent, variable)] for parent in parents[variable]), bias_values[variable]),
            noise_values[variable],
        )
        for variable in variables
    }
    return SCM(nodes)


__all__ = ["linear_scm"]
