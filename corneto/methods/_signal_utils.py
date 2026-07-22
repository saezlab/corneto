"""Private vectorized building blocks shared by signaling formulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from corneto._constants import VarType
from corneto.backend._base import Backend, ProblemDef


@dataclass(frozen=True)
class SignedEdgeState:
    """Positive, negative, signed, and selected edge-state expressions."""

    positive: Any
    negative: Any
    value: Any
    selected: Any


def add_signed_edge_state(
    backend: Backend,
    problem: ProblemDef,
    shape: tuple[int, ...],
    *,
    positive_name: str,
    negative_name: str,
    value_alias: str | None = None,
    selected_alias: str | None = None,
) -> SignedEdgeState:
    """Create mutually exclusive positive and negative binary edge states."""
    positive = backend.Variable(positive_name, shape, vartype=VarType.BINARY)
    negative = backend.Variable(negative_name, shape, vartype=VarType.BINARY)
    problem += positive + negative <= 1
    value = positive - negative
    selected = positive + negative
    if value_alias is not None:
        problem.register(value_alias, value)
    if selected_alias is not None:
        problem.register(selected_alias, selected)
    return SignedEdgeState(
        positive=positive,
        negative=negative,
        value=value,
        selected=selected,
    )
