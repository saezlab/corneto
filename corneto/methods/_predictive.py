"""Small private helpers shared by reusable mechanistic methods.

The public :class:`~corneto.methods._base.Method` deliberately remains an
optimization-only abstraction.  These helpers are used by the few method
classes that expose a fitted mechanistic model as well as the normal CORNETO
``build``/``ProblemDef`` lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def solve_options(value: Mapping[str, Any] | None, extra: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the keyword options accepted by a convenience ``fit`` call."""
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("solve_options must be a mapping or None.")
    if value and extra:
        raise TypeError("Pass solver arguments either in solve_options or as keyword arguments, not both.")
    return dict(value or extra)


def validate_solve_result(result: Any, expressions: Any, names: tuple[str, ...], method: str) -> Any:
    """Validate a solver result while accepting time-limited incumbents.

    Backends use different result classes and status strings.  A non-failure
    status is therefore not enough: every expression needed for prediction
    must also contain a finite incumbent value.  This permits useful
    ``user_limit``/``time_limit`` results when the solver supplied primals.
    """
    status = str(getattr(result, "status", "")).lower()
    failed = ("infeasible", "unbounded", "error", "invalid", "unknown")
    if not status or any(token in status for token in failed):
        raise ValueError(f"{method}.fit() did not produce a usable solution (status={status or 'unknown'!r}).")
    for name in names:
        try:
            value = getattr(expressions, name).value
        except (AttributeError, KeyError) as error:
            raise ValueError(f"{method}.fit() solution does not expose required expression {name!r}.") from error
        if value is None:
            raise ValueError(f"{method}.fit() solver returned no value for required expression {name!r}.")
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{method}.fit() solver returned non-finite values for {name!r}.")
    return result


def require_expression_value(problem: Any, name: str, method: str) -> np.ndarray:
    """Return a solved expression value with a consistent public error."""
    if problem is None:
        raise ValueError(f"{method} has not been built; call build/build_many or fit first.")
    try:
        value = getattr(problem.expr, name).value
    except (AttributeError, KeyError) as error:
        raise ValueError(f"{method} problem does not expose required expression {name!r}.") from error
    if value is None:
        raise ValueError(f"{method} has not been solved or has no usable solution for {name!r}.")
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{method} has no usable finite solution for {name!r}.")
    return array
