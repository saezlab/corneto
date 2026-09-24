"""Small callable mechanism helpers for structural causal models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable

import numpy as np


def _finite_real(value: Any, name: str) -> float:
    """Validate a finite real-valued mechanism parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class Linear:
    """Add a weighted sum of the parents and a bias to the node noise."""

    weights: tuple[float, ...]
    bias: float = 0.0

    def __post_init__(self) -> None:
        try:
            weights = tuple(_finite_real(value, "weights") for value in self.weights)
        except TypeError as error:
            raise TypeError("weights must be a sequence of finite real numbers.") from error
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", _finite_real(self.bias, "bias"))

    def __call__(self, parents: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Evaluate ``parents @ weights + bias + noise``."""
        if parents.shape[1] != len(self.weights):
            raise ValueError(f"Linear mechanism expects {len(self.weights)} parents, got {parents.shape[1]}.")
        return parents @ np.asarray(self.weights, dtype=float) + self.bias + noise


@dataclass(frozen=True)
class Additive:
    """Wrap a deterministic parent function in additive noise."""

    function: Callable[[np.ndarray], Any]

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("function must be callable.")

    def __call__(self, parents: np.ndarray, noise: np.ndarray) -> Any:
        """Evaluate the deterministic function and add node noise."""
        return self.function(parents) + noise


@dataclass(frozen=True)
class _Constant:
    value: float

    def __call__(self, parents: np.ndarray, noise: np.ndarray) -> np.ndarray:
        return np.full(noise.shape, self.value, dtype=float)


@dataclass(frozen=True)
class _Shifted:
    mechanism: Callable[[np.ndarray, np.ndarray], Any]
    shift: float

    def __call__(self, parents: np.ndarray, noise: np.ndarray) -> Any:
        return self.mechanism(parents, noise) + self.shift
