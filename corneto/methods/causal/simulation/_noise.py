"""Noise distributions and random-state helpers for causal simulation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


def _sample_count(n: int) -> int:
    """Validate and normalize a sample count."""
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be a nonnegative integer.")
    if n < 0:
        raise ValueError("n must be a nonnegative integer.")
    return int(n)


def _rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    """Return a generator from a seed, generator, or ``None``."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is not None and (isinstance(rng, bool) or not isinstance(rng, Integral)):
        raise TypeError("rng must be an integer seed, numpy.random.Generator, or None.")
    return np.random.default_rng(rng)


def _finite_real(value: Any, name: str) -> float:
    """Validate a finite real-valued simulator parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _location_scale(loc: Any, scale: Any, name: str) -> tuple[float, float]:
    """Validate location and a nonnegative distribution scale."""
    loc = _finite_real(loc, f"{name} location")
    scale = _finite_real(scale, f"{name} scale")
    if scale < 0:
        raise ValueError(f"{name} scale must be nonnegative.")
    return loc, scale


@dataclass(frozen=True)
class Zero:
    """A deterministic zero-noise distribution."""

    def __call__(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` zeros."""
        return np.zeros(_sample_count(n), dtype=float)


@dataclass(frozen=True)
class Normal:
    """Draw independent normal noise with a location and standard deviation."""

    loc: float = 0.0
    std: float = 1.0

    def __post_init__(self) -> None:
        loc, std = _location_scale(self.loc, self.std, "Normal")
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "std", std)

    def __call__(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` independent normal values."""
        return rng.normal(self.loc, self.std, _sample_count(n))


@dataclass(frozen=True)
class Laplace:
    """Draw independent Laplace noise with a location and scale."""

    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        loc, scale = _location_scale(self.loc, self.scale, "Laplace")
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)

    def __call__(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` independent Laplace values."""
        return rng.laplace(self.loc, self.scale, _sample_count(n))
