"""Tests for alternative-solution sampling utilities."""

import numpy as np
import pytest

from corneto.methods.sampler import _objective_value


@pytest.mark.parametrize("value", [1.5, np.array(1.5), np.array([1.5]), np.array([[1.5]])])
def test_objective_value_accepts_backend_scalar_shapes(value):
    """Scalar and one-element backend values have the same representation."""
    assert _objective_value(value) == 1.5


def test_objective_value_rejects_vector_objectives():
    """A non-scalar objective indicates an invalid optimization model."""
    with pytest.raises(ValueError, match="Expected a scalar objective value"):
        _objective_value(np.array([1.0, 2.0]))
