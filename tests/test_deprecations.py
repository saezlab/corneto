"""Tests for compatibility APIs retained during the CORNETO 1.x line."""

import pytest

import corneto as cn
from corneto.backend._base import ProblemDef


@pytest.mark.parametrize("name", ["K", "ops"])
def test_deprecated_backend_alias_warns_and_returns_canonical_backend(name):
    """Legacy backend aliases warn and preserve object identity."""
    with pytest.warns(FutureWarning, match=rf"corneto\.{name} is deprecated"):
        backend = getattr(cn, name)

    assert backend is cn.opt


def test_undocumented_backend_compatibility_apis_are_removed():
    """Unused backend compatibility APIs do not survive into the 1.0 surface."""
    assert not hasattr(cn.opt, "Indicators")
    assert not hasattr(cn.opt, "Xor")

    with pytest.raises(TypeError, match="unexpected keyword argument 'graph'"):
        ProblemDef(graph=object())
