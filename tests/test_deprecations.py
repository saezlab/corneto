"""Tests for compatibility APIs retained during the CORNETO 1.x line."""

import importlib

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


def test_signal_package_warns_and_reexports_canonical_cellnopt():
    """The old CellNOpt path warns and preserves function identity."""
    canonical = importlib.import_module("corneto.methods.signaling.cellnopt_ilp")

    with pytest.warns(FutureWarning, match=r"corneto\.methods\.signal is deprecated"):
        legacy = importlib.import_module("corneto.methods.signal.cellnopt_ilp")

    assert legacy.cellnoptILP is canonical.cellnoptILP
    assert legacy.expand_graph_for_flows is canonical.expand_graph_for_flows
