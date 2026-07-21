"""Compatibility tests for the deprecated methods namespace."""

import importlib

import pytest


def test_future_namespace_warns_and_reexports_canonical_methods():
    """The namespace warns and exposes the canonical class objects."""
    import corneto.methods.future as future

    with pytest.warns(FutureWarning, match=r"corneto\.methods\.future is deprecated"):
        future = importlib.reload(future)

    from corneto.methods import (
        CarnivalFlow,
        CarnivalILP,
        MultiSampleFBA,
        MultiSampleIMAT,
        PrizeCollectingSteinerTree,
        SteinerTreeFlow,
    )

    assert future.CarnivalFlow is CarnivalFlow
    assert future.CarnivalILP is CarnivalILP
    assert future.MultiSampleFBA is MultiSampleFBA
    assert future.MultiSampleIMAT is MultiSampleIMAT
    assert future.PrizeCollectingSteinerTree is PrizeCollectingSteinerTree
    assert future.SteinerTreeFlow is SteinerTreeFlow


def test_future_submodules_reexport_identical_classes():
    """Documented future submodules preserve class identity."""
    from corneto.methods.carnival import CarnivalFlow, CarnivalILP
    from corneto.methods.fba import MultiSampleFBA
    from corneto.methods.future.carnival import (
        CarnivalFlow as FutureCarnivalFlow,
    )
    from corneto.methods.future.carnival import CarnivalILP as FutureCarnivalILP
    from corneto.methods.future.fba import MultiSampleFBA as FutureFBA
    from corneto.methods.future.imat import MultiSampleIMAT as FutureIMAT
    from corneto.methods.future.pcst import PrizeCollectingSteinerTree as FuturePCST
    from corneto.methods.future.steiner import SteinerTreeFlow as FutureSteiner
    from corneto.methods.imat import MultiSampleIMAT
    from corneto.methods.pcst import PrizeCollectingSteinerTree
    from corneto.methods.steiner import SteinerTreeFlow

    assert FutureCarnivalFlow is CarnivalFlow
    assert FutureCarnivalILP is CarnivalILP
    assert FutureFBA is MultiSampleFBA
    assert FutureIMAT is MultiSampleIMAT
    assert FuturePCST is PrizeCollectingSteinerTree
    assert FutureSteiner is SteinerTreeFlow
