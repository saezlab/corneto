"""Contract tests for the CORNETO 1.0 public API."""

import importlib.util

import corneto as cn
import corneto.backend as backend
import corneto.io as io
import corneto.methods as methods
import corneto.ml as ml
from corneto.methods import sampler


def test_root_api_surface():
    """The documented root namespace remains explicit and stable."""
    assert cn.__all__ == [
        "DEFAULT_BACKEND",
        "DEFAULT_SOLVER",
        "Attr",
        "Attributes",
        "Data",
        "Direction",
        "EdgeType",
        "Feature",
        "Graph",
        "GraphData",
        "K",
        "Sample",
        "VarType",
        "available_backends",
        "disable_logging",
        "enable_logging",
        "info",
        "ops",
        "opt",
        "pl",
        "set_verbosity",
        "suppress_output",
    ]


def test_backend_api_surface():
    """Backend exports include supported defaults and omit settings internals."""
    assert backend.__all__ == [
        "DEFAULT_BACKEND",
        "DEFAULT_SOLVER",
        "Backend",
        "CvxpyBackend",
        "PicosBackend",
        "VarType",
        "available_backends",
    ]
    assert "s" not in backend.__all__


def test_methods_api_surface_and_identity():
    """Canonical method imports expose the exact documented objects."""
    assert methods.__all__ == [
        "CarnivalFlow",
        "CarnivalILP",
        "MultiSampleFBA",
        "MultiSampleIMAT",
        "PHONEMeS",
        "PrizeCollectingSteinerTree",
        "SteinerTreeFlow",
        "create_multisample_shortest_path",
        "milp_carnival",
        "shortest_path",
        "solve_shortest_path",
    ]

    from corneto.methods.shortest_path import create_multisample_shortest_path

    assert methods.create_multisample_shortest_path is create_multisample_shortest_path


def test_ml_api_replaces_private_module():
    """KPNN helpers live in the public ML module without a private-path shim."""
    assert ml.__all__ == [
        "build_dagnn",
        "index_selector",
        "kfold_nonzero_splits",
        "plot_model",
        "signed_dense",
        "toposort",
    ]
    assert importlib.util.find_spec("corneto._ml") is None


def test_io_surface_uses_graph_methods_for_serialization():
    """Duplicate module-level graph serialization helpers are removed."""
    assert io.__all__ == [
        "cobra_model_to_graph",
        "import_cobra_model",
        "import_miom_model",
        "load_graph_from_sif",
        "load_graph_from_sif_tuples",
        "parse_cobra_model",
    ]
    assert not hasattr(io, "load_corneto_graph")
    assert not hasattr(io, "save_corneto_graph")
    assert callable(cn.Graph.load)
    assert callable(cn.Graph.save)


def test_sampler_remains_a_supported_module():
    """The indexed sampling tutorials retain their public implementation."""
    assert sampler.__all__ == ["sample_alternative_solutions"]
    assert callable(sampler.sample_alternative_solutions)
