Input and Output (:mod:`corneto.io`)
=====================================
.. currentmodule:: corneto.io

This page documents the public API for the corneto.io package.

File-based loaders accept local paths, HTTP(S) URLs, and readable streams. A
string is treated as a URL only when it has an HTTP(S) scheme; path-like
objects are always local paths. Binary loaders require binary streams, while
the SIF loaders also accept text streams. Borrowed streams are read from their
current position and remain open after loading. Compression defaults to
``"auto"``, which detects gzip, bz2, and xz by magic bytes; pass
``compression=None`` to disable decompression or choose an explicit codec.
HTTP(S) requests use a 30 second timeout by default, configurable with
``timeout``. It limits each blocking HTTP operation, not the total time for a
download. Inline XML or SIF strings are not treated as file contents.

For example, SIF data can be read from a remote URL or an already-open stream:

.. code-block:: python

    from io import BytesIO
    from corneto.io import load_graph_from_sif

    graph_from_url = load_graph_from_sif("https://example.org/network.sif.gz")
    graph_from_stream = load_graph_from_sif(BytesIO(b"A\t1\tB\n"))

For static SBML Level 3 Version 1 Core models with FBC Version 2 and optional
Groups Version 1, the native ``import_sbml_model`` reader parses with the
Python standard library and does not require COBRApy. It preserves SBML
identifiers and records boundary or constant species in graph metadata while
excluding them from mass-balance vertices. The active FBC objective is stored
in the graph's ``fba_objective`` attribute using FBA's minimization convention;
pass it explicitly to ``MultiSampleFBA.build(..., objectives=...)`` when it
should drive optimization. Use ``import_cobra_model`` for SBML features outside
the native reader's supported subset.

.. code-block:: python

    from corneto.io import import_sbml_model
    from corneto.methods.fba import MultiSampleFBA

    model = import_sbml_model("metabolic_model.xml")
    objective = model.get_graph_attributes()["fba_objective"]
    problem = MultiSampleFBA().build(model, objectives=objective)

.. automodule:: corneto.io
    :no-members:

Metabolism
----------

.. autosummary::
    :toctree: generated/

    parse_cobra_model
    cobra_model_to_graph
    import_cobra_model
    import_miom_model
    MetabolicModel
    read_sbml
    sbml_model_to_graph
    import_sbml_model


Signaling
----------

.. autosummary::
    :toctree: generated/

    load_graph_from_sif
    load_graph_from_sif_tuples
