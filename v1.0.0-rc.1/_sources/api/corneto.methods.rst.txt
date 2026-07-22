Network inference (:mod:`corneto.methods`)
==========================================
.. currentmodule:: corneto.methods

.. automodule:: corneto.methods
    :no-members:

Signaling
---------

Network methods to infer signaling networks from omics data.

``milp_carnival`` is the simple single-condition formulation.
``CarnivalILP`` is the general formulation for one or more conditions, while
``CarnivalFlow`` provides the flow-based multi-condition alternative.

.. autosummary::
    :toctree: generated/


    CarnivalFlow
    CarnivalILP
    milp_carnival
    signaling.cellnopt_ilp.cellnoptILP

Metabolism
----------

Network methods for flux balance analysis in metabolic networks.

.. autosummary::
    :toctree: generated/

    MultiSampleFBA
    MultiSampleIMAT

Graph optimization
------------------

Methods for extracting optimal subnetworks.

.. autosummary::
    :toctree: generated/

    PrizeCollectingSteinerTree
    SteinerTreeFlow
    create_multisample_shortest_path
    shortest_path
    solve_shortest_path

Solution sampling
-----------------

Utilities used by the indexed alternative-solution tutorials.

.. autosummary::
    :toctree: generated/

    sampler.sample_alternative_solutions
