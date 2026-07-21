Network inference (:mod:`corneto.methods`)
==========================================
.. currentmodule:: corneto.methods

.. automodule:: corneto.methods
    :no-members:

Signaling
---------

Network methods to infer signaling networks from omics data.

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
