Network inference (:mod:`corneto.methods`)
==========================================
.. currentmodule:: corneto.methods

.. automodule:: corneto.methods
    :no-members:

Signaling
---------

Network methods to infer signaling networks from omics data.

``CarnivalILP.build`` and ``CarnivalFlow.build`` accept explicit perturbation
and transcription-factor mappings for one condition. Their ``build_many``
methods accept named conditions. ``milp_carnival`` remains as a compatibility
formulation.

``CellNOptILP.build`` accepts binary input, measurement, and optional inhibitor
mappings. ``build_many`` infers one shared connected reaction model while
evaluating its Boolean state independently in every named condition.

.. autosummary::
    :toctree: generated/


    CarnivalFlow
    CarnivalILP
    signaling.CellNOptILP
    BidirectionalPHONEMeS
    PHONEMeS
    compute_phonemes_scores
    milp_carnival

CellNOpt visualization
~~~~~~~~~~~~~~~~~~~~~~

CellNOpt plotting utilities use the standard CORNETO graph renderers for
network views and return Matplotlib figure/axes objects for data-fit views.

.. autosummary::
    :toctree: generated/

    signaling.plot_cellnopt_model
    signaling.plot_cellnopt_fit

Metabolism
----------

Network methods for flux balance analysis in metabolic networks.
Use ``build`` with explicit objectives, bounds, or expression values;
``build_from_data`` provides the advanced generic-data interface.

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
