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

``CellNOptDAG.build`` accepts binary input, measurement, and optional inhibitor
mappings. ``build_many`` infers one shared connected reaction model while
evaluating its Boolean state independently in every named condition.

``LinearDAGDiscovery`` and ``signaling.CellNOptDAG`` are reusable mechanistic
models. Both provide a thin ``fit(...)`` convenience wrapper that returns the
same method instance, while retaining the composable
``build(...)``/``build_many(...) -> ProblemDef -> solve(...)`` lifecycle. After
a usable solution has been obtained, ``predict(...)`` applies the fixed model
to new externally specified conditions and ``evaluate(...)`` compares those
predictions with supplied measurements. ``LinearDAGDiscovery`` also exposes
``residuals(...)`` for local structural-equation diagnostics: its forward loss
and local loss have different semantics. ``CellNOptDAG`` evaluates its fixed
selected reactions by deterministic Boolean propagation. Most other
``Method`` subclasses remain explanatory optimization methods; the base class
does not expose an sklearn-style fit/predict API.

.. autosummary::
    :toctree: generated/


    CarnivalFlow
    CarnivalILP
    signaling.CellNOptDAG
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

Causal discovery
----------------

``LinearDAGDiscovery`` infers a sparse, acyclic linear structural model from
continuous observational and perfect-intervention measurements. Candidate
interactions come from a directed prior-knowledge network, and commodity flows
ensure that selected edges are supported by intervention-to-response paths.

.. autosummary::
    :toctree: generated/

    LinearDAGDiscovery

CellNOpt and AnnNet
~~~~~~~~~~~~~~~~~~~

These helpers keep the signed network, perturbation conditions, and fitted
CellNOptDAG results in one AnnNet object.

.. autosummary::
    :toctree: generated/

    signaling.add_cellnopt_conditions
    signaling.build_cellnopt_from_annnet
    signaling.add_cellnopt_results

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
