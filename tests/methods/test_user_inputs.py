"""Tests for method-specific user input interfaces."""

import pytest

from corneto.data import Data
from corneto.graph import EdgeType, Graph
from corneto.methods import (
    CarnivalILP,
    MultiSampleFBA,
    MultiSampleIMAT,
    PrizeCollectingSteinerTree,
    SteinerTreeFlow,
)


def _metabolic_model():
    graph = Graph()
    graph.add_edge("A", "B", id="R1", default_lb=0, default_ub=10, GPR="G1")
    return graph


def _network_graph():
    graph = Graph()
    graph.add_edge("A", "B", type=EdgeType.UNDIRECTED, value=1.0)
    graph.add_edge("B", "C", type=EdgeType.UNDIRECTED, value=2.0)
    return graph


def test_carnival_explicit_single_and_many_inputs(backend):
    pkn = Graph.from_tuples([("EGFR", 1, "JUN")])
    single = CarnivalILP(beta_weight=0, backend=backend)
    single.build(
        pkn,
        perturbations={"EGFR": 1},
        transcription_factors={"JUN": -1},
    )

    features = {feature.id: feature for feature in single.processed_data.samples["condition"].features}
    assert features["EGFR"].data["role"] == "input"
    assert features["JUN"].data["role"] == "output"

    many = CarnivalILP(beta_weight=0, backend=backend)
    many.build_many(
        pkn,
        perturbations={"control": {"EGFR": 1}, "treated": {"EGFR": -1}},
        transcription_factors={"control": {"JUN": 1}, "treated": {"JUN": -1}},
    )
    assert list(many.processed_data.samples) == ["control", "treated"]


def test_carnival_rejects_mismatched_conditions(backend):
    pkn = Graph.from_tuples([("EGFR", 1, "JUN")])
    with pytest.raises(ValueError, match="Condition names in transcription_factors must match perturbations"):
        CarnivalILP(backend=backend).build_many(
            pkn,
            perturbations={"control": {"EGFR": 1}},
            transcription_factors={"treated": {"JUN": 1}},
        )


def test_fba_explicit_inputs_and_validation(backend):
    model = _metabolic_model()
    method = MultiSampleFBA(backend=backend)
    method.build(model, objectives={"R1": -1}, reaction_bounds={"R1": (1, 8)})

    feature = method.processed_data.samples["condition"].features[0]
    assert feature.data == {
        "id": "R1",
        "value": -1.0,
        "mapping": "edge",
        "role": "objective",
        "lower_bound": 1.0,
        "upper_bound": 8.0,
    }

    with pytest.raises(ValueError, match="Unknown reaction 'missing' in reaction_bounds"):
        method.build(model, reaction_bounds={"missing": (0, 1)})
    with pytest.raises(ValueError, match=r"Lower bound 2\.0 exceeds upper bound 1\.0"):
        method.build(model, reaction_bounds={"R1": (2, 1)})

    many = MultiSampleFBA(backend=backend)
    many.build_many(
        model,
        objectives={"control": {"R1": -1}, "treated": {"R1": -1}},
        reaction_bounds={"control": {}, "treated": {"R1": (0, 0)}},
    )
    assert list(many.processed_data.samples) == ["control", "treated"]


def test_imat_separates_objectives_and_expression(backend):
    model = _metabolic_model()
    method = MultiSampleIMAT(backend=backend)
    method.build(model, reaction_scores={"R1": 2}, objectives={"R1": -1})

    feature = method.processed_data.samples["condition"].features[0]
    assert feature.data["role"] == "objective"
    assert feature.data["value"] == -1.0
    assert feature.data["imat_score"] == 2.0
    assert {objective.name for objective in method.problem.objectives} >= {
        "objective_condition__R1",
        "imat_fit_pos_condition_0",
    }

    gene_method = MultiSampleIMAT(backend=backend)
    gene_method.build(model, gene_expression={"G1": 2}, objectives={"R1": -1})
    gene_feature = next(
        feature
        for feature in gene_method.processed_data.samples["condition"].features
        if feature.id == "R1"
    )
    assert gene_feature.data["role"] == "objective"
    assert gene_feature.data["imat_score"] == 2.0

    many = MultiSampleIMAT(backend=backend)
    many.build_many(
        model,
        reaction_scores={"control": {"R1": 2}, "treated": {"R1": -2}},
    )
    assert list(many.processed_data.samples) == ["control", "treated"]

    with pytest.raises(ValueError, match="exactly one"):
        method.build(model, gene_expression={"G1": 1}, reaction_scores={"R1": 1})
    with pytest.raises(ValueError, match="Unknown gene 'missing'"):
        method.build(model, gene_expression={"missing": 1})


def test_pcst_and_steiner_explicit_inputs(backend):
    graph = _network_graph()

    pcst = PrizeCollectingSteinerTree(root_vertex="A", strict_acyclic=False, backend=backend)
    pcst.build(graph, prizes={"C": 5}, terminals=["A"], edge_costs={0: 1, 1: 2})
    pcst_features = {feature.id: feature for feature in pcst.processed_data.samples["condition"].features}
    assert pcst_features["C"].data["role"] == "prize"
    assert pcst_features["A"].data["role"] == "terminal"

    steiner = SteinerTreeFlow(root_vertex="A", strict_acyclic=False, backend=backend)
    steiner.build(graph, terminals=["A", "C"], edge_costs={0: 1, 1: 2})
    terminals = {
        feature.id
        for feature in steiner.processed_data.samples["condition"].features
        if feature.data.get("role") == "terminal"
    }
    assert terminals == {"A", "C"}

    with pytest.raises(ValueError, match="Invalid edge index 2"):
        steiner.build(graph, terminals=["A", "C"], edge_costs={2: 1})

    pcst_many = PrizeCollectingSteinerTree(root_vertex="A", strict_acyclic=False, backend=backend)
    pcst_many.build_many(
        graph,
        prizes={"first": {"C": 5}, "second": {"B": 3}},
        terminals={"first": ["A"], "second": ["A"]},
        edge_costs={"first": {0: 1}, "second": {1: 2}},
    )
    assert list(pcst_many.processed_data.samples) == ["first", "second"]

    steiner_many = SteinerTreeFlow(root_vertex="A", strict_acyclic=False, backend=backend)
    steiner_many.build_many(
        graph,
        terminals={"first": ["A", "B"], "second": ["A", "C"]},
    )
    assert list(steiner_many.processed_data.samples) == ["first", "second"]


def test_legacy_data_path_warns_and_named_path_does_not(backend):
    model = _metabolic_model()
    data = Data.from_cdict({"sample": {"R1": {"value": -1, "role": "objective"}}})
    method = MultiSampleFBA(backend=backend)

    with pytest.warns(DeprecationWarning, match="build_from_data"):
        method.build(model, data)
    method.build_from_data(model, data)
