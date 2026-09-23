"""Focused tests for fixed-model mechanistic prediction APIs."""

import numpy as np
import pytest

from corneto.data import Data
from corneto.graph import Graph
from corneto.methods.causal import LinearDAGDiscovery
from corneto.methods.signaling import CellNOptDAG


def _feature_value(data, sample, vertex):
    return next(feature.value for feature in data.samples[sample].features if feature.id == vertex)


def _feature(data, sample, vertex):
    return next(feature for feature in data.samples[sample].features if feature.id == vertex)


def _linear_training_data():
    return Data.from_cdict(
        {
            "baseline": {
                "A": {"mapping": "vertex", "value": 0.0},
                "B": {"mapping": "vertex", "value": 0.0},
            },
            "intervention": {
                "A": {"mapping": "vertex", "value": 2.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 6.0},
            },
        }
    )


def test_linear_fit_predicts_forward_without_refitting(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    model = LinearDAGDiscovery(fit_intercept=False, backend=backend).fit(
        graph,
        _linear_training_data(),
        solve_options={"verbosity": 0},
    )

    def no_solve(*args, **kwargs):
        raise AssertionError("predict must not solve")

    model.problem.solve = no_solve
    held_out = Data.from_cdict(
        {
            "test": {
                "A": {"mapping": "vertex", "value": 3.0},
                # This endogenous observation must not be used as a predictor.
                "B": {"mapping": "vertex", "value": 999.0},
            }
        }
    )
    prediction = model.predict(held_out)
    assert np.isclose(_feature_value(prediction, "test", "A"), 0.0)
    assert np.isclose(_feature_value(prediction, "test", "B"), 0.0)
    assert _feature(prediction, "test", "A").data["predicted"]
    assert not _feature(prediction, "test", "A").data["clamped"]
    metrics = model.evaluate(held_out)
    assert metrics["forward_count"] == 2
    assert metrics["local_count"] == 2
    changed_root = Data.from_cdict(
        {
            "test": {
                "A": {"mapping": "vertex", "value": 4.0},
                "B": {"mapping": "vertex", "value": 999.0},
            }
        }
    )
    changed_prediction = model.predict(changed_root)
    assert changed_prediction.to_dict() == prediction.to_dict()
    assert model.evaluate(changed_root)["forward_loss"] != metrics["forward_loss"]


def test_linear_manual_build_and_solve_supports_prediction(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    model = LinearDAGDiscovery(fit_intercept=False, backend=backend)
    problem = model.build(graph, _linear_training_data())
    problem.solve()
    prediction = model.predict(
        Data.from_cdict({"test": {"A": {"mapping": "vertex", "value": 1.0, "intervened": True}}})
    )
    assert np.isclose(_feature_value(prediction, "test", "B"), 3.0)


def test_linear_soft_target_can_be_missing_and_estimated_group_is_reused(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    training = Data.from_cdict(
        {
            "obs": {
                "A": {"mapping": "vertex", "value": 0.0},
                "B": {"mapping": "vertex", "value": 0.0},
            },
            "do": {
                "A": {"mapping": "vertex", "value": 1.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 2.0},
            },
            "shift_1": {
                "A": {"mapping": "vertex", "value": 1.0},
                "B": {
                    "mapping": "vertex",
                    "value": 5.0,
                    "intervention": "shift",
                    "intervention_group": "drug",
                },
            },
            "shift_2": {
                "A": {"mapping": "vertex", "value": 2.0},
                "B": {
                    "mapping": "vertex",
                    "value": 7.0,
                    "intervention": "shift",
                    "intervention_group": "drug",
                },
            },
        }
    )
    model = LinearDAGDiscovery(fit_intercept=False, backend=backend).fit(graph, training)
    prediction = model.predict(
        Data.from_cdict(
            {
                "held_out": {
                    "A": {"mapping": "vertex", "value": 3.0, "intervened": True},
                    "B": {
                        "mapping": "vertex",
                        "value": None,
                        "intervention": "shift",
                        "intervention_group": "drug",
                    },
                }
            }
        )
    )
    assert np.isclose(_feature_value(prediction, "held_out", "B"), 9.0)
    with pytest.raises(ValueError, match="No fitted soft-intervention shift"):
        model.predict(
            Data.from_cdict(
                {
                    "held_out": {
                        "A": {"mapping": "vertex", "value": 3.0, "intervened": True},
                        "B": {
                            "mapping": "vertex",
                            "value": None,
                            "intervention": "shift",
                            "intervention_group": "new_drug",
                        },
                    }
                }
            )
        )


def test_linear_known_shift_group_is_not_reused_without_test_shift(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    training = Data.from_cdict(
        {
            "obs": {
                "A": {"mapping": "vertex", "value": 0.0},
                "B": {"mapping": "vertex", "value": 0.0},
            },
            "do": {
                "A": {"mapping": "vertex", "value": 1.0, "intervened": True},
                "B": {"mapping": "vertex", "value": 2.0},
            },
            "shift": {
                "A": {"mapping": "vertex", "value": 1.0},
                "B": {
                    "mapping": "vertex",
                    "value": 5.0,
                    "intervention": "shift",
                    "shift": 3.0,
                    "intervention_group": "known_drug",
                },
            },
        }
    )
    model = LinearDAGDiscovery(fit_intercept=False, backend=backend).fit(graph, training)
    with pytest.raises(ValueError, match="No fitted soft-intervention shift"):
        model.predict(
            Data.from_cdict(
                {
                    "held_out": {
                        "A": {"mapping": "vertex", "value": 1.0, "intervened": True},
                        "B": {
                            "mapping": "vertex",
                            "value": None,
                            "intervention": "shift",
                            "intervention_group": "known_drug",
                        },
                    }
                }
            )
        )
    explicit = model.predict(
        Data.from_cdict(
            {
                "held_out": {
                    "A": {"mapping": "vertex", "value": 1.0, "intervened": True},
                    "B": {
                        "mapping": "vertex",
                        "value": None,
                        "intervention": "shift",
                        "shift": 3.0,
                        "intervention_group": "known_drug",
                    },
                }
            }
        )
    )
    assert np.isclose(_feature_value(explicit, "held_out", "B"), 5.0)


def test_linear_rejects_prediction_before_build_or_with_malformed_data(backend):
    model = LinearDAGDiscovery(backend=backend)
    with pytest.raises(ValueError, match="not been built"):
        model.predict(Data.from_cdict({"test": {}}))
    graph = Graph.from_tuples([("A", 1, "B")])
    model.build(graph, _linear_training_data()).solve()
    with pytest.raises(TypeError, match="Data object"):
        model.predict({"test": {}})


def test_linear_evaluation_honors_vertex_and_held_out_sample_weights(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    model = LinearDAGDiscovery(
        fit_intercept=False,
        vertex_weights={"A": 5.0, "B": 1.0},
        backend=backend,
    ).fit(graph, _linear_training_data())
    held_out = Data.from_cdict(
        {
            "s1": {
                "A": {"mapping": "vertex", "value": 1.0},
                "B": {"mapping": "vertex", "value": 2.0},
            },
            "s2": {
                "A": {"mapping": "vertex", "value": 2.0},
                "B": {"mapping": "vertex", "value": 4.0},
            },
        }
    )
    scales = model.loss_scales
    expected_vertex_weighted = (5.0 * (1.0 / scales[0] + 2.0 / scales[0]) + (2.0 / scales[1] + 4.0 / scales[1])) / 12.0
    metrics = model.evaluate(held_out)
    assert np.isclose(metrics["forward_loss"], expected_vertex_weighted)
    expected_sample_weighted = (
        5.0 * (4.0 * 1.0 / scales[0] + 1.0 * 2.0 / scales[0]) + (4.0 * 2.0 / scales[1] + 1.0 * 4.0 / scales[1])
    ) / 30.0
    weighted = model.evaluate(held_out, sample_weights={"s1": 4.0, "s2": 1.0})
    assert np.isclose(weighted["forward_loss"], expected_sample_weighted)
    with pytest.raises(ValueError, match="unknown keys"):
        model.evaluate(held_out, sample_weights={"unseen": 1.0})
    with pytest.raises(ValueError, match="positive"):
        model.evaluate(held_out, sample_weights={"s1": 0.0})
    with pytest.raises(TypeError, match="mapping"):
        model.evaluate(held_out, sample_weights=[1.0, 1.0])


def test_cellnopt_fit_prediction_and_inhibitor_clamping(backend):
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "Y")])
    model = CellNOptDAG(lambda_reg=0, backend=backend).fit(
        graph,
        inputs={"train": {"A": 1}},
        measurements={"train": {"Y": 1}},
        solve_options={"verbosity": 0},
    )
    prediction = model.predict(
        inputs={"active": {"A": 1}, "blocked": {"A": 1}},
        inhibitors={"active": {}, "blocked": {"B": 1}},
    )
    assert _feature_value(prediction, "active", "Y") == 1.0
    assert _feature_value(prediction, "blocked", "Y") == 0.0
    metrics = model.evaluate(
        inputs={"active": {"A": 1}, "blocked": {"A": 1}},
        inhibitors={"active": {}, "blocked": {"B": 1}},
        measurements={"blocked": {"Y": 0}, "active": {"Y": 1}},
    )
    assert metrics["loss"] == 0.0
    assert metrics["by_condition"] == {"active": 0.0, "blocked": 0.0}


def test_fit_solver_options_validate_mapping_and_mutual_exclusion(backend):
    graph = Graph.from_tuples([("A", 1, "Y")])
    model = CellNOptDAG(lambda_reg=0, backend=backend)
    kwargs = {
        "inputs": {"train": {"A": 1}},
        "measurements": {"train": {"Y": 1}},
    }
    with pytest.raises(TypeError, match="mapping"):
        model.fit(graph, **kwargs, solve_options=[])
    with pytest.raises(TypeError, match="either in solve_options"):
        model.fit(graph, **kwargs, solve_options={"verbosity": 0}, verbosity=0)


def test_cellnopt_manual_build_predicts_without_solver(backend):
    graph = Graph.from_tuples([("A", 1, "Y")])
    model = CellNOptDAG(lambda_reg=0, backend=backend)
    problem = model.build(graph, inputs={"A": 1}, measurements={"Y": 1})
    with pytest.raises(ValueError, match="not been solved"):
        model.predict(inputs={"test": {"A": 1}})
    problem.solve()
    problem.solve = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("predict must not solve"))
    prediction = model.predict(inputs={"test": {"A": 1}})
    assert _feature_value(prediction, "test", "Y") == 1.0
    assert not _feature(prediction, "test", "A").data["predicted"]
    assert _feature(prediction, "test", "A").data["forced"]
    assert _feature(prediction, "test", "Y").data["predicted"]


def test_rebuilding_linear_model_invalidates_the_previous_solution(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    model = LinearDAGDiscovery(fit_intercept=False, backend=backend).fit(graph, _linear_training_data())
    model.build(graph, _linear_training_data())
    assert model.solve_result is None
    with pytest.raises(ValueError, match="not been solved"):
        model.predict(Data.from_cdict({"test": {"A": {"mapping": "vertex", "value": 1.0, "intervened": True}}}))


def test_rebuilding_cellnopt_model_invalidates_the_previous_solution(backend):
    graph = Graph.from_tuples([("A", 1, "Y")])
    model = CellNOptDAG(lambda_reg=0, backend=backend).fit(
        graph,
        inputs={"train": {"A": 1}},
        measurements={"train": {"Y": 1}},
    )
    model.build(graph, inputs={"A": 1}, measurements={"Y": 1})
    assert model.solve_result is None
    with pytest.raises(ValueError, match="not been solved"):
        model.predict(inputs={"test": {"A": 1}})


def test_cellnopt_fixed_prediction_matches_solved_training_values(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "Y"),
            ("B", 1, "Y"),
            ("C", -1, "Y"),
            ("D", 1, "AND1"),
            ("E", 1, "AND1"),
            ("AND1", 1, "Z"),
        ]
    )
    inputs = {
        "a": {"A": 1, "C": 1},
        "b": {"B": 1, "C": 1},
        "c_off": {"C": 0},
        "c_on": {"C": 1},
        "and": {"D": 1, "E": 1, "C": 1},
        "d_only": {"D": 1, "C": 1},
        "e_only": {"E": 1, "C": 1},
        "inhibited": {"A": 1, "C": 1},
    }
    inhibitors = {condition: ({"Y": 1} if condition == "inhibited" else {}) for condition in inputs}
    measurements = {
        "a": {"Y": 1, "Z": 0},
        "b": {"Y": 1, "Z": 0},
        "c_off": {"Y": 1, "Z": 0},
        "c_on": {"Y": 0, "Z": 0},
        "and": {"Y": 0, "Z": 1},
        "d_only": {"Y": 0, "Z": 0},
        "e_only": {"Y": 0, "Z": 0},
        "inhibited": {"Y": 0, "Z": 0},
    }
    model = CellNOptDAG(lambda_reg=0, backend=backend)
    problem = model.build_many(graph, inputs=inputs, measurements=measurements, inhibitors=inhibitors)
    problem += problem.expr.reaction_selected == 1
    problem.solve()
    assert np.allclose(np.asarray(problem.expr.reaction_selected.value).reshape(-1), [1, 1, 1, 1])
    solved = np.asarray(problem.expr.vertex_value.value, dtype=float).reshape(problem.expr.vertex_value.shape)
    model.problem.solve = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("predict must not solve"))
    predicted = model.predict(inputs=inputs, inhibitors=inhibitors)
    for condition_index, condition in enumerate(inputs):
        for vertex_index, vertex in enumerate(model.processed_graph.V):
            assert np.isclose(_feature_value(predicted, condition, vertex), solved[vertex_index, condition_index])
