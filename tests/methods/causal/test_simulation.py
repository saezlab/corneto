"""Tests for generic structural causal model simulation."""

import numpy as np
import pytest

from corneto.data import Data
from corneto.graph import EdgeType, Graph
from corneto.methods.causal import LinearDAGDiscovery
from corneto.methods.causal.simulation import SCM, Additive, Laplace, Linear, Node, Normal, Zero, linear_scm


def test_evaluate_preserves_output_and_declared_parent_order():
    scm = SCM(
        {
            "C": Node(("B", "A"), Linear((10, 1)), Zero()),
            "A": Node((), lambda parents, noise: noise, Zero()),
            "B": Node((), lambda parents, noise: noise, Zero()),
        }
    )

    assert scm.variables == ("C", "A", "B")
    assert scm.topological_order == ("A", "B", "C")
    assert scm.parents("C") == ("B", "A")
    values = scm.evaluate({"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0]), "C": np.zeros(2)})
    np.testing.assert_array_equal(values, [[31.0, 1.0, 3.0], [42.0, 2.0, 4.0]])


def test_arbitrary_nonlinear_interaction_and_noise_replay():
    scm = SCM(
        {
            "x": Node((), lambda parents, noise: noise, Normal()),
            "y": Node((), lambda parents, noise: noise, Normal()),
            "z": Node(
                ("x", "y"),
                Additive(lambda parents: np.tanh(parents[:, 0] * parents[:, 1])),
                Laplace(scale=0.1),
            ),
        }
    )
    noise = {"x": np.array([-1.0, 0.0, 1.0]), "y": np.array([2.0, 3.0, 4.0]), "z": np.array([0.1, -0.2, 0.3])}
    values = scm.evaluate(noise)
    np.testing.assert_allclose(values[:, 2], np.tanh(noise["x"] * noise["y"]) + noise["z"])

    noise = scm.draw_noise(10, rng=21)
    np.testing.assert_array_equal(scm.evaluate(noise), scm.evaluate(noise))
    np.testing.assert_array_equal(scm.sample(10, rng=21), scm.sample(10, rng=21))


def test_paired_interventions_reuse_exogenous_noise():
    scm = SCM(
        {
            "A": Node((), lambda parents, noise: noise, Normal()),
            "B": Node(("A",), Linear((2.0,)), Normal(std=0.2)),
        }
    )
    noise = scm.draw_noise(25, rng=7)
    observed = scm.evaluate(noise)
    intervened = scm.do({"A": 3.0}).evaluate(noise)

    np.testing.assert_array_equal(intervened[:, 0], np.full(25, 3.0))
    np.testing.assert_allclose(intervened[:, 1] - observed[:, 1], 6.0 - 2.0 * observed[:, 0])


def test_validation_reports_missing_parents_cycles_and_bad_shapes():
    with pytest.raises(ValueError, match="missing parent"):
        SCM({"A": Node(("missing",), Linear((1.0,)))})
    with pytest.raises(ValueError, match="duplicates"):
        Node(("A", "A"), Linear((1.0, 1.0)))
    with pytest.raises(ValueError, match="cycle"):
        SCM({"A": Node(("B",), Linear((1.0,))), "B": Node(("A",), Linear((1.0,)))})
    malformed = SCM({"A": Node((), lambda parents, noise: np.zeros((len(noise), 1)), Zero())})
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        malformed.sample(3, rng=0)


def test_graph_adapter_keeps_isolates_and_orders_parents_by_vertex_order():
    graph = Graph()
    graph.add_vertices(["A", "B", "C", "isolated"])
    graph.add_edge("B", "C")
    graph.add_edge("A", "C")
    scm = SCM.from_graph(
        graph,
        {
            "A": lambda parents, noise: noise,
            "B": lambda parents, noise: noise,
            "C": lambda parents, noise: parents[:, 0] + 10 * parents[:, 1] + noise,
            "isolated": lambda parents, noise: noise,
        },
        noises=Zero(),
    )

    assert scm.variables == ("A", "B", "C", "isolated")
    assert scm.parents("C") == ("A", "B")
    result = scm.evaluate({name: np.array([value]) for name, value in zip(scm.variables, (1, 2, 0, 0), strict=True)})
    np.testing.assert_array_equal(result, [[1.0, 2.0, 21.0, 0.0]])


@pytest.mark.parametrize(
    "build_graph, message",
    [
        (lambda: _graph_with_edge("A", "B", type=EdgeType.UNDIRECTED), "not directed"),
        (lambda: _graph_with_edge(["A", "B"], "C"), "hyperedge"),
        (lambda: _graph_with_edge([], "A"), "boundary edge"),
        (lambda: _graph_with_edge("A", "A"), "self-loop"),
        (lambda: _parallel_graph(), "parallel edge"),
    ],
)
def test_graph_adapter_rejects_unsupported_edges(build_graph, message):
    graph = build_graph()
    mechanisms = {variable: lambda parents, noise: noise for variable in graph.V}
    with pytest.raises(ValueError, match=message):
        SCM.from_graph(graph, mechanisms)


def test_graph_adapter_rejects_cycles():
    graph = Graph.from_tuples([("A", 1, "B"), ("B", 1, "A")])
    mechanisms = {variable: lambda parents, noise: noise for variable in graph.V}
    with pytest.raises(ValueError, match="cycle"):
        SCM.from_graph(graph, mechanisms)


def test_linear_factory_uses_weights_or_explicit_weight_attribute():
    graph = Graph()
    graph.add_vertices(["A", "B"])
    graph.add_edge("A", "B", interaction=-1, coefficient=1.5)
    model = linear_scm(graph, weight_attribute="coefficient", biases={"B": 0.25}, noise=Zero())
    result = model.do({"A": 2}).sample(1, rng=3)
    np.testing.assert_array_equal(result, [[2.0, 3.25]])

    random_model = linear_scm(graph, rng=4, noise=Zero())
    weight = random_model.nodes["B"].mechanism.weights[0]
    assert abs(weight) >= 0.5


def test_linear_factory_rejects_missing_explicit_weights():
    graph = Graph.from_tuples([("A", 1, "B")])
    with pytest.raises(ValueError, match="Weight keys must match"):
        linear_scm(graph, weights={})


def test_replacements_are_immutable_and_general_replacement_cannot_be_misannotated():
    scm = SCM({"A": Node((), lambda parents, noise: noise, Zero())})
    replaced = scm.replace({"A": Node((), lambda parents, noise: np.ones(len(noise)), Zero())})
    np.testing.assert_array_equal(scm.sample(2), [[0.0], [0.0]])
    np.testing.assert_array_equal(replaced.sample(2), [[1.0], [1.0]])
    with pytest.raises(ValueError, match="arbitrary node replacements"):
        replaced.to_data(replaced.sample(2))
    with pytest.raises(AttributeError, match="immutable"):
        scm._nodes = {}


def test_sequential_shifts_accumulate_the_effect_and_metadata():
    shifted = SCM({"A": Node((), lambda parents, noise: noise, Zero())}).shift({"A": 1.0}).shift({"A": 2.0})
    np.testing.assert_array_equal(shifted.sample(2, rng=0), [[3.0], [3.0]])
    data = shifted.sample_data(1, rng=0)
    feature = data.samples[0].features[0]
    assert feature.data["intervention"] == "shift"
    assert feature.data["shift"] == 3.0


def test_data_conversion_retains_hard_and_known_shift_metadata():
    scm = (
        SCM(
            {
                "A": Node((), lambda parents, noise: noise, Zero()),
                "B": Node(("A",), Linear((2.0,)), Zero()),
                "C": Node(("B",), Linear((0.5,)), Zero()),
            }
        )
        .do({"A": 1.0})
        .shift({"B": 3.0})
    )
    data = scm.sample_data(2, rng=0, sample_ids=("rep1", "rep2"), intervention_group="drug")

    assert isinstance(data, Data)
    hard = data.samples["rep1"].query.filter(lambda feature: feature.id == "A").to_list()[0]
    shift = data.samples["rep1"].query.filter(lambda feature: feature.id == "B").to_list()[0]
    assert hard.data["intervention"] == "hard"
    assert hard.data["intervention_group"] == "drug"
    assert shift.data["intervention"] == "shift"
    assert shift.data["shift"] == 3.0
    assert shift.data["intervention_group"] == "drug"
    method = LinearDAGDiscovery()
    method.preprocess(Graph.from_tuples([("A", 1, "B"), ("B", 1, "C")]), data)
    design = method._intervention_design
    assert design.hard[0].all()
    assert design.shift[1].all()
    np.testing.assert_array_equal(design.known_shift[1], [3.0, 3.0])


def _graph_with_edge(source, target, **kwargs):
    graph = Graph()
    graph.add_edge(source, target, **kwargs)
    return graph


def _parallel_graph():
    graph = Graph()
    graph.add_edge("A", "B")
    graph.add_edge("A", "B")
    return graph
