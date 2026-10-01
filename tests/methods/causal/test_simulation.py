"""Tests for generic structural causal model simulation."""

from copy import deepcopy

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


def test_evaluate_snapshots_input_noise_and_reused_mechanism_buffers():
    scratch = np.empty(3)

    def mutate_noise(parents, noise):
        noise += 1.0
        return noise

    def reuse_output(parents, noise):
        np.copyto(scratch, noise)
        return scratch

    scm = SCM(
        {
            "mutated": Node((), mutate_noise, Zero()),
            "first": Node((), reuse_output, Zero()),
            "second": Node((), reuse_output, Zero()),
        }
    )
    supplied = {
        "mutated": np.array([1.0, 2.0, 3.0]),
        "first": np.array([4.0, 5.0, 6.0]),
        "second": np.array([7.0, 8.0, 9.0]),
    }
    original = {variable: values.copy() for variable, values in supplied.items()}
    first = scm.evaluate(supplied)
    second = scm.evaluate(supplied)

    np.testing.assert_array_equal(first, [[2, 4, 7], [3, 5, 8], [4, 6, 9]])
    np.testing.assert_array_equal(second, first)
    for variable in scm.variables:
        np.testing.assert_array_equal(supplied[variable], original[variable])


def test_draw_noise_copies_sampler_buffers_between_nodes():
    scratch = np.empty(4)

    def reusable_sampler(rng, n):
        scratch[:] = rng.normal(size=n)
        return scratch

    scm = SCM(
        {
            "A": Node((), lambda parents, noise: noise, reusable_sampler),
            "B": Node((), lambda parents, noise: noise, reusable_sampler),
        }
    )
    draws = scm.draw_noise(4, rng=9)
    expected_rng = np.random.default_rng(9)

    np.testing.assert_array_equal(draws["A"], expected_rng.normal(size=4))
    np.testing.assert_array_equal(draws["B"], expected_rng.normal(size=4))
    assert not np.shares_memory(draws["A"], draws["B"])


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


def test_invalid_sample_counts_noise_inputs_and_nonfinite_outputs():
    scm = SCM({"A": Node((), lambda parents, noise: noise, Zero())})
    with pytest.raises(ValueError, match="nonnegative integer"):
        scm.sample(-1)
    with pytest.raises(TypeError, match="nonnegative integer"):
        scm.sample(2.5)
    with pytest.raises(TypeError, match="nonnegative integer"):
        scm.sample(True)
    with pytest.raises(ValueError, match="one-dimensional"):
        scm.evaluate({"A": np.zeros((2, 1))})
    with pytest.raises(ValueError, match="Noise keys must match"):
        scm.evaluate({"extra": np.zeros(2)})
    with pytest.raises(ValueError, match="same sample count"):
        SCM({"A": Node((), lambda parents, noise: noise), "B": Node((), lambda parents, noise: noise)}).evaluate(
            {"A": np.zeros(2), "B": np.zeros(3)}
        )
    with pytest.raises(ValueError, match="finite values"):
        scm.evaluate({"A": np.array([np.nan])})
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        SCM({"A": Node((), lambda parents, noise: noise, lambda rng, n: np.zeros((n, 1)))}).draw_noise(2)
    with pytest.raises(ValueError, match="finite values"):
        SCM({"A": Node((), lambda parents, noise: noise, lambda rng, n: np.full(n, np.inf))}).draw_noise(2)
    with pytest.raises(ValueError, match="finite values"):
        SCM({"A": Node((), lambda parents, noise: np.full(len(noise), np.nan), Zero())}).sample(2)


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


def test_linear_factory_respects_graph_parent_order_for_explicit_weights():
    graph = Graph()
    graph.add_vertices(["A", "B", "C"])
    graph.add_edge("B", "C")
    graph.add_edge("A", "C")
    scm = linear_scm(graph, weights={("A", "C"): 2.0, ("B", "C"): 3.0}, noise=Zero())

    assert scm.parents("C") == ("A", "B")
    np.testing.assert_array_equal(scm.do({"A": 1.0, "B": 4.0}).sample(1), [[1.0, 4.0, 14.0]])


def test_linear_factory_rejects_missing_explicit_weights():
    graph = Graph.from_tuples([("A", 1, "B")])
    with pytest.raises(ValueError, match="Weight keys must match"):
        linear_scm(graph, weights={})


def test_hashable_nonstring_labels_work_in_models_and_data():
    tuple_label, integer_label = ("protein", "A"), 17
    scm = SCM(
        {
            tuple_label: Node((), lambda parents, noise: noise, Zero()),
            integer_label: Node((tuple_label,), Linear((2.0,)), Zero()),
        }
    ).do({tuple_label: 3.0})
    data = scm.sample_data(1, rng=0, sample_ids=("sample",))

    np.testing.assert_array_equal(scm.sample(1, rng=0), [[3.0, 6.0]])
    assert data.samples["sample"].features[0].id == tuple_label
    assert data.samples["sample"].features[1].id == integer_label


def test_factory_generator_reproducibility_and_separate_sample_state():
    graph = Graph.from_tuples([("A", 1, "B")])
    first = linear_scm(graph, rng=15, noise=Normal())
    second = linear_scm(graph, rng=15, noise=Normal())
    assert first.nodes["B"].mechanism.weights == second.nodes["B"].mechanism.weights

    generation_rng = np.random.default_rng(18)
    initial_state = deepcopy(generation_rng.bit_generator.state)
    generated = linear_scm(graph, rng=generation_rng, noise=Normal())
    generated_state = deepcopy(generation_rng.bit_generator.state)
    assert generated_state != initial_state

    sample_rng = np.random.default_rng(23)
    initial_sample_state = deepcopy(sample_rng.bit_generator.state)
    generated.sample(5, rng=sample_rng)
    assert sample_rng.bit_generator.state != initial_sample_state
    assert generation_rng.bit_generator.state == generated_state


def test_node_replace_preserves_fields_and_runs_node_validation():
    original = Node(("A",), Linear((2.0,)), Normal(std=0.5))
    noise_changed = original.replace(noise=Zero())
    assert noise_changed.parents == ("A",)
    assert noise_changed.mechanism is original.mechanism
    assert isinstance(noise_changed.noise, Zero)
    assert original.noise.std == 0.5

    fully_changed = original.replace(parents=(), mechanism=lambda parents, noise: noise + 3.0, noise=Zero())
    assert fully_changed.parents == ()
    assert fully_changed.mechanism is not original.mechanism
    assert isinstance(fully_changed.noise, Zero)

    with pytest.raises(AttributeError):
        noise_changed.parents = ()
    with pytest.raises(TypeError, match="unexpected keyword"):
        original.replace(unknown_field=True)
    with pytest.raises(ValueError, match="duplicates"):
        original.replace(parents=("A", "A"))
    with pytest.raises(TypeError, match="mechanism must be callable"):
        original.replace(mechanism=None)


def test_replace_edits_baseline_and_exports_data_without_intervention_metadata():
    scm = SCM(
        {
            "A": Node((), lambda parents, noise: np.ones(len(noise)), Zero()),
            "B": Node(("A",), Linear((2.0,)), Zero()),
        }
    )
    changed = scm.replace({"B": scm.nodes["B"].replace(mechanism=Linear((3.0,)))})
    np.testing.assert_array_equal(scm.sample(2), [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_array_equal(changed.sample(2), [[1.0, 3.0], [1.0, 3.0]])
    data = changed.sample_data(1, rng=0)

    assert all("intervention" not in feature.data for feature in data.samples[0].features)
    with pytest.raises(AttributeError, match="immutable"):
        scm._nodes = {}


def test_intervene_marks_arbitrary_equations_and_do_supersedes_marker():
    scm = SCM({"A": Node((), lambda parents, noise: noise, Zero())})
    arbitrary_node = Node((), lambda parents, noise: np.full(len(noise), 4.0), Zero())
    intervened = scm.intervene({"A": arbitrary_node})

    np.testing.assert_array_equal(scm.sample(2), [[0.0], [0.0]])
    np.testing.assert_array_equal(intervened.sample(2), [[4.0], [4.0]])
    with pytest.raises(ValueError, match="arbitrary node interventions"):
        intervened.to_data(intervened.sample(2))
    with pytest.raises(ValueError, match="arbitrary node interventions"):
        intervened.shift({"A": 1.0}).to_data(intervened.shift({"A": 1.0}).sample(2))

    hard = intervened.do({"A": 7.0})
    feature = hard.sample_data(1, rng=0).samples[0].features[0]
    assert feature.value == 7.0
    assert feature.data["intervention"] == "hard"


def test_replace_clears_only_edited_intervention_metadata():
    scm = SCM(
        {
            "A": Node((), lambda parents, noise: noise, Zero()),
            "B": Node((), lambda parents, noise: noise, Zero()),
            "C": Node((), lambda parents, noise: noise, Zero()),
        }
    )
    arbitrary_c = Node((), lambda parents, noise: np.full(len(noise), 5.0), Zero())
    mixed = scm.do({"A": 1.0}).shift({"B": 2.0}).intervene({"C": arbitrary_c})
    changed_b = mixed.replace({"B": scm.nodes["B"].replace(mechanism=lambda parents, noise: noise + 10.0)})

    assert changed_b._interventions["A"].kind == "hard"
    assert "B" not in changed_b._interventions
    assert changed_b._interventions["C"].kind == "replace"
    with pytest.raises(ValueError, match="arbitrary node interventions"):
        changed_b.sample_data(1, rng=0)

    changed_c = changed_b.replace({"C": scm.nodes["C"].replace(mechanism=lambda parents, noise: noise + 20.0)})
    assert changed_c._interventions == {"A": mixed._interventions["A"]}
    data = changed_c.sample_data(1, rng=0)
    features = {feature.id: feature for feature in data.samples[0].features}
    assert features["A"].data["intervention"] == "hard"
    assert "intervention" not in features["B"].data
    assert "intervention" not in features["C"].data
    assert "B" not in changed_c._interventions
    assert mixed._interventions["B"].kind == "shift"
    assert mixed._interventions["C"].kind == "replace"


def test_replace_and_intervene_both_validate_acyclic_topology():
    scm = SCM(
        {
            "A": Node((), lambda parents, noise: noise, Zero()),
            "B": Node(("A",), Linear((1.0,)), Zero()),
        }
    )
    cyclic_a = Node(("B",), Linear((1.0,)), Zero())

    with pytest.raises(ValueError, match="cycle"):
        scm.replace({"A": cyclic_a})
    with pytest.raises(ValueError, match="cycle"):
        scm.intervene({"A": cyclic_a})

    np.testing.assert_array_equal(scm.sample(1), [[0.0, 0.0]])


def test_sequential_shifts_accumulate_the_effect_and_metadata():
    shifted = SCM({"A": Node((), lambda parents, noise: noise, Zero())}).shift({"A": 1.0}).shift({"A": 2.0})
    np.testing.assert_array_equal(shifted.sample(2, rng=0), [[3.0], [3.0]])
    data = shifted.sample_data(1, rng=0)
    feature = data.samples[0].features[0]
    assert feature.data["intervention"] == "shift"
    assert feature.data["shift"] == 3.0


def test_hard_intervention_shift_composition_preserves_exported_semantics():
    base = SCM({"A": Node((), lambda parents, noise: noise + 1.0, Normal())})
    hard_then_shift = base.do({"A": 2.0}).shift({"A": 3.0})
    shifted_then_hard = base.shift({"A": 20.0}).do({"A": 7.0})

    hard_data = hard_then_shift.sample_data(2, rng=4, intervention_group="clamp")
    hard_feature = hard_data.samples[0].features[0]
    assert hard_feature.value == 5.0
    assert hard_feature.data["intervention"] == "hard"
    assert "shift" not in hard_feature.data
    np.testing.assert_array_equal(hard_then_shift.sample(2), [[5.0], [5.0]])
    np.testing.assert_array_equal(shifted_then_hard.sample(2), [[7.0], [7.0]])
    shifted_hard = shifted_then_hard.sample_data(1).samples[0].features[0].data
    assert shifted_hard["intervention"] == "hard"
    assert "shift" not in shifted_hard


def test_intervention_composition_rejects_overflow():
    scm = SCM({"A": Node((), lambda parents, noise: noise, Zero())})
    with pytest.raises(ValueError, match="Accumulated shift"):
        scm.shift({"A": 1e308}).shift({"A": 1e308})
    with pytest.raises(ValueError, match="Accumulated intervention"):
        scm.do({"A": 1e308}).shift({"A": 1e308})


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
