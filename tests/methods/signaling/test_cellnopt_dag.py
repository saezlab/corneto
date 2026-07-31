import numpy as np

from corneto.backend import PicosBackend
from corneto.graph import Graph
from corneto.methods.signaling.cellnopt_dag import CellNOptDAG


def _vertex_values(method, problem, vertex):
    index = method.processed_graph.V.index(vertex)
    values = np.asarray(problem.expr.vertex_value.value, dtype=float)
    return values.reshape(problem.expr.vertex_value.shape)[index].reshape(-1)


def _flat_values(expression):
    return np.asarray(expression.value, dtype=float).reshape(expression.shape).reshape(-1)


def _assert_infeasible(problem, backend):
    if isinstance(backend, PicosBackend):
        result = problem.solve(primals=None)
    else:
        result = problem.solve()
    assert result.status == "infeasible"


def _solve(method, graph, *, inputs, measurements, inhibitors=None):
    problem = method.build_many(
        graph,
        inputs=inputs,
        measurements=measurements,
        inhibitors=inhibitors,
    )
    result = problem.solve()
    assert result.status == "optimal"
    return problem


def test_and_gate_is_one_reaction_with_conjunctive_truth(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "AND1"),
            ("B", 1, "AND1"),
            ("AND1", 1, "Y"),
        ]
    )
    inputs = {
        "neither": {"A": 0, "B": 0},
        "a_only": {"A": 1, "B": 0},
        "b_only": {"A": 0, "B": 1},
        "both": {"A": 1, "B": 1},
    }
    measurements = {
        "neither": {"Y": 0},
        "a_only": {"Y": 0},
        "b_only": {"Y": 0},
        "both": {"Y": 1},
    }

    method = CellNOptDAG(lambda_reg=1e-3, backend=backend)
    problem = _solve(
        method,
        graph,
        inputs=inputs,
        measurements=measurements,
    )

    assert len(method.reactions) == 1
    assert method.reactions[0].positive_literals == ("A", "B")
    assert np.allclose(_flat_values(problem.expr.reaction_selected), [1])
    assert np.allclose(
        np.asarray(problem.expr.reaction_active.value).reshape(1, -1),
        [[0, 0, 0, 1]],
    )
    assert np.allclose(_vertex_values(method, problem, "Y"), [0, 0, 0, 1])
    assert np.all(np.asarray(problem.expr.flow.value)[:2] >= method.epsilon)


def test_or_is_induced_by_multiple_active_producing_reactions(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "Y"),
            ("B", 1, "Y"),
        ]
    )
    inputs = {
        "neither": {"A": 0, "B": 0},
        "a_only": {"A": 1, "B": 0},
        "b_only": {"A": 0, "B": 1},
        "both": {"A": 1, "B": 1},
    }
    measurements = {
        "neither": {"Y": 0},
        "a_only": {"Y": 1},
        "b_only": {"Y": 1},
        "both": {"Y": 1},
    }

    method = CellNOptDAG(lambda_reg=1e-3, backend=backend)
    problem = _solve(
        method,
        graph,
        inputs=inputs,
        measurements=measurements,
    )

    assert np.allclose(_flat_values(problem.expr.reaction_selected), [1, 1])
    assert np.allclose(
        np.asarray(problem.expr.reaction_active.value),
        [[0, 1, 0, 1], [0, 0, 1, 1]],
    )
    assert np.allclose(_vertex_values(method, problem, "Y"), [0, 1, 1, 1])


def test_positive_and_negative_alternatives_fit_complementary_conditions(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "Y"),
            ("A", -1, "Y"),
        ]
    )
    method = CellNOptDAG(lambda_reg=1e-3, backend=backend)
    problem = _solve(
        method,
        graph,
        inputs={"off": {"A": 0}, "on": {"A": 1}},
        measurements={"off": {"Y": 1}, "on": {"Y": 1}},
    )

    assert np.allclose(_flat_values(problem.expr.reaction_selected), [1, 1])
    assert np.allclose(
        np.asarray(problem.expr.reaction_active.value),
        [[0, 1], [1, 0]],
    )
    assert np.allclose(_vertex_values(method, problem, "Y"), [1, 1])


def test_same_inputs_with_conflicting_measurements_cannot_both_be_fit(backend):
    graph = Graph.from_tuples([("A", 1, "Y")])
    method = CellNOptDAG(lambda_reg=0, backend=backend)
    problem = _solve(
        method,
        graph,
        inputs={"first": {"A": 1}, "second": {"A": 1}},
        measurements={"first": {"Y": 0}, "second": {"Y": 1}},
    )

    assert np.isclose(problem.objectives[0].value, 1)
    predictions = _vertex_values(method, problem, "Y")
    assert np.allclose(predictions, [0, 0]) or np.allclose(predictions, [1, 1])


def test_irrelevant_active_input_does_not_force_a_disconnected_branch(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "B"),
            ("C", 1, "D"),
        ]
    )
    method = CellNOptDAG(lambda_reg=1e-3, backend=backend)
    problem = _solve(
        method,
        graph,
        inputs={"condition": {"A": 1, "C": 1}},
        measurements={"condition": {"B": 1}},
    )

    assert np.allclose(_flat_values(problem.expr.reaction_selected), [1, 0])
    assert np.allclose(_vertex_values(method, problem, "C"), [1])
    assert np.allclose(_vertex_values(method, problem, "D"), [0])


def test_flow_rejects_a_selected_component_without_experimental_boundaries(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "B"),
            ("C", 1, "D"),
        ]
    )
    method = CellNOptDAG(lambda_reg=0, backend=backend)
    problem = method.build(
        graph,
        inputs={"A": 1},
        measurements={"B": 1},
    )
    problem += problem.expr.reaction_selected[1] == 1

    _assert_infeasible(problem, backend)


def test_flow_rejects_a_selected_dangling_branch(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "B"),
            ("A", 1, "C"),
        ]
    )
    method = CellNOptDAG(lambda_reg=0, backend=backend)
    problem = method.build(
        graph,
        inputs={"A": 1},
        measurements={"B": 1},
    )
    problem += problem.expr.reaction_selected[1] == 1

    _assert_infeasible(problem, backend)


def test_inhibition_overrides_product_without_disabling_reaction_truth(backend):
    graph = Graph.from_tuples([("A", 1, "B")])
    method = CellNOptDAG(lambda_reg=0, backend=backend)
    problem = method.build(
        graph,
        inputs={"A": 1},
        inhibitors={"B": 1},
        measurements={"B": 0},
    )
    problem += problem.expr.reaction_selected[0] == 1

    result = problem.solve()

    assert result.status == "optimal"
    assert np.allclose(problem.expr.reaction_active.value, [[1]])
    assert np.allclose(_vertex_values(method, problem, "B"), [0])


def test_acyclicity_rejects_a_selected_feedback_cycle(backend):
    graph = Graph.from_tuples(
        [
            ("A", 1, "B"),
            ("B", 1, "A"),
        ]
    )
    method = CellNOptDAG(lambda_reg=0, backend=backend)
    problem = method.build(
        graph,
        inputs={"A": 1},
        measurements={"B": 1},
    )
    problem += problem.expr.reaction_selected == 1

    _assert_infeasible(problem, backend)


def test_constraint_blocks_are_vectorized_over_reactions_and_conditions(backend):
    small_graph = Graph.from_tuples([("A", 1, "B")])
    large_graph = Graph.from_tuples([(f"v{i}", 1, f"v{i + 1}") for i in range(30)])
    small = CellNOptDAG(lambda_reg=0, backend=backend).build(
        small_graph,
        inputs={"A": 1},
        measurements={"B": 1},
    )
    large = CellNOptDAG(lambda_reg=0, backend=backend).build_many(
        large_graph,
        inputs={
            "one": {"v0": 1},
            "two": {"v0": 1},
            "three": {"v0": 1},
        },
        measurements={
            "one": {"v30": 1},
            "two": {"v30": 1},
            "three": {"v30": 1},
        },
    )

    assert len(small.constraints) == len(large.constraints)
