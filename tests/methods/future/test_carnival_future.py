import os
import pickle

import numpy as np
import pytest

from corneto._data import Data
from corneto.backend import Backend, CvxpyBackend, PicosBackend
from corneto.graph import Graph
from corneto.methods.future.carnival import CarnivalFlow


@pytest.fixture
def large_dataset():
    """Load a large test graph from the pickle file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "large_graph_carnival.pkl.xz")
    with open(
        os.path.join(current_dir, "data", "large_graph_carnival_single_input.pkl"), "rb"
    ) as f:
        input_data = pickle.load(f)
    return Graph.load(file_path), input_data


@pytest.fixture
def graph_two_samples():
    G = Graph.from_tuples(
        [
            ("r1", 1, "a"),
            ("r1", 1, "b"),
            ("a", 1, "c"),
            ("c", 1, "tf1"),
            ("c", -1, "b"),
            ("b", -1, "a"),
            ("b", -1, "tf2"),
            ("r2", -1, "b"),
            ("r2", 1, "d"),
            ("d", 1, "e"),
            ("e", 1, "a"),
            ("r2", -1, "tf1"),
        ]
    )
    samples = {
        "s1": {
            "r1": {
                "value": 1,
                "role": "input",
                "mapping": "vertex"
            },
            "tf1": {
                "value": 1,
                "role": "output",
                "mapping": "vertex"
            },
            "tf2": {
                "value": 1,
                "role": "output",
                "mapping": "vertex"
            },
        },
        "s2": {
            "r2": {
                "value": 1,
                "role": "input",
                "mapping": "vertex"
            },
            "tf1": {
                "value": 1,
                "role": "output",
                "mapping": "vertex"
            },
            "tf2": {
                "value": -1,
                "role": "output",
                "mapping": "vertex"
            },
        },
    }
    return G, samples


def test_carnivalflow_large_dataset_one_sample(backend, large_dataset):
    G, samples = large_dataset
    data = Data.from_cdict(samples)
    carnival = CarnivalFlow(lambda_reg=0, backend=backend, data_type_key="type")
    P = carnival.build(G, data)
    P.solve()
    # Check that the first objective is 32.6251 (up to 4 decimal places)
    assert np.isclose(P.objectives[0].value, 32.6251, atol=1e-4)


def test_carnivalflow_two_samples_inverse(backend, graph_two_samples):
    G, samples = graph_two_samples
    samples["s1"]["r1"]["value"] = 0
    samples["s2"]["r2"]["value"] = 0
    data = Data.from_cdict(samples)
    carnival = CarnivalFlow(lambda_reg=1e-3, backend=backend)
    P = carnival.build(G, data)
    P.solve()
    sol1 = np.array(P.expr.edge_value.value[:, 0]).flatten()
    sol2 = np.array(P.expr.edge_value.value[:, 1]).flatten()
    gsol1 = carnival.processed_graph.edge_subgraph(np.flatnonzero(sol1))
    gsol2 = carnival.processed_graph.edge_subgraph(np.flatnonzero(sol2))
    vertex_values_s1 = {
        carnival.processed_graph.V[i]: P.expr.vertex_value.value[i, 0]
        for i in range(P.expr.vertex_value.shape[0])
    }
    vertex_values_s2 = {
        carnival.processed_graph.V[i]: P.expr.vertex_value.value[i, 1]
        for i in range(P.expr.vertex_value.shape[0])
    }
    assert P.objectives[0].value == 0.0
    assert P.objectives[1].value == 0.0
    assert P.objectives[2].value == 9.0
    assert P.expr.edge_value.shape == (16, 2)
    assert carnival.processed_graph.shape == (9, 16)
    # TODO: This fails with PICOS/GLPK only on the CI
    sol = [
        vertex_values_s1["r1"],
        vertex_values_s1["r2"],
        vertex_values_s2["r1"],
        vertex_values_s2["r2"],
    ]
    assert np.allclose(sol, [1, 0, 0, -1], atol=1e-4) or np.allclose(
        sol, [-1, 0, 0, -1], atol=1e-4
    )


def test_carnivalflow_two_samples(backend, graph_two_samples):
    G, samples = graph_two_samples
    data = Data.from_cdict(samples)
    carnival = CarnivalFlow(lambda_reg=1e-3, backend=backend)
    P = carnival.build(G, data)
    P.solve()
    sol1 = np.array(P.expr.edge_value.value[:, 0]).flatten()
    sol2 = np.array(P.expr.edge_value.value[:, 1]).flatten()
    gsol1 = carnival.processed_graph.edge_subgraph(np.flatnonzero(sol1))
    gsol2 = carnival.processed_graph.edge_subgraph(np.flatnonzero(sol2))
    assert P.objectives[0].value == 0.0
    assert P.objectives[1].value == 1.0
    assert P.objectives[2].value == 9.0
    assert P.expr.edge_value.shape == (16, 2)
    assert carnival.processed_graph.shape == (9, 16)
    assert set(gsol1.V) == set(["c", "a", "r1", "tf1", "b", "tf2"])
    assert set(gsol2.V) == set(["c", "r2", "a", "tf1", "b"])
