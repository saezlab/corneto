import numpy as np
import pytest

from corneto import Graph
from corneto.methods import PHONEMeS, compute_phonemes_scores


def test_compute_phonemes_scores_flat_mapping():
    scores = compute_phonemes_scores(
        {
            "supported": 0.005,
            "threshold": 0.05,
            "unsupported": 0.5,
        },
        scale=False,
    )

    expected = np.log2(10)
    assert scores == pytest.approx(
        {
            "supported": -expected,
            "threshold": 0,
            "unsupported": expected,
        }
    )


def test_compute_phonemes_scores_uses_fold_change_direction():
    scores = compute_phonemes_scores(
        {"up": 0.001, "down": 0.001, "small": 0.001},
        fold_changes={"up": 2, "down": -2, "small": 0.2},
        fold_change_threshold=1,
        direction="up",
        scale=False,
    )

    assert scores["up"] < 0
    assert scores["down"] > 0
    assert scores["small"] > 0


def test_compute_phonemes_scores_nested_mapping_scales_per_condition():
    scores = compute_phonemes_scores(
        {
            "a": {"site_1": 0.005, "site_2": 0.5},
            "b": {"site_1": 0.0005, "site_2": 0.005},
        }
    )

    assert scores["a"] == pytest.approx({"site_1": -1, "site_2": 1})
    assert scores["b"]["site_1"] == pytest.approx(-1)
    assert -1 < scores["b"]["site_2"] < 0


def test_compute_phonemes_scores_preserves_numpy_shape():
    pvalues = np.array([[0.005, 0.5], [0.05, 0.005]])
    scores = compute_phonemes_scores(pvalues, scale=False)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == pvalues.shape
    assert scores[0, 0] < 0
    assert scores[0, 1] > 0
    assert scores[1, 0] == pytest.approx(0)


def test_compute_phonemes_scores_preserves_pandas_labels():
    pd = pytest.importorskip("pandas")
    pvalues = pd.DataFrame(
        {
            "control": [0.005, 0.5],
            "treated": [0.0005, 0.005],
        },
        index=["site_1", "site_2"],
    )

    scores = compute_phonemes_scores(pvalues)

    assert isinstance(scores, pd.DataFrame)
    assert scores.index.equals(pvalues.index)
    assert scores.columns.equals(pvalues.columns)
    assert scores.loc["site_1", "control"] == pytest.approx(-1)
    assert scores.loc["site_2", "control"] == pytest.approx(1)


def test_phonemes_accepts_pandas_scores_directly(backend):
    pd = pytest.importorskip("pandas")
    graph = Graph()
    graph.add_edge("r", "m")
    method = PHONEMeS(default_edge_cost=0, backend=backend)

    single = method.build(
        graph,
        perturbations=["r"],
        phosphosite_scores=pd.Series({"m": -1.0}),
    )
    assert single.expr.vertex_selected.shape == (2, 1)

    many = method.build_many(
        graph,
        perturbations={"control": ["r"], "treated": ["r"]},
        phosphosite_scores=pd.DataFrame({"control": {"m": -1.0}, "treated": {"m": -0.5}}),
    )
    assert many.expr.vertex_selected.shape == (2, 2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pvalues": {"site": np.nan}}, "finite"),
        ({"pvalues": {"site": 1.1}}, "between 0 and 1"),
        (
            {
                "pvalues": {"site": 0.01},
                "fold_change_threshold": 1,
            },
            "fold_changes is required",
        ),
        (
            {
                "pvalues": {"site": 0.01, "condition": {"site": 0.02}},
            },
            "uniformly nested",
        ),
    ],
)
def test_compute_phonemes_scores_validates_inputs(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        compute_phonemes_scores(**kwargs)


def test_phonemes_prunes_unreachable_measurements_and_remaps_costs(backend):
    graph = Graph()
    graph.add_edges(
        [
            ("r", "dead"),
            ("r", "middle"),
            ("middle", "measured"),
            ("other", "unreachable"),
        ]
    )
    method = PHONEMeS(default_edge_cost=0, backend=backend)
    problem = method.build(
        graph,
        perturbations=["r"],
        phosphosite_scores={"measured": -1, "unreachable": -10},
        edge_costs={0: -100, 1: 0.2, 2: 0.3, 3: -100},
    )

    biological_edges = method.processed_graph.E[: problem.expr.edge_selected.shape[0]]
    assert biological_edges == (
        (frozenset({"r"}), frozenset({"middle"})),
        (frozenset({"middle"}), frozenset({"measured"})),
    )
    assert method._edge_costs.tolist() == pytest.approx([0.2, 0.3])
    assert "dead" not in method.processed_graph.V
    assert "unreachable" not in method.processed_graph.V
    assert all(feature.id != "unreachable" for feature in method.processed_data.samples["condition"].features)
    assert problem.expr.edge_selected.shape == (2, 1)


def test_phonemes_rejects_perturbation_without_measured_path(backend):
    graph = Graph()
    graph.add_edges([("r1", "m"), ("r2", "dead")])

    with pytest.raises(ValueError, match="cannot reach any measured phosphosite"):
        PHONEMeS(backend=backend).build(
            graph,
            perturbations=["r1", "r2"],
            phosphosite_scores={"m": -1},
        )
