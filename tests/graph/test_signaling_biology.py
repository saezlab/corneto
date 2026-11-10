import numpy as np
import pytest

from corneto._plotting import _resolve_theme, to_dot_source
from corneto.graph import EdgeType, Graph
from corneto.plotting.signaling import BIOLOGY_THEME, biology_signaling_style


class _DotPlot(str):
    def _repr_mimebundle_(self):
        return {}


@pytest.fixture
def graph():
    g = Graph()
    g.add_edge("receptor", "regulator", interaction=-1)
    g.add_edge("regulator", "protein", interaction=1)
    return g


def test_biology_encodes_activity_mechanism_and_annotations(graph):
    edges, vertices = biology_signaling_style(
        graph,
        {
            "vertex_values": [-1, 1, 0],
            "receptors": ["receptor", "regulator"],
            "regulators": ["regulator"],
            "measured": ["protein"],
        },
        BIOLOGY_THEME,
    )
    assert vertices["receptor"]["shape"] == "box"
    assert vertices["regulator"]["shape"] == "hexagon"
    assert vertices["protein"]["shape"] == "ellipse"
    assert [vertices[v]["fillcolor"] for v in graph.V] == ["#CCE2F2", "#F6D8B5", "#F2F2F2"]
    assert vertices["protein"]["penwidth"] == "1.8"
    assert edges[0]["arrowhead"] == "tee"
    assert edges[0]["style"] == "dashed"
    assert edges[0]["tooltip"] == "Release of inhibition"
    assert edges[1]["arrowhead"] == "normal"
    assert edges[1]["style"] == "solid"
    assert edges[0]["penwidth"] == edges[1]["penwidth"]


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf])
def test_processor_validates_threshold_without_helper(graph, threshold):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        biology_signaling_style(graph, {"vertex_values": [0, 0, 0], "activity_threshold": threshold}, {})


@pytest.mark.parametrize("values", [[1, 0], [np.nan, 0, 0], [[1, 0], [0, 0], [0, 1]]])
def test_processor_validates_activity(graph, values):
    with pytest.raises(ValueError, match="finite values"):
        biology_signaling_style(graph, {"vertex_values": values}, {})


def test_threshold_boundaries_and_column_vector(graph):
    edges, vertices = biology_signaling_style(
        graph, {"vertex_values": np.array([[-0.2], [0.2], [0]]), "activity_threshold": 0.2}, {}
    )
    assert all(v["fillcolor"] == BIOLOGY_THEME["zero_color"] for v in vertices.values())
    assert edges[0]["style"] == "solid"
    assert "No inferred change" in edges[0]["tooltip"]


def test_missing_activity_is_actionable(graph):
    with pytest.raises(ValueError, match="vertex_values"):
        biology_signaling_style(graph, {}, {})


def test_boundary_hyperedges_and_undirected_edges_keep_base_rendering():
    g = Graph()
    g.add_edge((), "A")
    g.add_edge({"A", "B"}, "C")
    g.add_edge("A", "B", type=EdgeType.UNDIRECTED)
    edges, _ = biology_signaling_style(g, {"vertex_values": [-1] * len(g.V)}, {})
    assert edges == {}


def test_preset_matches_direct_use_and_preserves_overrides(monkeypatch, graph):
    monkeypatch.setattr(Graph, "to_graphviz", lambda self, **kwargs: _DotPlot(to_dot_source(self, **kwargs)))
    data = {"vertex_values": [-1, 1, 0], "regulators": ["regulator"]}
    source = graph.plot(
        renderer="graphviz",
        preset="signaling_biology",
        data=data,
        theme={"reduced_source_edge_color": "gray"},
        graph_attr={"rankdir": "LR"},
        custom_vertex_attr={"protein": {"fillcolor": "yellow"}},
        custom_edge_attr={0: {"penwidth": "3"}},
    )
    assert 'color="gray"' in source
    assert 'fillcolor="#F6D8B5"' in source
    assert 'fillcolor="yellow"' in source
    assert 'penwidth="3"' in source
    assert 'rankdir="LR"' in source
    assert 'bgcolor="white"' in source
    assert 'fixedsize="false"' in source
    assert BIOLOGY_THEME["reduced_source_edge_color"] == "#888888"
    edges, vertices = biology_signaling_style(graph, data, BIOLOGY_THEME)
    direct = graph.plot(renderer="graphviz", processor=biology_signaling_style, theme=BIOLOGY_THEME, data=data)
    assert direct == to_dot_source(graph, custom_edge_attr=edges, custom_vertex_attr=vertices)


def test_named_theme_replaces_preset_theme():
    assert _resolve_theme("default", default="signaling_biology")["positive_color"] == "firebrick4"
    assert _resolve_theme({"edge_width": 2}, default="signaling_biology")["positive_color"] == "#F6D8B5"


def test_preset_extracts_solution_sample_and_keeps_selection_explicit(monkeypatch, graph):
    monkeypatch.setattr(Graph, "to_graphviz", lambda self, **kwargs: _DotPlot(to_dot_source(self, **kwargs)))
    source = graph.plot(
        renderer="graphviz",
        preset="signaling_biology",
        solution={"vertex_value": np.array([[-1, 1], [1, -1], [0, 0]])},
        sample=1,
        edge_indexes=[1],
    )
    assert '"receptor" -> "regulator"' not in source
    assert 'style="dashed"' in source


def test_integer_vertex_identifiers():
    g = Graph()
    g.add_edge(10, 20)
    edges, vertices = biology_signaling_style(g, {"vertex_values": [-1, 1]}, {})
    source = to_dot_source(g, custom_edge_attr=edges, custom_vertex_attr=vertices)
    assert '"10" [shape="ellipse", style="filled", fillcolor="#CCE2F2"' in source
