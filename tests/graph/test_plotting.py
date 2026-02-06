import sys

from corneto._plotting import to_dot_source
from corneto.contrib._util import dot_vizjs_html
from corneto.graph import Graph


def test_to_dot_source_generates_basic_dot():
    g = Graph()
    g.add_edge("A", "B")
    src = to_dot_source(g)
    assert src.startswith("digraph {")
    assert '"A"' in src
    assert '"B"' in src
    assert 'arrowhead="normal"' in src


def test_dot_vizjs_html_accepts_raw_dot_source():
    html = dot_vizjs_html('digraph {"A" -> "B";}')
    assert "renderGraph(" in html
    assert "unpkg.com/viz.js" in html


def test_plot_vizjs_works_without_graphviz(monkeypatch):
    g = Graph()
    g.add_edge("A", "B")

    def _raise_graphviz(*args, **kwargs):
        raise ImportError("graphviz is not available")

    monkeypatch.setattr(Graph, "to_graphviz", _raise_graphviz)
    monkeypatch.setattr("corneto._util.supports_html", lambda: True)

    obj = g.plot(renderer="vizjs")
    html = obj._repr_html_()
    assert "renderGraph(" in html
    assert "unpkg.com/viz.js" in html


def test_plot_auto_uses_vizjs_on_emscripten(monkeypatch):
    g = Graph()
    g.add_edge("A", "B")

    def _raise_graphviz(*args, **kwargs):
        raise ImportError("graphviz is not available")

    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.setattr(Graph, "to_graphviz", _raise_graphviz)
    monkeypatch.setattr("corneto._util.supports_html", lambda: True)

    obj = g.plot()
    assert hasattr(obj, "_repr_html_")
