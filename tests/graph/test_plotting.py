import sys

import pytest

from corneto._plotting import to_dot_source
from corneto._util import supports_html
from corneto.contrib._util import DEFAULT_WASM_GRAPHVIZ_JS_URL, dot_wasm_html
from corneto.graph import Graph


def test_to_dot_source_generates_basic_dot():
    g = Graph()
    g.add_edge("A", "B")
    src = to_dot_source(g)
    assert src.startswith("digraph {")
    assert '"A"' in src
    assert '"B"' in src
    assert 'arrowhead="normal"' in src


def test_dot_wasm_html_accepts_raw_dot_source():
    html = dot_wasm_html('digraph {"A" -> "B";}')
    assert DEFAULT_WASM_GRAPHVIZ_JS_URL in html
    assert 'script type="module"' in html


def test_plot_wasm_works_without_graphviz(monkeypatch):
    g = Graph()
    g.add_edge("A", "B")

    def _raise_graphviz(*args, **kwargs):
        raise ImportError("graphviz is not available")

    monkeypatch.setattr(Graph, "to_graphviz", _raise_graphviz)
    monkeypatch.setattr("corneto._util.supports_html", lambda: True)

    obj = g.plot(renderer="wasm")
    html = obj._repr_html_()
    mime, payload = obj._mime_()
    assert mime == "text/html"
    assert payload == html
    assert DEFAULT_WASM_GRAPHVIZ_JS_URL in html


def test_plot_auto_uses_wasm_on_emscripten(monkeypatch):
    g = Graph()
    g.add_edge("A", "B")

    def _raise_graphviz(*args, **kwargs):
        raise ImportError("graphviz is not available")

    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.setattr(Graph, "to_graphviz", _raise_graphviz)
    monkeypatch.setattr("corneto._util.supports_html", lambda: True)

    obj = g.plot()
    assert hasattr(obj, "_repr_html_")
    assert hasattr(obj, "_mime_")


def test_plot_auto_falls_back_to_wasm_when_graphviz_missing(monkeypatch):
    g = Graph()
    g.add_edge("A", "B")

    def _raise_graphviz(*args, **kwargs):
        raise ImportError("graphviz is not available")

    monkeypatch.setattr(Graph, "to_graphviz", _raise_graphviz)
    monkeypatch.setattr("corneto._util.supports_html", lambda: True)

    obj = g.plot()
    assert hasattr(obj, "_repr_html_")
    assert hasattr(obj, "_mime_")


def test_plot_networkx_renderer(monkeypatch):
    g = Graph()
    g.add_edge("A", "B")
    sentinel_fig = object()

    def _fake_plot_with_networkx(graph, **kwargs):
        del graph, kwargs
        return sentinel_fig

    monkeypatch.setattr("corneto._plotting._plot_with_networkx", _fake_plot_with_networkx)

    obj = g.plot(renderer="networkx")
    assert obj is sentinel_fig


def test_supports_html_true_when_marimo_loaded(monkeypatch):
    monkeypatch.setitem(sys.modules, "marimo", object())
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    monkeypatch.delitem(sys.modules, "IPython.display", raising=False)
    assert supports_html() is True


def test_to_dot_accepts_backend_for_backward_compatibility():
    pytest.importorskip("pydot")
    g = Graph()
    g.add_edge("A", "B")
    dot_obj = g.to_dot(backend="pydot")
    assert hasattr(dot_obj, "create_svg")


def test_plot_with_simple_preset_applies_edge_style(monkeypatch):
    g = Graph()
    g.add_edge("A", "B")

    class _FakeGV:
        def __init__(self, source):
            self.source = source

        def _repr_mimebundle_(self):
            return {}

    def _fake_to_graphviz(self, **kwargs):
        from corneto._plotting import to_dot_source

        return _FakeGV(to_dot_source(self, **kwargs))

    monkeypatch.setattr(Graph, "to_graphviz", _fake_to_graphviz)
    obj = g.plot(
        renderer="graphviz",
        preset="simple",
        data={"edge_values": [1.0], "vertex_values": [1.0, -1.0]},
    )
    assert 'color="firebrick4"' in obj.source
    assert 'color="dodgerblue4"' in obj.source


def test_plot_with_metabolism_flux_preset_applies_penwidth(monkeypatch):
    g = Graph()
    g.add_edge("A", "B")

    class _FakeGV:
        def __init__(self, source):
            self.source = source

        def _repr_mimebundle_(self):
            return {}

    def _fake_to_graphviz(self, **kwargs):
        from corneto._plotting import to_dot_source

        return _FakeGV(to_dot_source(self, **kwargs))

    monkeypatch.setattr(Graph, "to_graphviz", _fake_to_graphviz)
    obj = g.plot(renderer="graphviz", preset="metabolism_flux", data={"edge_values": [10.0]})
    assert "penwidth=" in obj.source


def test_plot_with_metabolism_flux_log_keeps_negative_sign(monkeypatch):
    g = Graph()
    g.add_edge("Glucose", "Pyruvate")
    g.add_edge("Pyruvate", "Lactate")
    g.add_edge("Pyruvate", "Acetyl-CoA")

    class _FakeGV:
        def __init__(self, source):
            self.source = source

        def _repr_mimebundle_(self):
            return {}

    def _fake_to_graphviz(self, **kwargs):
        from corneto._plotting import to_dot_source

        return _FakeGV(to_dot_source(self, **kwargs))

    monkeypatch.setattr(Graph, "to_graphviz", _fake_to_graphviz)
    obj = g.plot(
        renderer="graphviz",
        preset="metabolism_flux",
        data={"edge_values": [20.0, -0.2, 3.0], "scale": "log", "clip_quantil": 0.05},
    )
    assert 'color="firebrick4"' in obj.source
    assert 'color="dodgerblue4"' in obj.source
