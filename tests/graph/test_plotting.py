import sys
from types import ModuleType

import pytest

from corneto._data import Data
from corneto._plotting import to_dot_source
from corneto._util import supports_html
from corneto.contrib._util import (
    DEFAULT_WASM_GRAPHVIZ_JS_URL,
    GraphRender,
    dot_wasm_html,
)
from corneto.graph import Graph


def _install_fake_marimo(monkeypatch, iframe_impl):
    marimo_module = ModuleType("marimo")
    output_module = ModuleType("marimo._output")
    formatting_module = ModuleType("marimo._output.formatting")
    formatting_module.iframe = iframe_impl

    monkeypatch.setitem(sys.modules, "marimo", marimo_module)
    monkeypatch.setitem(sys.modules, "marimo._output", output_module)
    monkeypatch.setitem(sys.modules, "marimo._output.formatting", formatting_module)


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


def test_graph_render_mime_uses_marimo_frame_mime(monkeypatch):
    expected = '<iframe srcdoc="<p>ok</p>"></iframe>'

    class _Frame:
        def _mime_(self):
            return "text/html", expected

    _install_fake_marimo(monkeypatch, lambda *_args, **_kwargs: _Frame())

    mime, payload = GraphRender("<div>graph</div>")._mime_()
    assert mime == "text/html"
    assert payload == expected


def test_graph_render_mime_prefers_html_field_over_text(monkeypatch):
    class _Frame:
        html = '<iframe srcdoc="<p>ok</p>"></iframe>'
        text = "&lt;iframe srcdoc=&quot;&lt;p&gt;escaped&lt;/p&gt;&quot;&gt;&lt;/iframe&gt;"

    _install_fake_marimo(monkeypatch, lambda *_args, **_kwargs: _Frame())

    mime, payload = GraphRender("<div>graph</div>")._mime_()
    assert mime == "text/html"
    assert payload == _Frame.html


def test_graph_render_repr_html_wraps_in_iframe_when_ipython_loaded(monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", object())
    html = GraphRender('<div><script type="module">console.log("x")</script></div>')._repr_html_()
    assert html.startswith("<iframe ")
    assert "srcdoc=" in html
    assert "&lt;script" in html


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


def test_plot_with_vertex_attribute_style_processor(monkeypatch):
    g = Graph()
    g.add_vertex("EGFR", type="receptor")
    g.add_vertex("AKT1", type="signaling")
    g.add_vertex("STAT3", type="tf")
    g.add_edge("EGFR", "AKT1")
    g.add_edge("AKT1", "STAT3")

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
        processor="vertex_attribute_style",
        data={
            "vertex_style_attr": "type",
            "vertex_style_map": {
                "receptor": {"shape": "triangle"},
                "signaling": "box",
                "tf": {"shape": "diamond"},
            },
        },
    )
    assert '"EGFR" [shape="triangle"]' in obj.source
    assert '"AKT1" [shape="box"]' in obj.source
    assert '"STAT3" [shape="diamond"]' in obj.source


def test_plot_with_composed_processors_combines_vertex_shape_and_sign_color(monkeypatch):
    g = Graph()
    g.add_vertex("EGFR", type="receptor")
    g.add_vertex("AKT1", type="signaling")
    g.add_edge("EGFR", "AKT1")

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
        processor=["sign_magnitude", "vertex_attribute_style"],
        data={
            "vertex_values": [1.0, -1.0],
            "vertex_style_attr": "type",
            "vertex_style_map": {"receptor": {"shape": "triangle"}},
        },
    )
    assert '"EGFR"' in obj.source
    assert 'shape="triangle"' in obj.source
    assert 'color="firebrick4"' in obj.source
    assert '"AKT1"' in obj.source
    assert 'shape="circle"' in obj.source
    assert 'color="dodgerblue4"' in obj.source


def test_plot_with_solution_and_solution_map(monkeypatch):
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
        solution={"v_sol": [1.0, -1.0], "e_sol": [1.0]},
        solution_map={"vertex": "v_sol", "edge": "e_sol"},
    )
    assert 'color="firebrick4"' in obj.source
    assert 'color="dodgerblue4"' in obj.source


def test_plot_with_metabolism_flux_preset_uses_solution_default_mapping(monkeypatch):
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
        preset="metabolism_flux",
        solution={"flow": [3.0]},
    )
    assert "penwidth=" in obj.source
    assert 'color="firebrick4"' in obj.source


def test_plot_data_overrides_solution_values(monkeypatch):
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
        data={"edge_values": [-1.0], "vertex_values": [0.0, 0.0]},
        solution={"edge_value": [1.0], "vertex_value": [1.0, -1.0]},
    )
    assert 'color="dodgerblue4"' in obj.source


def test_plot_with_signaling_preset_applies_edge_style(monkeypatch):
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
        preset="signaling",
        data={"edge_values": [1.0], "vertex_values": [1.0, -1.0]},
    )
    assert 'color="firebrick4"' in obj.source
    assert 'color="dodgerblue4"' in obj.source


def test_plot_with_metabolism_preset_uses_solution_default_mapping(monkeypatch):
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
        preset="metabolism",
        solution={"flow": [3.0]},
    )
    assert "penwidth=" in obj.source
    assert 'color="firebrick4"' in obj.source


def test_plot_with_signaling_preset_role_shapes_from_node_roles(monkeypatch):
    g = Graph()
    g.add_edge("TGFBR1", "AKT1")
    g.add_edge("AKT1", "STAT3")

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
        preset="signaling",
        node_roles={"TGFBR1": "input", "STAT3": "output"},
    )
    assert '"TGFBR1" [shape="triangle"' in obj.source
    assert '"STAT3" [shape="diamond"' in obj.source
    assert '"AKT1" [shape="circle"]' in obj.source


def test_plot_with_signaling_preset_role_shapes_from_feature_data(monkeypatch):
    g = Graph()
    g.add_edge("TGFBR1", "AKT1")
    g.add_edge("AKT1", "STAT3")

    feature_data = Data.from_cdict(
        {
            "sample1": {
                "TGFBR1": {"value": 1.0, "role": "input", "mapping": "vertex"},
                "STAT3": {"value": -1.0, "role": "output", "mapping": "vertex"},
            }
        }
    )

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
        preset="signaling",
        feature_data=feature_data,
    )
    assert '"TGFBR1" [shape="triangle"' in obj.source
    assert '"STAT3" [shape="diamond"' in obj.source
    assert '"AKT1" [shape="circle"]' in obj.source
