import sys

from corneto._util import _get_info


def test_get_info_includes_plotting_renderers():
    info = _get_info()

    assert "plot_default" in info
    assert "plot_available" in info
    assert isinstance(info["plot_available"]["value"], list)


def test_get_info_detects_marimo_html_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "marimo", object())
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    monkeypatch.delitem(sys.modules, "IPython.display", raising=False)

    info = _get_info()
    assert "graphviz-wasm" in info["plot_available"]["value"]
