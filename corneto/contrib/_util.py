import base64
import html
import os
import sys
import uuid
from typing import Any, Optional, Tuple

DEFAULT_WASM_GRAPHVIZ_JS_URL = "https://cdn.jsdelivr.net/npm/@hpcc-js/wasm-graphviz@1.21.0/dist/index.min.js"


class GraphRender:
    """Small rich display wrapper compatible with Jupyter and marimo."""

    def __init__(self, html: str, plain_text: str = "", iframe_height: str = "220px"):
        self._html = html
        self._plain_text = plain_text
        self._iframe_height = iframe_height

    def _mime_(self) -> Tuple[str, str]:
        # marimo does not execute inline scripts in raw text/html; wrap in
        # iframe so WASM bootstrap scripts can run.
        try:
            from marimo._output.formatting import iframe

            frame = iframe(self._html, height=self._iframe_height)

            frame_mime = getattr(frame, "_mime_", None)
            if callable(frame_mime):
                mime_type, payload = frame_mime()
                if mime_type == "text/html" and isinstance(payload, str):
                    return mime_type, payload

            html_payload = getattr(frame, "html", None)
            if isinstance(html_payload, str):
                return "text/html", html_payload

            text_payload = getattr(frame, "text", None)
            if isinstance(text_payload, str):
                return "text/html", text_payload

            if "IPython" in sys.modules:
                return "text/html", self._iframe_html(self._html)
            return "text/html", self._html
        except Exception:
            if "IPython" in sys.modules:
                return "text/html", self._iframe_html(self._html)
            return "text/html", self._html

    def _repr_html_(self) -> str:
        # In Jupyter/IPython, run WASM bootstrap in an isolated iframe to avoid
        # blocking the notebook document thread.
        if "IPython" in sys.modules:
            return self._iframe_html(self._html)
        return self._html

    def _iframe_html(self, body: str) -> str:
        escaped = html.escape(body, quote=True)
        return f"<iframe srcdoc='{escaped}' width='100%' height='{self._iframe_height}' frameborder='0'></iframe>"

    def __repr__(self) -> str:
        return self._plain_text if self._plain_text else "GraphRender(html)"


def _to_dot_source(dot_input: Any) -> str:
    if isinstance(dot_input, str):
        if os.path.exists(dot_input):
            with open(dot_input, "r") as file:
                return file.read()
        return dot_input
    dot_source = getattr(dot_input, "source", None)
    if dot_source is None:
        raise TypeError("Provided object is neither DOT source/path nor a graphviz object with '.source'.")
    return dot_source


def dot_wasm_html(
    dot_input: Any,
    container_id: Optional[str] = None,
    wasm_graphviz_js_url: Optional[str] = None,
) -> str:
    if container_id is None:
        container_id = f"container-{uuid.uuid4()}"

    dot_string = _to_dot_source(dot_input)
    dot_string_base64 = base64.b64encode(dot_string.encode()).decode("utf-8")

    if wasm_graphviz_js_url is None:
        wasm_graphviz_js_url = DEFAULT_WASM_GRAPHVIZ_JS_URL

    return f"""
    <div id="{container_id}" style="overflow:auto"></div>
    <script type="module">
    (async () => {{
      const target = document.getElementById("{container_id}");
      const resizeFrameToContent = () => {{
        try {{
          let h = target ? target.scrollHeight : 0;
          const svg = target ? target.querySelector("svg") : null;
          if (svg) {{
            const rect = svg.getBoundingClientRect();
            if (rect && Number.isFinite(rect.height)) {{
              h = Math.max(h, rect.height);
            }}
          }}
          h = Math.max(120, Math.ceil(h) + 24);
          if (window.frameElement) {{
            window.frameElement.style.height = `${{h}}px`;
          }}
        }} catch (_err) {{
          // best effort only
        }}
      }};
      try {{
        const mod = await import("{wasm_graphviz_js_url}");
        const Graphviz = mod.Graphviz || (mod.default && mod.default.Graphviz) || mod.default;
        if (!Graphviz || !Graphviz.load) {{
          throw new Error("Graphviz WASM module does not expose Graphviz.load()");
        }}
        const graphviz = await Graphviz.load();
        const dot = atob("{dot_string_base64}");
        const svg = await graphviz.dot(dot, "svg", "dot");
        target.innerHTML = svg;
        resizeFrameToContent();
        requestAnimationFrame(resizeFrameToContent);
        setTimeout(resizeFrameToContent, 50);
      }} catch (error) {{
        target.innerHTML = "<pre style='white-space:pre-wrap;color:#b00020'>Graph rendering failed: " +
          String(error) + "</pre>";
        resizeFrameToContent();
      }}
    }})();
    </script>
    """


def dot_wasm_render(
    dot_input: Any,
    container_id: Optional[str] = None,
    wasm_graphviz_js_url: Optional[str] = None,
    iframe_height: str = "220px",
) -> GraphRender:
    dot_source = _to_dot_source(dot_input)
    return GraphRender(
        html=dot_wasm_html(
            dot_source,
            container_id=container_id,
            wasm_graphviz_js_url=wasm_graphviz_js_url,
        ),
        plain_text=dot_source,
        iframe_height=iframe_height,
    )
