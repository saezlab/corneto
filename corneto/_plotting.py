import contextlib
import io
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from corneto.backend._base import EXPR_NAME_FLOW
from corneto.graph import Attr, BaseGraph, EdgeType

Theme = Dict[str, Any]
ProcessorOutput = Tuple[Dict[int, Dict[str, str]], Dict[Union[int, str], Dict[str, str]]]
ProcessorFn = Callable[[BaseGraph, Dict[str, Any], Theme], ProcessorOutput]
ProcessorArg = Optional[
    Union[
        str,
        ProcessorFn,
        List[Union[str, ProcessorFn]],
        Tuple[Union[str, ProcessorFn], ...],
    ]
]

_THEMES: Dict[str, Theme] = {
    "default": {
        "positive_color": "firebrick4",
        "negative_color": "dodgerblue4",
        "zero_color": "black",
        "min_edge_width": 0.25,
        "max_edge_width": 5.0,
    },
    "red_blue": {
        "positive_color": "firebrick4",
        "negative_color": "dodgerblue4",
        "zero_color": "black",
        "min_edge_width": 0.25,
        "max_edge_width": 5.0,
    },
}


def _resolve_theme(theme: Optional[Union[str, Theme]]) -> Theme:
    if theme is None:
        return dict(_THEMES["default"])
    if isinstance(theme, str):
        if theme not in _THEMES:
            raise ValueError(f"Unknown theme '{theme}'. Available themes: {list(_THEMES.keys())}")
        return dict(_THEMES[theme])
    out = dict(_THEMES["default"])
    out.update(theme)
    return out


def _as_values(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    if value is None:
        return None
    return np.array(value)


def _processor_sign_magnitude(graph: BaseGraph, data: Dict[str, Any], theme: Theme) -> ProcessorOutput:
    edge_attrs: Dict[int, Dict[str, str]] = {}
    vertex_attrs: Dict[Union[int, str], Dict[str, str]] = {}

    edge_values = _as_values(data.get("edge_values"))
    if edge_values is not None:
        edge_attrs = create_graphviz_edge_attributes(
            edge_values=edge_values,
            min_edge_width=float(theme["min_edge_width"]),
            max_edge_width=float(theme["max_edge_width"]),
            negative_color=str(theme["negative_color"]),
            positive_color=str(theme["positive_color"]),
        )
        for attrs in edge_attrs.values():
            if attrs.get("color") == "black":
                attrs["color"] = str(theme["zero_color"])

    vertex_values = _as_values(data.get("vertex_values"))
    if vertex_values is not None:
        vertex_attrs = create_graphviz_vertex_attributes(
            graph_vertices=list(graph.V),
            vertex_values=vertex_values,
            negative_color=str(theme["negative_color"]),
            positive_color=str(theme["positive_color"]),
        )
    return edge_attrs, vertex_attrs


def _processor_metabolism_flux(graph: BaseGraph, data: Dict[str, Any], theme: Theme) -> ProcessorOutput:
    del graph
    edge_attrs: Dict[int, Dict[str, str]] = {}
    flux_values = _as_values(data.get("edge_values"))
    if flux_values is None:
        flux_values = _as_values(data.get("flux_values"))
    if flux_values is None:
        return {}, {}

    zero_flow_threshold = float(data.get("zero_flow_threshold", 1e-6))
    scale = data.get("scale", "log")
    clip_quantil = data.get("clip_quantil", 0.05)

    flow = np.array(flux_values, dtype=float)
    flow[np.abs(flow) < zero_flow_threshold] = 0
    if scale is not None:
        if scale == "log":
            flow = np.sign(flow) * np.log10(np.abs(flow) + 1.0)
        elif scale == "std":
            std = float(np.std(flow))
            if std > 0:
                flow = flow / std
        else:
            raise ValueError(f"Unknown scale '{scale}'. Use None, 'log', or 'std'.")
    if clip_quantil is not None:
        flow = clip_quantiles(flow, float(clip_quantil))
    max_flow = max(float(np.max(np.abs(flow))), 1e-6)
    min_w = float(theme["min_edge_width"])
    max_w = float(theme["max_edge_width"])

    for i, v in enumerate(flow):
        if scale is None:
            edge_width = max_w if abs(v) > 0 else min_w
        else:
            edge_width = min_w + (max_w - min_w) * abs(v / max_flow)
        edge_attrs[i] = {"penwidth": str(edge_width)}
        if v > 0:
            edge_attrs[i]["color"] = str(theme["positive_color"])
        elif v < 0:
            edge_attrs[i]["color"] = str(theme["negative_color"])
        else:
            edge_attrs[i]["color"] = str(theme["zero_color"])
    return edge_attrs, {}


def _processor_vertex_attribute_style(graph: BaseGraph, data: Dict[str, Any], theme: Theme) -> ProcessorOutput:
    del theme
    vertex_style_map = data.get("vertex_style_map")
    if not vertex_style_map:
        return {}, {}
    if not isinstance(vertex_style_map, Mapping):
        raise ValueError("data['vertex_style_map'] must be a mapping from attribute values to Graphviz attrs.")

    vertex_style_attr = str(data.get("vertex_style_attr", "type"))
    default_vertex_style = data.get("default_vertex_style")
    if default_vertex_style is not None and not isinstance(default_vertex_style, Mapping):
        raise ValueError("data['default_vertex_style'] must be a dict with Graphviz attrs.")

    vertex_attrs: Dict[Union[int, str], Dict[str, str]] = {}
    for v in graph.V:
        attr = graph.get_attr_vertex(v)
        attr_value = attr.get(vertex_style_attr, None)
        style = vertex_style_map.get(attr_value, default_vertex_style)
        if style is None:
            continue
        if isinstance(style, str):
            style = {"shape": style}
        elif not isinstance(style, Mapping):
            raise ValueError(
                f"Style for vertex attribute value {attr_value!r} must be a dict or a shape string, got {type(style)}."
            )
        vertex_attrs[v] = {str(k): str(val) for k, val in style.items()}
    return {}, vertex_attrs


_PROCESSORS: Dict[str, ProcessorFn] = {
    "sign_magnitude": _processor_sign_magnitude,
    "metabolism_flux": _processor_metabolism_flux,
    "vertex_attribute_style": _processor_vertex_attribute_style,
}

_PRESETS: Dict[str, Dict[str, str]] = {
    # Preferred names
    "default": {"theme": "default", "processor": "sign_magnitude"},
    "signaling": {"theme": "default", "processor": "sign_magnitude"},
    "metabolism": {"theme": "red_blue", "processor": "metabolism_flux"},
    # Backward-compatible aliases
    "simple": {"theme": "default", "processor": "sign_magnitude"},
    "metabolism_flux": {"theme": "red_blue", "processor": "metabolism_flux"},
}

# Default role styles for domain presets.
_PRESET_ROLE_STYLES: Dict[str, Dict[str, Dict[str, str]]] = {
    "signaling": {
        "input": {"shape": "triangle", "style": "filled", "fillcolor": "#dff3ff"},
        "output": {"shape": "diamond", "style": "filled", "fillcolor": "#ffe8d6"},
        "internal": {"shape": "circle"},
        "mixed": {"shape": "circle", "style": "dashed"},
    }
}


def _solution_get(solution: Any, name: str) -> Any:
    if solution is None:
        return None
    if isinstance(solution, Mapping):
        return solution.get(name, None)
    expr = getattr(solution, "expr", None)
    if expr is None:
        return None
    try:
        if hasattr(expr, "__contains__") and name in expr:
            return expr[name]
    except Exception:
        pass
    if hasattr(expr, name):
        return getattr(expr, name)
    return None


def _merge_solution_data(
    base_data: Dict[str, Any],
    solution: Any,
    solution_map: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    if solution is None:
        return base_data
    merged = dict(base_data)
    semantic_to_data_key = {
        "vertex": "vertex_values",
        "edge": "edge_values",
        "flux": "flux_values",
    }
    for semantic_key, data_key in semantic_to_data_key.items():
        var_name = None
        if solution_map is not None:
            var_name = solution_map.get(semantic_key, None)
        if not var_name:
            continue
        if data_key in merged:
            # Keep explicit user-provided plotting data over inferred solution values.
            continue
        value = _solution_get(solution, var_name)
        if value is not None:
            merged[data_key] = value
    return merged


def _resolve_role_styles(
    preset: Optional[str],
    role_styles: Optional[Dict[str, Union[str, Dict[str, str]]]],
) -> Dict[str, Dict[str, str]]:
    defaults = dict(_PRESET_ROLE_STYLES.get(preset or "", {}))
    if role_styles is None:
        return defaults
    out: Dict[str, Dict[str, str]] = {}
    out.update(defaults)
    for role, style in role_styles.items():
        if isinstance(style, str):
            out[role] = {"shape": style}
        else:
            out[role] = {str(k): str(v) for k, v in style.items()}
    return out


def _infer_node_roles_from_feature_data(
    feature_data: Any,
    sample: Optional[Union[int, str]] = None,
    role_key: str = "role",
) -> Dict[Any, str]:
    samples = getattr(feature_data, "samples", None)
    if samples is None:
        raise ValueError("feature_data must be a Data-like object with a 'samples' attribute.")

    items = list(samples.items())
    if len(items) == 0:
        return {}

    if sample is None:
        selected_items = items
    elif isinstance(sample, int):
        if sample < 0 or sample >= len(items):
            raise ValueError(f"sample index {sample} out of range for feature_data with {len(items)} samples.")
        selected_items = [items[sample]]
    else:
        if sample not in samples:
            raise ValueError(f"sample '{sample}' not found in feature_data samples.")
        selected_items = [(sample, samples[sample])]

    per_vertex_roles: Dict[Any, set] = {}
    for _, sample_obj in selected_items:
        for feat in sample_obj.features:
            feat_mapping = getattr(feat, "mapping", None)
            feat_role = feat.data.get(role_key, None) if hasattr(feat, "data") else None
            if feat_mapping != "vertex" or feat_role is None:
                continue
            per_vertex_roles.setdefault(feat.id, set()).add(str(feat_role))

    node_roles: Dict[Any, str] = {}
    for v, roles in per_vertex_roles.items():
        if len(roles) == 1:
            node_roles[v] = next(iter(roles))
        else:
            node_roles[v] = "mixed"
    return node_roles


def _build_vertex_attrs_from_roles(
    graph: BaseGraph,
    node_roles: Dict[Any, str],
    role_styles: Dict[str, Dict[str, str]],
) -> Dict[Union[int, str], Dict[str, str]]:
    attrs: Dict[Union[int, str], Dict[str, str]] = {}
    for v in graph.V:
        role = node_roles.get(v, None)
        if role is None:
            role = "internal" if "internal" in role_styles else None
        if role is None:
            continue
        style = role_styles.get(role, None)
        if style is None:
            continue
        attrs[v] = dict(style)
    return attrs


def _resolve_processors(processor: ProcessorArg) -> List[ProcessorFn]:
    if processor is None:
        return []
    candidates: List[Union[str, ProcessorFn]]
    if isinstance(processor, (list, tuple)):
        candidates = list(processor)
    else:
        candidates = [processor]

    resolved: List[ProcessorFn] = []
    for p in candidates:
        if callable(p):
            resolved.append(p)
        else:
            if p not in _PROCESSORS:
                raise ValueError(f"Unknown processor '{p}'. Available processors: {list(_PROCESSORS.keys())}")
            resolved.append(_PROCESSORS[p])
    return resolved


def _merge_attrs(
    base: Optional[Dict[Any, Dict[str, str]]],
    override: Optional[Dict[Any, Dict[str, str]]],
) -> Optional[Dict[Any, Dict[str, str]]]:
    if not base and not override:
        return None
    out: Dict[Any, Dict[str, str]] = {}
    for source in (base or {}, override or {}):
        for key, value in source.items():
            out.setdefault(key, {}).update(value)
    return out


def _plot_with_networkx(graph: BaseGraph, **kwargs) -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx

    from corneto.contrib.networkx import corneto_graph_to_networkx

    skip_unsupported_edges = bool(kwargs.get("skip_unsupported_edges", True))
    nx_graph = corneto_graph_to_networkx(graph, skip_unsupported_edges=skip_unsupported_edges)

    pos = kwargs.get("pos")
    if pos is None:
        pos = nx.spring_layout(nx_graph)

    ax = kwargs.get("ax")
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize"))
    else:
        fig = ax.figure

    nx.draw_networkx(nx_graph, pos=pos, ax=ax, with_labels=kwargs.get("node_labels") is None)
    if kwargs.get("node_labels") is not None:
        nx.draw_networkx_labels(nx_graph, pos=pos, labels=kwargs["node_labels"], ax=ax)
    ax.set_axis_off()
    return fig


def plot_graph(graph: BaseGraph, renderer: str = "auto", **kwargs) -> Any:
    import sys

    from corneto._settings import LOGGER
    from corneto._util import supports_html
    from corneto.contrib._util import dot_wasm_render

    preset = kwargs.pop("preset", None)
    theme_arg = kwargs.pop("theme", None)
    processor_arg = kwargs.pop("processor", None)
    data = kwargs.pop("data", None) or {}
    solution = kwargs.pop("solution", None)
    solution_map = kwargs.pop("solution_map", None)
    feature_data = kwargs.pop("feature_data", None)
    node_roles = kwargs.pop("node_roles", None)
    role_styles_arg = kwargs.pop("role_styles", None)
    sample = kwargs.pop("sample", None)
    role_key = kwargs.pop("role_key", "role")

    if preset is not None:
        if preset not in _PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {list(_PRESETS.keys())}")
        preset_cfg = _PRESETS[preset]
        if theme_arg is None:
            theme_arg = preset_cfg["theme"]
        if processor_arg is None:
            processor_arg = preset_cfg["processor"]

    preset_solution_map = None
    if preset in ("simple", "default", "signaling"):
        preset_solution_map = {"vertex": "vertex_value", "edge": "edge_value"}
    elif preset in ("metabolism_flux", "metabolism"):
        preset_solution_map = {"flux": EXPR_NAME_FLOW}
    merged_solution_map = dict(preset_solution_map or {})
    if solution_map is not None:
        merged_solution_map.update(solution_map)
    data = _merge_solution_data(data, solution, merged_solution_map)

    if node_roles is None and feature_data is not None:
        node_roles = _infer_node_roles_from_feature_data(feature_data, sample=sample, role_key=role_key)
    if node_roles is not None:
        role_styles = _resolve_role_styles(preset, role_styles_arg)
        role_vertex_attr = _build_vertex_attrs_from_roles(graph, node_roles, role_styles)
        kwargs["custom_vertex_attr"] = _merge_attrs(role_vertex_attr, kwargs.get("custom_vertex_attr"))

    processors = _resolve_processors(processor_arg)
    if processors:
        theme = _resolve_theme(theme_arg)
        for processor in processors:
            proc_edge_attr, proc_vertex_attr = processor(graph, data, theme)
            kwargs["custom_edge_attr"] = _merge_attrs(proc_edge_attr, kwargs.get("custom_edge_attr"))
            kwargs["custom_vertex_attr"] = _merge_attrs(proc_vertex_attr, kwargs.get("custom_vertex_attr"))

    def _as_wasm_plot() -> Any:
        if not supports_html():
            raise RuntimeError("HTML rendering is required for browser WASM plotting.")

        dot_input: Any
        try:
            dot_input = graph.to_graphviz(**kwargs)
        except Exception:
            dot_input = to_dot_source(graph, **kwargs)
        return dot_wasm_render(dot_input)

    if renderer == "wasm":
        return _as_wasm_plot()
    if renderer == "networkx":
        return _plot_with_networkx(graph, **kwargs)
    if renderer != "auto" and renderer != "graphviz":
        raise ValueError("Unknown renderer specified. Must be 'auto', 'graphviz', 'networkx', or 'wasm'.")

    # Pyodide/wasm notebooks cannot run dot subprocesses; prefer browser WASM.
    if renderer == "auto" and sys.platform == "emscripten":
        return _as_wasm_plot()

    try:
        graphviz_obj = graph.to_graphviz(**kwargs)
        graphviz_obj._repr_mimebundle_()
        return graphviz_obj
    except (OSError, Exception) as e:
        LOGGER.debug(f"Graphviz rendering failed: {e}.")
        if supports_html():
            LOGGER.debug("Falling back to browser WASM rendering.")
            return _as_wasm_plot()
        LOGGER.debug("HTML rendering not supported.")
        raise e


def suppress_repr_warnings(g):
    """Monkey-patch the _repr_* methods of an object instance (e.g. a graphviz.Digraph)
    so that any output written to stderr during their execution is suppressed.

    Parameters:
        g : object
            The instance whose _repr_* methods should have their warnings suppressed.
    """
    # Identify all representation methods (names starting with '_repr_') on the instance.
    repr_methods = [m for m in dir(g) if m.startswith("_repr_") and callable(getattr(g, m))]
    for method_name in repr_methods:
        original = getattr(g, method_name)

        def make_wrapper(orig_func):
            def wrapper(*args, **kwargs):
                with contextlib.redirect_stderr(io.StringIO()):
                    return orig_func(*args, **kwargs)

            return wrapper

        # Directly set the wrapped function on the instance
        setattr(g, method_name, make_wrapper(original))


def clip_quantiles(arr, q):
    if q < 0 or q > 1:
        raise ValueError(f"Clipping value must be between 0 and 1, got {q}")
    # compute the quantiles at clipping and 1-clipping and clip the flow
    q = np.quantile(arr, [q, 1 - q])
    return np.clip(arr, q[0], q[1])


def vertex_style(
    P,
    G: Optional[BaseGraph] = None,  # TODO: G should not be required
    vertex_var: str = "vertex_values",
    negative_color: str = "dodgerblue4",
    positive_color: str = "firebrick4",
    sample: Optional[int] = None,
):
    v_values = np.array(P.expr[vertex_var].value)
    if v_values is None:
        raise ValueError(
            f"""Variable {vertex_var} in the problem, but values are None.
            Has the problem been solved?"""
        )
    if sample is not None:
        if len(v_values.shape) < 2:
            raise ValueError(
                f"""Variable {vertex_var} in the problem is not a matrix.
                Cannot select sample {sample}"""
            )
        v_values = v_values[:, sample]
    if G is None:
        vertices = list(range(len(v_values)))
    else:
        vertices = G.V
    return create_graphviz_vertex_attributes(
        vertices, v_values, negative_color=negative_color, positive_color=positive_color
    )


def edge_style(
    P,
    max_edge_width: float = 5,
    min_edge_width: float = 0.25,
    edge_var: str = "edge_values",
    negative_color: str = "dodgerblue4",
    positive_color: str = "firebrick4",
    sample: Optional[int] = None,
):
    e_values = P.expr[edge_var].value
    if e_values is None:
        raise ValueError(
            f"""Variable {edge_var} in the problem, but values are None.
            Has the problem been solved?"""
        )
    if sample is not None:
        if len(e_values.shape) < 2:
            raise ValueError(
                f"""Variable {edge_var} in the problem is not a matrix.
                Cannot select sample {sample}"""
            )
        e_values = e_values[:, sample]
    return create_graphviz_edge_attributes(
        e_values,
        max_edge_width=max_edge_width,
        min_edge_width=min_edge_width,
        negative_color=negative_color,
        positive_color=positive_color,
    )


def create_graphviz_edge_attributes(
    edge_values: Union[List, np.ndarray],
    max_edge_width: float = 5,
    min_edge_width: float = 0.25,
    negative_color: str = "dodgerblue4",
    positive_color: str = "firebrick4",
):
    e_values = np.array(edge_values)
    edge_attrs = dict()
    for i, v in enumerate(e_values):
        if abs(v) > 0:
            edge_width = max_edge_width
        else:
            edge_width = min_edge_width
        edge_attrs[i] = {"penwidth": str(edge_width)}
        if e_values[i] > 0:
            edge_attrs[i]["color"] = positive_color
        elif e_values[i] < 0:
            edge_attrs[i]["color"] = negative_color
        else:
            edge_attrs[i]["color"] = "black"
    return edge_attrs


def create_graphviz_vertex_attributes(
    graph_vertices: List,
    vertex_values: Union[List, np.ndarray],
    negative_color: str = "dodgerblue4",
    positive_color: str = "firebrick4",
):
    v_values = np.array(vertex_values)
    if len(v_values) != len(graph_vertices):
        raise ValueError(
            f"""Length of vertex_values ({len(v_values)}) does not match the number
            of vertices ({len(graph_vertices)})"""
        )
    vertex_attrs = dict()
    for vn, v in zip(graph_vertices, v_values):
        vertex_attrs[vn] = dict()
        if v > 0:
            vertex_attrs[vn]["color"] = positive_color
            vertex_attrs[vn]["penwidth"] = "2"
        elif v < 0:
            vertex_attrs[vn]["color"] = negative_color
            vertex_attrs[vn]["penwidth"] = "2"
    return vertex_attrs


def flow_style(
    P,
    max_edge_width: float = 5,
    min_edge_width: float = 0.25,
    flow_name: str = EXPR_NAME_FLOW,
    positive_color: str = "dodgerblue4",
    negative_color: str = "firebrick4",
    zero_flow_threshold: float = 1e-6,
    scale: Optional[Literal["log", "std"]] = "log",
    clip_quantil: Optional[float] = 0.05,
):
    flow = np.array(P.expr[flow_name].value)
    flow[np.abs(flow) < zero_flow_threshold] = 0
    if scale is not None:
        if scale == "log":
            flow = np.sign(flow) * np.log10(np.abs(flow) + 1.0)
        elif scale == "std":
            flow = flow / np.std(flow)
        else:
            raise ValueError(f"Unknown normalization method: {scale}")
    if clip_quantil is not None:
        flow = clip_quantiles(flow, clip_quantil)
    max_flow = max(np.max(np.abs(flow)), 1e-6)
    edge_attrs = dict()
    for i, v in enumerate(flow):
        # Apply threshold edge width
        if abs(v) > 0:
            edge_width = max_edge_width
        else:
            edge_width = min_edge_width
        if scale is not None:
            edge_width = min_edge_width + (max_edge_width - min_edge_width) * abs(v / max_flow)
        edge_attrs[i] = {"penwidth": str(edge_width)}
        if flow[i] > 0:
            edge_attrs[i]["color"] = positive_color
        elif flow[i] < 0:
            edge_attrs[i]["color"] = negative_color
        else:
            edge_attrs[i]["color"] = "black"
    return edge_attrs


def _create_vertices(g, e, vertex_props=None):
    if vertex_props is None:
        vertex_props = {}
    v_s, v_t = [], []
    i, (s, t) = e

    # Function to update node properties with user-provided properties
    def update_node_props(v_name, default_shape):
        # Merge default shape with user provided properties, if any
        props = {"shape": default_shape}
        if v_name in vertex_props:
            props.update(vertex_props[v_name])
        g.node(v_name, **props)

    if len(s) == 0:
        v_name = f"e_{i}_source"
        update_node_props(v_name, "point")  # Updated to use helper function
        v_s.append(v_name)

    if len(t) == 0:
        v_name = f"e_{i}_target"
        update_node_props(v_name, "point")  # Updated to use helper function
        v_t.append(v_name)

    for v in s:
        v_name = str(v)
        v_s.append(v_name)
        update_node_props(v_name, "circle")  # Updated to use helper function

    for v in t:
        v_name = str(v)
        v_t.append(v_name)
        update_node_props(v_name, "circle")  # Updated to use helper function

    return v_s, v_t


def _dot_quote(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _dot_attr_dict(attrs: Optional[Dict[str, str]]) -> str:
    if not attrs:
        return ""
    parts = [f"{k}={_dot_quote(v)}" for k, v in attrs.items()]
    return "[" + ", ".join(parts) + "]"


def _normalize_custom_vertex_attr(
    graph: BaseGraph,
    custom_vertex_attr: Optional[Dict[Union[int, str], Dict[str, str]]],
) -> Dict[str, Dict[str, str]]:
    if custom_vertex_attr is None:
        return {}
    out = dict(custom_vertex_attr)
    if len(out) > 0:
        keys = list(out.keys())
        if all(isinstance(k, int) for k in keys):
            vertices = graph.V
            out = {str(v): out[i] for i, v in enumerate(vertices)}
    return out


def _build_plot_model(
    graph: BaseGraph,
    graph_attr: Optional[Dict[str, str]] = None,
    node_attr: Optional[Dict[str, str]] = None,
    edge_attr: Optional[Dict[str, str]] = None,
    custom_edge_attr: Optional[Dict[int, Dict[str, str]]] = None,
    custom_vertex_attr: Optional[Dict[Union[int, str], Dict[str, str]]] = None,
    edge_indexes: Optional[List[int]] = None,
    orphan_edges: bool = True,
) -> Dict[str, Any]:
    custom_edge_attr = custom_edge_attr or {}
    custom_vertex_attr_str = _normalize_custom_vertex_attr(graph, custom_vertex_attr)
    node_defaults = dict(fixedsize="true") if node_attr is None else node_attr

    nodes: List[Tuple[str, Dict[str, str]]] = []
    edges: List[Tuple[str, str, Dict[str, str]]] = []
    is_hypergraph = False

    for i, e in enumerate(graph.edges()):
        if edge_indexes is not None and i not in edge_indexes:
            continue
        i, (s, t) = e
        if not orphan_edges and (len(s) == 0 or len(t) == 0):
            continue

        v_s: List[str] = []
        v_t: List[str] = []

        def add_node(v_name: str, default_shape: str) -> None:
            attrs = {}
            if default_shape != "circle" or "shape" not in node_defaults:
                attrs["shape"] = default_shape
            if v_name in custom_vertex_attr_str:
                attrs.update(custom_vertex_attr_str[v_name])
            nodes.append((v_name, attrs))

        if len(s) == 0:
            v_name = f"e_{i}_source"
            add_node(v_name, "point")
            v_s.append(v_name)
        if len(t) == 0:
            v_name = f"e_{i}_target"
            add_node(v_name, "point")
            v_t.append(v_name)

        for v in s:
            v_name = str(v)
            v_s.append(v_name)
            add_node(v_name, "circle")
        for v in t:
            v_name = str(v)
            v_t.append(v_name)
            add_node(v_name, "circle")

        if len(s) > 1 or len(t) > 1:
            is_hypergraph = True
            edge_center = f"e_{i}_center"
            nodes.append((edge_center, {"shape": "square", "width": "0.1", "height": "0.1", "label": ""}))
            for v in v_s:
                attrs = dict(arrowtail="none", arrowhead="none", dir="both")
                attrs.update(custom_edge_attr.get(i, {}))
                edges.append((v, edge_center, attrs))
            for v in v_t:
                attrs = dict(custom_edge_attr.get(i, {}))
                edges.append((edge_center, v, attrs))
        else:
            if graph.get_attr_edge(i).get_attr(Attr.EDGE_TYPE, "") == EdgeType.UNDIRECTED.value:
                attrs = dict(arrowhead="none", dir="none")
                attrs.update(custom_edge_attr.get(i, {}))
            else:
                head = "normal"
                if graph.get_attr_edge(i).get("interaction", 0) < 0:
                    head = "tee"
                attrs = dict(arrowhead=head)
                attrs.update(custom_edge_attr.get(i, {}))
            edges.append((v_s[0], v_t[0], attrs))

    return {
        "graph_attr": graph_attr,
        "node_defaults": node_defaults,
        "edge_attr": edge_attr,
        "nodes": nodes,
        "edges": edges,
        "is_hypergraph": is_hypergraph,
    }


def to_dot_source(
    graph: BaseGraph,
    graph_attr: Optional[Dict[str, str]] = None,
    node_attr: Optional[Dict[str, str]] = None,
    edge_attr: Optional[Dict[str, str]] = None,
    custom_edge_attr: Optional[Dict[int, Dict[str, str]]] = None,
    custom_vertex_attr: Optional[Dict[Union[int, str], Dict[str, str]]] = None,
    edge_indexes: Optional[List[int]] = None,
    orphan_edges: bool = True,
) -> str:
    """Create DOT source without requiring graphviz/pydot."""
    model = _build_plot_model(
        graph=graph,
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
        custom_edge_attr=custom_edge_attr,
        custom_vertex_attr=custom_vertex_attr,
        edge_indexes=edge_indexes,
        orphan_edges=orphan_edges,
    )

    lines = ["digraph {"]
    if model["graph_attr"]:
        lines.append(f"  graph {_dot_attr_dict(model['graph_attr'])};")
    if model["node_defaults"]:
        lines.append(f"  node {_dot_attr_dict(model['node_defaults'])};")
    if model["edge_attr"]:
        lines.append(f"  edge {_dot_attr_dict(model['edge_attr'])};")
    for node_name, attrs in model["nodes"]:
        lines.append(f"  {_dot_quote(node_name)} {_dot_attr_dict(attrs)};")
    for source, target, attrs in model["edges"]:
        lines.append(f"  {_dot_quote(source)} -> {_dot_quote(target)} {_dot_attr_dict(attrs)};")

    if model["is_hypergraph"] and graph_attr is None:
        lines.append('  graph [splines="true"];')
    lines.append("}")
    return "\n".join(lines)


def to_python_graphviz(
    graph: BaseGraph,
    graph_attr: Optional[Dict[str, str]] = None,
    node_attr: Optional[Dict[str, str]] = None,
    edge_attr: Optional[Dict[str, str]] = None,
    custom_edge_attr: Optional[Dict[int, Dict[str, str]]] = None,
    custom_vertex_attr: Optional[Dict[Union[int, str], Dict[str, str]]] = None,
    edge_indexes: Optional[List[int]] = None,
    layout: str = "dot",
    orphan_edges: bool = True,
    supress_warnings: bool = True,
) -> Any:
    import graphviz  # type: ignore

    model = _build_plot_model(
        graph=graph,
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
        custom_edge_attr=custom_edge_attr,
        custom_vertex_attr=custom_vertex_attr,
        edge_indexes=edge_indexes,
        orphan_edges=orphan_edges,
    )
    g = graphviz.Digraph(
        engine=layout,
        graph_attr=model["graph_attr"],
        node_attr=model["node_defaults"],
        edge_attr=model["edge_attr"],
    )
    for node_name, attrs in model["nodes"]:
        g.node(node_name, **attrs)
    for source, target, attrs in model["edges"]:
        g.edge(source, target, **attrs)
    if model["is_hypergraph"] and graph_attr is None:
        g.graph_attr["splines"] = "true"
    if supress_warnings:
        suppress_repr_warnings(g)
    return g


def _pydot_create_vertices(g, e, vertex_props=None):
    import pydot

    if vertex_props is None:
        vertex_props = {}
    v_s, v_t = [], []
    i, (s, t) = e

    def update_node_props(v_name, default_shape):
        props = {"shape": default_shape}
        if v_name in vertex_props:
            props.update(vertex_props[v_name])
        node = pydot.Node(v_name, **props)
        g.add_node(node)

    if len(s) == 0:
        v_name = f"e_{i}_source"
        update_node_props(v_name, "point")
        v_s.append(v_name)
    if len(t) == 0:
        v_name = f"e_{i}_target"
        update_node_props(v_name, "point")
        v_t.append(v_name)
    for v in s:
        v_name = str(v)
        v_s.append(v_name)
        update_node_props(v_name, "circle")
    for v in t:
        v_name = str(v)
        v_t.append(v_name)
        update_node_props(v_name, "circle")
    return v_s, v_t


def to_pydot(
    graph: BaseGraph,
    graph_attr: Optional[Dict[str, str]] = None,
    node_attr: Optional[Dict[str, str]] = None,
    edge_attr: Optional[Dict[str, str]] = None,
    custom_edge_attr: Optional[Dict[int, Dict[str, str]]] = None,
    custom_vertex_attr: Optional[Dict[Union[int, str], Dict[str, str]]] = None,
    edge_indexes: Optional[List[int]] = None,
    layout: str = "dot",
    orphan_edges: bool = True,
) -> Any:
    import pydot

    model = _build_plot_model(
        graph=graph,
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
        custom_edge_attr=custom_edge_attr,
        custom_vertex_attr=custom_vertex_attr,
        edge_indexes=edge_indexes,
        orphan_edges=orphan_edges,
    )
    # Create a pydot.Dot graph
    g = pydot.Dot(graph_type="digraph", **(model["graph_attr"] if model["graph_attr"] else {}))
    if node_attr is not None:
        g.set_node_defaults(**node_attr)
    if model["edge_attr"] is not None:
        g.set_edge_defaults(**model["edge_attr"])
    for node_name, attrs in model["nodes"]:
        g.add_node(pydot.Node(node_name, **attrs))
    for source, target, attrs in model["edges"]:
        g.add_edge(pydot.Edge(source, target, **attrs))
    if model["is_hypergraph"] and graph_attr is None:
        g.set_splines("true")
    return g


def to_graphviz(
    graph: BaseGraph,
    graph_attr: Optional[Dict[str, str]] = None,
    node_attr: Optional[Dict[str, str]] = None,
    edge_attr: Optional[Dict[str, str]] = None,
    custom_edge_attr: Optional[Dict[int, Dict[str, str]]] = None,
    custom_vertex_attr: Optional[Dict[Union[int, str], Dict[str, str]]] = None,
    edge_indexes: Optional[List[int]] = None,
    layout: str = "dot",
    orphan_edges: bool = True,
    supress_warnings: bool = True,
    backend: Literal["graphviz", "pydot"] = "graphviz",
) -> Any:
    """Generate and return a graph using the selected backend.
    backend:
       - 'graphviz' (default) returns a graphviz.Digraph object.
       - 'pydot' returns a pydot.Dot graph.
    """
    if backend == "graphviz":
        return to_python_graphviz(
            graph,
            graph_attr,
            node_attr,
            edge_attr,
            custom_edge_attr,
            custom_vertex_attr,
            edge_indexes,
            layout,
            orphan_edges,
            supress_warnings,
        )
    elif backend == "pydot":
        return to_pydot(
            graph,
            graph_attr,
            node_attr,
            edge_attr,
            custom_edge_attr,
            custom_vertex_attr,
            edge_indexes,
            layout,
            orphan_edges,
        )
    else:
        raise ValueError("Unknown backend specified. Must be 'graphviz' or 'pydot'.")
