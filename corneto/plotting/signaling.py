"""Reusable biological styling for signed CORNETO signaling graphs.

The processor uses CORNETO's public ``processor(graph, data, theme)`` API.
It does not fetch annotations or depend on CARNIVAL/OmniPath/Decoupler.
Node classes must be supplied by the caller; regulator annotations win overlaps.
Arrowheads encode the prior interaction, dashes encode reduced source activity,
and fills encode inferred node activity. Widths do not encode confidence.
"""

import numpy as np

from corneto.graph import Attr, EdgeType

__all__ = ["BIOLOGY_THEME", "biology_signaling_style"]


BIOLOGY_THEME = {
    "positive_color": "#F6D8B5",
    "negative_color": "#CCE2F2",
    "zero_color": "#F2F2F2",
    "node_border": "#555555",
    "edge_color": "#555555",
    "reduced_source_edge_color": "#888888",
    "edge_width": 1.1,
    "arrow_size": 0.65,
    "node_border_width": 0.8,
    "measured_border_width": 1.8,
    "receptor_shape": "box",
    "regulator_shape": "hexagon",
    "protein_shape": "ellipse",
}


def _activity_vector(values, size, name):
    """Accept one sample in graph order, including column vectors."""
    values = np.asarray(values, dtype=float)
    if values.shape == (size, 1):
        values = values[:, 0]
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain {size} finite values for one sample in graph order.")
    return values


def biology_signaling_style(graph, plot_data, theme):
    """Style signed signaling activity using explicit biological annotations.

    ``plot_data`` requires finite ``vertex_values`` in ``graph.V`` order and
    optionally accepts ``receptors``, ``regulators``, and ``measured`` vertex
    collections. Regulators take precedence over receptors. The nonnegative
    ``activity_threshold`` defaults to 0.5; values at either threshold are neutral.
    Only directed edges with one source and one target receive edge styling.
    Edge selection and boundary filtering remain the caller's responsibility.
    """
    theme = {**BIOLOGY_THEME, **theme}
    threshold = float(plot_data.get("activity_threshold", 0.5))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("activity_threshold must be finite and nonnegative.")
    if "vertex_values" not in plot_data:
        raise ValueError("signaling_biology requires data['vertex_values'] in graph order.")
    values = _activity_vector(plot_data["vertex_values"], len(graph.V), "vertex_values")
    activity = dict(zip(graph.V, values, strict=True))
    receptors = set(plot_data.get("receptors", ()))
    regulators = set(plot_data.get("regulators", ()))
    measured = set(plot_data.get("measured", ()))
    edge_attrs, vertex_attrs = {}, {}

    for vertex in graph.V:
        # Regulator annotations take precedence over the broad LigRecExtra
        # target list, which can also contain intracellular proteins (e.g. SMAD3).
        if vertex in regulators:
            shape, style = theme["regulator_shape"], "filled"
            node_class = "TF / transcriptional regulator"
        elif vertex in receptors:
            shape, style = theme["receptor_shape"], "rounded,filled"
            node_class = "Receptor"
        else:
            shape, style = theme["protein_shape"], "filled"
            node_class = "Other signaling protein"
        value = activity[vertex]
        fill = (
            theme["positive_color"]
            if value > threshold
            else theme["negative_color"]
            if value < -threshold
            else theme["zero_color"]
        )
        vertex_attrs[str(vertex)] = {
            "shape": shape,
            "style": style,
            "fillcolor": fill,
            "color": theme["node_border"],
            "penwidth": str(theme["measured_border_width"] if vertex in measured else theme["node_border_width"]),
            "tooltip": f"{vertex}: {node_class}; inferred activity {value:+g}",
        }

    for index, (sources, targets) in graph.edges():
        # Boundary edges and hyperedges have no unambiguous single-source activity.
        if len(sources) != 1 or len(targets) != 1:
            continue
        if graph.get_attr_edge(index).get_attr(Attr.EDGE_TYPE, "") == EdgeType.UNDIRECTED.value:
            continue
        source = next(iter(sources))
        reduced = activity[source] < -threshold
        inhibitory = graph.get_attr_edge(index).get("interaction", 0) < 0
        mechanism = "inhibition" if inhibitory else "activation"
        effect = f"Reduced {mechanism}" if reduced else f"Increased {mechanism}"
        if reduced and inhibitory:
            effect = "Release of inhibition"
        if abs(activity[source]) <= threshold:
            effect = f"No inferred change in upstream activity ({mechanism})"
        edge_attrs[index] = {
            "arrowhead": "tee" if inhibitory else "normal",
            "style": "dashed" if reduced else "solid",
            "color": theme["reduced_source_edge_color"] if reduced else theme["edge_color"],
            "penwidth": str(theme["edge_width"]),
            "arrowsize": str(theme["arrow_size"]),
            "tooltip": effect,
        }
    return edge_attrs, vertex_attrs
