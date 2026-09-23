"""Visualization utilities for :class:`~corneto.methods.signaling.CellNOptDAG`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np

from corneto._plotting import _scaled_magnitudes
from corneto.graph import Graph
from corneto.methods.signaling.cellnopt_dag import CellNOptDAG

__all__ = ["plot_cellnopt_fit", "plot_cellnopt_model"]

_STIMULUS_COLOR = "#9ACD32"
_MEASUREMENT_COLOR = "#ADD8E6"
_INHIBITOR_COLOR = "#FF6846"
_POSITIVE_COLOR = "#222222"
_NEGATIVE_COLOR = "#C43C39"
_INACTIVE_POSITIVE_COLOR = "#A8A8A8"
_INACTIVE_NEGATIVE_COLOR = "#E3A19F"
_UNSELECTED_POSITIVE_COLOR = "#D5D5D5"
_UNSELECTED_NEGATIVE_COLOR = "#F0CECD"


@dataclass(frozen=True)
class _CellNOptPlotData:
    conditions: tuple[str, ...]
    vertices: tuple[Any, ...]
    selected: np.ndarray
    active: np.ndarray
    states: np.ndarray
    flow_by_reaction: tuple[np.ndarray, ...]
    measurement_mask: np.ndarray
    measurements: np.ndarray
    input_mask: np.ndarray
    input_values: np.ndarray
    inhibitor_mask: np.ndarray


@dataclass(frozen=True)
class _FitSeries:
    """Observed and predicted responses on an explicit time axis.

    CellNOptDAG currently infers one steady-state endpoint per condition, so
    extraction produces a singleton axis labelled ``Endpoint``. Keeping time
    as a real array dimension lets a future time-resolved formulation reuse
    the renderer without assigning scientific meaning to series identity.
    """

    times: np.ndarray
    time_labels: tuple[str, ...]
    observed: np.ndarray
    measured: np.ndarray
    predicted: np.ndarray


@dataclass(frozen=True)
class _ModelPlotSpec:
    graph: Graph
    vertex_attributes: dict[Any, dict[str, str]]
    edge_attributes: dict[int, dict[str, str]]
    graph_attributes: dict[str, str]
    node_attributes: dict[str, str]


def _expression_values(problem: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if problem is None:
        raise ValueError("CellNOptDAG has not been built. Build and solve the method before plotting.")
    try:
        expression = problem.expr[name]
    except (KeyError, TypeError, AttributeError):
        try:
            expression = getattr(problem.expr, name)
        except AttributeError as exc:
            raise ValueError(f"CellNOpt solution does not contain expression {name!r}.") from exc
    value = getattr(expression, "value", None)
    if value is None:
        raise ValueError(f"CellNOpt expression {name!r} has no value. Solve a feasible problem before plotting.")
    array = np.asarray(value, dtype=float)
    try:
        array = array.reshape(shape)
    except ValueError as exc:
        raise ValueError(f"CellNOpt expression {name!r} has shape {array.shape}; expected {shape}.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"CellNOpt expression {name!r} contains non-finite values.")
    return array


def _validate_binary_values(values: np.ndarray, name: str, tolerance: float = 1e-5) -> np.ndarray:
    if np.any(values < -tolerance) or np.any(values > 1 + tolerance):
        raise ValueError(f"CellNOpt expression {name!r} contains values outside the binary domain.")
    return values >= 0.5


def _extract_plot_data(method: CellNOptDAG, problem: Any = None) -> _CellNOptPlotData:
    if not isinstance(method, CellNOptDAG):
        raise TypeError("method must be a CellNOptDAG instance.")
    if problem is None:
        problem = method.problem
    conditions = tuple(method._condition_names)
    vertices = tuple(method.processed_graph.V) if method.processed_graph is not None else ()
    if not conditions or not vertices or not method.reactions:
        raise ValueError("CellNOptDAG has not been built. Build and solve the method before plotting.")

    num_reactions = len(method.reactions)
    num_conditions = len(conditions)
    num_vertices = len(vertices)
    selected_values = _expression_values(problem, "reaction_selected", (num_reactions,))
    active_values = _expression_values(problem, "reaction_active", (num_reactions, num_conditions))
    state_values = _expression_values(problem, "vertex_value", (num_vertices, num_conditions))
    flow_values = _expression_values(problem, "flow", tuple(problem.expr.flow.shape)).reshape(-1)
    if flow_values.size < method._biological_num_edges:
        raise ValueError("CellNOpt flow vector is shorter than the compiled biological dependency graph.")

    selected = _validate_binary_values(selected_values, "reaction_selected")
    active = _validate_binary_values(active_values, "reaction_active")
    states = _validate_binary_values(state_values, "vertex_value")

    dependency_flow = flow_values[: method._biological_num_edges]
    flow_by_reaction = []
    cursor = 0
    for reaction in method.reactions:
        next_cursor = cursor + len(reaction.literals)
        flow_by_reaction.append(dependency_flow[cursor:next_cursor].copy())
        cursor = next_cursor
    if cursor != method._biological_num_edges:
        raise ValueError("Compiled reactions do not align with CellNOpt dependency-flow variables.")

    input_mask = np.zeros((num_vertices, num_conditions), dtype=bool)
    input_values = np.zeros((num_vertices, num_conditions), dtype=float)
    inhibitor_mask = np.zeros((num_vertices, num_conditions), dtype=bool)
    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    for condition_index, condition in enumerate(conditions):
        sample = method.processed_data.samples[condition]
        for feature in sample.features:
            index = vertex_index[feature.id]
            role = feature.data.get("role")
            if role in {"input", "input_output"}:
                input_mask[index, condition_index] = True
                input_values[index, condition_index] = float(feature.data.get("input_value", feature.value))
            if feature.data.get("intervention") == "inhibitor":
                inhibitor_mask[index, condition_index] = True

    return _CellNOptPlotData(
        conditions=conditions,
        vertices=vertices,
        selected=selected,
        active=active,
        states=states,
        flow_by_reaction=tuple(flow_by_reaction),
        measurement_mask=np.asarray(method._measurement_mask, dtype=bool),
        measurements=np.asarray(method._measurements, dtype=float),
        input_mask=input_mask,
        input_values=input_values,
        inhibitor_mask=inhibitor_mask,
    )


def _condition_index(
    data: _CellNOptPlotData,
    condition: Optional[Union[int, str]],
) -> Optional[int]:
    if condition is None:
        return None
    if isinstance(condition, int):
        if condition < 0 or condition >= len(data.conditions):
            raise ValueError(f"condition index {condition} out of range for {len(data.conditions)} conditions.")
        return condition
    if condition not in data.conditions:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {list(data.conditions)!r}.")
    return data.conditions.index(condition)


def _reaction_text(reaction: Any) -> str:
    literals = [str(vertex) for vertex in reaction.positive_literals]
    literals.extend(f"NOT {vertex}" for vertex in reaction.negative_literals)
    return f"{' AND '.join(literals)} -> {reaction.product}"


def _node_roles(
    data: _CellNOptPlotData,
    condition_index: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if condition_index is None:
        inputs = data.input_mask.any(axis=1)
        outputs = data.measurement_mask.any(axis=1)
        inhibitors = data.inhibitor_mask.any(axis=1)
    else:
        inputs = data.input_mask[:, condition_index]
        outputs = data.measurement_mask[:, condition_index]
        inhibitors = data.inhibitor_mask[:, condition_index]
    return inputs, outputs, inhibitors


def _merge_attributes(
    generated: dict[Any, dict[str, str]],
    supplied: Optional[dict[Any, dict[str, str]]],
) -> dict[Any, dict[str, str]]:
    merged = {key: dict(value) for key, value in generated.items()}
    for key, attrs in (supplied or {}).items():
        merged.setdefault(key, {}).update({str(name): str(value) for name, value in attrs.items()})
    return merged


def _build_cellnopt_model_plot(
    method: CellNOptDAG,
    problem: Any = None,
    *,
    condition: Optional[Union[int, str]] = None,
    show_unselected: bool = False,
    show_inactive: bool = True,
    width_by: Literal["selection", "flow"] = "selection",
) -> _ModelPlotSpec:
    if width_by not in {"selection", "flow"}:
        raise ValueError("width_by must be 'selection' or 'flow'.")
    data = _extract_plot_data(method, problem)
    condition_index = _condition_index(data, condition)

    included_reactions = []
    segment_flows = []
    for reaction_index, reaction in enumerate(method.reactions):
        selected = bool(data.selected[reaction_index])
        active = selected if condition_index is None else bool(data.active[reaction_index, condition_index])
        include = selected or show_unselected
        if condition_index is not None and selected and not active and not show_inactive:
            include = False
        if not include:
            continue
        included_reactions.append((reaction_index, reaction, selected, active))
        flows = data.flow_by_reaction[reaction_index]
        segment_flows.extend(flows.tolist())
        if len(reaction.literals) > 1:
            segment_flows.append(float(np.sum(flows)))

    flow_scale = (
        _scaled_magnitudes(
            np.asarray(segment_flows, dtype=float),
            zero_threshold=1e-9,
            scale="log",
            clip_quantile=0.0,
        )
        if width_by == "flow"
        else np.empty(0)
    )
    flow_scale_cursor = 0

    graph = Graph()
    included_species = set()
    for _, reaction, _, _ in included_reactions:
        included_species.update(reaction.literals)
        included_species.add(reaction.product)
    inputs, outputs, inhibitors = _node_roles(data, condition_index)
    experimental_species = {
        vertex for vertex, is_experimental in zip(data.vertices, inputs | outputs | inhibitors) if is_experimental
    }
    included_species.update(experimental_species)
    for vertex in data.vertices:
        if vertex in included_species:
            graph.add_vertex(vertex)

    existing_names = {str(vertex) for vertex in data.vertices}
    and_nodes = {}
    for reaction_index, reaction, _, _ in included_reactions:
        if len(reaction.literals) <= 1:
            continue
        candidate = f"__cellnopt_and_{reaction_index}"
        while candidate in existing_names:
            candidate = f"_{candidate}"
        existing_names.add(candidate)
        and_nodes[reaction_index] = candidate
        graph.add_vertex(candidate)

    vertex_attributes: dict[Any, dict[str, str]] = {}
    vertex_index = {vertex: index for index, vertex in enumerate(data.vertices)}
    for vertex in included_species:
        index = vertex_index[vertex]
        is_input = bool(inputs[index])
        is_output = bool(outputs[index])
        is_inhibitor = bool(inhibitors[index])
        is_stimulus = is_input and not is_inhibitor
        if is_inhibitor:
            fillcolor = _INHIBITOR_COLOR
            role = "inhibitor target"
        elif is_stimulus:
            fillcolor = _STIMULUS_COLOR
            role = "stimulus"
        elif is_output:
            fillcolor = _MEASUREMENT_COLOR
            role = "measurement"
        else:
            fillcolor = "white"
            role = "internal species"
        attrs = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": fillcolor,
            "color": "#4C4C4C",
            "fontcolor": "#111111",
            "label": str(vertex),
            "tooltip": role,
            "margin": "0.08,0.05",
        }
        if is_stimulus and is_output:
            attrs["peripheries"] = "2"
            attrs["color"] = "#4C78A8"
            attrs["tooltip"] = "stimulus and measurement"
        elif is_inhibitor and is_output:
            attrs["peripheries"] = "2"
            attrs["color"] = "#4C78A8"
            attrs["tooltip"] = "inhibitor target and measurement"
        if condition_index is not None:
            value = int(data.states[index, condition_index])
            attrs["penwidth"] = "2.5" if value else "1.0"
            attrs["tooltip"] += f"; state={value}"
        vertex_attributes[str(vertex)] = attrs
    for reaction_index, node in and_nodes.items():
        vertex_attributes[node] = {
            "shape": "circle",
            "style": "filled",
            "fillcolor": "white",
            "color": "#333333",
            "label": "AND",
            "width": "0.38",
            "height": "0.38",
            "fixedsize": "true",
            "fontsize": "8",
            "tooltip": _reaction_text(method.reactions[reaction_index]),
        }

    edge_attributes: dict[int, dict[str, str]] = {}

    def state_attributes(
        *,
        interaction: int,
        selected: bool,
        active: bool,
        raw_flow: float,
        aggregate: bool,
        reaction_text: str,
    ) -> dict[str, str]:
        nonlocal flow_scale_cursor
        if not selected:
            color = _UNSELECTED_NEGATIVE_COLOR if interaction < 0 else _UNSELECTED_POSITIVE_COLOR
            style = "dotted"
            width = 0.6
        elif condition_index is not None and not active:
            color = _INACTIVE_NEGATIVE_COLOR if interaction < 0 else _INACTIVE_POSITIVE_COLOR
            style = "dashed"
            width = 1.0
        else:
            color = _NEGATIVE_COLOR if interaction < 0 else _POSITIVE_COLOR
            style = "solid"
            width = 2.8 if condition_index is not None else 2.2
        if width_by == "flow":
            magnitude = float(flow_scale[flow_scale_cursor])
            flow_scale_cursor += 1
            if selected:
                width = 0.8 + 4.2 * magnitude
        tooltip = (
            f"{reaction_text}; selected={int(selected)}"
            + ("" if condition_index is None else f"; active={int(active)}")
            + f"; {'aggregate ' if aggregate else ''}structural flow={raw_flow:.5g}"
        )
        return {
            "color": color,
            "fontcolor": color,
            "style": style,
            "penwidth": f"{width:.3g}",
            "arrowhead": "tee" if interaction < 0 else "normal",
            "tooltip": tooltip,
        }

    for reaction_index, reaction, selected, active in included_reactions:
        reaction_text = _reaction_text(reaction)
        flows = data.flow_by_reaction[reaction_index]
        literals = [
            *((vertex, 1) for vertex in reaction.positive_literals),
            *((vertex, -1) for vertex in reaction.negative_literals),
        ]
        if len(literals) == 1:
            literal, interaction = literals[0]
            edge_index = graph.add_edge(
                literal,
                reaction.product,
                interaction=interaction,
                reaction=reaction_index,
            )
            edge_attributes[edge_index] = state_attributes(
                interaction=interaction,
                selected=selected,
                active=active,
                raw_flow=float(flows[0]),
                aggregate=False,
                reaction_text=reaction_text,
            )
            continue

        and_node = and_nodes[reaction_index]
        for literal_index, (literal, interaction) in enumerate(literals):
            edge_index = graph.add_edge(
                literal,
                and_node,
                interaction=interaction,
                reaction=reaction_index,
            )
            edge_attributes[edge_index] = state_attributes(
                interaction=interaction,
                selected=selected,
                active=active,
                raw_flow=float(flows[literal_index]),
                aggregate=False,
                reaction_text=reaction_text,
            )
        aggregate_flow = float(np.sum(flows))
        edge_index = graph.add_edge(
            and_node,
            reaction.product,
            interaction=1,
            reaction=reaction_index,
        )
        edge_attributes[edge_index] = state_attributes(
            interaction=1,
            selected=selected,
            active=active,
            raw_flow=aggregate_flow,
            aggregate=True,
            reaction_text=reaction_text,
        )

    return _ModelPlotSpec(
        graph=graph,
        vertex_attributes=vertex_attributes,
        edge_attributes=edge_attributes,
        graph_attributes={"rankdir": "LR", "pad": "0.2", "nodesep": "0.35", "ranksep": "0.5"},
        node_attributes={"fixedsize": "false", "fontname": "Helvetica", "fontsize": "10"},
    )


def plot_cellnopt_model(
    method: CellNOptDAG,
    problem: Any = None,
    *,
    condition: Optional[Union[int, str]] = None,
    show_unselected: bool = False,
    show_inactive: bool = True,
    width_by: Literal["selection", "flow"] = "selection",
    renderer: str = "auto",
    **plot_kwargs: Any,
) -> Any:
    """Plot a solved CellNOpt model using CORNETO's graph renderers.

    ``condition=None`` shows the shared selected structure. Selecting a
    condition overlays reaction activity and predicted species states.
    Structural flow may optionally control edge widths, but never edge color.
    """
    spec = _build_cellnopt_model_plot(
        method,
        problem,
        condition=condition,
        show_unselected=show_unselected,
        show_inactive=show_inactive,
        width_by=width_by,
    )
    graph_attributes = dict(spec.graph_attributes)
    graph_attributes.update(plot_kwargs.pop("graph_attr", {}) or {})
    node_attributes = dict(spec.node_attributes)
    node_attributes.update(plot_kwargs.pop("node_attr", {}) or {})
    edge_attributes = _merge_attributes(
        spec.edge_attributes,
        plot_kwargs.pop("custom_edge_attr", None),
    )
    vertex_attributes = _merge_attributes(
        spec.vertex_attributes,
        {str(vertex): attrs for vertex, attrs in (plot_kwargs.pop("custom_vertex_attr", None) or {}).items()},
    )
    return spec.graph.plot(
        renderer=renderer,
        graph_attr=graph_attributes,
        node_attr=node_attributes,
        custom_edge_attr=edge_attributes,
        custom_vertex_attr=vertex_attributes,
        **plot_kwargs,
    )


def _selected_indices(
    values: Optional[Union[Any, Sequence[Any]]],
    available: Sequence[Any],
    *,
    argument: str,
) -> list[int]:
    if values is None:
        return list(range(len(available)))
    if isinstance(values, (str, int)) or values in available:
        values = [values]
    indices = []
    for value in values:
        if isinstance(value, int) and argument == "conditions":
            if value < 0 or value >= len(available):
                raise ValueError(f"{argument} index {value} out of range.")
            indices.append(value)
        else:
            if value not in available:
                raise ValueError(f"Unknown {argument[:-1]} {value!r}; expected one of {list(available)!r}.")
            indices.append(available.index(value))
    if not indices:
        raise ValueError(f"{argument} must select at least one value.")
    return indices


def _fit_selection(
    data: _CellNOptPlotData,
    conditions: Optional[Union[Union[int, str], Sequence[Union[int, str]]]],
    signals: Optional[Union[Any, Sequence[Any]]],
) -> tuple[list[int], list[int]]:
    condition_indices = _selected_indices(conditions, data.conditions, argument="conditions")
    measured_vertices = [
        vertex for vertex, measured in zip(data.vertices, data.measurement_mask.any(axis=1)) if measured
    ]
    selected_signals = measured_vertices if signals is None else signals
    signal_indices = _selected_indices(selected_signals, data.vertices, argument="signals")
    return condition_indices, signal_indices


def _cue_data(
    data: _CellNOptPlotData,
    condition_indices: Sequence[int],
) -> tuple[np.ndarray, list[str], list[str]]:
    selected_inputs = data.input_mask[:, condition_indices]
    selected_inhibitors = data.inhibitor_mask[:, condition_indices]
    stimulus_vertices = [
        index for index in range(len(data.vertices)) if np.any(selected_inputs[index] & ~selected_inhibitors[index])
    ]
    inhibitor_vertices = [index for index in range(len(data.vertices)) if np.any(selected_inhibitors[index])]
    labels = [str(data.vertices[index]) for index in stimulus_vertices]
    labels.extend(f"{data.vertices[index]} (inh)" for index in inhibitor_vertices)
    kinds = ["stimulus"] * len(stimulus_vertices) + ["inhibitor"] * len(inhibitor_vertices)
    cues = np.zeros((len(condition_indices), len(labels)), dtype=float)
    for row, condition_index in enumerate(condition_indices):
        for column, vertex_index in enumerate(stimulus_vertices):
            if not data.inhibitor_mask[vertex_index, condition_index]:
                cues[row, column] = data.input_values[vertex_index, condition_index]
        offset = len(stimulus_vertices)
        for column, vertex_index in enumerate(inhibitor_vertices):
            cues[row, offset + column] = float(data.inhibitor_mask[vertex_index, condition_index])
    return cues, labels, kinds


def _import_matplotlib():
    try:
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError(
            "CellNOpt fit plots require Matplotlib. Install CORNETO with `pip install 'corneto[plot]'`."
        ) from exc
    return plt, colors, Line2D


def _endpoint_fit_series(
    data: _CellNOptPlotData,
    condition_indices: Sequence[int],
    signal_indices: Sequence[int],
) -> _FitSeries:
    """Return current steady-state results with an explicit endpoint axis."""
    observed = data.measurements[np.ix_(signal_indices, condition_indices)].T
    measured = data.measurement_mask[np.ix_(signal_indices, condition_indices)].T
    predicted = data.states[np.ix_(signal_indices, condition_indices)].T.astype(float)
    return _FitSeries(
        times=np.array([0.0]),
        time_labels=("Endpoint",),
        observed=observed[..., np.newaxis],
        measured=measured[..., np.newaxis],
        predicted=predicted[..., np.newaxis],
    )


def _plot_fit_grid(
    data: _CellNOptPlotData,
    condition_indices: Sequence[int],
    signal_indices: Sequence[int],
    *,
    figsize: Optional[tuple[float, float]],
):
    plt, colors, Line2D = _import_matplotlib()
    num_conditions = len(condition_indices)
    num_signals = len(signal_indices)
    cues, cue_labels, cue_kinds = _cue_data(data, condition_indices)
    series = _endpoint_fit_series(data, condition_indices, signal_indices)
    if figsize is None:
        condition_label_width = min(
            4.0,
            0.09 * max(len(data.conditions[index]) for index in condition_indices),
        )
        figsize = (
            max(
                6.0,
                2.15 * num_signals + max(2.6, 0.65 * len(cue_labels)) + condition_label_width,
            ),
            max(3.2, 1.65 * num_conditions + 0.45),
        )
    width_ratios = [1.0] * num_signals + [max(1.2, 0.35 * max(len(cue_labels), 1))]
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    layout = figure.add_gridspec(2, 1, height_ratios=[0.12, 1])
    legend_axis = figure.add_subplot(layout[0])
    legend_axis.axis("off")
    panel_layout = layout[1].subgridspec(
        num_conditions,
        num_signals + 1,
        width_ratios=width_ratios,
    )
    axes = np.empty((num_conditions, num_signals + 1), dtype=object)
    for row in range(num_conditions):
        for column in range(num_signals + 1):
            axes[row, column] = figure.add_subplot(panel_layout[row, column])
    error_cmap = colors.LinearSegmentedColormap.from_list(
        "cellnopt_absolute_error",
        ["#E7F5E7", "#FFF0B3", "#F4B7B2"],
    )
    error_norm = colors.Normalize(vmin=0, vmax=1)

    for row, condition_index in enumerate(condition_indices):
        for column, signal_index in enumerate(signal_indices):
            axis = axes[row, column]
            predicted = series.predicted[row, column]
            observed = series.observed[row, column]
            measured = series.measured[row, column]
            if np.any(measured):
                error = float(np.mean(np.abs(observed[measured] - predicted[measured])))
                axis.set_facecolor(error_cmap(error_norm(error)))
                for time, observed_value, predicted_value, is_measured in zip(
                    series.times,
                    observed,
                    predicted,
                    measured,
                ):
                    if is_measured:
                        axis.plot(
                            [time, time],
                            [observed_value, predicted_value],
                            color="#777777",
                            linewidth=1.0,
                            label="_nolegend_",
                            zorder=1,
                        )
            else:
                axis.set_facecolor("#F2F2F2")
                axis.patch.set_hatch("//")
                axis.patch.set_edgecolor("#C8C8C8")

            prediction_line = "-" if len(series.times) > 1 else "none"
            axis.plot(
                series.times,
                predicted,
                color="#2C7FB8",
                marker="s",
                markersize=5,
                linewidth=1.5,
                linestyle=prediction_line,
                label="Model",
                zorder=2,
            )
            if np.any(measured):
                observed_values = np.ma.array(observed, mask=~measured)
                observation_line = "-" if np.count_nonzero(measured) > 1 else "none"
                axis.plot(
                    series.times,
                    observed_values,
                    color=_NEGATIVE_COLOR,
                    marker="o",
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                    markersize=7,
                    linewidth=1.2,
                    linestyle=observation_line,
                    label="Observed",
                    zorder=3,
                )

            if len(series.times) == 1:
                axis.set_xlim(series.times[0] - 0.5, series.times[0] + 0.5)
            else:
                span = float(series.times[-1] - series.times[0])
                margin = max(0.03 * span, 1e-6)
                axis.set_xlim(series.times[0] - margin, series.times[-1] + margin)
            axis.set_ylim(-0.05, 1.05)
            axis.set_yticks([0, 0.5, 1])
            if row == 0:
                axis.set_title(str(data.vertices[signal_index]), fontsize=10)
            if row == num_conditions - 1:
                axis.set_xticks(series.times, series.time_labels)
            else:
                axis.set_xticks([])
            if column == 0:
                axis.set_ylabel(
                    data.conditions[condition_index],
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=10,
                )
            else:
                axis.set_yticklabels([])
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)

        cue_axis = axes[row, -1]
        if cue_labels:
            colors = [_STIMULUS_COLOR if kind == "stimulus" else _INHIBITOR_COLOR for kind in cue_kinds]
            cue_axis.bar(
                np.arange(len(cue_labels)),
                cues[row],
                color=colors,
                edgecolor="#555555",
                linewidth=0.5,
            )
            cue_axis.set_xlim(-0.6, len(cue_labels) - 0.4)
        cue_axis.set_ylim(0, 1.05)
        cue_axis.set_yticks([])
        if row == 0:
            cue_axis.set_title("Cues", fontsize=10)
        if row == num_conditions - 1:
            cue_axis.set_xticks(
                np.arange(len(cue_labels)),
                cue_labels,
                rotation=45,
                ha="right",
            )
        else:
            cue_axis.set_xticks([])

    legend_axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color=_NEGATIVE_COLOR,
                marker="o",
                markerfacecolor="none",
                linestyle="none",
                label="Observed",
            ),
            Line2D([], [], color="#2C7FB8", marker="s", linestyle="none", label="Model"),
        ],
        loc="center",
        ncols=2,
        frameon=False,
    )
    error_scale = plt.cm.ScalarMappable(norm=error_norm, cmap=error_cmap)
    error_scale.set_array([])
    colorbar = figure.colorbar(
        error_scale,
        ax=axes.ravel().tolist(),
        fraction=0.018,
        pad=0.02,
    )
    colorbar.set_label("Mean absolute error")
    return figure, axes


def _annotate_heatmap(axis: Any, values: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
    if values.size > 100:
        return
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if mask is not None and mask[row, column]:
                continue
            axis.text(
                column,
                row,
                f"{values[row, column]:.2g}",
                ha="center",
                va="center",
                fontsize=8,
            )


def _plot_fit_heatmaps(
    data: _CellNOptPlotData,
    condition_indices: Sequence[int],
    signal_indices: Sequence[int],
    *,
    figsize: Optional[tuple[float, float]],
):
    plt, colors, _ = _import_matplotlib()
    observed = data.measurements[np.ix_(signal_indices, condition_indices)].T
    measured = data.measurement_mask[np.ix_(signal_indices, condition_indices)].T
    predicted = data.states[np.ix_(signal_indices, condition_indices)].T.astype(float)
    error = np.abs(observed - predicted)
    cues, cue_labels, cue_kinds = _cue_data(data, condition_indices)
    condition_labels = [data.conditions[index] for index in condition_indices]
    signal_labels = [str(data.vertices[index]) for index in signal_indices]

    if figsize is None:
        figsize = (
            max(9.5, 1.0 * len(signal_labels) + 0.55 * len(cue_labels)),
            max(3.0, 0.52 * len(condition_labels) + 1.8),
        )
    figure, axes = plt.subplots(
        1,
        4,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, max(0.7, len(cue_labels) / max(len(signal_labels), 1))]},
    )
    axes = axes[0]

    value_cmap = plt.get_cmap("viridis").copy()
    value_cmap.set_bad("#E8E8E8")
    error_cmap = plt.get_cmap("RdYlGn_r").copy()
    error_cmap.set_bad("#E8E8E8")
    observed_masked = np.ma.array(observed, mask=~measured)
    error_masked = np.ma.array(error, mask=~measured)
    matrices = [observed_masked, predicted, error_masked]
    titles = ["Observed", "Model", "Absolute error"]
    cmaps = [value_cmap, value_cmap, error_cmap]
    for axis, matrix, title, cmap in zip(axes[:3], matrices, titles, cmaps):
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=cmap)
        axis.set_title(title)
        axis.set_xticks(np.arange(len(signal_labels)), signal_labels, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(condition_labels)), condition_labels)
        _annotate_heatmap(
            axis,
            np.asarray(matrix.filled(0) if np.ma.isMaskedArray(matrix) else matrix),
            np.ma.getmaskarray(matrix) if np.ma.isMaskedArray(matrix) else None,
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)

    cue_axis = axes[3]
    if cue_labels:
        cue_codes = np.zeros_like(cues)
        for column, kind in enumerate(cue_kinds):
            cue_codes[:, column] = cues[:, column] * (1 if kind == "stimulus" else 2)
        cue_cmap = colors.ListedColormap(["#FFFFFF", _STIMULUS_COLOR, _INHIBITOR_COLOR])
        cue_axis.imshow(cue_codes, aspect="auto", vmin=0, vmax=2, cmap=cue_cmap)
    else:
        cue_axis.imshow(np.zeros((len(condition_labels), 1)), aspect="auto", vmin=0, vmax=1, cmap="Greys")
        cue_labels = ["None"]
    cue_axis.set_title("Cues")
    cue_axis.set_xticks(np.arange(len(cue_labels)), cue_labels, rotation=45, ha="right")
    cue_axis.set_yticks(np.arange(len(condition_labels)), condition_labels)
    return figure, axes


def plot_cellnopt_fit(
    method: CellNOptDAG,
    problem: Any = None,
    *,
    view: Literal["cellnopt", "heatmap"] = "cellnopt",
    conditions: Optional[Union[Union[int, str], Sequence[Union[int, str]]]] = None,
    signals: Optional[Union[Any, Sequence[Any]]] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Compare CellNOpt measurements and predictions across conditions.

    The ``cellnopt`` view overlays observations and model predictions at the
    inferred endpoint for every selected condition and signal. Unmeasured
    signals show model predictions only. ``heatmap`` provides aligned matrices
    for larger datasets.
    """
    data = _extract_plot_data(method, problem)
    condition_indices, signal_indices = _fit_selection(data, conditions, signals)
    if view == "cellnopt":
        return _plot_fit_grid(
            data,
            condition_indices,
            signal_indices,
            figsize=figsize,
        )
    if view == "heatmap":
        return _plot_fit_heatmaps(
            data,
            condition_indices,
            signal_indices,
            figsize=figsize,
        )
    raise ValueError("view must be 'cellnopt' or 'heatmap'.")
