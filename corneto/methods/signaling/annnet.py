"""AnnNet integration for CellNOptDAG signaling analyses."""

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from corneto.contrib.annnet import from_annnet
from corneto.utils import import_optional_module

if TYPE_CHECKING:
    from annnet import AnnNet

    from corneto.backend._base import ProblemDef
    from corneto.methods.signaling import CellNOptDAG


@dataclass(frozen=True)
class _CellNOptAnnNetContext:
    condition_layers: dict[str, tuple[str, ...]]
    source_edge_ids: tuple[str, ...]
    source_edges: dict[str, tuple[str, str, int]]


def _matching_condition_keys(**collections: Mapping[str, Any]) -> tuple[str, ...]:
    names: tuple[str, ...] | None = None
    for argument, values in collections.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"{argument} must be a mapping from condition names to protein values.")
        current = tuple(values)
        if names is None:
            names = current
        elif set(current) != set(names):
            raise ValueError(f"{argument} must contain the same conditions as inputs.")
    return names or ()


def _condition_layer_map(
    graph: "AnnNet",
    *,
    condition_aspect: str,
    condition_layers: Mapping[str, tuple[str, ...]] | None,
) -> dict[str, tuple[str, ...]]:
    aspects = tuple(graph.layers.list_aspects())
    if condition_layers is not None:
        result = {str(name): tuple(layer) for name, layer in condition_layers.items()}
        if not result:
            raise ValueError("condition_layers must contain at least one condition.")
        if any(len(layer) != len(aspects) for layer in result.values()):
            raise ValueError("Each condition layer must provide one value for every AnnNet layer aspect.")
        return result

    if aspects != (condition_aspect,):
        raise ValueError(
            f"Automatic condition discovery requires one AnnNet layer aspect named {condition_aspect!r}. "
            "Pass condition_layers explicitly when the network uses several aspects."
        )
    names = graph.layers.list_layers(aspect=condition_aspect)
    if not names:
        raise ValueError(f"The AnnNet object has no layers for the {condition_aspect!r} aspect.")
    return {str(name): (str(name),) for name in names}


def add_cellnopt_conditions(
    graph: "AnnNet",
    *,
    inputs: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, Mapping[str, Any]],
    inhibitors: Mapping[str, Mapping[str, Any]] | None = None,
    condition_aspect: str = "condition",
    input_attr: str = "input",
    inhibitor_attr: str = "inhibited",
    measurement_attr: str = "observed",
) -> dict[str, tuple[str, ...]]:
    """Add CellNOpt perturbations and measurements as AnnNet condition layers.

    Every condition receives a copy of the graph's existing vertices. Input,
    inhibitor, and measurement values are stored as vertex-layer attributes.
    The returned mapping can be passed to :func:`build_cellnopt_from_annnet`
    when a caller wants to name or select layers explicitly.
    """
    import_optional_module("annnet")
    if inhibitors is None:
        inhibitors = {condition: {} for condition in inputs}
    condition_names = _matching_condition_keys(
        inputs=inputs,
        measurements=measurements,
        inhibitors=inhibitors,
    )
    if not condition_names:
        raise ValueError("At least one experimental condition is required.")

    aspects = tuple(graph.layers.list_aspects())
    if aspects and aspects != (condition_aspect,):
        raise ValueError(
            "add_cellnopt_conditions can initialize an unlayered AnnNet object or reuse its single "
            f"{condition_aspect!r} aspect. Use AnnNet directly for experiments with several layer aspects."
        )
    if not aspects:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Declared aspects")
            graph.layers.set_aspects(
                [condition_aspect],
                {condition_aspect: list(condition_names)},
            )

    known_vertices = set(graph.vertices())
    base_vertices = list(graph.vertices())
    layers = {condition: (condition,) for condition in condition_names}
    for condition, layer in layers.items():
        graph.add_vertices(base_vertices, layer=layer)
        values_by_role = (
            (inputs[condition], input_attr),
            (inhibitors[condition], inhibitor_attr),
            (measurements[condition], measurement_attr),
        )
        for values, attribute in values_by_role:
            if not isinstance(values, Mapping):
                raise TypeError(f"Values for condition {condition!r} must be mappings keyed by protein.")
            unknown = set(values) - known_vertices
            if unknown:
                protein = sorted(unknown, key=str)[0]
                raise ValueError(f"Unknown protein {protein!r} in condition {condition!r}.")
            for protein, value in values.items():
                graph.layers.set_vertex_layer_attrs(str(protein), layer, **{attribute: value})
    return layers


def _cellnopt_source_graph(graph: "AnnNet", network_slice: str | None):
    selected = set(graph.edges()) if network_slice is None else set(graph.slices.edges(network_slice))
    edge_ids = [edge_id for edge_id in graph.edges() if edge_id in selected]
    directed_edge_ids = set(graph.get_edges_by_direction(True))
    source = graph.__class__(directed=None)

    def biological_id(vertex):
        if isinstance(vertex, tuple) and len(vertex) == 2 and isinstance(vertex[1], tuple):
            return str(vertex[0])
        return str(vertex)

    for edge_id in edge_ids:
        edge_source, edge_target = graph.get_edge(edge_id)
        sources = [biological_id(vertex) for vertex in edge_source]
        targets = [biological_id(vertex) for vertex in edge_target]
        if len(sources) != 1 or len(targets) != 1:
            raise ValueError(
                "CellNOptDAG requires signed interactions with one source and one target; "
                f"AnnNet edge {edge_id!r} has {len(sources)} sources and {len(targets)} targets."
            )
        source.add_edges(
            sources[0],
            targets[0],
            edge_id=edge_id,
            directed=edge_id in directed_edge_ids,
            parallel="parallel",
        )
        attributes = dict(graph.attrs.get_edge_attrs(edge_id))
        attributes.pop("edge_id", None)
        if attributes:
            source.attrs.set_edge_attrs(edge_id, **deepcopy(attributes))

    source.uns.update(deepcopy(dict(graph.uns)))
    return source


def build_cellnopt_from_annnet(
    method: "CellNOptDAG",
    graph: "AnnNet",
    *,
    network_slice: str | None = None,
    condition_aspect: str = "condition",
    condition_layers: Mapping[str, tuple[str, ...]] | None = None,
    input_attr: str = "input",
    inhibitor_attr: str = "inhibited",
    measurement_attr: str = "observed",
) -> "ProblemDef":
    """Build a CellNOptDAG problem from a signed network and data in AnnNet.

    The signed interactions are read from ``network_slice`` when supplied.
    Inputs, inhibitors, and measurements are read from vertex attributes in
    the condition layers. Internal graph preparation is kept inside this
    integration function so that AnnNet remains the user-facing graph object.
    """
    from corneto.methods.signaling import CellNOptDAG

    if not isinstance(method, CellNOptDAG):
        raise TypeError("method must be a CellNOptDAG instance.")
    layers = _condition_layer_map(
        graph,
        condition_aspect=condition_aspect,
        condition_layers=condition_layers,
    )
    source = _cellnopt_source_graph(graph, network_slice)
    pkn = from_annnet(source)

    inputs: dict[str, dict[str, Any]] = {}
    inhibitors: dict[str, dict[str, Any]] = {}
    measurements: dict[str, dict[str, Any]] = {}
    for condition, layer in layers.items():
        inputs[condition] = {}
        inhibitors[condition] = {}
        measurements[condition] = {}
        for protein in graph.layers.layer_vertex_set(layer):
            protein_id = str(protein[0]) if isinstance(protein, tuple) else str(protein)
            attributes = graph.layers.get_vertex_layer_attrs(protein_id, layer)
            if input_attr in attributes:
                inputs[condition][protein_id] = attributes[input_attr]
            if attributes.get(inhibitor_attr):
                inhibitors[condition][protein_id] = 1
            if measurement_attr in attributes:
                measurements[condition][protein_id] = attributes[measurement_attr]

    source_edge_ids = tuple(source.edges())
    source_edges = {}
    for edge_id in source_edge_ids:
        edge_source, edge_target = source.get_edge(edge_id)
        if len(edge_source) != 1 or len(edge_target) != 1:
            continue
        attributes = source.attrs.get_edge_attrs(edge_id)
        source_edges[edge_id] = (
            str(next(iter(edge_source))),
            str(next(iter(edge_target))),
            int(attributes.get("interaction", 1)),
        )

    problem = method.build_many(
        pkn,
        inputs=inputs,
        measurements=measurements,
        inhibitors=inhibitors,
    )
    problem._annnet_cellnopt_context = _CellNOptAnnNetContext(
        condition_layers=layers,
        source_edge_ids=source_edge_ids,
        source_edges=source_edges,
    )
    return problem


def _reaction_text(reaction) -> str:
    literals = [str(node) for node in reaction.positive_literals]
    literals.extend(f"NOT {node}" for node in reaction.negative_literals)
    return f"{' AND '.join(literals)} -> {reaction.product}"


def add_cellnopt_results(
    graph: "AnnNet",
    method: "CellNOptDAG",
    problem: "ProblemDef",
    *,
    solution: Any = None,
    model_slice: str = "cellnopt_selected",
    prediction_attr: str = "predicted",
    activity_attr: str = "active",
    measurement_attr: str = "observed",
    error_attr: str = "absolute_error",
) -> dict[str, Any]:
    """Add a solved CellNOptDAG model and its condition results to AnnNet.

    The selected interactions are added to ``model_slice``. Predictions are
    stored on protein-layer pairs, while copies of selected interactions in
    each condition layer record whether their reaction is active.
    """
    context = getattr(problem, "_annnet_cellnopt_context", None)
    if context is None:
        raise ValueError("Build the problem with build_cellnopt_from_annnet before adding its results.")
    required = ("reaction_selected", "reaction_active", "vertex_value")
    missing = [name for name in required if getattr(problem.expr, name).value is None]
    if missing:
        raise ValueError("Solve the CellNOptDAG problem before adding results to AnnNet.")
    if graph.slices.exists(model_slice):
        raise ValueError(f"AnnNet slice {model_slice!r} already exists.")

    selected = np.rint(np.asarray(problem.expr.reaction_selected.value)).astype(int).reshape(-1)
    activity = np.rint(np.asarray(problem.expr.reaction_active.value)).astype(int)
    predictions = np.rint(np.asarray(problem.expr.vertex_value.value)).astype(int)
    selected_indices = np.flatnonzero(selected)
    condition_names = tuple(context.condition_layers)
    vertices = [str(vertex) for vertex in method.processed_graph.V]

    graph.slices.add(model_slice, role="inferred_model", method="CellNOptDAG")
    selected_source_edges = {
        context.source_edge_ids[edge_index]
        for reaction_index in selected_indices
        for edge_index in method.reactions[reaction_index].source_edges
    }
    for edge_id in selected_source_edges:
        graph.slices.add_edge_to_slice(model_slice, edge_id)

    result_edge_ids = []
    condition_errors = {}
    for condition_index, condition in enumerate(condition_names):
        layer = context.condition_layers[condition]
        graph.add_vertices(vertices, layer=layer)
        endpoint_error = 0.0
        for vertex_index, protein in enumerate(vertices):
            predicted = float(predictions[vertex_index, condition_index])
            attributes = {prediction_attr: predicted}
            existing = graph.layers.get_vertex_layer_attrs(protein, layer)
            if measurement_attr in existing:
                error = abs(predicted - float(existing[measurement_attr]))
                attributes[error_attr] = error
                endpoint_error += error
            graph.layers.set_vertex_layer_attrs(protein, layer, **attributes)

        condition_edges = []
        for reaction_index in selected_indices:
            reaction = method.reactions[reaction_index]
            is_active = bool(activity[reaction_index, condition_index])
            for source_edge_index in reaction.source_edges:
                prior_edge_id = context.source_edge_ids[source_edge_index]
                source, target, interaction = context.source_edges[prior_edge_id]
                condition_edges.append(
                    {
                        "source": (source, layer),
                        "target": (target, layer),
                        "edge_id": (
                            f"{model_slice}__condition_{condition_index}__reaction_{reaction_index}"
                            f"__edge_{source_edge_index}"
                        ),
                        "interaction": interaction,
                        "prior_edge_id": prior_edge_id,
                        "reaction_index": int(reaction_index),
                        "reaction": _reaction_text(reaction),
                        "selected": True,
                        activity_attr: is_active,
                        "source_method": "CellNOptDAG",
                    }
                )
        result_edge_ids.extend(
            graph.add_edges(
                condition_edges,
                slice=model_slice,
                default_edge_directed=True,
            )
        )

        active_count = int(activity[selected_indices, condition_index].sum())
        layer_attributes = {
            "endpoint_absolute_error": float(endpoint_error),
            "selected_reactions": len(selected_indices),
            "active_reactions": active_count,
        }
        if solution is not None and getattr(solution, "status", None) is not None:
            layer_attributes["solver_status"] = str(solution.status)
        graph.layers.set_layer_attrs(layer, **layer_attributes)
        condition_errors[condition] = endpoint_error

    graph.history.snapshot("cellnopt_results_added")
    return {
        "selected_reactions": len(selected_indices),
        "selected_prior_edges": len(selected_source_edges),
        "condition_edges": len(result_edge_ids),
        "condition_errors": condition_errors,
    }


__all__ = [
    "add_cellnopt_conditions",
    "add_cellnopt_results",
    "build_cellnopt_from_annnet",
]
