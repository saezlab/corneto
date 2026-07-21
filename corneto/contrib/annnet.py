"""Conversion helpers for CORNETO and AnnNet graphs.

The converters focus on the flat, directed graphs and hypergraphs commonly
used by CORNETO. AnnNet-only concepts such as layers, slices, and edge-entities
are flattened or ignored. Undirected hyperedges preserve their members, but
not a CORNETO-specific source/target partition.
"""

import warnings
from copy import deepcopy
from numbers import Number
from typing import TYPE_CHECKING

from corneto.graph import BaseGraph, EdgeType, Graph
from corneto.utils import Attr, import_optional_module

if TYPE_CHECKING:
    from annnet import AnnNet


_ANNNET_VERTEX_RESERVED = {"layer", "slice", "vertex_id"}
_ANNNET_EDGE_RESERVED = {
    "as_entity",
    "default_edge_directed",
    "default_edge_type",
    "default_propagate",
    "default_slice_weight",
    "default_weight",
    "directed",
    "edge_directed",
    "edge_id",
    "edge_type",
    "flexible",
    "head",
    "kind",
    "members",
    "parallel",
    "propagate",
    "slice",
    "slice_weight",
    "source",
    "src",
    "tail",
    "target",
    "tgt",
    "weight",
}
_CORNETO_EDGE_RESERVED = {
    Attr.EDGE_TYPE.value,
    Attr.SOURCE_ATTR.value,
    Attr.TARGET_ATTR.value,
}
_ANNNET_EDGE_ID = "_annnet_edge_id"


def _copy_supported_attributes(attributes, reserved, element):
    copied = {k: deepcopy(v) for k, v in attributes.items() if k not in reserved}
    omitted = sorted(set(attributes).intersection(reserved))
    if omitted:
        warnings.warn(
            f"Omitting {element} attributes reserved by AnnNet: {', '.join(omitted)}",
            UserWarning,
            stacklevel=3,
        )
    return copied


def _endpoint_magnitude(edge_attributes, side: Attr, vertex) -> float:
    side_attributes = edge_attributes.get(side.value, {})
    vertex_attributes = side_attributes.get(vertex, {})
    if isinstance(vertex_attributes, dict):
        value = vertex_attributes.get(Attr.VALUE.value, 1.0)
    else:
        value = vertex_attributes
    if not isinstance(value, Number):
        return 1.0
    return float(abs(value))


def _uniform_magnitude(source, target, edge_attributes):
    magnitudes = [
        *(_endpoint_magnitude(edge_attributes, Attr.SOURCE_ATTR, v) for v in source),
        *(_endpoint_magnitude(edge_attributes, Attr.TARGET_ATTR, v) for v in target),
    ]
    if not magnitudes:
        return 1.0
    first = magnitudes[0]
    return first if all(value == first for value in magnitudes[1:]) else None


def to_annnet(graph: BaseGraph, *, copy_attributes: bool = True) -> "AnnNet":
    """Convert a CORNETO graph to an AnnNet graph.

    Directed edges and hyperedges, parallel edges, endpoint coefficients, and
    ordinary attributes are preserved. Vertex identifiers are converted to
    strings because AnnNet uses string identifiers. A collision after string
    conversion raises ``ValueError``.

    Undirected hyperedges are represented by their combined member set, so a
    CORNETO-specific source/target partition is not preserved.
    """
    annnet = import_optional_module("annnet")
    result = annnet.AnnNet(directed=None)

    vertex_ids = {}
    used_ids = set()
    for vertex in graph.V:
        vertex_id = str(vertex)
        if vertex_id in used_ids:
            raise ValueError(
                f"Multiple CORNETO vertices map to AnnNet vertex {vertex_id!r}."
            )
        used_ids.add(vertex_id)
        vertex_ids[vertex] = vertex_id
        attributes = {}
        if copy_attributes:
            attributes = _copy_supported_attributes(
                graph.get_attr_vertex(vertex),
                _ANNNET_VERTEX_RESERVED,
                "vertex",
            )
        result.add_vertices(vertex_id, **attributes)

    if copy_attributes:
        result.uns.update(deepcopy(dict(graph.get_graph_attributes())))

    for index, (source, target) in graph.edges():
        if not source or not target:
            raise ValueError(
                f"AnnNet conversion requires both endpoint sets; CORNETO edge {index} "
                "has an empty source or target."
            )

        edge_attributes = graph.get_attr_edge(index)
        edge_type = edge_attributes.get(Attr.EDGE_TYPE.value, EdgeType.DIRECTED)
        directed = edge_type in {EdgeType.DIRECTED, EdgeType.DIRECTED.value}
        annnet_edge_id = str(
            edge_attributes.get(_ANNNET_EDGE_ID, f"corneto_edge_{index}")
        )
        attributes = {}
        if copy_attributes:
            ordinary_attributes = {
                key: value
                for key, value in edge_attributes.items()
                if key not in _CORNETO_EDGE_RESERVED
            }
            attributes = _copy_supported_attributes(
                ordinary_attributes,
                _ANNNET_EDGE_RESERVED | {_ANNNET_EDGE_ID},
                "edge",
            )

        annnet_source = [vertex_ids[v] for v in source]
        annnet_target = [vertex_ids[v] for v in target]
        uniform_weight = _uniform_magnitude(source, target, edge_attributes)

        if not directed:
            members = list(dict.fromkeys([*annnet_source, *annnet_target]))
            if len(source) == len(target) == 1:
                result.add_edges(
                    annnet_source[0],
                    annnet_target[0],
                    edge_id=annnet_edge_id,
                    directed=False,
                    parallel="parallel",
                    weight=uniform_weight if uniform_weight is not None else 1.0,
                    **attributes,
                )
            else:
                result.add_edges(
                    members,
                    edge_id=annnet_edge_id,
                    directed=False,
                    parallel="parallel",
                    weight=uniform_weight if uniform_weight is not None else 1.0,
                    **attributes,
                )
            continue

        if uniform_weight is not None:
            source_arg = annnet_source[0] if len(annnet_source) == 1 else annnet_source
            target_arg = annnet_target[0] if len(annnet_target) == 1 else annnet_target
        else:
            source_arg = {
                vertex_ids[v]: -_endpoint_magnitude(
                    edge_attributes, Attr.SOURCE_ATTR, v
                )
                for v in source
            }
            target_arg = {
                vertex_ids[v]: _endpoint_magnitude(edge_attributes, Attr.TARGET_ATTR, v)
                for v in target
            }

        result.add_edges(
            source_arg,
            target_arg,
            edge_id=annnet_edge_id,
            directed=True,
            parallel="parallel",
            weight=uniform_weight if uniform_weight is not None else 1.0,
            **attributes,
        )

    return result


def _edge_directed(graph, edge, edge_id, directed_edge_ids) -> bool:
    directed = getattr(edge, "directed", None)
    return bool(directed) if directed is not None else edge_id in directed_edge_ids


def _edge_coefficient(matrix, row_by_vertex, edge_index, vertex) -> float:
    row = row_by_vertex.get(vertex)
    if row is None:
        return 1.0
    value = matrix[row, edge_index]
    try:
        value = value.item()
    except AttributeError:
        pass
    return float(abs(value)) if isinstance(value, Number) and value != 0 else 1.0


def from_annnet(graph: "AnnNet", *, copy_attributes: bool = True) -> Graph:
    """Convert a flat AnnNet graph to a CORNETO graph.

    AnnNet layers, slice membership, edge-entities, and direction policies do
    not have direct CORNETO equivalents and are not reproduced. Undirected
    hyperedges use the complete member set on both CORNETO endpoint sides.
    """
    import_optional_module("annnet")
    graph_attributes = deepcopy(dict(graph.uns)) if copy_attributes else {}
    result = Graph()
    result.get_graph_attributes().update(graph_attributes)

    vertices = list(graph.vertices())
    for vertex in vertices:
        attributes = {}
        if copy_attributes:
            attributes = dict(graph.attrs.get_vertex_attrs(vertex))
            attributes.pop("vertex_id", None)
        result.add_vertex(vertex, **deepcopy(attributes))

    edge_ids = list(graph.edges())
    directed_edge_ids = set(graph.get_edges_by_direction(True))
    matrix = graph.X()
    row_by_vertex = {graph.get_vertex(i): i for i in range(graph.nv)}

    for edge_index, edge_id in enumerate(edge_ids):
        edge = graph.get_edge(edge_id)
        source, target = edge
        directed = _edge_directed(graph, edge, edge_id, directed_edge_ids)
        attributes = {}
        if copy_attributes:
            attributes = dict(graph.attrs.get_edge_attrs(edge_id))
            attributes.pop("edge_id", None)
            attributes = _copy_supported_attributes(
                attributes,
                _CORNETO_EDGE_RESERVED,
                "edge",
            )
            attributes[_ANNNET_EDGE_ID] = edge_id

        if directed:
            corneto_source = {
                vertex: _edge_coefficient(matrix, row_by_vertex, edge_index, vertex)
                for vertex in source
            }
            corneto_target = {
                vertex: _edge_coefficient(matrix, row_by_vertex, edge_index, vertex)
                for vertex in target
            }
        else:
            members = sorted(set(source) | set(target), key=str)
            if len(members) == 2:
                corneto_source = {
                    members[0]: _edge_coefficient(
                        matrix, row_by_vertex, edge_index, members[0]
                    )
                }
                corneto_target = {
                    members[1]: _edge_coefficient(
                        matrix, row_by_vertex, edge_index, members[1]
                    )
                }
            else:
                corneto_source = {
                    vertex: _edge_coefficient(matrix, row_by_vertex, edge_index, vertex)
                    for vertex in members
                }
                corneto_target = dict(corneto_source)

        result.add_edge(
            corneto_source,
            corneto_target,
            type=EdgeType.DIRECTED if directed else EdgeType.UNDIRECTED,
            **attributes,
        )

    return result


__all__ = ["from_annnet", "to_annnet"]
