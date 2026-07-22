"""Private graph-layout and incidence helpers shared by network methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix

from corneto.graph import BaseGraph, EdgeType


@dataclass(frozen=True)
class BoundaryFlowLayout:
    """An augmented graph and stable mappings for its boundary edges."""

    graph: BaseGraph
    original_num_edges: int
    inflow_edges: dict[Any, int]
    outflow_edges: dict[Any, int]

    @property
    def biological_edge_indices(self) -> np.ndarray:
        return np.arange(self.original_num_edges, dtype=int)

    @property
    def boundary_edge_indices(self) -> np.ndarray:
        return np.arange(self.original_num_edges, self.graph.num_edges, dtype=int)


@dataclass(frozen=True)
class DirectedIncidence:
    """Sparse directed incidence matrices over a selected set of edges."""

    outgoing: csr_matrix
    incoming: csr_matrix
    edge_indices: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray


def augment_with_boundaries(
    graph: BaseGraph,
    *,
    inflow_vertices: Iterable[Any] = (),
    outflow_vertices: Iterable[Any] = (),
    inflow_type: EdgeType = EdgeType.DIRECTED,
    outflow_type: EdgeType = EdgeType.DIRECTED,
    boundary_order: tuple[str, str] = ("inflow", "outflow"),
) -> BoundaryFlowLayout:
    """Copy a graph and add one boundary edge for each requested vertex."""
    augmented = graph.copy()
    inflow_vertices = tuple(dict.fromkeys(inflow_vertices))
    outflow_vertices = tuple(dict.fromkeys(outflow_vertices))
    inflow_edges = {}
    outflow_edges = {}
    if set(boundary_order) != {"inflow", "outflow"}:
        raise ValueError("boundary_order must contain 'inflow' and 'outflow' exactly once.")
    for direction in boundary_order:
        if direction == "inflow":
            inflow_edges.update(
                (vertex, augmented.add_edge((), vertex, type=inflow_type))
                for vertex in inflow_vertices
            )
        else:
            outflow_edges.update(
                (vertex, augmented.add_edge(vertex, (), type=outflow_type))
                for vertex in outflow_vertices
            )
    return BoundaryFlowLayout(
        graph=augmented,
        original_num_edges=graph.num_edges,
        inflow_edges=inflow_edges,
        outflow_edges=outflow_edges,
    )


def directed_incidence(
    graph: BaseGraph,
    edge_indices: Iterable[int] | None = None,
) -> DirectedIncidence:
    """Return unambiguous sparse tail/head matrices for simple directed edges."""
    indexes = np.asarray(
        list(range(graph.num_edges)) if edge_indices is None else list(edge_indices),
        dtype=int,
    )
    vertex_index = {vertex: index for index, vertex in enumerate(graph.V)}
    source_indices = []
    target_indices = []
    outgoing_rows = []
    outgoing_columns = []
    incoming_rows = []
    incoming_columns = []
    for column, edge_index in enumerate(indexes):
        source, target = graph.get_edge(int(edge_index))
        if len(source) > 1 or len(target) > 1 or (not source and not target):
            raise ValueError(
                "Directed incidence requires simple edges; "
                f"edge {edge_index} has {len(source)} source and {len(target)} target vertices."
            )
        source_index = vertex_index[next(iter(source))] if source else -1
        target_index = vertex_index[next(iter(target))] if target else -1
        if source:
            outgoing_rows.append(source_index)
            outgoing_columns.append(column)
        if target:
            incoming_rows.append(target_index)
            incoming_columns.append(column)
        source_indices.append(source_index)
        target_indices.append(target_index)

    shape = (graph.num_vertices, len(indexes))
    outgoing = csr_matrix(
        (np.ones(len(outgoing_rows)), (outgoing_rows, outgoing_columns)),
        shape=shape,
    )
    incoming = csr_matrix(
        (np.ones(len(incoming_rows)), (incoming_rows, incoming_columns)),
        shape=shape,
    )
    return DirectedIncidence(
        outgoing=outgoing,
        incoming=incoming,
        edge_indices=indexes,
        source_indices=np.asarray(source_indices, dtype=int),
        target_indices=np.asarray(target_indices, dtype=int),
    )
