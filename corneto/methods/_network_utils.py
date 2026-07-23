"""Private graph-layout and incidence helpers shared by network methods."""

from __future__ import annotations

from collections.abc import Mapping
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


@dataclass(frozen=True)
class PathPruning:
    """A graph restricted to condition-specific source-to-target paths."""

    graph: BaseGraph
    original_vertex_indices: np.ndarray
    original_edge_indices: np.ndarray
    vertices_by_condition: dict[str, frozenset[Any]]
    unreachable_sources: dict[str, frozenset[Any]]
    unreachable_targets: dict[str, frozenset[Any]]


def prune_to_paths(
    graph: BaseGraph,
    *,
    sources: Mapping[str, Iterable[Any]],
    targets: Mapping[str, Iterable[Any]],
) -> PathPruning:
    """Keep vertices and edges on a directed source-to-target path.

    Reachability is computed independently for every condition. The returned
    graph is the union of those condition-specific path subgraphs, preserving
    the input graph's vertex and edge order.
    """
    if tuple(sources) != tuple(targets):
        if set(sources) != set(targets):
            raise ValueError("sources and targets must contain the same condition names.")
        targets = {condition: targets[condition] for condition in sources}

    graph_vertices = set(graph.V)
    vertices_by_condition: dict[str, frozenset[Any]] = {}
    unreachable_sources: dict[str, frozenset[Any]] = {}
    unreachable_targets: dict[str, frozenset[Any]] = {}
    retained_vertices: set[Any] = set()
    retained_edges: set[int] = set()

    for condition in sources:
        condition_sources = set(sources[condition]) & graph_vertices
        condition_targets = set(targets[condition]) & graph_vertices
        forward = set(graph.bfs(condition_sources))
        backward = set(graph.bfs(condition_targets, reverse=True))
        condition_vertices = forward & backward & graph_vertices
        vertices_by_condition[condition] = frozenset(condition_vertices)
        unreachable_sources[condition] = frozenset(condition_sources - condition_vertices)
        unreachable_targets[condition] = frozenset(condition_targets - condition_vertices)
        retained_vertices.update(condition_vertices)

        for edge_index, (source, target) in enumerate(graph.E):
            edge_vertices = set(source) | set(target)
            if edge_vertices and edge_vertices <= condition_vertices:
                retained_edges.add(edge_index)

    vertex_indices = np.asarray(
        [index for index, vertex in enumerate(graph.V) if vertex in retained_vertices],
        dtype=int,
    )
    edge_indices = np.asarray(
        [index for index in range(graph.num_edges) if index in retained_edges],
        dtype=int,
    )
    ordered_vertices = [graph.V[index] for index in vertex_indices]
    ordered_edges = edge_indices.tolist()

    ordered_extract = getattr(graph, "_extract_subgraph_keep_order", None)
    if callable(ordered_extract):
        pruned_graph = ordered_extract(vertices=ordered_vertices, edges=ordered_edges)
    else:
        pruned_graph = graph.extract_subgraph(vertices=ordered_vertices, edges=ordered_edges)

    return PathPruning(
        graph=pruned_graph,
        original_vertex_indices=vertex_indices,
        original_edge_indices=edge_indices,
        vertices_by_condition=vertices_by_condition,
        unreachable_sources=unreachable_sources,
        unreachable_targets=unreachable_targets,
    )


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
