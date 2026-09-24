"""Shared algorithms for ordering graph vertices."""

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any


def topological_sort(vertices: Iterable[Any], adjacency: Mapping[Any, Iterable[Any]]) -> list[Any]:
    """Return a stable topological order for an ordered vertex set.

    The input vertex order determines the initial order of ready vertices, and
    each adjacency iterable determines the order in which newly-ready
    successors are considered. Duplicate successors are retained and treated
    as parallel edges; callers that model simple adjacency should deduplicate
    their successor lists before calling this function.

    Raises:
        TypeError: If a vertex or successor identifier is not hashable.
        ValueError: If vertices are duplicated, adjacency references unknown
            vertices, or the graph contains a cycle.
    """
    ordered_vertices = tuple(vertices)
    try:
        vertex_set = set(ordered_vertices)
    except TypeError as error:
        raise TypeError("Topological-sort vertex identifiers must be hashable.") from error
    if len(vertex_set) != len(ordered_vertices):
        raise ValueError("Topological-sort vertices must be unique.")

    unknown_sources = set(adjacency).difference(vertex_set)
    if unknown_sources:
        raise ValueError(f"Adjacency contains unknown source vertices: {unknown_sources!r}.")

    indegree = dict.fromkeys(ordered_vertices, 0)
    successors = {}
    for vertex in ordered_vertices:
        try:
            adjacent = tuple(adjacency.get(vertex, ()))
        except TypeError as error:
            raise TypeError(f"Adjacency for vertex {vertex!r} must be iterable.") from error
        for successor in adjacent:
            try:
                is_known = successor in vertex_set
            except TypeError as error:
                raise TypeError(f"Successor identifier {successor!r} must be hashable.") from error
            if not is_known:
                raise ValueError(f"Adjacency for vertex {vertex!r} references unknown vertex {successor!r}.")
            indegree[successor] += 1
        successors[vertex] = adjacent

    ready = deque(vertex for vertex in ordered_vertices if indegree[vertex] == 0)
    order = []
    while ready:
        vertex = ready.popleft()
        order.append(vertex)
        for successor in successors[vertex]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)

    if len(order) != len(ordered_vertices):
        cyclic = tuple(vertex for vertex in ordered_vertices if indegree[vertex] > 0)
        raise ValueError(f"Graph contains a cycle involving {cyclic!r}, so topological sort is not possible.")
    return order
