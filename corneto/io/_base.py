from typing import List, Union

import numpy as np

from corneto.graph import Graph


def graph_from_vertex_incidence(
    A: np.ndarray,
    vertex_ids: Union[List[str], np.ndarray],
    edge_ids: Union[List[str], np.ndarray],
):
    """Create graph from vertex incidence matrix and labels.

    Args:
        A: Vertex incidence matrix. Rows are vertices, columns are edges.
            Non-zero entries indicate edge-vertex connections.
        vertex_ids: Labels for vertices corresponding to matrix rows
        edge_ids: Labels for edges corresponding to matrix columns

    Returns:
        Graph instance constructed from incidence matrix

    Raises:
        ValueError: If dimensions of inputs don't match
    """
    g = Graph()
    if len(vertex_ids) != A.shape[0]:
        raise ValueError(
            """The number of rows in A matrix is different from
            the number of vertex ids"""
        )
    if len(edge_ids) != A.shape[1]:
        raise ValueError(
            """The number of columns in A matrix is different from
            the number of edge ids"""
        )
    for v in vertex_ids:
        g.add_vertex(v)
    for j, v in enumerate(edge_ids):
        values = A[:, j]
        idx = np.flatnonzero(values)
        coeffs = values[idx]
        v_names = [vertex_ids[i] for i in idx]
        s = {n: val for n, val in zip(v_names, coeffs) if val < 0}
        t = {n: val for n, val in zip(v_names, coeffs) if val > 0}
        g.add_edge(s, t, id=v)
    return g
