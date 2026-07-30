"""Private vectorized building blocks shared by network-flow methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from corneto._constants import VarType
from corneto.backend._base import Backend, ProblemDef
from corneto.graph import BaseGraph
from corneto.methods._network_utils import directed_incidence


@dataclass(frozen=True)
class VertexSelection:
    """Expressions created by :func:`add_vertex_selection`."""

    selected: Any
    outgoing: Any
    incoming: Any


def add_vertex_selection(
    backend: Backend,
    problem: ProblemDef,
    graph: BaseGraph,
    edge_selected,
    *,
    edge_indices: Iterable[int],
    force_selected: Any,
    require_incoming: np.ndarray,
    require_outgoing: np.ndarray,
    name: str = "vertex_selected",
    reverse: bool = False,
) -> VertexSelection:
    """Link explicit vertex selection to selected directed biological edges."""
    incidence = directed_incidence(graph, edge_indices)
    num_conditions = force_selected.shape[1]
    selected = backend.Variable(
        name,
        (graph.num_vertices, num_conditions),
        vartype=VarType.BINARY,
    )
    outgoing_incidence = incidence.incoming if reverse else incidence.outgoing
    incoming_incidence = incidence.outgoing if reverse else incidence.incoming
    outgoing = backend.Constant(outgoing_incidence) @ edge_selected
    incoming = backend.Constant(incoming_incidence) @ edge_selected
    out_degree = np.asarray(outgoing_incidence.sum(axis=1)).reshape(-1, 1)
    in_degree = np.asarray(incoming_incidence.sum(axis=1)).reshape(-1, 1)

    problem += outgoing <= selected.multiply(np.broadcast_to(out_degree, selected.shape))
    problem += incoming <= selected.multiply(np.broadcast_to(in_degree, selected.shape))
    forced = force_selected.astype(float) if isinstance(force_selected, np.ndarray) else force_selected
    problem += selected >= forced
    problem += selected.multiply(require_outgoing.astype(float)) <= outgoing
    problem += selected.multiply(require_incoming.astype(float)) <= incoming
    return VertexSelection(selected=selected, outgoing=outgoing, incoming=incoming)
