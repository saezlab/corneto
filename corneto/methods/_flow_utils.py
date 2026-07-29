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


@dataclass(frozen=True)
class SelectedFlow:
    """Flow-selection expressions over all and biological edges."""

    all_edges: Any
    biological_edges: Any
    dag_layer: Any | None


def add_acyclic_flow_selection(
    backend: Backend,
    problem: ProblemDef,
    graph: BaseGraph,
    *,
    epsilon: float,
):
    """Add flow indicators and DAG constraints, returning edge selection."""
    problem += backend.NonZeroIndicator(problem.expr._flow, tolerance=epsilon)
    backend.Acyclic(
        graph,
        problem,
        indicator_negative_var_name="_flow_ineg",
        indicator_positive_var_name="_flow_ipos",
    )
    return problem.expr._flow_ipos + problem.expr._flow_ineg


def add_selected_flow(
    backend: Backend,
    problem: ProblemDef,
    graph: BaseGraph,
    *,
    biological_edge_indices: Iterable[int],
    epsilon: float,
    acyclic: bool,
) -> SelectedFlow:
    """Attach edge-selection indicators and optionally DAG constraints to a flow."""
    biological_edge_indices = list(biological_edge_indices)
    if acyclic:
        selected_all = add_acyclic_flow_selection(
            backend,
            problem,
            graph,
            epsilon=epsilon,
        )
        dag_layer = problem.expr._dag_layer
    else:
        flow = problem.expr._flow
        indicator_indexes = biological_edge_indices if len(flow.shape) == 1 else (biological_edge_indices, slice(None))
        problem += backend.Indicator(flow, indexes=indicator_indexes)
        selected_all = problem.expr._flow_i
        dag_layer = None
    biological = (
        selected_all[biological_edge_indices]
        if len(selected_all.shape) == 1
        else selected_all[biological_edge_indices, :]
    )
    return SelectedFlow(
        all_edges=selected_all,
        biological_edges=biological,
        dag_layer=dag_layer,
    )


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
