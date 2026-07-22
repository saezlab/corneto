"""Private helpers shared by flow-based network methods."""

from corneto.backend._base import Backend, ProblemDef
from corneto.graph import BaseGraph


def add_acyclic_flow_selection(
    backend: Backend,
    problem: ProblemDef,
    graph: BaseGraph,
    *,
    epsilon: float,
):
    """Add flow indicators and DAG constraints, returning edge selection."""
    problem += backend.NonZeroIndicator(problem.expr._flow, tolerance=epsilon)
    problem += backend.Acyclic(
        graph,
        problem,
        indicator_negative_var_name="_flow_ineg",
        indicator_positive_var_name="_flow_ipos",
    )
    return problem.expr._flow_ipos + problem.expr._flow_ineg
