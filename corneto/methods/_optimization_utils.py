"""Private optimization-expression helpers shared by method families."""

from corneto.backend._base import Backend, ProblemDef


def add_condition_union(
    backend: Backend,
    problem: ProblemDef,
    selected,
    *,
    name: str,
):
    """Return the linear OR over conditions, avoiding variables for one column."""
    if len(selected.shape) == 1:
        union = selected
    elif selected.shape[1] == 1:
        union = selected[:, 0]
    else:
        internal_name = f"_{name}"
        problem += backend.linear_or(selected, axis=1, varname=internal_name, ignore_type=True)
        union = problem.expr[internal_name]
    problem.register(name, union)
    return union
