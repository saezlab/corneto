"""Deprecated compatibility imports for the CellNOpt ILP implementation."""

from corneto.methods.signaling.cellnopt_ilp import (
    cellnoptILP,
    cno_style,
    expand_graph_for_flows,
    plot_data,
    plot_fitness,
)

__all__ = [
    "cellnoptILP",
    "cno_style",
    "expand_graph_for_flows",
    "plot_data",
    "plot_fitness",
]
