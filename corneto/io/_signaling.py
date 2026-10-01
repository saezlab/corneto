"""Compatibility imports for SIF helpers now implemented in :mod:`._sif`."""

from ._sif import _read_sif as _read_sif
from ._sif import _read_sif_iter as _read_sif_iter
from ._sif import load_graph_from_sif, load_graph_from_sif_tuples

__all__ = ["load_graph_from_sif", "load_graph_from_sif_tuples"]
