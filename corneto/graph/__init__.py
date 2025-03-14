r"""Graph (:mod:`corneto.graph`)
==================================

.. currentmodule:: corneto.graph

Graph module containing base graph and concrete graph implementations.

This module provides the core graph data structures used in corneto. It includes:

- **BaseGraph**: Abstract base class defining the graph interface
- **Graph**: Concrete implementation supporting directed/undirected/hypergraphs
- **EdgeType**: Enum for edge types

The graph implementations support:

- Directed and undirected edges
- Hyperedges (edges connecting multiple vertices)
- Edge and vertex attributes
- Graph attributes
- Common graph operations like BFS traversal and subgraph extraction

.. autosummary::
   :toctree: generated/

   BaseGraph
   Graph
   EdgeType
   
"""

from ._base import BaseGraph, EdgeType
from ._graph import Graph

__all__ = ["BaseGraph", "EdgeType", "Graph"]
