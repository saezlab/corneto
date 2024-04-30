"""This module enables conversion between Corneto and NetworkX graph representations.

It offers functions to transform Corneto graphs to NetworkX graphs and back,
and a class that interfaces with NetworkX to utilize its features directly on
Corneto graphs through transparent conversions.

The module employs lazy loading for NetworkX, optimizing load times and memory
when NetworkX is not immediately needed.

Functions:
    corneto_graph_to_networkx(G, skip_unsupported_edges=False):
        Converts a Corneto graph to a NetworkX graph, supporting directed and
        undirected edges, with an option to skip unsupported edge types.
    networkx_to_corneto_graph(G):
        Converts a NetworkX graph to a Corneto graph, maintaining edge data.

Classes:
    NetworkXWrapper:
        Wraps NetworkX, providing dynamic access to attributes and methods.
        Automatically converts Corneto graphs for NetworkX methods, allowing
        seamless integration.

Example Usage:
    # Convert Corneto graph to NetworkX
    G_corneto = cn.Graph()
    G_corneto.add_edge(1, 2)
    G_networkx = corneto_graph_to_networkx(G_corneto)

    # Use NetworkX function on a Corneto graph via the wrapper
    networkx.shortest_path(corneto_graph_to_networkx(G_corneto), source=1, target=2)

Note:
    Assumes 'corneto' library availability with specific structure. Designed for
    basic edge types; does not support hybrid or hypergraphs unless skipped.
"""

from typing import Union

import corneto as cn
from corneto._graph import BaseGraph, EdgeType
from corneto._types import NxDiGraph, NxGraph
from corneto.utils import Attr, import_optional_module


def corneto_graph_to_networkx(G: BaseGraph, skip_unsupported_edges: bool = False):
    """Converts a Corneto graph to a NetworkX graph.

    Args:
        G (BaseGraph): The Corneto graph to convert.
        skip_unsupported_edges (bool): If True, skip hyperedges and other unsupported edge types. Defaults to False.

    Returns:
        Union[NxGraph, NxDiGraph]: A NetworkX graph.

    Raises:
        ValueError: If the graph contains hyperedges or unsupported hybrid graph types.
    """
    nx = import_optional_module("networkx")
    dir_edges = []
    undir_edges = []
    for i, (s, t) in G.edges():
        if len(s) > 1 or len(t) > 1:
            if skip_unsupported_edges:
                continue
            else:
                raise ValueError("Hyperedges are not supported by NetworkX")
        attr = G.get_attr_edge(i)
        etype = attr.get_attr(Attr.EDGE_TYPE)
        if etype == EdgeType.DIRECTED:
            dir_edges.append(i)
        elif etype == EdgeType.UNDIRECTED:
            undir_edges.append(i)
    if len(dir_edges) == 0 and len(undir_edges) > 0:
        Gnx = nx.Graph()
    elif len(dir_edges) > 0 and len(undir_edges) == 0:
        Gnx = nx.DiGraph()
    else:
        raise ValueError("Hybrid graphs are not supported by NetworkX")
    for i, (s, t) in G.edges():
        attr = G.get_attr_edge(i)
        if len(s) == 0 or len(t) == 0:
            if skip_unsupported_edges:
                continue
            else:
                raise ValueError("Edges with no source or target are not supported")
        s = list(s)[0]
        t = list(t)[0]
        Gnx.add_edge(s, t, **attr)
    return Gnx


def networkx_to_corneto_graph(G: Union[NxGraph, NxDiGraph]):
    """Converts a NetworkX graph to a Corneto graph.

    Args:
        G (Union[NxGraph, NxDiGraph]): The NetworkX graph to convert.

    Returns:
        cn.Graph: A Corneto graph.
    """
    Gc = cn.Graph()
    for edge in G.edges():
        e_data = G.get_edge_data(edge[0], edge[1], default=dict())
        Gc.add_edge(edge[0], edge[1], **e_data)
    return Gc


class NetworkXWrapper:
    """Wrapper class to interface between NetworkX and Corneto graph types.

    This class dynamically imports NetworkX when needed and converts Corneto graph arguments
    to NetworkX graphs for function calls.
    """

    def __init__(self):
        """Initializes the NetworkX wrapper."""
        self.networkx = None

    def __getattr__(self, name):
        """Dynamically handles attribute accesses and method calls on the NetworkX module.

        Args:
            name (str): The attribute name to fetch from NetworkX.

        Returns:
            Any: The attribute or wrapped method from the NetworkX library.
        """
        if self.networkx is None:
            self.networkx = import_optional_module("networkx")
        original_attr = getattr(self.networkx, name)

        if callable(original_attr):

            def wrapped(*args, **kwargs):
                # Convert all corneto graph arguments to networkx graphs
                new_args = [
                    (
                        corneto_graph_to_networkx(arg)
                        if isinstance(arg, BaseGraph)
                        else arg
                    )
                    for arg in args
                ]
                # Call the original NetworkX function with the possibly converted arguments
                return original_attr(*new_args, **kwargs)

            return wrapped
        return original_attr


networkx = NetworkXWrapper()
