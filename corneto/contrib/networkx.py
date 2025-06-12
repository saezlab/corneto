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

from typing import Callable, Optional, Union

import corneto as cn
from corneto._types import NxDiGraph, NxGraph
from corneto.graph import BaseGraph
from corneto.utils import Attr, import_optional_module


def corneto_graph_to_networkx(
    G: BaseGraph,
    skip_unsupported_edges: bool = False,
    graph_class: Optional[Callable] = None,
    copy_attributes: bool = True,
):
    nx = import_optional_module("networkx")
    if graph_class is not None:
        Gx = graph_class()
    else:
        gtype = None
        for attr in G.get_attr_edges():
            etype = attr.get_attr(Attr.EDGE_TYPE)
            if gtype is None:
                gtype = etype
            else:
                if etype != gtype:
                    raise ValueError(
                        "Hybrid graphs are not supported by NetworkX. "
                        "Please provide `graph_class` as either nx.Graph or nx.DiGraph."
                    )
        if gtype == cn.EdgeType.DIRECTED:
            Gx = nx.DiGraph()
        else:
            Gx = nx.Graph()
    for i, (source, target) in G.edges():
        source_vertex, target_vertex = None, None
        if source:
            source = list(source)
            source_vertex = source[0]
            if len(source) > 1 and not skip_unsupported_edges:
                raise ValueError(f"Edge {i} has {len(source)} source vertices")
        if target:
            target = list(target)
            target_vertex = target[0]
            if len(target) > 1 and not skip_unsupported_edges:
                raise ValueError(f"Edge {i} has {len(source)} source vertices")
        props = dict()
        if copy_attributes:
            props = G.get_attr_edge(i)
        if source_vertex is not None and target_vertex is not None:
            Gx.add_edge(source_vertex, target_vertex, **props)
    return Gx


def networkx_to_corneto_graph(G: Union[NxGraph, NxDiGraph]):
    """Converts a NetworkX graph to a Corneto graph.

    Args:
        G (Union[NxGraph, NxDiGraph]): The NetworkX graph to convert.

    Returns:
        cn.Graph: A Corneto graph.
    """
    from corneto.graph import Graph

    Gc = Graph()
    edge_type = cn.EdgeType.DIRECTED if G.is_directed() else cn.EdgeType.UNDIRECTED
    for edge in G.edges():
        e_data = G.get_edge_data(edge[0], edge[1], default=dict())
        e_data[Attr.EDGE_TYPE] = edge_type
        Gc.add_edge(edge[0], edge[1], **e_data)
    # Also add node attributes
    for node, data in G.nodes(data=True):
        Gc.add_vertex(node, **data)
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
                new_args = [(corneto_graph_to_networkx(arg) if isinstance(arg, BaseGraph) else arg) for arg in args]
                # Call the original NetworkX function with the possibly converted arguments
                return original_attr(*new_args, **kwargs)

            return wrapped
        return original_attr


networkx = NetworkXWrapper()
