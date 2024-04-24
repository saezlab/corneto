import corneto as cn
from corneto._graph import BaseGraph
from corneto._types import NxGraph, NxDiGraph
from corneto.utils import import_optional_module, Attr
from corneto._graph import EdgeType
from typing import Union


def corneto_graph_to_networkx(G: BaseGraph, skip_unsupported_edges: bool = False):
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
    Gc = cn.Graph()
    for edge in G.edges():
        e_data = G.get_edge_data(edge[0], edge[1], default=dict())
        Gc.add_edge(edge[0], edge[1], **e_data)
    return Gc


class NetworkXWrapper:
    def __init__(self):
        self.networkx = None

    def __getattr__(self, name):
        if self.networkx is None:
            self.networkx = import_optional_module("networkx")
        # This method is called whenever an attribute is accessed
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
