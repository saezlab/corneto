from collections import OrderedDict
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from corneto._types import Edge

from ._base import Attr, Attributes, BaseGraph, EdgeType


class Graph(BaseGraph):
    """Concrete graph implementation supporting directed/undirected edges and hyperedges.

    Allows parallel edges (multiple edges between same vertices) and hyperedges
    (edges connecting multiple vertices). Edges and vertices can have attributes.

    Examples:
        >>> graph = corneto.Graph() 
        >>> graph.add_edge(1, 2)
        >>> graph.plot()
    """

    def __init__(
        self, default_edge_type: EdgeType = EdgeType.DIRECTED, **kwargs
    ) -> None:
        """Initialize Graph.

        Args:
            default_edge_type: Default type for new edges
            **kwargs: Additional graph attributes
        """
        super().__init__(default_edge_type=default_edge_type)
        # Allow edges with same s/t vertices. Edges are represented as a tuple
        # (S, T) where S is the set of nodes as the source/head of the edge, and T the
        # set of target nodes at the tail of the edge. There can be many edges with
        # the same set of source/target node(s).
        self._edges: List[Edge] = []
        # Edge properties, including vertex-edge properties
        self._edge_attr: List[Attributes] = []
        # Vertices (in order of addition). Since vertices are unique
        # they are indexed. The vertex has to be any indexable object.
        # Vertex -> Indexes of edges where they appear
        self._vertices: Dict[Any, Set[int]] = OrderedDict()
        # Vertex properties
        self._vertex_attr: Dict[Any, Attributes] = dict()
        # Global graph attributes
        self._graph_attr: Attributes = Attributes()
        # Add custom graph params
        self._graph_attr.update(kwargs)

    def _add_edge(
        self,
        source: Iterable,
        target: Iterable,
        type: EdgeType,
        edge_source_attr: Optional[Attributes] = None,
        edge_target_attr: Optional[Attributes] = None,
        **kwargs,
    ) -> int:
        """Add edge to graph.

        Args:
            source: Source vertices
            target: Target vertices 
            type: Edge type
            edge_source_attr: Optional attributes for source vertices
            edge_target_attr: Optional attributes for target vertices
            **kwargs: Additional edge attributes

        Returns:
            Index of new edge
        """
        sv = frozenset(source)
        tv = frozenset(target)
        # uv = sv | tv
        edge = (sv, tv)
        self._edges.append(edge)
        idx = len(self._edges) - 1
        # Properties related to the edge
        edge_attr = Attributes()
        edge_attr.set_attr(Attr.EDGE_TYPE, type)
        if len(kwargs) > 0:
            edge_attr.update(kwargs)
        if edge_source_attr is not None:
            edge_attr.set_attr(Attr.SOURCE_ATTR, edge_source_attr)
        if edge_target_attr is not None:
            edge_attr.set_attr(Attr.TARGET_ATTR, edge_target_attr)
        self._edge_attr.append(edge_attr)

        seen = set()
        for v in list(source) + list(target):
            if v in seen:
                continue
            if v in self._vertices:
                self._vertices[v].add(idx)
            else:
                self._vertices[v] = {idx}
            seen.add(v)
        return idx

    def _add_vertex(self, vertex: Any, **kwargs) -> int:
        """Add vertex to the graph.

        Args:
            vertex: Vertex to add
            **kwargs: Additional vertex attributes

        Returns:
            Index of the new vertex
        """
        if vertex not in self._vertices:
            self._vertices[vertex] = set()
            self._vertex_attr[vertex] = Attributes(kwargs)
            idx = len(self._vertices) - 1
        else:
            idx = list(self._vertices).index(vertex)
            if vertex in self._vertex_attr:
                va = self._vertex_attr[vertex]
            else:
                va = Attributes()
                self._vertex_attr[vertex] = va
            va.update(kwargs)
        return idx

    def get_edge(self, index: int) -> Edge:
        """Get edge at specified index.

        Args:
            index: Edge index to retrieve

        Returns:
            Edge tuple (source vertices, target vertices)
        """
        return self._edges[index]

    def _get_vertices(self) -> Iterable:
        """Get iterator over all vertices in graph.

        Returns:
            Iterator yielding graph vertices
        """
        return iter(self._vertices.keys())

    def _get_incident_edges(self, vertex) -> Iterable[int]:
        """Get edges incident to vertex.

        Args:
            vertex: Vertex to find incident edges for

        Returns:
            Iterator yielding indices of incident edges
        """
        return self._vertices[vertex]

    def _get_edge_attributes(self, index: int) -> Attributes:
        """Get attributes of edge at index.

        Args:
            index: Edge index to get attributes for

        Returns:
            Attributes object containing edge properties
        """
        return self._edge_attr[index]

    def _get_vertex_attributes(self, v) -> Attributes:
        """Get attributes of vertex.

        Args:
            v: Vertex to get attributes for

        Returns:
            Attributes object containing vertex properties
        """
        if v in self._vertex_attr:
            return self._vertex_attr[v]
        return Attributes()

    def get_graph_attributes(self) -> Attributes:
        """Get global graph attributes.

        Returns:
            Attributes object containing graph-level properties
        """
        return self._graph_attr

    def _num_vertices(self) -> int:
        """Get number of vertices in graph.

        Returns:
            Total vertex count
        """
        return len(self._vertices)

    def _num_edges(self) -> int:
        """Get number of edges in graph.

        Returns:
            Total edge count
        """
        return len(self._edges)

    def extract_subgraph(
        self, vertices: Optional[Iterable] = None, edges: Optional[Iterable[int]] = None
    ):
        """Extract subgraph induced by vertices and/or edges.

        Args:
            vertices: Optional vertices to include
            edges: Optional edge indices to include

        Returns:
            New Graph containing only specified vertices/edges and their incident edges
        """
        # Graph induced by the set of vertices + selected edges
        n_v = 0
        g = Graph()
        g._graph_attr = deepcopy(self._graph_attr)
        if vertices is not None:
            vertices = set(vertices)
            n_v = len(vertices)
        else:
            vertices = set()
        if edges is not None:
            edges = set(edges)
            v_edges = set()
            for i in edges:
                s, t = self.get_edge(i)
                v_edges |= set().union(*[s, t])
            vertices |= v_edges
        else:
            edges = set()
        # Get all incident edges to the selected vertices (if provided)
        incident_edges = set()
        if n_v > 0:
            for v in vertices:
                incident_edges |= set(self.get_incident_edges([v]))
        # Get edges induced by the set of vertices
        # In the case of hyperedges, if an edge e.g. {A, B} - {C, D}
        # all nodes A, B, C, D have to be present in the list to include the edge
        for i in incident_edges:
            # Flatten and merge vertices
            s, t = self.get_edge(i)
            v_edges = set().union(*[s, t])
            if v_edges.intersection(vertices) == v_edges:
                # Select edge
                edges.add(i)
        # Copy by vertices
        for v in vertices:
            g.add_vertex(v)
            if v in self._vertex_attr:
                v_attr = self.get_attr_vertex(v)
                if len(v_attr) > 0:
                    g._vertex_attr[v] = deepcopy(v_attr)
        # Copy edges
        if edges is None:
            edges = set(range(self.ne))
        for i in edges:
            s, t = self.get_edge(i)
            attr = deepcopy(self.get_attr_edge(i))
            g.add_edge(s, t, **attr)
        return g

    def _extract_subgraph_keep_order(
        self, vertices: Optional[Iterable] = None, edges: Optional[Iterable[int]] = None
    ):
        """Extract subgraph while preserving vertex and edge order.

        Internal method that maintains insertion order of vertices and edges.

        Args:
            vertices: Optional vertices to include
            edges: Optional edge indices to include

        Returns:
            New Graph with preserved ordering
        """
        # Create a new graph
        g = Graph()
        g._graph_attr = deepcopy(self._graph_attr)

        def append_unique(lst, items):
            for item in items:
                if item not in lst:
                    lst.append(item)

        # Initialize lists to preserve order
        if vertices is not None:
            vertices = list(vertices)
        else:
            vertices = []
        if edges is not None:
            edges = list(edges)
            # Collect vertices from edges
            for i in edges:
                s, t = self.get_edge(i)
                v_set = s.union(t)
                append_unique(vertices, v_set)
        else:
            edges = []
            # If edges are not specified but vertices are, include edges induced by the vertices
            if vertices:
                # Collect edges induced by the set of vertices
                for i in range(self.ne):
                    s, t = self.get_edge(i)
                    v_set = s.union(t)
                    if all(v in vertices for v in v_set):
                        edges.append(i)

        # Copy vertices
        for v in vertices:
            g.add_vertex(v)
            if v in self._vertex_attr:
                v_attr = self.get_attr_vertex(v)
                if len(v_attr) > 0:
                    g._vertex_attr[v] = deepcopy(v_attr)

        # Copy edges
        for i in edges:
            s, t = self.get_edge(i)
            attr = deepcopy(self.get_attr_edge(i))
            g.add_edge(s, t, **attr)

        return g

    def reverse(self) -> "Graph":
        """Create new graph with all edge directions reversed.

        Returns:
            New Graph with reversed edges
        """
        G = self.copy()
        rev_edges = [(t, s) for s, t in G.E]
        G._edges = rev_edges
        # TODO: Simplify handling edge attributes
        for attr in G._edge_attr:
            s = attr.get_attr(Attr.SOURCE_ATTR, Attributes())
            t = attr.get_attr(Attr.TARGET_ATTR, Attributes())
            attr.set_attr(Attr.SOURCE_ATTR, t)
            attr.set_attr(Attr.TARGET_ATTR, s)
        return G

    def filter_graph(
        self,
        filter_vertex: Optional[Callable[[Any], bool]] = None,
        filter_edge: Optional[Callable[[int], bool]] = None,
    ):
        """Create filtered graph based on vertex and edge predicates.

        Args:
            filter_vertex: Optional function that returns True for vertices to keep
            filter_edge: Optional function that returns True for edges to keep

        Returns:
            New Graph containing only elements that pass filters
        """
        g = Graph()
        g._graph_attr = deepcopy(self._graph_attr)
        if filter_vertex is not None:
            vertices = set(filter(filter_vertex, self._get_vertices()))
        else:
            vertices = set(self._get_vertices())
        for v in vertices:
            g.add_vertex(v)
            if v in self._vertex_attr:
                v_attr = self.get_attr_vertex(v)
                if len(v_attr) > 0:
                    g._vertex_attr[v] = deepcopy(v_attr)
        if filter_edge is not None:
            edges = (i for i, _ in self.edges() if filter_edge(i))
        else:
            edges = (i for i in range(self.ne))
            for i in edges:
                s, t = self.get_edge(i)
                attr = deepcopy(self.get_attr_edge(i))
                g.add_edge(s, t, **attr)

    def _subgraph(self, vertices: Iterable):
        """Internal method to create vertex-induced subgraph.

        Args:
            vertices: Vertices to include

        Returns:
            New Graph containing vertices and their incident edges
        """
        g = Graph()
        g._graph_attr = deepcopy(self._graph_attr)
        if isinstance(vertices, str):
            vertices = [vertices]
        vertices = set(vertices)
        for v in vertices:
            g.add_vertex(v)
            if v in self._vertex_attr:
                v_attr = self.get_attr_vertex(v)
                if len(v_attr) > 0:
                    g._vertex_attr[v] = deepcopy(v_attr)
        # Get all common edges between vertices
        edges = set()
        for v in vertices:
            edges |= set(self.get_incident_edges([v]))

        for i in edges:
            s, t = self.get_edge(i)
            # Merge if many sets
            if len(s) > 1:
                s = set().union(*s)
            if len(t) > 1:
                t = set().union(*t)
            if len(s.intersection(vertices)) > 0 and len(t.intersection(vertices)) > 0:
                # Copy the edge
                attr = deepcopy(self.get_attr_edge(i))
                g.add_edge(s, t, **attr)
        return g

    def _edge_subgraph(self, edges: Iterable[int]):
        """Internal method to create edge-induced subgraph.

        Args:
            edges: Edge indices to include

        Returns:
            New Graph containing edges and their incident vertices
        """
        g = Graph()
        g._graph_attr = deepcopy(self._graph_attr)
        vertices = set()
        for s, t in self.get_edges(edges):
            for v in s:
                vertices.add(v)
            for v in t:
                vertices.add(v)
        for v in vertices:
            g.add_vertex(v)
            if v in self._vertex_attr:
                v_attr = self.get_attr_vertex(v)
                if len(v_attr) > 0:
                    g._vertex_attr[v] = deepcopy(v_attr)
        for i in edges:
            (s, t) = self.get_edge(i)
            attr = deepcopy(self.get_attr_edge(i))
            g.add_edge(s, t, **attr)
        return g

    def copy(self) -> "Graph":
        """Create deep copy of graph.

        Returns:
            New Graph with copied structure and attributes
        """
        return deepcopy(self)

    @staticmethod
    def from_tuples(tuples: Iterable[Tuple]):
        """Alias for corneto.io.load_graph_from_sif_tuples for backwards compatibility.

        Args:
            tuples: Iterable of (source, interaction, target) tuples

        Returns:
            New Graph created from tuple data
        """
        from corneto.io import load_graph_from_sif_tuples
        return load_graph_from_sif_tuples(tuples)
