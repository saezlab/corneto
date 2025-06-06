import abc
import pickle
from collections import OrderedDict, deque
from copy import deepcopy
from enum import Enum
from itertools import chain
from numbers import Number
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from corneto._io import import_cobra_model
from corneto._types import CobraModel, Edge, NxDiGraph, NxGraph
from corneto._util import obj_content_hash, unique_iter
from corneto.utils import Attr, Attributes

T = TypeVar("T")


class EdgeType(str, Enum):
    """Edge type enumeration.

    Attributes:
        DIRECTED: Represents a directed edge
        UNDIRECTED: Represents an undirected edge
    """

    DIRECTED = "directed"
    UNDIRECTED = "undirected"


def _wrap(
    elements: Any,
    container: Callable[[Iterable[Any]], T],
    skip: Optional[Type[T]] = None,
) -> T:
    """Helper function to wrap elements in a container type.

    Args:
        elements: Element or iterable of elements to wrap
        container: Container type constructor (e.g. frozenset, tuple)
        skip: Optional type to skip wrapping if elements is already that type

    Returns:
        Container instance containing the elements
    """
    if skip is not None and isinstance(elements, skip):
        return elements
    if isinstance(elements, str):
        return container((elements,))
    if isinstance(elements, Iterable):
        return container(elements)
    return container((elements,))


def _fset(elements: Union[Any, Iterable[Any]]) -> FrozenSet[Any]:
    """Convert elements to a frozenset.

    Args:
        elements: Element or iterable of elements

    Returns:
        FrozenSet containing the elements
    """
    return _wrap(elements, frozenset, frozenset)


def _tpl(elements: Union[Any, Iterable[Any]]) -> Tuple[Any, ...]:
    """Convert elements to a tuple.

    Args:
        elements: Element or iterable of elements

    Returns:
        Tuple containing the elements
    """
    return _wrap(elements, tuple, tuple)


class BaseGraph(abc.ABC):
    """Abstract base class for graphs and hypergraphs.

    Defines the interface and common functionality for graph implementations.
    Supports directed/undirected/mixed edges and self-edges.
    """

    def __init__(self, default_edge_type: EdgeType = EdgeType.DIRECTED) -> None:
        """Initialize BaseGraph.

        Args:
            default_edge_type: Default type for edges when not specified
        """
        super().__init__()
        self._default_edge_type = default_edge_type

    @staticmethod
    def _parse_vertices(s):
        """Parse vertices from different input formats.

        Args:
            s: Input that could be dict, str, number or iterable

        Returns:
            Dict mapping vertices to their attributes
        """
        if isinstance(s, dict):
            return {k: v if isinstance(v, (dict, Number)) else ValueError() for k, v in s.items()}
        elif isinstance(s, (str, Number, Iterable)):
            return {v: {} for v in (s if isinstance(s, Iterable) else [s])}
        else:
            raise ValueError()

    @staticmethod
    def _extract_ve_attr(s):
        """Extract vertex attributes from input.

        Args:
            s: Input that could be dict, str, number or iterable

        Returns:
            Dict mapping vertices to their attributes
        """
        if isinstance(s, dict):
            return {k: v if isinstance(v, (dict, Number)) else ValueError() for k, v in s.items()}
        elif isinstance(s, (str, Number, Iterable)):
            return {v: {} for v in (s if not isinstance(s, str) and isinstance(s, Iterable) else [s])}
        else:
            raise ValueError()

    @abc.abstractmethod
    def _add_edge(
        self,
        source: Iterable,
        target: Iterable,
        type: EdgeType,
        edge_source_attr: Optional[Attributes] = None,
        edge_target_attr: Optional[Attributes] = None,
        **kwargs,
    ) -> int:
        """Add edge to the graph.

        Args:
            source: Source vertices
            target: Target vertices
            type: Edge type
            edge_source_attr: Optional attributes for source vertices
            edge_target_attr: Optional attributes for target vertices
            **kwargs: Additional edge attributes

        Returns:
            Index of the new edge
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _add_vertex(self, vertex: Any, **kwargs) -> int:
        """Add vertex to the graph.

        Args:
            vertex: Vertex to add
            **kwargs: Additional vertex attributes

        Returns:
            Index of the new vertex
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_edge(self, index: int) -> Edge:
        """Get edge by index.

        Args:
            index: Index of the edge

        Returns:
            Edge at the specified index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_vertices(self) -> Iterable:
        """Get all vertices in the graph.

        Returns:
            Iterable of vertices
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_incident_edges(self, vertex) -> Iterable[int]:
        """Get incident edges for a vertex.

        Args:
            vertex: Vertex to get incident edges for

        Returns:
            Iterable of incident edge indices
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_edge_attributes(self, index: int) -> Attributes:
        """Get attributes for an edge.

        Args:
            index: Index of the edge

        Returns:
            Attributes of the edge
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_vertex_attributes(self, v) -> Attributes:
        """Get attributes for a vertex.

        Args:
            v: Vertex to get attributes for

        Returns:
            Attributes of the vertex
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_graph_attributes(self) -> Attributes:
        """Get global graph attributes.

        Returns:
            Attributes of the graph
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _num_vertices(self) -> int:
        """Get number of vertices in the graph.

        Returns:
            Number of vertices
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _num_edges(self) -> int:
        """Get number of edges in the graph.

        Returns:
            Number of edges
        """
        raise NotImplementedError()

    def edge_subgraph(self, edges: Union[Iterable[int], np.ndarray]):
        """Create subgraph induced by a set of edges.

        Args:
            edges: Indices of edges to include in the subgraph

        Returns:
            Subgraph induced by the specified edges
        """
        if isinstance(edges, np.ndarray):
            # Check if its a logical array:
            if edges.dtype == bool:
                edges = np.where(edges)[0]
        return self.extract_subgraph(vertices=None, edges=edges)

    def subgraph(self, vertices: Iterable):
        """Create subgraph induced by a set of vertices.

        Args:
            vertices: Vertices to include in the subgraph

        Returns:
            Subgraph induced by the specified vertices
        """
        return self.extract_subgraph(vertices=vertices, edges=None)

    @abc.abstractmethod
    def extract_subgraph(self, vertices: Optional[Iterable] = None, edges: Optional[Iterable[int]] = None):
        """Extract subgraph induced by a set of vertices and/or edges.

        Args:
            vertices: Optional vertices to include in the subgraph
            edges: Optional edges to include in the subgraph

        Returns:
            Subgraph induced by the specified vertices and/or edges
        """
        raise NotImplementedError()

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the graph.

        Returns:
            Number of vertices
        """
        return self._num_vertices()

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph.

        Returns:
            Number of edges
        """
        return self._num_edges()

    def hash(self) -> str:
        """Compute hash of the graph based on its content.

        Returns:
            Hash string representing the graph content
        """
        return obj_content_hash(self)

    def get_attr_edge(self, index: int) -> Attributes:
        """Get attributes of an edge.

        Args:
            index: Index of the edge

        Returns:
            Attributes of the edge
        """
        return self._get_edge_attributes(index)

    def get_attr_vertex(self, v) -> Attributes:
        """Get attributes of a vertex.

        Args:
            v: Vertex to get attributes for

        Returns:
            Attributes of the vertex
        """
        return self._get_vertex_attributes(v)

    def get_attr_edges(self, indexes: Optional[Iterable[int]] = None) -> List[Attributes]:
        """Get attributes of multiple edges.

        Args:
            indexes: Optional indices of edges to get attributes for

        Returns:
            List of attributes for the specified edges
        """
        if indexes is None:
            indexes = range(self.num_edges)
        return [self._get_edge_attributes(i) for i in indexes]

    def get_attr_from_edges(self, attr: str, default: Any = None) -> List[Any]:
        """Get specific attribute from all edges.

        Args:
            attr: Attribute name to get
            default: Default value if attribute is not present

        Returns:
            List of attribute values for all edges
        """
        attrs = self.get_attr_edges()
        return [a.get(attr, default) for a in attrs]

    def get_edges_by_attr(self, key: str, value: Any) -> Iterable[int]:
        """Get edges by specific attribute value.

        Args:
            key: Attribute name to filter by
            value: Attribute value to filter by

        Returns:
            Iterable of edge indices with the specified attribute value
        """
        for i, e in enumerate(self.get_attr_edges()):
            if key in e and e[key] == value:
                yield i

    def get_attr_vertices(self, vertices: Optional[Iterable] = None) -> List[Attributes]:
        """Get attributes of multiple vertices.

        Args:
            vertices: Optional vertices to get attributes for

        Returns:
            List of attributes for the specified vertices
        """
        if vertices is None:
            vertices = self.V
        return [self._get_vertex_attributes(v) for v in vertices]

    def get_edges(self, indexes: Iterable[int]) -> Iterable[Edge]:
        """Get multiple edges by their indices.

        Args:
            indexes: Indices of edges to get

        Returns:
            Iterable of edges at the specified indices
        """
        return (self.get_edge(i) for i in indexes)

    def get_vertex(self, index: int) -> Any:
        """Get vertex by its index.

        Args:
            index: Index of the vertex

        Returns:
            Vertex at the specified index

        Raises:
            IndexError: If index is out of range
        """
        for i, v in enumerate(self._get_vertices()):
            if i == index:
                return v
        raise IndexError(f"Vertex index {index} out of range [0 - {self.num_vertices - 1}]")

    def get_incident_edges(self, vertices) -> Iterable[int]:
        """Get incident edges for multiple vertices.

        Args:
            vertices: Vertices to get incident edges for

        Returns:
            Iterable of incident edge indices
        """
        combined = chain.from_iterable(self._get_incident_edges(v) for v in vertices)
        seen: Set[int] = set()
        for e in combined:
            if e not in seen:
                seen.add(e)
                yield e

    def get_common_incident_edges(self, vertices) -> Iterable[int]:
        """Get common incident edges for multiple vertices.

        Args:
            vertices: Vertices to get common incident edges for

        Returns:
            Iterable of common incident edge indices
        """
        vset = set(vertices)
        seen: Set[int] = set()
        for e in self.get_incident_edges(vertices):
            if e not in seen:
                (s, t) = self.get_edge(e)
                if (s | t).intersection(vset) == vset:
                    seen.add(e)
                    yield e

    def edges(self, vertices=None) -> Iterable[Tuple[int, Edge]]:
        """Get edges in the graph.

        Args:
            vertices: Optional vertices to get edges for

        Returns:
            Iterable of edge indices and edges
        """
        if vertices is not None:
            vertices = _tpl(vertices)
            eidx = self.get_incident_edges(vertices)
            return ((i, self.get_edge(i)) for i in eidx)
        else:
            return ((i, self.get_edge(i)) for i in range(self.num_edges))

    def copy(self) -> "BaseGraph":
        """Create a deep copy of the graph.

        Returns:
            Deep copy of the graph
        """
        return deepcopy(self)

    @abc.abstractmethod
    def reverse(self) -> "BaseGraph":
        """Reverse the direction of all edges in the graph.

        Returns:
            Graph with reversed edges
        """
        return NotImplementedError()

    def _edges_by_dir(self, vertices, direction: Optional[str] = None) -> Iterable[Tuple[int, Edge]]:
        """Get edges by direction relative to vertices.

        Args:
            vertices: Vertices to get edges for
            direction: Direction of edges - 'in' for incoming, 'out' for outgoing,
                      None for both directions

        Returns:
            Iterator yielding (edge index, Edge tuple) pairs
        """
        vertices = _tpl(vertices)
        for idx, (s, t) in self.edges(vertices=vertices):
            attr = self.get_attr_edge(idx)
            etype = attr.get_attr(Attr.EDGE_TYPE)
            if etype == EdgeType.DIRECTED:
                if direction == "in":
                    check = t
                elif direction == "out":
                    check = s
                else:
                    check = s | t
                if len(check.intersection(vertices)) > 0:
                    yield idx, (s, t)
            else:
                yield idx, (s, t)

    def in_edges(self, vertices) -> Iterable[Tuple[int, Edge]]:
        """Get incoming edges for vertices.

        Args:
            vertices: Vertices to get incoming edges for

        Returns:
            Iterator yielding (edge index, Edge tuple) pairs for incoming edges
        """
        yield from self._edges_by_dir(vertices, "in")

    def out_edges(self, vertices) -> Iterable[Tuple[int, Edge]]:
        """Get outgoing edges for vertices.

        Args:
            vertices: Vertices to get outgoing edges for

        Returns:
            Iterator yielding (edge index, Edge tuple) pairs for outgoing edges
        """
        yield from self._edges_by_dir(vertices, "out")

    def _successors(self, vertex) -> Iterable:
        """Get successor vertices for a vertex.

        Args:
            vertex: Vertex to get successors for

        Returns:
            Iterable of successor vertices
        """

        def succ():
            for i, (s, t) in self._edges_by_dir(vertex, direction="out"):
                attr = self.get_attr_edge(i)
                etype = attr.get_attr(Attr.EDGE_TYPE)
                if etype == EdgeType.DIRECTED:
                    if vertex == s or vertex in s:
                        yield t
                else:
                    if vertex == s or vertex in s:
                        yield t
                    elif vertex == t or vertex in t:
                        yield s

        return unique_iter(chain.from_iterable(succ()))

    def successors(self, vertices) -> Iterable:
        """Get successor vertices for multiple vertices.

        Args:
            vertices: Vertices to get successors for

        Returns:
            Iterable of successor vertices
        """
        vertices = _tpl(vertices)
        return unique_iter(chain.from_iterable((self._successors(v) for v in vertices)))

    def _predecessors(self, vertex) -> Iterable:
        """Get predecessor vertices for a vertex.

        Args:
            vertex: Vertex to get predecessors for

        Returns:
            Iterable of predecessor vertices
        """

        def succ():
            for i, (s, t) in self._edges_by_dir(vertex, direction="in"):
                attr = self.get_attr_edge(i)
                etype = attr.get_attr(Attr.EDGE_TYPE)
                if etype == EdgeType.DIRECTED:
                    if vertex == t or vertex in t:
                        yield s
                else:
                    if vertex == s or vertex in s:
                        yield t
                    elif vertex == t or vertex in t:
                        yield s

        return unique_iter(chain.from_iterable(succ()))

    def predecessors(self, vertices) -> Iterable:
        """Get predecessor vertices for multiple vertices.

        Args:
            vertices: Vertices to get predecessors for

        Returns:
            Iterable of predecessor vertices
        """
        vertices = _tpl(vertices)
        return unique_iter(chain.from_iterable((self._predecessors(v) for v in vertices)))

    def neighbors(self, vertex) -> Iterable:
        """Get neighbors of a vertex (ignoring edge direction).

        Args:
            vertex: Vertex to get neighbors for

        Returns:
            Iterable of neighbor vertices
        """
        iter_vertices = (s | t for _, (s, t) in self._edges_by_dir(vertex, direction=None))
        return unique_iter(chain.from_iterable(iter_vertices))

    def is_hypergraph(self) -> bool:
        """Check if the graph is a hypergraph.

        Returns:
            True if the graph is a hypergraph, False otherwise
        """
        return any(len(s) > 1 or len(t) > 1 for _, (s, t) in self.edges())

    @property
    def E(self) -> Tuple[Edge, ...]:
        """Edges in the graph.

        Returns:
            Tuple of edges
        """
        return tuple(e for _, e in self.edges())

    @property
    def V(self) -> Tuple[Any, ...]:
        """Vertices in the graph.

        Returns:
            Tuple of vertices
        """
        return tuple(self._get_vertices())

    @property
    def vertices(self) -> Tuple[Any, ...]:
        """Vertices in the graph.

        Returns:
            Tuple of vertices

        Note:
            Alias for V property
        """
        return self.V

    @property
    def nv(self) -> int:
        """Number of vertices in the graph.

        Returns:
            Number of vertices

        Note:
            Alias for num_vertices property
        """
        return self.num_vertices

    @property
    def ne(self) -> int:
        """Number of edges in the graph.

        Returns:
            Number of edges

        Note:
            Alias for num_edges property
        """
        return self.num_edges

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the graph (number of vertices, number of edges).

        Returns:
            Tuple of number of vertices and number of edges
        """
        return (self.nv, self.ne)

    def add_edge(
        self,
        source: Union[Any, Iterable[Any]],
        target: Union[Any, Iterable[Any]],
        type: Optional[EdgeType] = EdgeType.DIRECTED,
        edge_source_attr: Optional[Attributes] = None,
        edge_target_attr: Optional[Attributes] = None,
        **kwargs,
    ) -> int:
        """Add edge to the graph.

        Args:
            source: Source vertices
            target: Target vertices
            type: Edge type
            edge_source_attr: Optional attributes for source vertices
            edge_target_attr: Optional attributes for target vertices
            **kwargs: Additional edge attributes

        Returns:
            Index of the new edge
        """
        if type is None:
            type = self._default_edge_type
        ve_s = self._extract_ve_attr(source)  # {vertex: value}
        ve_t = self._extract_ve_attr(target)
        if edge_source_attr is None:
            edge_source_attr = Attributes()
        if edge_target_attr is None:
            edge_target_attr = Attributes()
        for vertex, value in ve_s.items():
            s_v_attr = edge_source_attr.get(vertex, Attributes())
            s_v_attr.set_attr(Attr.VALUE, value)
            edge_source_attr[vertex] = s_v_attr
        for vertex, value in ve_t.items():
            t_v_attr = edge_target_attr.get(vertex, Attributes())
            t_v_attr.set_attr(Attr.VALUE, value)
            edge_target_attr[vertex] = t_v_attr
        return self._add_edge(
            set(ve_s.keys()),
            set(ve_t.keys()),
            type=type,
            edge_source_attr=edge_source_attr,
            edge_target_attr=edge_target_attr,
            **kwargs,
        )

    def add_edges(
        self,
        edges: Iterable,
        type: EdgeType = EdgeType.DIRECTED,
        **kwargs,
    ) -> List[int]:
        """Add multiple edges to the graph.

        Args:
            edges: Iterable of (source, target) pairs
            type: Edge type
            **kwargs: Additional edge attributes

        Returns:
            List of indices of the new edges
        """
        eidxs = []
        for s, t in edges:
            eidxs.append(self.add_edge(s, t, type=type, **kwargs))
        return eidxs

    def add_vertex(self, v: Any, **kwargs) -> int:
        """Add vertex to the graph.

        Args:
            v: Vertex to add
            **kwargs: Additional vertex attributes

        Returns:
            Index of the new vertex
        """
        return self._add_vertex(v, **kwargs)

    def add_vertices(self, vertices: List, **kwargs) -> List[int]:
        """Add multiple vertices to the graph.

        Args:
            vertices: List of vertices to add
            **kwargs: Additional vertex attributes

        Returns:
            List of indices of the new vertices
        """
        return [self.add_vertex(v, **kwargs) for v in vertices]

    def get_vertex_incidence_matrix_as_lists(self, values: bool = False):
        """Get vertex incidence matrix as lists.

        Args:
            values: Whether to include edge values in the matrix

        Returns:
            Tuple of data, (row indices, column indices)
        """
        row_ind = []
        col_ind = []
        data = []
        V_indexes = {v: i for i, v in enumerate(self.V)}
        for j, e in enumerate(self.E):
            attr = self.get_attr_edge(j)
            s, t = e
            for v in s:
                value = -1
                if values:
                    v_attr = attr.get_attr(Attr.SOURCE_ATTR)
                    if v in v_attr:
                        if Attr.VALUE.value in v_attr[v]:
                            coeff = v_attr[v].get_attr(Attr.VALUE, 1)
                            # If coeff is not a number:
                            if not isinstance(coeff, Number):
                                coeff = 1
                            value = -1 * abs(coeff)
                if value != 0:
                    row_ind.append(V_indexes[v])
                    col_ind.append(j)
                    data.append(value)
            for v in t:
                value = 1
                if values:
                    v_attr = attr.get_attr(Attr.TARGET_ATTR)
                    if v in v_attr:
                        if Attr.VALUE.value in v_attr[v]:
                            coeff = v_attr[v].get_attr(Attr.VALUE, 1)
                            # If coeff is not a number:
                            if not isinstance(coeff, Number):
                                coeff = 1
                            value = abs(coeff)
                if value != 0:
                    row_ind.append(V_indexes[v])
                    col_ind.append(j)
                    data.append(value)
        return data, (row_ind, col_ind)

    import numpy as np

    def vertex_incidence_matrix(self, values: bool = False, sparse: bool = False):
        """Get vertex incidence matrix.

        Args:
            values: Whether to include edge values in the matrix.
            sparse: If True, returns the matrix as a sparse CSR matrix (default is False, which returns a dense NumPy array).

        Returns:
            Vertex incidence matrix as a numpy array or a sparse matrix.
        """
        data, (row_ind, col_ind) = self.get_vertex_incidence_matrix_as_lists(values)

        if sparse:
            from scipy.sparse import csr_matrix

            # Create a sparse CSR matrix
            A = csr_matrix((data, (row_ind, col_ind)), shape=(self.num_vertices, self.num_edges))
        else:
            # Create a dense matrix
            A = np.zeros((self.num_vertices, self.num_edges))
            A[row_ind, col_ind] = data

        return A

    def bfs(self, starting_vertices: Any, reverse: bool = False, undirected: bool = False) -> Dict[Any, int]:
        """Perform breadth-first search (BFS) traversal.

        Args:
            starting_vertices: Starting vertices for BFS
            reverse: Whether to traverse in reverse direction
            undirected: Whether to treat edges as undirected

        Returns:
            Dictionary mapping vertices to their BFS layer
        """
        starting_vertices = _tpl(starting_vertices)
        next_vertices = self.successors
        if reverse and undirected:
            raise ValueError("Reverse and undirected are mutually exclusive")
        if reverse:
            next_vertices = self.predecessors
        if undirected:
            next_vertices = self.neighbors
        layer = 0
        visited = {v: layer for v in starting_vertices}
        succ = next_vertices(starting_vertices)
        while succ:
            layer += 1
            new = set()
            for s in succ:
                curr_layer = visited.get(s, np.inf)
                if layer < curr_layer:
                    visited[s] = layer
                    new.add(s)
            if len(new) == 0:
                break
            succ = next_vertices(new)
        return visited

    def prune(
        self,
        source: Optional[List] = None,
        target: Optional[List] = None,
    ) -> "Graph":
        """Prune the graph to include only reachable vertices.

        Args:
            source: Source vertices to start pruning from
            target: Target vertices to reach

        Returns:
            Pruned subgraph
        """
        if source is None:
            source = list(self.V)
        if target is None:
            target = list(self.V)
        forward = set(self.bfs(source).keys())
        backward = set(self.bfs(target, reverse=True).keys())
        reachable = list(forward.intersection(backward))
        return self.subgraph(reachable)

    def plot(self, **kwargs):
        """Plot the graph using Graphviz.

        Renders the graph structure visually using Graphviz.
        Falls back to Viz.js rendering if SVG+XML rendering fails.

        Args:
            **kwargs: Additional plotting options passed to Graphviz

        Returns:
            Graphviz plot object

        Raises:
            OSError: If Graphviz rendering fails
        """
        Gv = self.to_graphviz(**kwargs)
        try:
            # Check if the object is able to produce a MIME bundle
            Gv._repr_mimebundle_()
            return Gv
        except (OSError, Exception) as e:
            from corneto._settings import LOGGER
            from corneto._util import supports_html

            LOGGER.debug(f"SVG+XML rendering failed: {e}.")
            # Detect if HTML support
            if supports_html():
                LOGGER.debug("Falling back to Viz.js rendering.")
                from corneto.contrib._util import dot_vizjs_html

                class _VizJS:
                    def _repr_html_(self):
                        return dot_vizjs_html(Gv)

                return _VizJS()
            else:
                LOGGER.debug("HTML rendering not supported.")
                raise e

    def plot_values(self, vertex_values=None, edge_values=None, vertex_props=None, edge_props=None):
        """Plot graph with vertex and edge values visualized.

        Args:
            vertex_values: Optional values to display on vertices
            edge_values: Optional values to display on edges
            vertex_props: Optional dict of vertex drawing properties
            edge_props: Optional dict of edge drawing properties

        Returns:
            Graphviz plot object with values visualized
        """
        from corneto._plotting import (
            create_graphviz_edge_attributes,
            create_graphviz_vertex_attributes,
        )

        vertex_props = vertex_props or {}
        edge_props = edge_props or {}
        vertex_drawing_props = None
        if vertex_values is not None:
            # Check if vertices has an attribute value to do vertices.value
            if hasattr(vertex_values, "value"):
                vertex_values = vertex_values.value
            vertex_drawing_props = create_graphviz_vertex_attributes(
                list(self.V), vertex_values=vertex_values, **vertex_props
            )
        edge_drawing_props = None
        if edge_values is not None:
            if hasattr(edge_values, "value"):
                edge_values = edge_values.value
            edge_drawing_props = create_graphviz_edge_attributes(edge_values=edge_values, **edge_props)
        return self.plot(
            custom_edge_attr=edge_drawing_props,
            custom_vertex_attr=vertex_drawing_props,
        )

    def to_graphviz(self, **kwargs):
        """Convert graph to Graphviz representation.

        Args:
            **kwargs: Additional options for Graphviz conversion

        Returns:
            Graphviz object representing the graph structure
        """
        from corneto._plotting import to_graphviz

        return to_graphviz(self, **kwargs)

    @staticmethod
    def from_vertex_incidence(
        A: np.ndarray,
        vertex_ids: Union[List[str], np.ndarray],
        edge_ids: Union[List[str], np.ndarray],
    ):
        """Create graph from vertex incidence matrix and labels.

        Args:
            A: Vertex incidence matrix. Rows are vertices, columns are edges.
               Non-zero entries indicate edge-vertex connections.
            vertex_ids: Labels for vertices corresponding to matrix rows
            edge_ids: Labels for edges corresponding to matrix columns

        Returns:
            Graph instance constructed from incidence matrix

        Raises:
            ValueError: If dimensions of inputs don't match
        """
        g = Graph()
        if len(vertex_ids) != A.shape[0]:
            raise ValueError(
                """The number of rows in A matrix is different from
                the number of vertex ids"""
            )
        if len(edge_ids) != A.shape[1]:
            raise ValueError(
                """The number of columns in A matrix is different from
                the number of edge ids"""
            )
        for v in vertex_ids:
            g.add_vertex(v)
        for j, v in enumerate(edge_ids):
            values = A[:, j]
            idx = np.flatnonzero(values)
            coeffs = values[idx]
            v_names = [vertex_ids[i] for i in idx]
            s = {n: val for n, val in zip(v_names, coeffs) if val < 0}
            t = {n: val for n, val in zip(v_names, coeffs) if val > 0}
            g.add_edge(s, t, id=v)
        return g

    def save(self, filename: str, compressed: Optional[bool] = True) -> None:
        """Save graph to file.

        Args:
            filename: Path to save graph to
            compressed: Whether to use compression. If True, uses LZMA compression.

        Raises:
            ValueError: If filename is empty

        Note:
            If compressed=True, '.xz' extension is added if not present
            If '.pkl' extension is missing, it will be added
        """
        if not filename:
            raise ValueError("Filename must not be empty.")

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        if compressed:
            import lzma

            if not filename.endswith(".xz"):
                filename += ".xz"
            with lzma.open(filename, "wb", preset=9) as f:
                pickle.dump(self, f)
        else:
            with open(filename, "wb") as f:
                pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> "BaseGraph":
        """Load graph from a saved file.

        Supports various compression formats:
        - .gz (gzip)
        - .bz2 (bzip2)
        - .xz (LZMA)
        - .zip (zip archive, reads first file)

        Args:
            filename: Path to saved graph file

        Returns:
            Loaded graph instance
        """
        if filename.endswith(".gz"):
            import gzip

            opener = gzip.open
        elif filename.endswith(".bz2"):
            import bz2

            opener = bz2.open
        elif filename.endswith(".xz"):
            import lzma

            opener = lzma.open
        elif filename.endswith(".zip"):
            import zipfile

            def opener(file, mode="r"):
                # Supports only reading the first file in a zip archive
                with zipfile.ZipFile(file, "r") as z:
                    return z.open(z.namelist()[0], mode=mode)
        else:
            opener = open

        with opener(filename, "rb") as f:
            return pickle.load(f)

    def toposort(self):
        """Perform topological sort on the graph using Kahn's algorithm.

        Returns:
            List of vertices in topological order

        Raises:
            ValueError: If graph contains cycles
        """
        # Topological sort using Kahn's algorithm
        in_degree = {v: len(set(self.predecessors(v))) for v in self._get_vertices()}

        # Initialize queue with nodes having zero in-degree
        queue = deque([v for v in in_degree.keys() if in_degree[v] == 0])

        result = []

        while queue:
            v = queue.popleft()
            result.append(v)

            # Decrease the in-degree of successor nodes by 1
            for successor in self.successors(v):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        # Check if topological sort is possible (i.e., graph has no cycles)
        if len(result) == self.num_vertices:
            return result
        else:
            raise ValueError("Graph contains a cycle, so topological sort is not possible.")

    def reachability_analysis(
        self,
        input_nodes,
        output_nodes,
        subset_edges=None,
        verbose=True,
        early_stop=False,
        expand_outputs=True,
        max_printed_outputs=10,
    ):
        """Perform reachability analysis from input nodes to output nodes.

        Args:
            input_nodes: Starting nodes for analysis
            output_nodes: Target nodes to reach
            subset_edges: Optional subset of edges to consider
            verbose: Whether to print progress information
            early_stop: Stop when all outputs are reached
            expand_outputs: Continue expanding from output nodes
            max_printed_outputs: Max outputs to show in verbose mode

        Returns:
            Set of edge indices used in paths from inputs to outputs
        """
        visited = set(input_nodes)
        current = set(input_nodes)
        unreached_outputs = set(output_nodes)
        outs = set(output_nodes)
        selected_edges = set()
        layer = 0
        if verbose:
            print("Starting reachability analysis...")
            print(f"L{layer:<3}: {len(input_nodes):<4} > input(s)")
        while current is not None and len(current) > 0:
            layer += 1
            new = set()
            for v in current:
                for i, (s, t) in self.out_edges(v):
                    if subset_edges is not None and i not in subset_edges:
                        continue
                    # Add only if t is a new node
                    nt = list(t)
                    if len(nt) == 0:
                        continue
                    vt = nt[0]
                    if vt not in visited:
                        new |= {vt}
                        selected_edges.add(i)
            # How many are output nodes?
            reached_outputs = outs.intersection(new)
            unreached_outputs -= reached_outputs
            if verbose:
                print(f"L{layer:<3}: {len(new):<4}", end="")
                if len(reached_outputs) > 0:
                    if len(reached_outputs) <= max_printed_outputs:
                        str_reached = "/".join(reached_outputs)
                    else:
                        # Get only the first max_printed_outputs items
                        str_reached = "/".join(list(reached_outputs)[:max_printed_outputs]) + "..."
                    print(f" > {len(reached_outputs):<4} output(s): {str_reached}")
                else:
                    print("")
            visited |= new
            current = set(new)
            if not expand_outputs:
                current -= reached_outputs
            if early_stop and len(unreached_outputs) == 0:
                break
        if verbose:
            print(f"Finished ({len(selected_edges)} selected edges).")
        return selected_edges

    def to_networkx(self):
        """Convert graph to NetworkX format.

        Raises:
            NotImplementedError: Not implemented in base class
        """
        raise NotImplementedError()

    @staticmethod
    def from_networkx(G: Union[NxGraph, NxDiGraph]):
        """Create graph from NetworkX graph.

        Args:
            G: NetworkX graph instance

        Returns:
            Graph instance with equivalent structure
        """
        Gc = Graph()
        is_directed = G.is_directed()
        for edge in G.edges():
            e_data = G.get_edge_data(edge[0], edge[1], default=dict())
            if is_directed:
                e_data[Attr.EDGE_TYPE.value] = EdgeType.DIRECTED.value
            else:
                e_data[Attr.EDGE_TYPE.value] = EdgeType.UNDIRECTED.value
            Gc.add_edge(edge[0], edge[1], **e_data)
        return Gc


class Graph(BaseGraph):
    """Concrete graph implementation supporting directed/undirected edges and hyperedges.

    Allows parallel edges (multiple edges between same vertices) and hyperedges
    (edges connecting multiple vertices). Edges and vertices can have attributes.

    Examples:
        >>> graph = corneto.Graph()
        >>> graph.add_edge(1, 2)
        >>> graph.plot()
    """

    def __init__(self, default_edge_type: EdgeType = EdgeType.DIRECTED, **kwargs) -> None:
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

    def extract_subgraph(self, vertices: Optional[Iterable] = None, edges: Optional[Iterable[int]] = None):
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

    def _extract_subgraph_keep_order(self, vertices: Optional[Iterable] = None, edges: Optional[Iterable[int]] = None):
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
    def from_sif(
        sif_file: str,
        delimiter: str = "\t",
        has_header: bool = False,
        discard_self_loops: Optional[bool] = True,
        column_order: List[int] = [0, 1, 2],
    ):
        """Create graph from Simple Interaction Format (SIF) file.

        Args:
            sif_file: Path to SIF file
            delimiter: Column delimiter in file
            has_header: Whether file has a header row
            discard_self_loops: Whether to ignore self-loops
            column_order: Order of source, interaction, target columns

        Returns:
            New Graph loaded from SIF file
        """
        from corneto._io import _read_sif_iter

        it = _read_sif_iter(
            sif_file,
            delimiter=delimiter,
            has_header=has_header,
            discard_self_loops=discard_self_loops,
            column_order=column_order,
        )
        return Graph.from_sif_tuples(it)

    @staticmethod
    def from_sif_tuples(tuples: Iterable[Tuple]):
        """Create graph from iterable of SIF tuples.

        Args:
            tuples: Iterable of (source, interaction, target) tuples

        Returns:
            New Graph created from SIF data
        """
        g = Graph()
        for s, v, t in tuples:
            g.add_edge(s, t, interaction=v)
        return g

    @staticmethod
    def from_cobra_model(model: CobraModel):
        """Create graph from COBRA metabolic model.

        Args:
            model: COBRApy model instance

        Returns:
            New Graph representing the metabolic network
        """
        S, R, M = import_cobra_model(model)
        G = Graph.from_vertex_incidence(S, M["id"], R["id"])
        # Add metadata to the graph, such as default lb/ub for reactions
        for i in range(G.num_edges):
            attr = G.get_attr_edge(i)
            attr["default_lb"] = R["lb"][i]
            attr["default_ub"] = R["ub"][i]
            attr["GPR"] = R["gpr"][i]
        return G

    @staticmethod
    def from_miom_model(model):
        """Create graph from MIOM metabolic model.

        Args:
            model: MIOM model instance or path to compressed model file

        Returns:
            New Graph representing the metabolic network
        """
        if isinstance(model, str):
            from corneto._io import _load_compressed_gem

            S, R, M = _load_compressed_gem(model)
        else:
            S = model.S, M = model.M, R = model.R
        G = Graph.from_vertex_incidence(S, M["id"], R["id"])
        # Add metadata to the graph, such as default lb/ub for reactions
        for i in range(G.num_edges):
            attr = G.get_attr_edge(i)
            attr["default_lb"] = R["lb"][i]
            attr["default_ub"] = R["ub"][i]
            attr["GPR"] = R["gpr"][i]
        return G
