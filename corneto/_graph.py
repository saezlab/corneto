import abc
import pickle
from collections import OrderedDict
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
    DIRECTED = "directed"
    UNDIRECTED = "undirected"


def _wrap(
    elements: Any,
    container: Callable[[Iterable[Any]], T],
    skip: Optional[Type[T]] = None,
) -> T:
    if skip is not None and isinstance(elements, skip):
        return elements
    if isinstance(elements, str):
        return container((elements,))
    if isinstance(elements, Iterable):
        return container(elements)
    return container((elements,))


def _fset(elements: Union[Any, Iterable[Any]]) -> FrozenSet[Any]:
    return _wrap(elements, frozenset, frozenset)


def _tpl(elements: Union[Any, Iterable[Any]]) -> Tuple[Any, ...]:
    return _wrap(elements, tuple, tuple)


class BaseGraph(abc.ABC):
    """BaseGraph class for graphs or hypergraphs with directed/undirected/mixed
    and self edges
    """

    def __init__(self, default_edge_type: EdgeType = EdgeType.DIRECTED) -> None:
        """Initialize BaseGraph with default edge type.

        Parameters
        ----------
        default_edge_type : EdgeType
            Default type for edges.

        """
        super().__init__()
        self._default_edge_type = default_edge_type

    @staticmethod
    def _parse_vertices(s):
        if isinstance(s, dict):
            return {
                k: v if isinstance(v, (dict, Number)) else ValueError()
                for k, v in s.items()
            }
        elif isinstance(s, (str, Number, Iterable)):
            return {v: {} for v in (s if isinstance(s, Iterable) else [s])}
        else:
            raise ValueError()

    @staticmethod
    def _extract_ve_attr(s):
        if isinstance(s, dict):
            return {
                k: v if isinstance(v, (dict, Number)) else ValueError()
                for k, v in s.items()
            }
        elif isinstance(s, (str, Number, Iterable)):
            return {
                v: {}
                for v in (
                    s if not isinstance(s, str) and isinstance(s, Iterable) else [s]
                )
            }
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
        raise NotImplementedError()

    @abc.abstractmethod
    def _add_vertex(self, vertex: Any, **kwargs) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_edge(self, index: int) -> Edge:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_vertices(self) -> Iterable:
        # Returns vertices by order of insertion
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_incident_edges(self, vertex) -> Iterable[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_edge_attributes(self, index: int) -> Attributes:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_vertex_attributes(self, v) -> Attributes:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_graph_attributes(self) -> Attributes:
        raise NotImplementedError()

    @abc.abstractmethod
    def _num_vertices(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def _num_edges(self) -> int:
        raise NotImplementedError()

    def edge_subgraph(self, edges: Union[Iterable[int], np.ndarray]):
        if isinstance(edges, np.ndarray):
            # Check if its a logical array:
            if edges.dtype == bool:
                edges = np.where(edges)[0]
        return self.extract_subgraph(vertices=None, edges=edges)

    def subgraph(self, vertices: Iterable):
        return self.extract_subgraph(vertices=vertices, edges=None)

    @abc.abstractmethod
    def extract_subgraph(
        self, vertices: Optional[Iterable] = None, edges: Optional[Iterable[int]] = None
    ):
        raise NotImplementedError()

    @property
    def num_vertices(self) -> int:
        return self._num_vertices()

    @property
    def num_edges(self) -> int:
        return self._num_edges()

    def hash(self) -> str:
        return obj_content_hash(self)

    def get_attr_edge(self, index: int) -> Attributes:
        return self._get_edge_attributes(index)

    def get_attr_vertex(self, v) -> Attributes:
        return self._get_vertex_attributes(v)

    def get_attr_edges(
        self, indexes: Optional[Iterable[int]] = None
    ) -> List[Attributes]:
        if indexes is None:
            indexes = range(self.num_edges)
        return [self._get_edge_attributes(i) for i in indexes]

    def get_attr_from_edges(self, attr: str, default: Any = None) -> List[Any]:
        attrs = self.get_attr_edges()
        return [a.get(attr, default) for a in attrs]

    def get_edges_by_attr(self, key: str, value: Any) -> Iterable[int]:
        for i, e in enumerate(self.get_attr_edges()):
            if key in e and e[key] == value:
                yield i

    def get_attr_vertices(
        self, vertices: Optional[Iterable] = None
    ) -> List[Attributes]:
        if vertices is None:
            vertices = self.V
        return [self._get_vertex_attributes(v) for v in vertices]

    def get_edges(self, indexes: Iterable[int]) -> Iterable[Edge]:
        return (self.get_edge(i) for i in indexes)

    def get_vertex(self, index: int) -> Any:
        # O(n) unless vertices are also indexed by position.
        # This method is added for convenience but shouldn't be required.
        for i, v in enumerate(self._get_vertices()):
            if i == index:
                return v
        raise IndexError(
            f"Vertex index {index} out of range [0 - {self.num_vertices - 1}]"
        )

    def get_incident_edges(self, vertices) -> Iterable[int]:
        combined = chain.from_iterable(self._get_incident_edges(v) for v in vertices)
        seen: Set[int] = set()
        for e in combined:
            if e not in seen:
                seen.add(e)
                yield e

    def get_common_incident_edges(self, vertices) -> Iterable[int]:
        vset = set(vertices)
        seen: Set[int] = set()
        for e in self.get_incident_edges(vertices):
            if e not in seen:
                (s, t) = self.get_edge(e)
                if (s | t).intersection(vset) == vset:
                    seen.add(e)
                    yield e

    def edges(self, vertices=None) -> Iterable[Tuple[int, Edge]]:
        # TODO: return in the order of the edges!
        # Add test case to check the order
        if vertices is not None:
            vertices = _tpl(vertices)
            eidx = self.get_incident_edges(vertices)
            return ((i, self.get_edge(i)) for i in eidx)
        else:
            return ((i, self.get_edge(i)) for i in range(self.num_edges))

    def copy(self) -> "BaseGraph":
        return deepcopy(self)

    @abc.abstractmethod
    def reverse(self) -> "BaseGraph":
        return NotImplementedError()

    def _edges_by_dir(
        self, vertices, direction: Optional[str] = None
    ) -> Iterable[Tuple[int, Edge]]:
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
        yield from self._edges_by_dir(vertices, "in")

    def out_edges(self, vertices) -> Iterable[Tuple[int, Edge]]:
        yield from self._edges_by_dir(vertices, "out")

    def _successors(self, vertex) -> Iterable:
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
        vertices = _tpl(vertices)
        return unique_iter(chain.from_iterable((self._successors(v) for v in vertices)))

    def _predecessors(self, vertex) -> Iterable:
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
        vertices = _tpl(vertices)
        return unique_iter(
            chain.from_iterable((self._predecessors(v) for v in vertices))
        )

    def neighbors(self, vertex) -> Iterable:
        # Ignores direction of edge
        iter_vertices = (
            s | t for _, (s, t) in self._edges_by_dir(vertex, direction=None)
        )
        return unique_iter(chain.from_iterable(iter_vertices))

    def is_hypergraph(self) -> bool:
        return any(len(s) > 1 or len(t) > 1 for _, (s, t) in self.edges())

    @property
    def E(self) -> Tuple[Edge, ...]:
        return tuple(e for _, e in self.edges())

    @property
    def V(self) -> Tuple[Any, ...]:
        return tuple(self._get_vertices())

    @property
    def vertices(self) -> Tuple[Any, ...]:
        return self.V

    @property
    def nv(self) -> int:
        return self.num_vertices

    @property
    def ne(self) -> int:
        return self.num_edges

    @property
    def shape(self) -> Tuple[int, int]:
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
        # In self loops, or hyperedges with partial self loops
        # important to have dupl. vertices!. E.g {A: -1, A: 1} A (-1)->(1) A
        if type is None:
            type = self._default_edge_type
        ve_s = Graph._extract_ve_attr(source)  # {vertex: value}
        ve_t = Graph._extract_ve_attr(target)
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
        eidxs = []
        for s, t in edges:
            eidxs.append(self.add_edge(s, t, type=type, **kwargs))
        return eidxs

    def add_vertex(self, v: Any, **kwargs) -> int:
        return self._add_vertex(v, **kwargs)

    def add_vertices(self, vertices: List, **kwargs) -> List[int]:
        return [self.add_vertex(v, **kwargs) for v in vertices]

    def get_vertex_incidence_matrix_as_lists(self, values: bool = False):
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

    def vertex_incidence_matrix(self, values: bool = False):
        # Returns V x E matrix
        A = np.zeros((self.num_vertices, self.num_edges))
        data, (row_ind, col_ind) = self.get_vertex_incidence_matrix_as_lists(values)
        A[row_ind, col_ind] = data
        return A

    def bfs(
        self, starting_vertices: Any, reverse: bool = False, undirected: bool = False
    ) -> Dict[Any, int]:
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
        if source is None:
            source = list(self.V)
        if target is None:
            target = list(self.V)
        forward = set(self.bfs(source).keys())
        backward = set(self.bfs(target, reverse=True).keys())
        reachable = list(forward.intersection(backward))
        return self.subgraph(reachable)

    def plot(self, **kwargs):
        Gv = self.to_graphviz(**kwargs)
        try:
            # Check if the object is able to produce a MIME bundle
            Gv._repr_mimebundle_()
            return Gv
        except Exception as e:
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

    def plot_values(
        self, vertex_values=None, edge_values=None, vertex_props=None, edge_props=None
    ):
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
            edge_drawing_props = create_graphviz_edge_attributes(
                edge_values=edge_values, **edge_props
            )
        return self.plot(
            custom_edge_attr=edge_drawing_props,
            custom_vertex_attr=vertex_drawing_props,
        )

    def to_graphviz(self, **kwargs):
        from corneto._plotting import to_graphviz

        return to_graphviz(self, **kwargs)

    @staticmethod
    def from_vertex_incidence(
        A: np.ndarray,
        vertex_ids: Union[List[str], np.ndarray],
        edge_ids: Union[List[str], np.ndarray],
    ):
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
                        str_reached = (
                            "/".join(list(reached_outputs)[:max_printed_outputs])
                            + "..."
                        )
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
        raise NotImplementedError()

    @staticmethod
    def from_networkx(G: Union[NxGraph, NxDiGraph]):
        Gc = Graph()
        for edge in G.edges():
            e_data = G.get_edge_data(edge[0], edge[1], default=dict())
            Gc.add_edge(edge[0], edge[1], **e_data)
        return Gc


class Graph(BaseGraph):
    """Default Graph class with support for undirected, directed, parallel and hypergedes.

    Parameters
    ----------
    default_edge_type
        Default edge type :class:`~corneto._graph.EdgeType`.

    Examples:
    --------
    >>> graph = corneto.Graph()
    >>> graph.add_edge(1, 2)
    >>> graph.plot()

    """

    def __init__(
        self, default_edge_type: EdgeType = EdgeType.DIRECTED, **kwargs
    ) -> None:
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
        return self._edges[index]

    def _get_vertices(self) -> Iterable:
        return iter(self._vertices.keys())

    def _get_incident_edges(self, vertex) -> Iterable[int]:
        return self._vertices[vertex]

    def _get_edge_attributes(self, index: int) -> Attributes:
        return self._edge_attr[index]

    def _get_vertex_attributes(self, v) -> Attributes:
        if v in self._vertex_attr:
            return self._vertex_attr[v]
        return Attributes()

    def get_graph_attributes(self) -> Attributes:
        return self._graph_attr

    def _num_vertices(self) -> int:
        return len(self._vertices)

    def _num_edges(self) -> int:
        return len(self._edges)

    def extract_subgraph(
        self, vertices: Optional[Iterable] = None, edges: Optional[Iterable[int]] = None
    ):
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

    def reverse(self) -> "Graph":
        """Create a new graph and reverse the direction of all edges in the graph."""
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
        return deepcopy(self)

    @staticmethod
    def from_sif(
        sif_file: str,
        delimiter: str = "\t",
        has_header: bool = False,
        discard_self_loops: Optional[bool] = True,
        column_order: List[int] = [0, 1, 2],
    ):
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
        g = Graph()
        for s, v, t in tuples:
            g.add_edge(s, t, interaction=v)
        return g

    @staticmethod
    def from_cobra_model(model: CobraModel):
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
