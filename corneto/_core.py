import abc
from copy import deepcopy
import numpy as np
from corneto._io import load_sif
from typing import Any, Optional, Iterable, Set, Tuple, Union, Dict, List
from corneto._typing import StrOrInt, TupleSIF
from corneto._settings import try_sparse
from corneto._constants import *
from corneto._decorators import jit
from numbers import Number
from collections import OrderedDict
from itertools import chain

def _set(e):
    if isinstance(e, set):
        return e
    if isinstance(e, Iterable) and not isinstance(e, str):
        e = frozenset(e)
    else:
        e = frozenset({e})
    return e


class BaseGraph(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _as_dict(s):
        if isinstance(s, dict):
            result = dict()
            for k, v in s.items():
                props = dict()
                if isinstance(v, Number):
                    props["v"] = v
                elif isinstance(v, dict):
                    props = dict(v)
                else:
                    raise ValueError()
                result[k] = props
            return result
        elif isinstance(s, str) or isinstance(s, Number):
            return {s: dict()}
        elif isinstance(s, Iterable):
            return {v: dict() for v in s}
        else:
            raise ValueError()


    @staticmethod
    def _as_dict2(s):
        if isinstance(s, str):
            return {s: dict()}
        if isinstance(s, Iterable):
            # e.g 'ab' -> {'ab': {}}
            if isinstance(s, str):
                return {s: dict()}
            if isinstance(s, dict):
                result = dict()
                for k, v in s.items():
                    props = dict()
                    if isinstance(v, Number):
                        props["v"] = v
                    elif isinstance(v, dict):
                        # Shallow copy
                        props = dict(v)
                    else:
                        raise ValueError()
                    result[k] = props
                return result
            else:
                # E.g.:
                #   ('a', 'b') -> {'a': {}, 'b': {}}
                #   (1, 2, 3) -> {1: {}, 2: {}, 3: {}}
                return {v: dict() for v in s}
        else:
            # e.g 'a' -> {'a': {}}, 1 -> {1: {}}
            return {s: dict()}

    @abc.abstractmethod
    def _add_edge(self, s: Dict, t: Dict, id: Optional[str] = None, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _add_vertex(self, v: Any, id: Optional[str] = None, **kwargs):
        pass

    @abc.abstractmethod
    def copy(self) -> 'BaseGraph':
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def edges(self) -> List[Tuple]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def vertices(self) -> List:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def num_vertices(self) -> int:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def num_edges(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_edges_from_vertex(self, v) -> Set[Tuple]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def edge_properties(self) -> Tuple[Dict, ...]:
        raise NotImplementedError()

    #@abc.abstractmethod
    def get_vertex_properties_for_edge(self, edge) -> Tuple[Dict, ...]:
        ev_props = self.edge_vertex_properties
        return tuple(ev_props[i] for i, e in enumerate(self.edges) if e==edge)

    @property
    @abc.abstractmethod
    def edge_vertex_properties(self) -> Tuple[Dict, ...]:
        raise NotImplementedError()
        
    def get_edges_with_source_vertex(self, v) -> Set[Tuple]:
        return {(s, t) for (s, t) in self.get_edges_from_vertex(v) if v in s}

    def get_edges_with_target_vertex(self, v) -> Set[Tuple]:
        return {(s, t) for (s, t) in self.get_edges_from_vertex(v) if v in t}

    def successors_of_vertex(self, v) -> Set:
        E = (t if len(t) > 0 else () for (s, t) in self.get_edges_from_vertex(v) if v in s)
        return set(chain.from_iterable(E))

    def predecessors_of_vertex(self, v) -> Set:
        E = (s if len(t) > 0 else () for (s, t) in self.get_edges_from_vertex(v) if v in t)
        return set(chain.from_iterable(E))

    def successors(self, vertices) -> Set:
        vertices = _set(vertices)
        succ = set()
        for v in vertices:
            succ |= self.successors_of_vertex(v)
        return succ

    def predecessors(self, vertices) -> Set:
        vertices = _set(vertices)
        succ = set()
        for v in vertices:
            succ |= self.predecessors_of_vertex(v)
        return succ

    def add_edge(self, s, t, id: Optional[str] = None, directed: bool = True, **kwargs):
        self._add_edge(
            BaseGraph._as_dict(s),
            BaseGraph._as_dict(t),
            id=id,
            directed=directed,
            **kwargs,
        )

    def add_edges(self, edges: List[Tuple], directed: bool = True, **kwargs):
        for (s, t) in edges:
            self.add_edge(s, t, directed=directed, **kwargs)

    def add_vertex(self, v: Any, id: Optional[str] = None, **kwargs):
        self._add_vertex(v, id, **kwargs)

    def vertex_incidence_matrix(self, values: bool = False, sparse: bool = True):
        A = np.zeros((self.num_vertices, self.num_edges))
        I = {v: i for i, v in enumerate(self.vertices)}
        for j, e in enumerate(self.edges):
            V = np.zeros(self.num_vertices)
            prop = self.edge_vertex_properties[j]
            s, t = e
            for v in s:
                value = -1
                if values:
                    if "v" in prop[v]:
                        value = -1 * abs(prop[v]["v"])
                    else:
                        raise ValueError(f"Vertex {v} does not have an assigned value.")
                V[I[v]] = value
            for v in t:
                value = 1
                if values:
                    if "v" in prop[v]:
                        value = abs(prop[v]["v"])
                    else:
                        raise ValueError(f"Vertex {v} does not have an assigned value.")
                V[I[v]] = value
            A[:, j] = V
        if sparse:
            return try_sparse(A)
        return A


    def bfs(
        self, starting_vertices: Any, rev: bool = False
    ) -> Dict[Any, int]:
        if isinstance(starting_vertices, Iterable) and not isinstance(starting_vertices, str):
            starting_vertices = frozenset(starting_vertices)
        else:
            starting_vertices = frozenset({starting_vertices})
        next_vertices = self.successors
        if rev:
            next_vertices = self.predecessors
        layer = 0
        visited = {v: layer for v in starting_vertices}
        succ = next_vertices(starting_vertices)
        while succ:
            layer += 1
            new = []
            for s in succ:
                l = visited.get(s, np.inf)
                if layer < l:
                    visited[s] = layer
                    new.append(s)
            succ = next_vertices(new)
        return visited
    
    @abc.abstractmethod
    def subgraph(self, nodes):
        raise NotImplementedError()
    
    def prune(
        self,
        source: List,
        target: List,
    ) -> 'Graph':
        forward = set(self.bfs(source).keys())
        backward = set(self.bfs(target, rev=True).keys())
        reachable = list(forward.intersection(backward))
        return self.subgraph(reachable)
    
    def plot(self, **kwargs):
        from corneto import legacy_plot
        legacy_plot(self, **kwargs)


class Graph(BaseGraph):
    def __init__(self) -> None:
        super().__init__()
        # Allow edges with same s/t vertices
        self._edges: List[Tuple] = []
        # Global edge propeties
        self._edge_properties: List[Dict] = []
        # Specific edge properties between vertices and edges
        self._edge_vertex_properties: List[Dict] = []
        # Vertices -> {edges where they appear}
        self._vertex_index: Dict[Any, Set[Tuple]] = OrderedDict()
        # Vertex properties
        self._vertex_properties: Dict = OrderedDict()

    def _add_edge(self, s: Dict, t: Dict, id: Optional[str] = None, **kwargs):
        # TODO: Self loops not supported, needed?
        uv = set().union(*[s, t])
        sv = frozenset(s.keys())
        tv = frozenset(t.keys())
        edge = (sv, tv)
        self._edges.append(edge)
        # Get vertex-edge properties
        ve_props = dict()
        for k, v in s.items():
            ve_props[k] = v
        for k, v in t.items():
            ve_props[k] = v
        self._edge_vertex_properties.append(ve_props)
        # Properties related to the edge
        edge_props = dict()
        if len(kwargs) > 0:
            edge_props.update(kwargs)
        if id:
            edge_props['id'] = id
        self._edge_properties.append(edge_props)
        # Add vertices to the index
        for v in uv:
            if v in self._vertex_index:
                self._vertex_index[v] |= {edge}
            else:
                self._vertex_index[v] = {edge}

    def _add_vertex(self, v: Any, id: Optional[str] = None, **kwargs):
        if v not in self._vertex_index:
            self._vertex_index[v] = set()
            self._vertex_properties[v] = dict(kwargs)
        else:
            if v in self._vertex_properties:
                props = self._vertex_properties[v]
            else:
                props = dict()
                self._vertex_properties[v] = props
            props.update(kwargs)
        if id:
            props[id] = id


    def _get_edge(self, edge) -> Dict:
        return self._edges[edge]
    
    def subgraph(self, vertices: List) -> 'Graph':
        g = Graph()
        g._edges = []
        g._edge_properties = []
        g._edge_vertex_properties = []
        g._vertex_index = OrderedDict()
        g._vertex_properties = OrderedDict()
        sv = set(vertices)
        eidx = {e: i for i, e in enumerate(self._edges)}
        E = set()
        for v in vertices:
            E |= self._vertex_index[v]
        selected = []
        for (s, t) in E:
            if len(sv.intersection(s)) > 0 and len(sv.intersection(t)) > 0:
                selected.append(eidx[(s, t)])
        # Preserve original order
        g._edges = [self._edges[i] for i in selected]
        g._edge_properties = [deepcopy(self._edge_properties[i]) for i in selected]
        g._edge_vertex_properties = [deepcopy(self._edge_vertex_properties[i]) for i in selected]
        g._vertex_index = OrderedDict()
        for v in self.vertices:
            if v in vertices:
                g._vertex_index[v] = deepcopy(self._vertex_index[v])
                props = self._vertex_properties.get(v, None)
                if props:
                    g._vertex_properties[v] = deepcopy(props)
        return g

    @property
    def edges(self):
        return list(self._edges)

    @property
    def vertices(self):
        return list(self._vertex_index.keys())

    @property
    def num_edges(self):
        return len(self._edges)

    @property
    def num_vertices(self):
        return len(self._vertex_index)

    def get_edges_from_vertex(self, v) -> Set[Tuple]:
        return self._vertex_index.get(v, set())

    @property
    def edge_vertex_properties(self) -> Tuple[Dict, ...]:
        return tuple(self._edge_vertex_properties)

    @property
    def edge_properties(self) -> Tuple[Dict, ...]:
        return tuple(self._edge_properties)

    def copy(self) -> 'Graph':
        return deepcopy(self)

    @staticmethod
    def from_vertex_incidence(A: np.ndarray, vertex_ids: List[str], edge_ids: List[str]):
        g = Graph()
        if len(vertex_ids) != A.shape[0]:
            raise ValueError("The number of rows in A matrix is different from the number of vertex ids")
        if len(edge_ids) != A.shape[1]:
            raise ValueError("The number of columns in A matrix is different from the number of edge ids")
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

    @staticmethod
    def from_sif(
        sif_file: str, 
        delimiter: str = "\t",
        has_header: bool = False,
        discard_self_loops: Optional[bool] = True,
        column_order: List[int] = [0, 1, 2]):
        from corneto._io import _read_sif
        tuples = _read_sif(
            sif_file,
            delimiter=delimiter, 
            has_header=has_header, 
            discard_self_loops=discard_self_loops,
            column_order=column_order
        )
        g = Graph()
        for (s, v, t) in tuples:
            g.add_edge(s, t, interaction=v)
        return g

