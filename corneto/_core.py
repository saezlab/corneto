import abc
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from corneto._constants import *
from corneto._settings import sparsify


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

    @abc.abstractmethod
    def _add_edge(self, s: Dict, t: Dict, id: Optional[str] = None, **kwargs) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def _add_vertex(self, v: Any, id: Optional[str] = None, **kwargs) -> int:
        pass

    @abc.abstractmethod
    def copy(self) -> "BaseGraph":
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

    # @abc.abstractmethod
    def get_vertex_properties_for_edge(self, edge) -> Tuple[Dict, ...]:
        ev_props = self.edge_vertex_properties
        return tuple(ev_props[i] for i, e in enumerate(self.edges) if e == edge)

    @property
    @abc.abstractmethod
    def edge_vertex_properties(self) -> Tuple[Dict, ...]:
        raise NotImplementedError()

    def get_edges_with_source_vertex(self, v) -> Set[Tuple]:
        return {(s, t) for (s, t) in self.get_edges_from_vertex(v) if v in s}

    def get_edges_with_target_vertex(self, v) -> Set[Tuple]:
        return {(s, t) for (s, t) in self.get_edges_from_vertex(v) if v in t}

    def successors_of_vertex(self, v) -> Set:
        # TODO: check if directed or not
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

    def add_edge(self, s, t, id: Optional[str] = None, directed: bool = True, **kwargs) -> int:
        return self._add_edge(
            BaseGraph._as_dict(s),
            BaseGraph._as_dict(t),
            id=id,
            directed=directed,
            **kwargs,
        )

    def add_edges(self, edges: List[Tuple], directed: bool = True, **kwargs) -> List[int]:
        idx = []
        for s, t in edges:
            idx.append(self.add_edge(s, t, directed=directed, **kwargs))
        return idx

    def add_vertex(self, v: Any, id: Optional[str] = None, **kwargs) -> int:
        return self._add_vertex(v, id, **kwargs)

    def vertex_incidence_matrix(self, values: bool = False, sparse: bool = False):
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
            return sparsify(A)
        return A

    def get_source_vertices(self) -> Set:
        sources = set()
        for v in self.vertices:
            pred = self.predecessors_of_vertex(v)
            if not pred or len(pred) == 0:
                sources.add(v)
            else:
                if len(pred) == 1 and (() in pred or frozenset() in pred):
                    sources.add(v)
        return sources

    def get_sink_vertices(self) -> Set:
        sinks = set()
        for v in self.vertices:
            succ = self.successors_of_vertex(v)
            if not succ or len(succ) == 0:
                sinks.add(v)
            else:
                if len(succ) == 1 and (() in succ or frozenset() in succ):
                    sinks.add(v)
        return sinks

    def bfs(self, starting_vertices: Any, rev: bool = False) -> Dict[Any, int]:
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
    ) -> "Graph":
        forward = set(self.bfs(source).keys())
        backward = set(self.bfs(target, rev=True).keys())
        reachable = list(forward.intersection(backward))
        return self.subgraph(reachable)

    def plot_legacy(self, **kwargs):
        from corneto import legacy_plot

        legacy_plot(self, **kwargs)

    def plot(self, **kwargs):
        return self.to_graphviz()

    def to_graphviz(
        self,
        problem=None,
        condition: str = None,
        graph_attr: Dict[str, str] = None,
        node_attr: Dict[str, str] = None,
        edge_attr: Dict[str, str] = None,
    ):
        import graphviz

        vertices, edges = self.vertices, self.edges
        custom_vertex = dict()
        custom_edge = dict()
        if problem:
            if hasattr(problem, "symbols"):
                problem = {k: v.value for k, v in problem.symbols.items()}
            # TODO: very ad-hoc, improve
            c = [k for k in problem.keys() if k.startswith("reaction_sends_activation")]
            if len(c) > 1 and condition is None:
                raise ValueError("Detected multiple conditions defined in problem, but a condition was not provided")
            if len(c) == 1 and condition is None:
                condition = c[0].split("activation_")[1]
            vertex_values = problem[f"species_activated_{condition}"] - problem[f"species_inhibited_{condition}"]
            edge_values = (
                problem[f"reaction_sends_activation_{condition}"] - problem[f"reaction_sends_inhibition_{condition}"]
            )
            # Add custom values per edge/vertex
            for v, value in zip(vertices, vertex_values):
                if value < 0:
                    custom_vertex[v] = dict(color="blue", penwidth="2", fillcolor="azure2", style="filled")
                elif value > 0:
                    custom_vertex[v] = dict(
                        color="red",
                        penwidth="2",
                        fillcolor="lightcoral",
                        style="filled",
                    )

            for e, value in zip(edges, edge_values):
                if value < 0:
                    custom_edge[e] = dict(color="blue", penwidth="2")
                elif value > 0:
                    custom_edge[e] = dict(color="red", penwidth="2")

        if node_attr is None:
            node_attr = dict(fixedsize="true")
        g = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
        for e, p in zip(edges, self.edge_properties):
            s, t = e
            s = list(s)
            if len(s) == 0:
                s = f"*_{t!s}"
                g.node(s, shape="point")
            elif len(s) == 1:
                s = str(s[0])
                props = custom_vertex.get(s, dict())
                g.node(s, shape="circle", **props)
            else:
                raise NotImplementedError("Represent- hyperedges as composite edges")
            t = list(t)
            if len(t) == 0:
                t = f"{s!s}_*"
                g.node(t, shape="point")
            elif len(t) == 1:
                t = str(t[0])
                props = custom_vertex.get(t, dict())
                g.node(t, shape="circle", **props)
            edge_type = p.get("interaction", 0)
            props = custom_edge.get(e, dict())
            if ("directed" in p and p["directed"] == False) or ("undirected" in p and p["undirected"] == True):
                props["dir"] = "none"
            if edge_type >= 0:
                g.edge(s, t, arrowhead="normal", **props)
            else:
                g.edge(s, t, arrowhead="tee", **props)
        return g


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

    def _add_edge(self, s: Dict, t: Dict, id: Optional[str] = None, **kwargs) -> int:
        # TODO: Self loops not supported, needed?
        uv = set().union(*[s, t])
        sv = frozenset(s.keys())
        tv = frozenset(t.keys())
        edge = (sv, tv)
        self._edges.append(edge)
        idx = len(self._edges) - 1
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
            edge_props["id"] = id
        self._edge_properties.append(edge_props)
        # Add vertices to the index
        for v in uv:
            if v in self._vertex_index:
                self._vertex_index[v] |= {edge}
            else:
                self._vertex_index[v] = {edge}
        return idx

    def _add_vertex(self, v: Any, id: Optional[str] = None, **kwargs) -> int:
        if v not in self._vertex_index:
            self._vertex_index[v] = set()
            self._vertex_properties[v] = dict(kwargs)
            idx = len(self._vertex_index) - 1
        else:
            idx = list(self._vertex_index).index(v)
            if v in self._vertex_properties:
                props = self._vertex_properties[v]
            else:
                props = dict()
                self._vertex_properties[v] = props
            props.update(kwargs)
        if id:
            props[id] = id
        return idx

    def _get_edge(self, edge) -> Dict:
        return self._edges[edge]

    def subgraph(self, vertices: List) -> "Graph":
        g = Graph()
        g._edges = []
        g._edge_properties = []
        g._edge_vertex_properties = []
        g._vertex_index = OrderedDict()
        g._vertex_properties = OrderedDict()
        sv = set(vertices)
        # TODO: Fix issues with edge indexes
        eidx = {e: i for i, e in enumerate(self._edges)}
        E = set()
        for v in vertices:
            E |= self._vertex_index[v]
        selected = []
        for s, t in E:
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
        return self._edges

    @property
    def vertices(self):
        return self._vertex_index.keys()

    def get_vertex_indexes(self):
        return {v: i for i, v in enumerate(self._vertex_index.keys())}

    def get_edge_indexes(self):
        # TODO: Fix
        return {e: i for i, e in enumerate(self._edges)}

    def get_vertex_edge_indexes(self):
        return self.get_vertex_indexes(), self.get_edge_indexes()

    @property
    def num_edges(self):
        return len(self._edges)

    @property
    def num_vertices(self):
        return len(self._vertex_index)

    # TODO: rename to get_adjacent_edges
    def get_edges_from_vertex(self, v) -> Set[Tuple]:
        return self._vertex_index.get(v, set())

    @property
    def edge_vertex_properties(self) -> Tuple[Dict, ...]:
        return tuple(self._edge_vertex_properties)

    @property
    def edge_properties(self) -> Tuple[Dict, ...]:
        return tuple(self._edge_properties)

    def copy(self) -> "Graph":
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
