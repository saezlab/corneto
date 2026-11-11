"""Steiner tree optimization."""

from collections.abc import Mapping
from typing import Any, List, Literal, Optional, Union

import numpy as np

from corneto._constants import VAR_FLOW
from corneto.backend._base import Backend
from corneto.graph import Attr, EdgeType
from corneto.methods._input_utils import (
    DEFAULT_CONDITION,
    data_from_features,
    legacy_data,
    require_mapping,
    validate_condition_keys,
    validate_edge_costs,
    validate_vertex_collection,
)
from corneto.methods.pcst import PrizeCollectingSteinerTree


class SteinerTreeFlow(PrizeCollectingSteinerTree):
    """Basic Steiner Tree optimization method as a flow-based problem.

    Given a graph and a set of terminal nodes, this method finds a minimal-weight
    connected subgraph (tree) that spans all terminals.
    """

    def __init__(
        self,
        max_flow: Optional[Union[float, List[float]]] = None,
        default_edge_cost: float = 1.0,
        flow_name: str = VAR_FLOW,
        root_vertex: Optional[Union[Any, List[Any]]] = None,
        root_selection_strategy: Literal["first", "best"] = "first",
        best_root_candidates: Literal["data", "graph"] = "data",
        epsilon: float = 1,
        strict_acyclic: bool = True,
        disable_structured_sparsity: bool = False,
        in_flow_edge_type: EdgeType = EdgeType.DIRECTED,
        out_flow_edge_type: EdgeType = EdgeType.DIRECTED,
        lambda_reg: float = 0.0,
        force_flow_through_root: bool = True,
        backend: Optional[Backend] = None,
    ):
        super().__init__(
            include_all_terminals=True,  # Steiner Tree requires all terminals
            max_flow=max_flow,
            default_edge_cost=default_edge_cost,
            flow_name=flow_name,
            root_vertex=root_vertex,
            root_selection_strategy=root_selection_strategy,
            best_root_candidates=best_root_candidates,
            epsilon=epsilon,
            strict_acyclic=strict_acyclic,
            disable_structured_sparsity=disable_structured_sparsity,
            in_flow_edge_type=in_flow_edge_type,
            out_flow_edge_type=out_flow_edge_type,
            lambda_reg=lambda_reg,
            force_flow_through_root=force_flow_through_root,
            backend=backend,
        )

    def build(
        self,
        graph,
        data=None,
        *,
        terminals=None,
        edge_costs=None,
    ):
        """Build a single-condition Steiner tree from explicit inputs."""
        old_data = legacy_data(data, method=self.__class__.__name__)
        if old_data is not None:
            if terminals is not None or edge_costs is not None:
                raise TypeError("Do not combine a Data object with explicit scientific inputs.")
            return self.build_from_data(graph, old_data)
        if terminals is None:
            raise TypeError("build() requires terminals=.")
        return self.build_many(
            graph,
            terminals={DEFAULT_CONDITION: terminals},
            edge_costs=None if edge_costs is None else {DEFAULT_CONDITION: edge_costs},
        )

    def build_many(self, graph, *, terminals, edge_costs=None):
        """Build multiple named Steiner tree conditions."""
        conditions = validate_condition_keys(terminals=terminals, edge_costs=edge_costs)
        features_by_condition = {}
        for condition in conditions:
            terminal_values = validate_vertex_collection(
                graph,
                terminals[condition],
                argument="terminals",
                condition=condition,
                required=True,
            )
            cost_values = validate_edge_costs(
                graph,
                {}
                if edge_costs is None
                else require_mapping(edge_costs[condition], argument="edge_costs", condition=condition),
                condition=condition,
            )
            features_by_condition[condition] = [
                {"id": identifier, "mapping": "vertex", "role": "terminal"} for identifier in terminal_values
            ] + [{"id": identifier, "mapping": "edge", "value": value} for identifier, value in cost_values.items()]
        return self.build_from_data(graph, data_from_features(features_by_condition))


def solve_steiner_tree(
    graph,
    *,
    terminals,
    edge_costs=None,
    root_vertex=None,
    engine: Literal["corneto", "steinerpy"] = "corneto",
    solver: Optional[str] = None,
    backend: Optional[Backend] = None,
    solver_kwargs: Optional[dict] = None,
):
    """Solve a single-condition Steiner tree using CORNETO or optional SteinerPy.

    Requires a simple directed or undirected graph and finite nonnegative
    edge costs (default 1). Directed graphs require ``root_vertex``; for
    undirected graphs it is an additional required vertex.

    Returns sorted original edge indices and the native solver result,
    or an empty array and ``None`` for a single required vertex.
    ``solver_kwargs`` go to the selected engine; ``backend`` is CORNETO-only.
    """
    if engine not in ("corneto", "steinerpy"):
        raise ValueError("engine must be 'corneto' or 'steinerpy'.")
    if engine == "steinerpy" and backend is not None:
        raise ValueError("backend is only supported with engine='corneto'.")
    if isinstance(terminals, Mapping):
        raise TypeError("terminals must be a collection of vertices, not a mapping of conditions or prizes.")
    terminals = validate_vertex_collection(
        graph, terminals, argument="terminals", condition=DEFAULT_CONDITION, required=True
    )
    costs = validate_edge_costs(
        graph,
        {} if edge_costs is None else require_mapping(edge_costs, argument="edge_costs"),
        condition=DEFAULT_CONDITION,
    )
    if any(cost < 0 for cost in costs.values()):
        raise ValueError("Ordinary Steiner trees require nonnegative edge costs.")

    edge_types = {attr.get_attr(Attr.EDGE_TYPE) for attr in graph.get_attr_edges()}
    if len(edge_types) > 1:
        raise ValueError("Ordinary Steiner trees do not support mixed edge types.")
    directed = EdgeType.DIRECTED in edge_types
    edge_indices = {}
    for index, (source, target) in graph.edges():
        if len(source) != 1 or len(target) != 1:
            raise ValueError("Ordinary Steiner trees do not support hyperedges or boundary edges.")
        u, v = next(iter(source)), next(iter(target))
        if u == v:
            raise ValueError("Ordinary Steiner trees do not support self-loops.")
        key = (u, v) if directed else frozenset((u, v))
        if key in edge_indices:
            raise ValueError("Ordinary Steiner trees do not support parallel edges.")
        edge_indices[key] = index

    if directed and root_vertex is None:
        raise ValueError("Directed Steiner trees require an explicit root_vertex.")
    root = terminals[0] if root_vertex is None else root_vertex
    if root not in graph.V:
        raise ValueError(f"Root vertex {root!r} is not present in the graph.")
    if root not in terminals:
        terminals.append(root)
    if not set(terminals).issubset(graph.bfs([root])):
        raise ValueError("No solution found: terminals are unreachable from the root.")
    if len(terminals) == 1:
        return np.array([], dtype=int), None

    if engine == "corneto":
        method = SteinerTreeFlow(root_vertex=root, disable_structured_sparsity=True, backend=backend)
        result = method.build(graph, terminals=terminals, edge_costs=costs)
        result.solve(solver=solver, **(solver_kwargs or {}))
        values = result.expr.with_flow.value
        if values is None or not np.all(np.isfinite(values)):
            raise ValueError("No solution found.")
        selected = np.flatnonzero(np.asarray(values).reshape(-1) > 0.5)
    else:
        try:
            from steinerpy import Problem
        except ImportError as exc:
            raise ImportError("SteinerPy is required; install it with 'pip install corneto[steinerpy]'.") from exc

        from corneto.contrib.networkx import corneto_graph_to_networkx

        network = corneto_graph_to_networkx(graph, copy_attributes=False)
        network.add_nodes_from(graph.V)
        for u, v in network.edges:
            key = (u, v) if directed else frozenset((u, v))
            network[u][v]["weight"] = costs.get(edge_indices[key], 1.0)
        problem = Problem(network, terminals=terminals, **({"root": root} if directed else {}))
        result = problem.get_solution(solver="highs" if solver is None else solver, **(solver_kwargs or {}))
        if not np.isfinite(result.objective):
            raise ValueError("No solution found.")
        selected = np.array(
            sorted(edge_indices[(u, v) if directed else frozenset((u, v))] for u, v in result.edges),
            dtype=int,
        )
    if not set(terminals).issubset(graph.edge_subgraph(selected).bfs([root])):
        raise ValueError("No solution found: selected edges do not connect all terminals from the root.")
    return selected, result
