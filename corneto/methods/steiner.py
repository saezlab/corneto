"""Steiner tree optimization."""

from typing import Any, List, Literal, Optional, Union

from corneto._constants import VAR_FLOW
from corneto.backend._base import Backend
from corneto.graph import EdgeType
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
                {} if edge_costs is None else require_mapping(
                    edge_costs[condition], argument="edge_costs", condition=condition
                ),
                condition=condition,
            )
            features_by_condition[condition] = [
                {"id": identifier, "mapping": "vertex", "role": "terminal"}
                for identifier in terminal_values
            ] + [
                {"id": identifier, "mapping": "edge", "value": value}
                for identifier, value in cost_values.items()
            ]
        return self.build_from_data(graph, data_from_features(features_by_condition))
