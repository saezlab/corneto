from typing import Any, List, Literal, Optional, Union

from corneto._constants import VAR_FLOW
from corneto._graph import EdgeType
from corneto.backend._base import Backend
from corneto.methods.future.pcst import PrizeCollectingSteinerTree


class SteinerTreeFlow(PrizeCollectingSteinerTree):
    """Basic Steiner Tree optimization method as a flow-based problem.

    This class implements the exact Steiner tree optimization method where, given a graph and a
    set of terminal nodes, it finds a minimal-weight connected subgraph (tree) that
    spans all terminals.
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
