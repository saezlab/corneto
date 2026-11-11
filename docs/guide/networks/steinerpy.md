# SteinerPy

Install [SteinerPy](https://github.com/berendmarkhorst/SteinerPy) to solve
ordinary single-condition Steiner trees:

```bash
pip install 'corneto[steinerpy]'
```

```python
import corneto as cn
from corneto.methods import solve_steiner_tree

graph = cn.Graph()
graph.add_edge("A", "B", type=cn.EdgeType.UNDIRECTED)
graph.add_edge("B", "C", type=cn.EdgeType.UNDIRECTED)
graph.add_edge("A", "C", type=cn.EdgeType.UNDIRECTED)
edges, result = solve_steiner_tree(
    graph, terminals=["A", "C"], edge_costs={0: 1, 1: 1, 2: 4}, engine="steinerpy",
)
tree = graph.edge_subgraph(edges)  # Original edges 0 and 1.
```

Returns original edge indices and the native solver result. Set `engine="corneto"`
(the default) to use CORNETO's flow formulation with the same inputs.

Requires a simple directed or undirected graph and finite nonnegative costs.
Missing costs default to 1; graph weight attributes are ignored. Directed graphs
require `root_vertex`. Use `SteinerTreeFlow` or `PrizeCollectingSteinerTree`
directly for richer models.

For SteinerPy heuristics, pass `solver_kwargs={"exact": False}`. Heuristic or
time-limited results may be nonoptimal; inspect `result.gap`.
