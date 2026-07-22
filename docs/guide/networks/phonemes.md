# PHONEMeS

`PHONEMeS` infers a directed acyclic signaling subnetwork connecting
perturbation targets to measured phosphosites. It expects a prepared directed
prior-knowledge network (PKN) and precomputed signed phosphosite scores.

## One condition

```python
from corneto.methods import PHONEMeS

method = PHONEMeS(default_edge_cost=1e-5)
problem = method.build(
    pkn,
    perturbations=["EGFR", "ERBB2"],
    phosphosite_scores={
        "ERK1_S123": -2.4,
        "AKT1_S473": 1.1,
        "JUN_S63": 0.0,
    },
    edge_costs={3: 0.01, 8: -0.001},
)
problem.solve()
```

The score is the coefficient of a binary node-selection variable and is
minimized directly. Negative values reward explaining a phosphosite; positive
values penalize its inclusion. Every key is considered measured and can extract
flow, including a site with score `0.0`.

`edge_costs` uses zero-based PKN edge indices. Unspecified interactions receive
`default_edge_cost`. Costs may be negative; PHONEMeS does not infer costs from
vertex naming conventions.

## Multiple conditions

```python
problem = method.build_many(
    pkn,
    perturbations={
        "control": ["EGFR"],
        "treated": ["ERBB2"],
    },
    phosphosite_scores={
        "control": {"ERK1_S123": -2.4, "AKT1_S473": 0.0},
        "treated": {"ERK1_S123": 0.8, "AKT1_S473": -3.1},
    },
    edge_costs={3: 0.01, 8: -0.001},
)
```

Each condition has its own flow, edge selections, and node selections. Edge
costs remain global: an interaction used in any number of conditions is charged
once through the union variable `edge_selected_any`.

## Formulation

For original PKN edges, let (Z_{ec}) indicate edge selection and (X_{vc})
indicate node selection. External boundary edges inject flow at every
perturbation and allow extraction at measured phosphosites. They participate in
flow conservation but not in the biological edge objective.

With sparse outgoing and incoming incidence matrices (T) and (H), CORNETO
constructs

\[
O = TZ, \qquad I = HZ.
\]

Matrix constraints link selected edges to their endpoints. Perturbation and
internal nodes require an outgoing PKN edge; measured and internal nodes require
an incoming PKN edge. Flow conservation connects every selected component to a
perturbation and a measured sink, while DAG-layer constraints exclude cycles.
Convergent signaling remains possible: a selected node may have several
parents.

The objective is

\[
\min \sum_{v,c} s_{vc}X_{vc} + \sum_e c_eY_e,
\qquad Y_e = \bigvee_c Z_{ec}.
\]

All node, edge, condition, and acyclicity constraints are constructed as
vectorized matrix expressions.

## Reference-data validation

The test suite includes a compact conversion of the MTOR inhibition example in
`inst/PHONEMeS_example/Cluster/data4cluster_3.RData` from the original
PHONEMeS-ILP repository. It contains the prepared 229-edge PKN, the
`MTOR_HUMAN` perturbation, and 17 phosphosite scores derived from the
`MTOR1 - Control` and `MTOR2 - Control` rows using the reference
`createSIF` and `buildDataMatrix` semantics.

The end-to-end test checks the optimum and selected-edge count with both
supported backends and verifies that every selected CORNETO edge is reachable
from the perturbation. The converted fixture records the reference commit and
source-file checksum, so its provenance is reproducible without requiring R or
loading multi-megabyte `.RData` files during testing. The expected result is
for CORNETO's conserved-flow formulation; it is not a cached solver output from
the reference R implementation.

## Reading the solution

After solving, the main registered expressions are:

| Expression | Meaning |
| --- | --- |
| `problem.expr.flow` | Flow on original and boundary edges by condition |
| `problem.expr.edge_selected` | Selected original PKN edges by condition |
| `problem.expr.vertex_selected` | Selected original PKN vertices by condition |
| `problem.expr.edge_selected_any` | Union of selected PKN edges |
| `problem.expr.dag_layer` | Topological layer assigned to each vertex |

The original PKN edges always occupy the first `pkn.num_edges` rows of `flow`.
The remaining rows are boundary inflow and outflow edges in
`method.processed_graph`.

## Advanced `Data` interface

Use `build_from_data` when inputs are already represented as CORNETO features:

```python
from corneto import Data

data = Data.from_dict(
    {
        "treated": {
            "features": [
                {"id": "EGFR", "mapping": "vertex", "role": "perturbation"},
                {
                    "id": "ERK1_S123",
                    "mapping": "vertex",
                    "role": "phosphosite",
                    "value": -2.4,
                },
                {"id": 3, "mapping": "edge", "value": 0.01},
            ]
        }
    }
)

problem = PHONEMeS().build_from_data(pkn, data)
```

A vertex that is both a target and measured uses role
`"perturbation_phosphosite"`. In multi-condition `Data`, edge features must be
identical across samples because their costs apply to the condition union.

This implementation covers the optimization core. Score preprocessing, PKN
construction, downsampling, and solver-specific solution pools remain separate
workflows. See the original [PHONEMeS publication](https://doi.org/10.1038/ncomms9033)
and [reference implementation](https://github.com/saezlab/PHONEMeS-ILP).
