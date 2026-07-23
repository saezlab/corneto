# PHONEMeS

`PHONEMeS` identifies a directed, acyclic signaling subnetwork that connects
known perturbation targets to phosphosites supported by an experiment. Typical
inputs are:

- a directed prior-knowledge network (PKN);
- one or more perturbed proteins, such as drug targets or inhibited kinases;
- a score for each measured phosphosite.

CORNETO expects the PKN and scores to be prepared beforehand. It does not
perform differential-expression analysis, map phosphosites to proteins, or
construct the PKN.

## Understanding phosphosite scores

PHONEMeS minimizes an objective. Consequently, the sign of a phosphosite score
has the following meaning:

| Score | Effect on the inferred network |
| --- | --- |
| Negative | Rewards inclusion of the phosphosite |
| Positive | Penalizes inclusion of the phosphosite |
| Zero | Marks the phosphosite as measured without favoring or penalizing it |

The magnitude controls the strength of that preference. For example, selecting
a site with score `-3.0` improves the objective more than selecting one with
score `-0.5`, provided that the required network connections do not cost too
much.

These signs are **optimization preferences**, not activity labels. A negative
score does not by itself mean that a phosphosite is activated, and a positive
score does not mean that it is inhibited. The preprocessing procedure used to
produce the scores determines their experimental interpretation.

Every dictionary key identifies a measured phosphosite, including keys whose
value is `0.0`. A positively scored site may still appear when it is needed as
an intermediate node on a better overall path.

## Inferring a network for one condition

```python
from corneto.methods import PHONEMeS

method = PHONEMeS()
problem = method.build(
    pkn,
    perturbations=["EGFR", "ERBB2"],
    phosphosite_scores={
        "ERK1_S123": -2.4,
        "AKT1_S473": 1.1,
        "JUN_S63": 0.0,
    },
)
problem.solve()
```

`perturbations` must contain at least one vertex from the PKN. Each
perturbation is treated as a required starting point in the inferred network.

`phosphosite_scores` must contain at least one measured PKN vertex. The method
chooses a connected explanation by balancing the phosphosite scores against
the cost of including interactions.

## Controlling network size with edge costs

By default, every interaction has the small positive cost specified by
`default_edge_cost`:

```python
method = PHONEMeS(default_edge_cost=1e-5)
```

Positive edge costs favor smaller subnetworks. Larger costs require stronger
negative phosphosite scores to justify additional interactions, so score and
edge-cost scales should be chosen together.

Individual interactions can be assigned different costs by their zero-based
index in `pkn.E`:

```python
problem = method.build(
    pkn,
    perturbations=["EGFR"],
    phosphosite_scores={"ERK1_S123": -2.4},
    edge_costs={
        3: 0.01,
        8: 0.2,
    },
)
```

Unspecified interactions retain `default_edge_cost`. Negative edge costs are
allowed and reward inclusion of an interaction, but should be used
deliberately: they can make the model include additional valid interactions
even when those interactions are not needed to explain a negatively scored
phosphosite.

## Reading the inferred subnetwork

For a single condition, `edge_selected` and `vertex_selected` are binary
solution arrays with one column:

```python
import numpy as np

selected_edge_mask = problem.expr.edge_selected.value[:, 0] > 0.5
selected_vertex_mask = problem.expr.vertex_selected.value[:, 0] > 0.5

selected_edge_indices = np.flatnonzero(selected_edge_mask)
selected_vertices = np.asarray(pkn.V, dtype=object)[selected_vertex_mask]
inferred_network = pkn.edge_subgraph(selected_edge_indices)
```

`inferred_network` contains only original PKN interactions. It can be inspected
or plotted with the same graph operations used elsewhere in CORNETO.

Because several subnetworks may have the same optimum, different solvers or
solver settings can return different but equally good interaction sets.

## Comparing multiple conditions

Use `build_many` with matching named dictionaries:

```python
problem = method.build_many(
    pkn,
    perturbations={
        "control": ["EGFR"],
        "treated": ["ERBB2"],
    },
    phosphosite_scores={
        "control": {
            "ERK1_S123": -2.4,
            "AKT1_S473": 0.0,
        },
        "treated": {
            "ERK1_S123": 0.8,
            "AKT1_S473": -3.1,
        },
    },
)
problem.solve()
```

Each condition receives its own inferred node and edge selections. Columns
follow the input condition order:

```python
condition_names = tuple(method.processed_data.samples)

for column, condition in enumerate(condition_names):
    selected = problem.expr.edge_selected.value[:, column] > 0.5
    condition_network = pkn.edge_subgraph(np.flatnonzero(selected))
```

### How joint fitting shares interactions

PHONEMeS infers a separate subnetwork for each condition, but it penalizes the
size of their **combined network**. The combined network is the union of all
interactions used in at least one condition.

For example:

| Condition | Selected interactions |
| --- | --- |
| `control` | A → B, B → C |
| `treated` | B → C, B → D |
| Combined network | A → B, B → C, B → D |

The shared interaction B → C contributes its edge cost once, not once per
condition. The example therefore pays for three distinct interactions rather
than four condition-interaction occurrences.

Phosphosite scores remain condition-specific and contribute separately in each
condition. Only the interaction-cost term is shared. This encourages conditions
to reuse a common signaling backbone when it explains the data well, but it
does not require their inferred subnetworks to be identical.

Consequently, `edge_costs` describes one global cost per PKN interaction.
PHONEMeS does not support assigning one cost to an interaction in `control` and
a different cost to the same interaction in `treated`.

## Advanced: implementation in CORNETO

This section describes how the biological problem is represented using
CORNETO's optimization framework. It is not necessary for routine use.

### Mathematical formulation

For original PKN edges, let \(Z_{ec}\) indicate whether edge \(e\) is selected
in condition \(c\), and let \(X_{vc}\) indicate whether vertex \(v\) is
selected. The objective is

\[
\min \sum_{v,c} s_{vc}X_{vc} + \sum_e c_eY_e,
\qquad Y_e = \bigvee_c Z_{ec}.
\]

Here, \(s_{vc}\) is the phosphosite score, \(c_e\) is the interaction cost, and
\(Y_e\) indicates whether an interaction is used by at least one condition.

CORNETO implements connectivity with a nonnegative conserved flow.
Condition-specific boundary edges inject flow at perturbation targets and
extract it at measured phosphosites. These auxiliary edges are implementation
devices: they are not PKN interactions and are not included in
`edge_selected`, `edge_selected_any`, or the edge-cost term.

Sparse incidence matrices link selected interactions to their endpoint
vertices. Biological constraints require perturbation and internal nodes to
have a selected outgoing interaction, and measured and internal nodes to have
a selected incoming interaction. Layer variables exclude directed cycles while
still allowing convergent signaling, so an inferred node may have more than one
parent. All constraints are constructed as vectorized matrix expressions.

### Internal solution variables

The registered expressions are:

| Expression | Meaning |
| --- | --- |
| `problem.expr.edge_selected` | Original PKN interactions selected per condition |
| `problem.expr.vertex_selected` | Original PKN vertices selected per condition |
| `problem.expr.edge_selected_any` | Union of selected PKN interactions |
| `problem.expr.flow` | Internal connectivity flow on PKN and boundary edges |
| `problem.expr.dag_layer` | Internal acyclicity layer for each vertex |

Most analyses only need `edge_selected` and `vertex_selected`. The `flow` and
`dag_layer` values encode feasibility of the optimization model and should not
normally be interpreted as biological measurements or signaling strength.

### Advanced `Data` interface

Use `build_from_data` when inputs are already represented as CORNETO features:

```python
from corneto import Data
from corneto.methods import PHONEMeS

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

A vertex that is both perturbed and measured uses role
`"perturbation_phosphosite"`. The PHONEMeS model defines one global cost for
each PKN interaction. Therefore, when the advanced `Data` representation
repeats edge-cost features in multiple samples, those repeated values must
agree; they cannot be used to define condition-specific edge costs.

### Reference-data validation

The test suite includes a compact conversion of the MTOR inhibition example in
`inst/PHONEMeS_example/Cluster/data4cluster_3.RData` from the original
PHONEMeS-ILP repository. It contains a prepared 229-edge PKN, the `MTOR_HUMAN`
perturbation, and 17 phosphosite scores. The test checks the optimum, selected
interaction count, and reachability from the perturbation with both supported
backends. The converted fixture also records the reference commit and source
checksum.

This implementation covers the optimization core. Score preprocessing, PKN
construction, downsampling, and solver-specific solution pools remain separate
workflows. See the original
[PHONEMeS publication](https://doi.org/10.1038/ncomms9033) and
[reference implementation](https://github.com/saezlab/PHONEMeS-ILP).
