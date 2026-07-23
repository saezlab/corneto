# Building method problems from explicit inputs

CORNETO methods accept ordinary Python mappings and collections for their
common scientific inputs. You do not need to construct `Data`, `Sample`, or
`Feature` objects for standard workflows.

## One condition

Use `build` with a positional network and keyword-only scientific inputs:

```python
from corneto.methods import CarnivalILP

problem = CarnivalILP().build(
    pkn,
    perturbations={"EGFR": 1},
    transcription_factors={"JUN": 1, "FOXO3": -1},
)
```

Metabolic and network methods expose inputs using their domain terminology:

```python
from corneto.methods import MultiSampleFBA, SteinerTreeFlow

fba_problem = MultiSampleFBA().build(
    model,
    objectives={"BIOMASS": -1},
    reaction_bounds={"EX_glc": (-10, 0)},
)

tree_problem = SteinerTreeFlow().build(
    graph,
    terminals=["EGFR", "JUN"],
    edge_costs={0: 1.5, 4: 0.8},
)
```

PHONEMeS uses perturbation targets and signed phosphosite scores. Because the
method minimizes its objective, negative scores favor including a site and
positive scores discourage it; zero-valued keys still identify measured sites:

```python
from corneto.methods import PHONEMeS

phonemes_problem = PHONEMeS().build(
    pkn,
    perturbations=["EGFR"],
    phosphosite_scores={"ERK1_S123": -2.4, "AKT1_S473": 0.0},
)
```

See the [PHONEMeS guide](networks/phonemes.md) for score interpretation, edge
costs, computing scores from differential results, optional pandas inputs, and
extracting the inferred subnetwork.

## Multiple named conditions

Use `build_many` and add one outer mapping keyed by condition name:

```python
problem = CarnivalILP().build_many(
    pkn,
    perturbations={
        "control": {"EGFR": 1},
        "treated": {"EGFR": -1},
    },
    transcription_factors={
        "control": {"JUN": 1},
        "treated": {"JUN": -1},
    },
)
```

All per-condition arguments must contain the same condition names. CORNETO
validates condition names, graph identifiers, numeric values, bounds, and edge
indices before constructing the optimization problem.

PHONEMeS also uses named mappings for `perturbations` and
`phosphosite_scores`. It infers one subnetwork per condition but charges edge
costs over their combined network: an interaction used by one or several
conditions is charged once. For this reason, `edge_costs` is one global mapping
rather than a mapping per condition.

## Advanced data interface

Use `build_from_data(graph, data)` when custom feature metadata or arbitrary
measurement roles require the general `Data` representation:

```python
problem = CarnivalILP().build_from_data(pkn, data)
```

Passing `Data` as the second positional argument to `build` remains temporarily
supported with a deprecation warning.
