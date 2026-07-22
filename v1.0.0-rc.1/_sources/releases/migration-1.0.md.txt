# Migrating to CORNETO 1.0

The documented guides, tutorials, and API index define the supported CORNETO
1.0 surface. Undocumented beta-era experiments and duplicate implementations
were removed instead of being carried as permanent compatibility aliases.

## Canonical imports

| Earlier import | CORNETO 1.0 import |
| --- | --- |
| `corneto.methods.future.CarnivalILP` | `corneto.methods.CarnivalILP` |
| `corneto.methods.future.CarnivalFlow` | `corneto.methods.CarnivalFlow` |
| `corneto.methods.future.MultiSampleFBA` | `corneto.methods.MultiSampleFBA` |
| `corneto.methods.future.MultiSampleIMAT` | `corneto.methods.MultiSampleIMAT` |
| `corneto.methods.future.PrizeCollectingSteinerTree` | `corneto.methods.PrizeCollectingSteinerTree` |
| `corneto.methods.future.SteinerTreeFlow` | `corneto.methods.SteinerTreeFlow` |
| `corneto.methods.signal.cellnopt_ilp` | `corneto.methods.signaling.cellnopt_ilp` |
| `corneto.K` or `corneto.ops` | `corneto.opt` |
| `corneto._ml.build_dagnn` | `corneto.ml.build_dagnn` |
| `corneto.methods.shortest_path.create_multisample_shortest_path` | `corneto.methods.create_multisample_shortest_path` |
| module-level graph serialization | `Graph.save` and `Graph.load` |

The old `future`, `signal`, `K`, and `ops` paths in this table remain warning
compatibility paths throughout 1.x. The private `_ml` path and duplicate graph
serialization functions do not have shims.

## Removed without compatibility shims

- The legacy `FBAProblem`/`fba_problem` formulation and the old
  `multicondition_imat` implementation.
- Undocumented beta-era method aliases such as `runVanillaCarnival`,
  `runInverseCarnival`, and `fast_carnival`.
- Duplicate private graph, data, I/O, NetworkX, and legacy implementation
  modules (`corneto._graph`, `corneto._data`, `corneto._io`, `corneto._nx`,
  `corneto._core`, and `corneto._legacy`).
- The British-spelling `corneto.methods.signalling` implementation and the old
  module-shaped `corneto.methods.signaling`; signaling methods now live in the
  `corneto.methods.signaling` package.
- Duplicate module-level graph serialization helpers from `corneto.io`.

## Multi-condition convention

Modern formulations accept a `corneto.Data` object containing one `Sample` per
condition. Features use `role="input"` or `role="output"` and
`mapping="vertex"` where appropriate. A method builds one condition axis while
sharing graph structure and any configured regularization across conditions.
Single-condition use is represented by a `Data` object with one sample.

## Choosing a CARNIVAL entry point

- `milp_carnival(graph, perturbations, measurements, ...)` is the supported
  simple single-condition ILP and accepts two mappings directly.
- `CarnivalILP(...).build(graph, data)` is the general ILP for one or many
  conditions and should be the default for new condition-aware workflows.
- `CarnivalFlow(...).build(graph, data)` is the flow-based multi-condition
  formulation for workflows that require explicit network flow structure.

## Alternative-solution sampling

The sampler used by the indexed tutorials remains supported:

```python
from corneto.methods.sampler import sample_alternative_solutions
```
