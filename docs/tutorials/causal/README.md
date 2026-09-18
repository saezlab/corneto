# Linear DAG discovery on the Sachs dataset

This tutorial applies CORNETO's `LinearDAGDiscovery` method to the continuous
Sachs protein-signaling dataset. It uses a deterministic, small stratified
sample so that the notebook is practical to run locally, while preserving all
observational and intervention labels present in the dataset.

The notebook demonstrates:

- preserving the original Sachs experimental conditions and correcting the
  legacy archive labels for measured intervention targets;
- representing those targets as hard interventions and additive shift
  interventions;
- fitting a sparse, acyclic linear structural model to all conditions;
- reading fitted coefficients and inferred signs from the solution graph;
- comparing exact coefficient support with structural support; and
- inspecting prediction error and intervention-to-response flow usage;
- comparing fixed hard-target semantics with condition-level shift estimates;
- distinguishing in-sample equation-fit diagnostics from held-out prediction.

When a shift value is supplied, it is treated as known. When the value is
omitted and an intervention group is supplied, `LinearDAGDiscovery` estimates
one bounded additive offset per target/group. The Sachs archive is natural-log
transformed, so these shifts are log-scale offsets. The tutorial uses the
original reagent condition as the group; the estimates are not calibrated
drug intensities or dose-response slopes.

## Run the tutorial

From the CORNETO repository root, install the tutorial environment and the
checked-out CORNETO source:

```bash
cd docs/tutorials/causal
pixi install
pixi run python -m pip install -e ../../..
mkdir -p build
pixi run python -m papermill linear-dag-discovery-sachs.ipynb build/linear-dag-discovery-sachs.ipynb
```

From the repository's notebook runner, the equivalent command is:

```bash
python docs/tutorials/run_notebooks.py causal \
  --editable-corneto --corneto-root .
```

The runner writes executed notebooks to `build/` unless `--rewrite` is used.

## Dataset provenance

The bundled archive is the continuous Sachs cytometry dataset used by the
earlier CORNETO Sachs experiment. It contains 7,466 observations of 11
signaling proteins and phospholipids, together with an `intervention_label`
column. The original study is:

> Sachs, K. et al. (2005). Causal protein-signaling networks derived from
> multiparameter single-cell data. *Science*, 308(5721), 523–529.

The archive is kept unchanged from that earlier experiment. Its SHA-256 is:

```text
fb8ea1ce445c53c5ba194fbf182606b27408b2aed4d11a875d36048c4ab7bece
```

See `data/sachs/README.md` and `data/sachs/condition_manifest.csv` for the
file-level details, original-condition mapping, and corrected measured targets.
