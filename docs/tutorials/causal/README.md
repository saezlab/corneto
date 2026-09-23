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
- comparing exact coefficient support with structural support;
- inspecting in-sample equation-fit diagnostics and intervention-to-response flow usage; and
- comparing fixed hard-target semantics with condition-level shift estimates.

For leakage-safe held-out prediction with a fixed fitted model, see the
[`LinearDAGDiscovery` guide](../../guide/causal/linear-dag-discovery.ipynb).

When a shift value is supplied, it is treated as known. When the value is
omitted and an intervention group is supplied, `LinearDAGDiscovery` estimates
one bounded additive offset per target/group. The Sachs data are natural-log
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

The `v1` dataset contains 7,466 observations of 11 signaling proteins and
phospholipids, together with an `intervention_label` column. It is a
natural-log-transformed, tutorial-oriented representation of the nine measured
conditions in the [Zenodo Sachs dataset](https://doi.org/10.5281/zenodo.7681811)
(version 2); the simulated conditions and ground-truth file are not included.
The source record identifies the data as [CC BY
4.0](https://creativecommons.org/licenses/by/4.0/). The versioned dataset
files are under `datasets/sachs/v1/`; installed users can resolve the same
files with `corneto.datasets.fetch_dataset("sachs")`.

The original source archive is `sachs.zip`, with MD5
`d50ccd7f88bb3705bcf6eda3978bcc41`. The transformed local representation is
not byte-for-byte identical to that ZIP; see `datasets/sachs/v1/README.md` for
the exact condition order, transformation, citation, and attribution.

See `datasets/sachs/v1/README.md` and
`datasets/sachs/v1/condition_manifest.csv` for the file-level details,
original-condition mapping, and corrected measured targets.
