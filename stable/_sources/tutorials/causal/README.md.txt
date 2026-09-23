# Running the LinearDAG Sachs tutorial

This tutorial uses `LinearDAGDiscovery` to find a network that helps explain
single-cell signaling measurements from the Sachs study. It includes two
general-stimulation conditions and six conditions with an annotated treatment
target. The notebook selects 100 cells from each condition for a practical run.

The notebook demonstrates:

- inspect the measured signals and treatment conditions;
- map treatments to their measured targets;
- fit a network using hard-intervention annotations;
- read the selected arrows and their coefficients; and
- fit an alternative model with estimated treatment offsets.

The [`LinearDAGDiscovery` guide](../../guide/causal/linear-dag-discovery.ipynb)
covers the method's other options and held-out prediction. The notebook uses
GUROBI, so running its fits requires a working Gurobi license.

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

See `datasets/sachs/v1/README.md` for the transformation, citation, and
attribution, and `datasets/sachs/v1/condition_manifest.csv` for the condition
to target mapping.
