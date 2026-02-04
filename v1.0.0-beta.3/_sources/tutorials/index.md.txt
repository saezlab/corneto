<style>
    .prev-next-footer {
        display: none;
    }
</style>

# Tutorials

```{toctree}
:maxdepth: 1

fba/context-specific-metabolic-omics.ipynb
carnival/single-sample-carnival-transcriptomics.ipynb
carnival/multi-receptor-integration.ipynb
carnival/network-sampler.ipynb
carnival/network-sampler-example.ipynb
ml/kpnn-with-sc.ipynb
```

## Additional Tutorials

```{toctree}
:maxdepth: 1

contrib/multi_condition_tutorial/multi_condition_carnival.ipynb
```

## Running tutorials locally

Each tutorial is self-contained and includes its own `pixi.toml`. To execute all notebooks via Pixi:

```bash
# Run all tutorials (outputs go to build/)
poetry run python docs/tutorials/run_notebooks.py

# Run a single tutorial by folder name
poetry run python docs/tutorials/run_notebooks.py ml

# Rewrite notebooks in place (overwrites original .ipynb files)
poetry run python docs/tutorials/run_notebooks.py ml --rewrite

# Use local CORNETO source instead of the pinned PyPI version
poetry run python docs/tutorials/run_notebooks.py ml --editable-corneto

# Point to a specific CORNETO checkout
poetry run python docs/tutorials/run_notebooks.py ml --editable-corneto --corneto-root /path/to/corneto
```
