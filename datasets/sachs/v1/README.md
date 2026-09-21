# Sachs continuous cytometry data (`v1`)

This version contains the continuous measurements used by the CORNETO Sachs
tutorial and the condition manifest used to map legacy intervention labels to
measured targets. The measurements are natural-log transformed.

The files are intentionally kept outside the `corneto` Python package. A
cloned repository can use them directly; installed users can obtain the same
version through `corneto.datasets.fetch_dataset("sachs")`.

The data originate from Sachs et al. (2005), “Causal protein-signaling
networks derived from multiparameter single-cell data,” *Science* 308(5721),
523–529. Confirm the distribution terms before redistributing these data
outside the repository.
