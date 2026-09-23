# Metabolism

CORNETO brings constraint-based metabolic modeling into the same optimization
framework used for other network-inference problems. It interoperates with
COBRApy models and supports standard flux balance analysis while exposing the
formulation for custom constraints, coupled conditions, shared sparsity, and
omics integration.

Start with standard FBA and model interoperability, continue with
multi-condition FBA and shared reaction selection, and then integrate
expression evidence first in one condition and later across several conditions.

```{toctree}
:maxdepth: 3

flux-balance-analysis.ipynb
multicondition-sfba.ipynb
imat.ipynb
multicondition-imat.ipynb
```
