# Sachs continuous cytometry data

`cytometryContinuous_with_intervention.tsv.tar.gz` contains the tab-separated
continuous measurements used by the Sachs tutorial. The archive contains one
file, `cytometryContinuous_with_intervention.tsv`, with these columns:

- `intervention_label`: `observational`, `pkc`, `raf`, `erk`, `pip2`, or `pka`;
- `raf`, `mek`, `plc`, `pip2`, `pip3`, `erk`, `akt`, `pka`, `pkc`, `p38`, and
  `jnk`: measured signaling variables.

The archive's `intervention_label` is a legacy label and is not always the
measured intervention target. `condition_manifest.csv` preserves the original
condition and records the corrected measured target: Akt inhibitor → `akt`,
Psitectorigenin → `pip2`, U0126 → `mek`, G0076/PMA → `pkc`, and β2cAMP →
`pka`. CD3/CD28, ICAM-2, and LY294002 remain no-target rows in the tutorial;
the latter acts through unmeasured PI3K.

The tutorial supports two encodings: hard interventions mark the measured
target with `intervened=True`, while additive shift interventions mark it with
`intervention="shift"` and use the original condition as
`intervention_group`. The measurements are natural-log transformed, so fitted
shifts are log-scale offsets rather than calibrated treatment intensities.

The archive was carried forward unchanged from the previous CORNETO Sachs
experiment. SHA-256:

```text
fb8ea1ce445c53c5ba194fbf182606b27408b2aed4d11a875d36048c4ab7bece
```

The data originate from Sachs et al. (2005), “Causal protein-signaling
networks derived from multiparameter single-cell data,” *Science* 308(5721),
523–529. Confirm the distribution terms before redistributing this archive
outside the repository.
