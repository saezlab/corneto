# Sachs continuous cytometry data (`v1`)

This revision contains the continuous measurements used by the CORNETO Sachs
tutorial: 7,466 observations of 11 signaling proteins and phospholipids. The
measurements are natural-log transformed and the first column,
`intervention_label`, preserves the condition grouping used by the tutorial.

The files are intentionally kept outside the `corneto` Python package. A
cloned repository can use them directly; installed users can obtain the same
version through `corneto.datasets.fetch_dataset("sachs")`.

## Files

- `measurements.tsv` contains the transformed measurements and condition
  labels.
- `condition_manifest.csv` records the source condition, row interval, legacy
  archive label, and intervention target used by the tutorial.
- `README.md` documents the source, transformation, citation, and license.

## Provenance and transformation

The source is the Zenodo record
[Sachs: Protein and Phospholipids Expressions](https://doi.org/10.5281/zenodo.7681811),
version 2, specifically its `sachs.zip` file
([direct download](https://zenodo.org/records/7681811/files/sachs.zip)). The
source archive is identified by MD5
`d50ccd7f88bb3705bcf6eda3978bcc41`.

The CORNETO `v1` files are a deterministic derived representation of the nine
measured experimental conditions in that archive, concatenated in this order:
`cd3cd28`, `cd3cd28_icam2`, `cd3cd28_aktinhib`, `cd3cd28_g0076`,
`cd3cd28_psitect`, `cd3cd28_u0126`, `cd3cd28_ly`, `pma`, and `b2camp`.
For each condition, the 11 numeric source columns were transformed with the
natural logarithm. The source column `Plcg` is named `plc` in the CORNETO
file, and the `intervention_label` column plus the manifest were added for the
tutorial.

The five simulated conditions and `GroundTruth.csv` from the Zenodo archive
are not included here. This explains why the CORNETO file is not byte-for-byte
identical to the Zenodo ZIP. The previous CORNETO archive had no recorded
external source; comparison with the Zenodo files establishes the derivation
described above.

## Citation and license

Please cite both the data record and the original study:

> Mathematical Research Data Initiative (2023). *Sachs: Protein and
> Phospholipids Expressions*. Zenodo. https://doi.org/10.5281/zenodo.7681811

> Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P.
> (2005). Causal protein-signaling networks derived from multiparameter
> single-cell data. *Science*, 308(5721), 523–529.
> https://doi.org/10.1126/science.1105809

The Zenodo record identifies the data as [Creative Commons Attribution 4.0
International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
granted by G. Nolan and D. Lauffenburger. The derived CORNETO dataset is
distributed under the same license with the attribution and source links
above.
