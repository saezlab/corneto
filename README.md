<div align="center">
<img alt="corneto logo" src="docs/_static/logo/corneto-logo-512px.png" height="200"/>
<br>
<h3>CORNETO: Unified knowledge-driven network inference from omics data.</h3>


<h4>

[Preprint](https://doi.org/10.1101/2024.10.26.620390) | [Documentation](https://saezlab.github.io/corneto/dev) | [Notebooks](https://saezlab.github.io/corneto/dev/tutorials/index.html)

</h4>

<!-- badges: start -->
[![GitHub stars](https://img.shields.io/github/stars/saezlab/corneto)](https://github.com/saezlab/corneto/stargazers)
[![main](https://github.com/saezlab/corneto/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/saezlab/corneto/actions)
<!-- badges: end -->

</div>


---

CORNETO is a Python framework for inferring biological networks from omics data by framing them as optimisation problems. It provides a unified, flexible, and extensible API to solve a wide range of network inference problems, including signalling pathway and metabolic model reconstruction.

## Why CORNETO?

| Feature | Why it matters |
|---|---|
| 🧩 **Unified optimisation core** | Express causal signaling, FBA, PCSF & more with the same primitives. |
| 🎯 **Exact, solver-backed answers** | LP/MILP formulations guarantee optimality |
| 📊 **Multi-sample power** | Borrow strength across conditions for cleaner, comparable subnetworks. |
| 🔧 **Modular & extensible** | Plug-in new constraints, priors, or scoring functions in a few lines of code. |
| ⚡ **Blazing-fast** | Supports CVXPY and PICOS backends with dozens of solvers. |



## 🚀 Installation

### Recommended Installation

For most users, we recommend creating a conda environment and installing the research flavor:

```bash
conda create -n corneto python>=3.10
conda activate corneto
conda install python-graphviz
pip install corneto[research]
```

This installs CORNETO with all research dependencies including Gurobi, PICOS, and visualization libraries.

### Standard Installation

The minimal installation via pip provides core functionalities:

```bash
pip install corneto
```

### Optional dependencies

CORNETO provides several optional dependency groups:

- **`research`**: Full research stack with Gurobi, PICOS, visualization, and network tools
- **`os`**: Open-source solvers (SCIP, HiGHS) with visualization and network tools
- **`ml`**: Machine learning dependencies (JAX, Keras, scikit-learn)
- **`vanilla`**: Minimal dependencies (same as base installation)

Install any combination with:
```bash
pip install corneto[research,ml]  # Multiple extras
```

### Gurobi Installation

For research problems, we strongly recommend using the Gurobi solver. Gurobi is a commercial solver that offers free academic licences. To install and configure Gurobi, please refer to the [official Gurobi documentation](https://www.gurobi.com/documentation/). After installation, you can verify that Gurobi is correctly set up by running:

```python
from corneto.utils import check_gurobi
check_gurobi()
```

### Development Installation

If you plan to contribute to CORNETO, we recommend using [Poetry](https://python-poetry.org) for dependency management.

```bash
git clone https://github.com/saezlab/corneto.git
cd corneto
poetry install --with dev
```

### Legacy Compatibility

The stable version used by [LIANA+](https://liana-py.readthedocs.io/) and [NetworkCommons](https://networkcommons.readthedocs.io/) remains available. However, we recommend using the latest version for new projects to access the latest features and improvements described in our manuscript.

## 🧪 Experiments

Notebooks with the experiments presented in the manuscript are available here: https://github.com/saezlab/corneto-manuscript-experiments

## 🎓 How to cite

```
@article{Rodriguez-Mier2025,
	author = {Rodriguez-Mier, Pablo and Garrido-Rodriguez, Martin and Gabor, Attila and Saez-Rodriguez, Julio},
	title = {Unifying multi-sample network inference from prior knowledge and omics data with CORNETO},
	journal = {Nature Machine Intelligence},
	year = {2025},
	doi = {10.1038/s42256-025-01069-9},
	url = {https://www.nature.com/articles/s42256-025-01069-9}
}
```

## 🙏 Acknowledgements

CORNETO is developed at the [Institute for Computational Biomedicine](https://saezlab.org) (Heidelberg University). We acknowledge funding from the European Unions Horizon 2020 Programme under the grant agreement No 951773 (PerMedCoE https://permedcoe.eu/) and under grant agreement No 965193 (DECIDER https://www.deciderproject.eu/)

<div align="left">
  <img src="https://raw.githubusercontent.com/saezlab/.github/main/profile/logos/saezlab.png" alt="Saez lab logo" height="64px" style="margin: 0 20px;">
  <img src="https://yt3.googleusercontent.com/ytc/AIf8zZSHTQJs12aUZjHsVBpfFiRyrK6rbPwb-7VIxZQk=s176-c-k-c0x00ffffff-no-rj" alt="UKHD logo" height="64px" style="margin: 0 20px;">
  <img src="https://lcsb-biocore.github.io/COBREXA.jl/stable/assets/permedcoe.svg" alt="PerMedCoE logo" height="64px" style="margin: 0 20px;">
  <img src="https://raw.githubusercontent.com/saezlab/corneto/refs/heads/main/docs/_static/decider-eu-logo.png" alt="UKHD logo" height="64px" style="margin: 0 20px;">
</div>
