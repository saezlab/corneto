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

CORNETO (Constrained Optimization for the Recovery of Networks from Omics) is a unified framework for multi-sample joint network inference, implemented in Python. It tackles common network inference problems in biology and extends them to support multiple samples or conditions simultaneously, enhancing network identification. The framework reformulates these problems using constrained optimization and mathematical programming, allowing them to be optimally solved with mathematical solvers. Additionally, it provides flexible modeling capabilities, enabling the exploration of hypotheses, modification, or development of new network inference problems through the use of modular constrained building blocks.

<p align="center">
  <img alt="CORNETO abstract" src="docs/_static/corneto-fig-abstract-v3.jpg" width="720" style="max-width: 100%; height: auto;">
</p>

## Installation

CORNETO provides flexible installation options depending on your needs:

### 🚀 Quick Start (Basic Installation)

For basic functionality with lightweight solvers:

```bash
pip install corneto
```

This installs the core CORNETO package with `cvxpy-base` and `scipy`.

⚠️ **Note**: Most CORNETO problems are MILPs. **For research problems, use `corneto[research]`** to install Gurobi solver for large-scale or multi-sample optimization problems. Please see the [Installing Gurobi solver](#installing-gurobi-solver) section for details.

### 📦 Installation with Extras

Choose the installation that best fits your use case:

#### **OS-level Solvers & Visualization** (`corneto[os]`)
For enhanced MILP solver support and basic plotting:

```bash
pip install corneto[os]
```

Includes: CVXPY (full), **SCIP**, **HiGHS**, NetworkX, Matplotlib, Graphviz

*Recommended for small-medium MILP problems*

#### **Research & Bioinformatics** (`corneto[research]`)
For biological research with commercial MILP solvers and specialized tools:

```bash
pip install corneto[research]
```

Includes: All OS extras + **Gurobi**, PICOS, COBRA, pandas, PCST-fast

⚠️ **Note**: Gurobi requires a [valid academic license](#installing-gurobi-solver)
🚀 **Recommended for large-scale MILP problems**

#### **Machine Learning** (`corneto[ml]`)
For ML applications, such as biologically-informed neural networks:

```bash
pip install corneto[ml]
```

Includes: CVXPY, JAX, Keras, scikit-learn, pandas

#### **Combined Installation**
You can combine multiple extras:

```bash
pip install corneto[os,ml]          # OS solvers + ML tools
pip install corneto[research,ml]    # Full research + ML stack
```

### 🐍 Conda Users

For Graphviz visualization support, users should install `python-graphviz` via conda instead of pip, to make sure that the Graphviz executables are installed and available in the PATH:

```bash
conda install python-graphviz  # Instead of pip install graphviz
pip install corneto[research]   # Then install corneto with other deps
```

### 🔧 Development Installation

To install the latest development version:

```bash
git clone https://github.com/saezlab/corneto.git
cd corneto
pip install -e .[research,ml]  # Install with desired extras
```

### 📚 Legacy Compatibility

The stable version used by [LIANA+](https://liana-py.readthedocs.io/) and [NetworkCommons](https://networkcommons.readthedocs.io/) remains available. However, we recommend using the latest version for new projects to access the latest features and improvements described in our manuscript.

### Installing Gurobi solver

CORNETO supports many different mathematical solvers for optimization. However, **for research problems, we strongly recommend GUROBI**.  **GUROBI is a commercial solver which offers free academic licenses**. If you have an academic email, getting a license is very quick - just a few minutes! Follow these steps:

1. Request the ["Academic Named-User License"](https://www.gurobi.com/features/academic-named-user-license/).
2. Register the license in your machine with the `grbgetkey` tool from GUROBI. For this, download the corresponding [license tool for your system](https://support.gurobi.com/hc/en-us/articles/360059842732-How-do-I-set-up-a-license-without-installing-the-full-Gurobi-package)
3. Run the `grbgetkey` tool and introduce your license key.

#### ✅ Verify Your Gurobi Installation

After installing Gurobi, verify everything works correctly:

```python
from corneto.utils import check_gurobi
check_gurobi()
```

This will test your Gurobi installation and license. You should see:
```
Gurobipy successfully imported.
Gurobi environment started successfully.
Starting optimization of the test model...
Test optimization was successful.
Gurobi environment disposed.
Gurobi is correctly installed and working.
```

If you find any issue, please check [this article](https://support.gurobi.com/hc/en-us/articles/13207658935185-How-do-I-retrieve-an-Academic-Named-User-license)

Please note that other high performance solvers like CPLEX, COPT, Mosek, etc are also supported. Please check the solver tables for supported backends:
- **CVXPY backend**: [CVXPY solver table](https://www.cvxpy.org/tutorial/solvers/index.html)
- **PICOS backend**: [PICOS solver table](https://picos-api.gitlab.io/picos/introduction.html#features)

### 🔍 MILP Solver Availability by Installation Tier

Most CORNETO problems are **Mixed-Integer Linear Programs (MILPs)**, which require specialized solvers. Here's what's available with each installation:

| Installation Tier | Included MILP Solvers | Solver Quality | Use Case |
|---|---|---|---|
| **Basic** (`pip install corneto`) | HiGHS (via scipy) | ⚠️ Limited | Toy problems, LP-only |
| **OS** (`corneto[os]`) | HiGHS, SCIP | ✅ Good | Small-medium MILPs |
| **Research** (`corneto[research]`) | All above + **Gurobi** | 🚀 Excellent | Large-scale research |

**Additional solvers available via CVXPY backend** (not recommended for MILPs):
- **CBC**: Install `cylp` package - basic MILP solver for small instances
- **GLPK**: Install `swiglpk` package - simple LP/MILP solver

```python
# Use specific MILP solvers explicitly
import corneto as cn

# Create and solve a MILP problem
problem = cn.Problem()
# ... define your problem ...
problem.solve(solver="GUROBI")   # Best for research (requires license)
problem.solve(solver="SCIP")     # Good open-source alternative
problem.solve(solver="HIGHS")    # Available in all installations
```

**Important**: If you have `gurobipy` installed but no valid license, specify an open-source solver explicitly to avoid license errors.

**🎯 Our Recommendation**: For research problems, use `corneto[research]` with Gurobi. It consistently outperforms open-source alternatives on large MILP instances and offers excellent academic licensing.

For complete lists of supported solvers by backend, see:
- **CVXPY backend**: [CVXPY solver table](https://www.cvxpy.org/tutorial/solvers/index.html)
- **PICOS backend**: [PICOS solver table](https://picos-api.gitlab.io/picos/introduction.html#features)

## Experiments

Notebooks with the experiments presented in the manuscript are available here: https://github.com/saezlab/corneto-manuscript-experiments

## How to cite

```
@article {Rodriguez-Mier2024,
	author = {Rodriguez-Mier, Pablo and Garrido-Rodriguez, Martin and Gabor, Attila and Saez-Rodriguez, Julio},
	title = {Unified knowledge-driven network inference from omics data},
	elocation-id = {2024.10.26.620390},
	year = {2024},
	doi = {10.1101/2024.10.26.620390},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/10/29/2024.10.26.620390},
	eprint = {https://www.biorxiv.org/content/early/2024/10/29/2024.10.26.620390.full.pdf},
	journal = {bioRxiv}
}
```

## Acknowledgements

CORNETO is developed at the [Institute for Computational Biomedicine](https://saezlab.org) (Heidelberg University). We acknowledge funding from the European Unions Horizon 2020 Programme under the grant agreement No 951773 (PerMedCoE https://permedcoe.eu/) and under grant agreement No 965193 (DECIDER https://www.deciderproject.eu/)

<div align="left">
  <img src="https://raw.githubusercontent.com/saezlab/.github/main/profile/logos/saezlab.png" alt="Saez lab logo" height="64px" style="margin: 0 20px;">
  <img src="https://yt3.googleusercontent.com/ytc/AIf8zZSHTQJs12aUZjHsVBpfFiRyrK6rbPwb-7VIxZQk=s176-c-k-c0x00ffffff-no-rj" alt="UKHD logo" height="64px" style="margin: 0 20px;">
  <img src="https://lcsb-biocore.github.io/COBREXA.jl/stable/assets/permedcoe.svg" alt="PerMedCoE logo" height="64px" style="margin: 0 20px;">
  <img src="https://raw.githubusercontent.com/saezlab/corneto/refs/heads/main/docs/_static/decider-eu-logo.png" alt="UKHD logo" height="64px" style="margin: 0 20px;">
</div>
