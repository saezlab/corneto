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

### Stable version

A stable version of the lib, which is used by [LIANA+](https://liana-py.readthedocs.io/) and [NetworkCommons](https://networkcommons.readthedocs.io/), is already available at Pypi and can be installed with `pip install corneto`. 

Plase note that this version lacks many of the developments discussed in the manuscript. To use the latest features, please install the development version. 

You can install it along with CVXPY, Scipy (for open source solver support) and Gurobipy for using GUROBI solver:

```bash
pip install corneto cvxpy-base scipy gurobipy
```

### Development version

To install the development version with support for the Gurobi solver and plotting capabilities using graphviz, please use conda or mamba to create an environment:

```
conda activate your-environment
conda install -c conda-forge python-graphviz scipy
pip install cvxpy-base gurobipy
```

Then, install the dev version of CORNETO:
```
git clone -b dev https://github.com/saezlab/corneto.git
cd corneto
pip install -e .
```

### Installing Gurobi solver

CORNETO supports many different mathematical solvers for optimization. However, for real world problems, we typically use GUROBI.  **GUROBI is a commercial solver which offers free academic licenses**. If you have an academic email, this step is very easy to do in just few minutes. Follow these steps:

1. Request the ["Academic Named-User License"](https://www.gurobi.com/features/academic-named-user-license/).
2. Register the license in your machine with the `grbgetkey` tool from GUROBI. For this, download the corresponding [license tool for your system](https://support.gurobi.com/hc/en-us/articles/360059842732-How-do-I-set-up-a-license-without-installing-the-full-Gurobi-package)
3. Run the `grbgetkey` tool and introduce your license key.

If you find any issue, please check [this article](https://support.gurobi.com/hc/en-us/articles/13207658935185-How-do-I-retrieve-an-Academic-Named-User-license)

Please note that other high performance solvers like CPLEX, COPT, Mosek, etc are also supported. Please check the [solver's table](https://www.cvxpy.org/tutorial/solvers/index.html) to see which solvers are supported by the CVXPY backend. 

### Installing open-source solvers

Alternatively, it is possible to use CORNETO with any free solver, such as HIGHS, included in Scipy. For this you don't need to install Gurobi. Please note that if `gurobipy` is installed but not registered with a valid license, CORNETO will choose it but the solver will fail due to license issues. If SCIPY is installed, when optimizing a problem, select SCIPY as the solver

```python
# P is a corneto problem
P.solve(solver="SCIPY")
```

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
