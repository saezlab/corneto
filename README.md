# CORNETO: An unified network learning framework from biological prior knowledge <img src="https://github.com/pablormier/resources/raw/main/images/logos/corneto-logo-512px.png" align="right" height="200" alt="logo">
<!-- badges: start -->
[![main](https://github.com/saezlab/corneto/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/saezlab/corneto/actions)
<!-- badges: end -->

CORNETO (CORe NETwork Optimization) is an unified network inference framework implemented in Python library that models biological network inference problems using convex and combinatorial optimization. It offers a comprehensive framework that facilitates the design and formulation of common optimization problems related to inference of biological networks from omics data. The library leverages domain-specific language frameworks, such as [CVXPY](https://www.cvxpy.org/index.html) or [PICOS](https://picos-api.gitlab.io/picos/), to translate high-level problem specifications in a clear manner and solves the problem using a wide range of supported free and commercial solvers.

> **NOTE**: This is an early preview of the library, which includes a very limited subset of methods for signalling, and an early version of the API to build optimization problems. We're currently working towards having a final version including additional and novel methods.

## Installation

The library will be uploaded to pypi once the API is stable. Meanwhile, it can be installed by downloading the wheel file from the repository. It's recommended to use also conda to create a new environment, although it's not mandatory.

### Recommended setup

CORNETO does not include any backend nor solver by default to avoid issues with architectures for which some of the required binaries are not available. The recommended setup for CORNETO requires CVXPY and Gurobi:

```bash
pip install corneto cvxpy gurobipy
```

Please note that **GUROBI is a commercial solver which offers free academic licenses**. If you have an academic email, this step is very easy to do in just few minutes: https://www.gurobi.com/features/academic-named-user-license/. 
Alternatively, it is possible to use CORNETO with any free solver, such as HIGHS, included in Scipy. To install CORNETO with support for HIGHs, you only need to install the latest version of scipy:

```bash
pip install corneto cvxpy scipy
```

> :warning: Please note that without any backend, you can't do much with CORNETO. There are two supported backends right now: [PICOS](https://picos-api.gitlab.io/picos/tutorial.html) and [CVXPY](https://www.cvxpy.org/). Both backends allow symbolic manipulation of expressions in matrix notation. 



## Acknowledgements

CORNETO is developed at the [Institute for Computational Biomedicine](https://saezlab.org) (Heidelberg University). The development of this project is supported by European Union's Horizon 2020 Programme under
PerMedCoE project ([permedcoe.eu](https://permedcoe.eu/)) agreement no. 951773.

<img src="https://raw.githubusercontent.com/saezlab/.github/main/profile/logos/saezlab.png" alt="Saez lab logo" height="64px" style="height:64px; width:auto"> <img src="https://lcsb-biocore.github.io/COBREXA.jl/stable/assets/permedcoe.svg" alt="PerMedCoE logo" height="64px" style="height:64px; width:auto"> <img src="https://www.klinikum.uni-heidelberg.de/typo3conf/ext/site_ukhd/Resources/Public/Images/Logo_ukhd_de.svg" alt="UKHD logo" height="64px" style="height:64px; width:auto">  

