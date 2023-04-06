# CORNETO: Core Network Optimization library <img src="https://github.com/pablormier/resources/raw/main/images/logos/corneto-logo-512px.png" align="right" height="200" alt="logo">
<!-- badges: start -->
[![main](https://github.com/saezlab/corneto/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/saezlab/corneto/actions)
<!-- badges: end -->

CORNETO (CORe NETwork Optimization) is a Python library that models biological network inference problems using convex and combinatorial optimization. It offers a comprehensive framework that facilitates the design and formulation of common optimization problems related to inference of biological networks from omics data. The library leverages domain-specific language frameworks, such as [CVXPY](https://www.cvxpy.org/index.html) or [PICOS](https://picos-api.gitlab.io/picos/), to translate high-level problem specifications in a clear manner and solves the problem using a wide range of supported free and commercial solvers.

## Installation

The library will be uploaded to pypi once the API is stable. Meanwhile, it can be installed by downloading the wheel file from the repository. It's recommended to use also conda to create a new environment, although it's not mandatory.

### Minimal setup

CORNETO does not include any backend nor solver by default to avoid issues with architectures for which some of the required binaries are not available. To install only the CORNETO API, just type:

```bash
conda create --name corneto python
conda activate corneto
pip install git+https://github.com/saezlab/corneto.git@0.9.1-alpha.0
```

Alternatively you can download the wheel (.whl) file from https://github.com/saezlab/corneto/releases and install it with `pip install file.whl`. 

> :warning: Please note that without any backend, you can't do much with CORNETO. There are two supported backends right now: [PICOS](https://picos-api.gitlab.io/picos/tutorial.html) and [CVXPY](https://www.cvxpy.org/). Both backends allow symbolic manipulation of expressions in matrix notation. 


#### CVXPY backend

CVXPY supports many solvers, including the open-source Coin-OR CBC solver through the `cylp` package. If you want to have support for solving problems both for non-commercial and commercial projects, you can use CORNETO with `cvxpy` and `cylp`:

```bash
conda activate corneto
pip install cvxpy cylp
```

To test if the solver is correctly installed, test the following command on the new environment:

```bash
conda activate corneto
python -c "import corneto; corneto.info()"
```

The CBC solver should appear on the list. Please see the CVXPY documentation for more information on how to install other solvers https://www.cvxpy.org/install/. Depending on the solver and the platform, you will need to take different actions. For example, for using Gurobi, you will need to install the Gurobi solver with a valid license and then installing `gurobipy` dependency on the environment:

```bash
conda activate corneto
pip install gurobipy
python -c "import corneto; corneto.info()"
```

#### Apple M1

CORNETO can be installed on M1 architecture but may need few extra steps. Suggested configuration is to use `PICOS` backend with `gurobipy`. Installing `PICOS` with pip requires `CVXOPT`, that can be compiled for `M1`, see https://cvxopt.org/install/:

```bash
brew install cmake
brew install suite-sparse
CVXOPT_SUITESPARSE_LIB_DIR=/opt/homebrew/Cellar/suite-sparse/VERSION/lib/ CVXOPT_SUITESPARSE_INC_DIR=/opt/homebrew/Cellar/suite-sparse/VERSION/include/ pip install cvxopt
pip install PICOS gurobipy
pip install corneto-{version}.whl
```

In order to properly install `brew` on M1, follow this guide: https://mac.install.guide/homebrew/index.html

Additionally, for plotting you will need to get also `graphviz` with `homebrew`:

```bash
brew install graphviz
```

### Recommended environment

A recommended setup for testing and generating plots with corneto requires additional dependencies. A recommended environment is created with the following command:

```bash
conda create --name corneto python=3.9 cylp matplotlib jupyter graphviz
conda activate corneto
pip install cvxpy gurobipy git+https://github.com/saezlab/corneto.git@0.9.1-alpha.0
python -c "import corneto; corneto.info()"
```



## Acknowledgements

CORNETO is developed at the [Institute for Computational Biomedicine](https://saezlab.org) (Heidelberg University). The development of this project is supported by European Union's Horizon 2020 Programme under
PerMedCoE project ([permedcoe.eu](https://permedcoe.eu/)) agreement no. 951773.

<img src="https://raw.githubusercontent.com/saezlab/.github/main/profile/logos/saezlab.png" alt="Saez lab logo" height="64px" style="height:64px; width:auto"> <img src="https://lcsb-biocore.github.io/COBREXA.jl/stable/assets/permedcoe.svg" alt="PerMedCoE logo" height="64px" style="height:64px; width:auto"> <img src="https://www.klinikum.uni-heidelberg.de/typo3conf/ext/site_ukhd/Resources/Public/Images/Logo_ukhd_de.svg" alt="UKHD logo" height="64px" style="height:64px; width:auto">  

