# CORNETO

CORNETO (CORe NETwork Optimization) is a first step towards unification of ILP based problems (steady state analysis of networks) with the long term goal of unifying also signaling and constraint-based modeling of metabolism.

CORNETO will translate the specification of a high level problem into an (I)LP formulation through different backends (e.g Python-MIP, PICOS and CVXPY) which are in charge of implementing specific backends for different free/commercial solvers in a transparent way. Different methods like CARNIVAL, CellNopt-ILP, Phonemes, etc could be reimplemented in a simple way on top of it, abstracting away all the low level details of ILP formulations.


## Installation

The library will be uploaded to pypi once the API is stable. Meanwhile, it can be installed by downloading the wheel file from the repository. It's recommended to use also conda to create a new environment, although it's not mandatory.

### Minimal setup

By default, CORNETO does not include any backend nor solver, in order to avoid issues with architectures for which some of the required binaries are not available by default. To install only the CORNETO API, just type:

```bash
conda create --name corneto python=3.8
conda activate corneto
pip install git+https://github.com/saezlab/corneto.git@0.9.0-alpha.3
```

Alternatively you can download the wheel file from https://github.com/saezlab/corneto/releases/download/0.9.0-alpha.3/corneto-0.9.0a3-py3-none-any.whl and install it with `pip install corneto-0.9.0a3-py3-none-any.whl`. 

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
python -c "import cvxpy; print(cvxpy.installed_solvers())"
```

The CBC solver should appear on the list. Please see the CVXPY documentation for more information on how to install other solvers https://www.cvxpy.org/install/. Depending on the solver and the platform, you will need to take different actions. For example, for using Gurobi, you will need to install the Gurobi solver with a valid license and then installing `gurobipy` dependency on the environment:

```bash
conda activate corneto
pip install gurobipy
python -c "import cvxpy; print(cvxpy.installed_solvers())"
```

#### PICOS backend

PICOS backend requires less dependencies than CVXPY and it can be a very good option for Apple M1 users. If you want to use `PICOS` instead of `CVXPY` backend, first install the dependency in your environment with:

```bash
conda activate corneto
pip install PICOS
```

If you don't have CVXPY installed, CORNETO will select PICOS automatically. If you have both, you can just instantiate whatever you like:

```python
import numpy as np
from corneto.backend import PicosBackend

backend = PicosBackend()
P = backend.Problem()

n = 20
A = np.random.rand(2, n)
b = np.array([1, 0])
x = backend.Variable('x', n)
P += x
P += sum(x) == 1, x >= 0
# Convex optimization problem
P.add_objectives(abs(A*x - b))
P.solve(solver="cvxopt", verbosity=1)
```

Check `PICOS` requirements for different solvers here: https://picos-api.gitlab.io/picos/introduction.html#features

#### Apple M1

CORNETO can be installed on M1 architecture but may need few extra steps. Suggested configuration is to use `PICOS` backend with `gurobipy`. Installing `PICOS` with pip requires `CVXOPT`, that can be compiled for `M1`, see https://cvxopt.org/install/:

```bash
brew install cmake
brew install suite-sparse
CVXOPT_SUITESPARSE_LIB_DIR=/opt/homebrew/Cellar/suite-sparse/VERSION/lib/ CVXOPT_SUITESPARSE_INC_DIR=/opt/homebrew/Cellar/suite-sparse/VERSION/include/ pip install cvxopt
pip install PICOS gurobipy
pip install corneto-0.9.0a3-py3-none-any.whl
```

In order to properly install `brew` on M1, follow this guide: https://mac.install.guide/homebrew/index.html

Additionally, for plotting you will need to get also `graphviz` with `homebrew`:

```bash
brew install graphviz
```

TODO: In future releases, this will be automatized through `setup.py`.

### Complete setup

A recommended setup for testing and generating plots with corneto requires additional dependencies. A recommended environment is created with the following command:

```bash
conda create --name corneto python=3.8 cvxpy=1.2.1 cylp=0.91.5 matplotlib=3.5.1 networkx=2.7.1 pandas=1.4.3 jupyter=1.0.0 pydot=1.4.1 graphviz=2.50.0
conda activate corneto
pip install corneto-0.9.0a3-py3-none-any.whl
```

Now the notebook included in the `tests` folder should be able to be run:

```bash
cd corneto
jupyter nbconvert --execute --to html tests/notebooks/tutorial.ipynb
```
