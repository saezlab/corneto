# CORNETO

CORNETO (CORe NETwork Optimization) is a first step towards unification of ILP based problems (steady state analysis of networks) with the long term goal of unifying also signaling and constraint-based modeling of metabolism.

CORNETO will translate the specification of a high level problem into an (I)LP formulation through different backends (e.g Python-MIP, PICOS and CVXPY) which are in charge of implementing specific backends for different free/commercial solvers in a transparent way. Different methods like CARNIVAL, CellNopt-ILP, Phonemes, etc could be reimplemented in a simple way on top of MIOM, abstracting away all the low level details of ILP formulations.


## Installation

The library will be uploaded to pypi once the API is stable. Meanwhile, it can be installed by downloading the wheel file from the repository. It's recommended to use also conda to create a new environment, although it's not mandatory.

### Minimal setup

With the minimal dependencies, corneto can be used with CVXPY and the Coin-OR CBC solver:

```bash
conda create --name corneto python=3.8
conda activate corneto
pip install corneto-0.9.0a0-py3-none-any.whl
```

To test if the solver is correctly installed, test the following command on the new environment:

```bash
conda activate corneto
python -c "import cvxpy; print(cvxpy.installed_solvers())"
```

The `CBC` solver should appear on the list.

### Complete setup

A recommended setup for testing and generating plots with corneto requires additional dependencies. A recommended environment is created with the following command:

```bash
conda create --name corneto python=3.8 matplotlib=3.5.1 networkx=2.7.1 pandas=1.4.3 jupyter=1.0.0 pydot=1.4.1 graphviz=2.50.0
conda activate corneto
pip install corneto-0.9.0a0-py3-none-any.whl
```

Now the notebook included in the `tests` folder should be able to be run:

```bash
cd corneto
jupyter nbconvert --execute --to html tests/notebooks/tutorial.ipynb
```

### Aditional solvers

Please see the CVXPY documentation for more information on how to install other solvers https://www.cvxpy.org/install/. Depending on the solver and the platform, you will need to take different actions. For example, for using Gurobi, you will need to install the Gurobi solver with a valid license and then installing `gurobipy` dependency on the environment.