<style>
    .prev-next-footer {
        display: none;
    }
</style>


# Install

CORNETO provides flexible installation options depending on your needs. You can install it with different mathematical solvers, each offering different capabilities and performance characteristics.

## Quick Installation

For a quick start with open-source solvers:

```bash
pip install git+https://github.com/saezlab/corneto.git@dev pyscipopt highspy
```

## Installation Options

### Basic installation (dev)

The basic installation includes the core functionality of CORNETO:

```bash
pip install git+https://github.com/saezlab/corneto.git@dev
```

### With commercial solvers

If you have access to commercial solvers like Gurobi (free for academic use), you can enhance CORNETO's capabilities:

```bash
# First install CORNETO
pip install git+https://github.com/saezlab/corneto.git@dev

# Then install Gurobi
pip install gurobipy
```

### With open source solvers

For open-source alternatives, you can use HIGHs and SCIP:

```bash
# Install CORNETO with HIGHs and SCIP
pip install git+https://github.com/saezlab/corneto.git@dev pyscipopt highspy
```

## Verifying installation

You can verify your installation by running Python and importing CORNETO:

```python
import corneto

corneto.info()
```

## Requirements

CORNETO requires:

- Python 3.10 or higher
- Compatible mathematical solver (HIGHs, SCIP, or Gurobi)
- The `cvxpy-base` or `cvxpy` backend

## Troubleshooting

If you encounter any issues during installation:

1. Ensure you have Python 3.10 or higher installed
2. Check if you have the required system dependencies for the solvers
3. For Gurobi users, verify that your license is properly configured

For more detailed troubleshooting, please visit our [GitHub Issues](https://github.com/saezlab/corneto/issues) page.