
<style>
    .prev-next-footer {
        display: none;
    }
</style>

# Install

**CORNETO** provides flexible installation options depending on your needs. It integrates with various mathematical optimization solvers, which differ in their licensing, capabilities, and performance. Below are several installation options tailored to different use cases.

## 🚀 Recommended Installation

For most users, we recommend creating a conda environment and installing the research flavor:

```bash
conda create -n corneto python>=3.10
conda activate corneto
conda install python-graphviz
pip install corneto[research]
```

This installs CORNETO with all research dependencies including Gurobi, PICOS, and visualization libraries.

---

## 🧩 Installation Options

### Standard Installation

The minimal installation via pip provides core functionalities:

```bash
pip install corneto
```

### Optional Dependencies

CORNETO provides several optional dependency groups:

- **`research`**: Full research stack with Gurobi, PICOS, visualization, and network tools
- **`os`**: Open-source solvers (SCIP, HiGHS) with visualization and network tools
- **`ml`**: Machine learning dependencies (JAX, Keras, scikit-learn)
- **`vanilla`**: Minimal dependencies (same as base installation)

Install any combination with:
```bash
pip install corneto[research,ml]  # Multiple extras
```

---

### Gurobi Installation

For research problems, we strongly recommend using the Gurobi solver. Gurobi is a commercial solver that offers free academic licences. To install and configure Gurobi, please refer to the [official Gurobi documentation](https://www.gurobi.com/documentation/). After installation, you can verify that Gurobi is correctly set up by running:

```python
from corneto.utils import check_gurobi
check_gurobi()
```

---

### Development Installation

If you plan to contribute to CORNETO, we recommend using [Poetry](https://python-poetry.org) for dependency management.

```bash
git clone https://github.com/saezlab/corneto.git
cd corneto
poetry install --with dev
```

---

## ✅ Verifying Installation

To check that CORNETO is installed and ready:

```python
import corneto

corneto.info()
```

This should print out CORNETO version info and configuration details.

---

### Legacy Compatibility

The stable version used by [LIANA+](https://liana-py.readthedocs.io/) and [NetworkCommons](https://networkcommons.readthedocs.io/) remains available. However, we recommend using the latest version for new projects to access the latest features and improvements described in our manuscript.

---

## 🛠 Troubleshooting

If you run into installation problems:

1. ✅ Make sure you're using **Python 3.10 or higher**
2. 🧱 Check that you have at least one solver backend installed (CVXPY is recommended)
3. 🔑 For Gurobi users, verify that your **license is properly installed**

In case of using Gurobi, you can check if the license and installation are correct by running:

```python
from corneto.utils import check_gurobi
check_gurobi()
```

This will perform a series of checks to ensure that Gurobi is correctly installed and configured. You should see:

```plain
Gurobipy successfully imported.
Gurobi environment started successfully.
Starting optimization of the test model...
Test optimization was successful.
Gurobi environment disposed.
Gurobi is correctly installed and working.
```

If you're still stuck, please open an issue on our [GitHub Issues](https://github.com/saezlab/corneto/issues) page—we're happy to help!
