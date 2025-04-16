
<style>
    .prev-next-footer {
        display: none;
    }
</style>

# Install

**CORNETO** provides flexible installation options depending on your needs. It integrates with various mathematical optimization solvers, which differ in their licensing, capabilities, and performance. Below are several installation options tailored to different use cases.

## 🔧 Quick Installation

To get started quickly with open-source solvers, run:

```bash
pip install git+https://github.com/saezlab/corneto.git@dev pyscipopt highspy cvxpy
```

This installs:
- **CORNETO** (from the development branch)
- **PySCIPOpt**: a Python interface for SCIP (a fast open-source solver)
- **highspy**: Python bindings for the HIGHs solver
- **CVXPY**: a modeling language for convex optimization problems

These libraries provide solid performance for most users and don’t require a commercial license.

---

## 🧩 Installation Options

### Basic Installation (CORNETO only)

If you only want the core CORNETO package without any solver backend (useful for development or adding solvers manually later):

```bash
pip install git+https://github.com/saezlab/corneto.git@dev
```

You’ll need to install and configure a solver separately to make CORNETO fully functional.

---

### ⚙️ Installation with Commercial Solvers

If you have access to **Gurobi** (a powerful commercial solver, free for academic use), you can install CORNETO along with Gurobi and its supported modeling tools:

```bash
pip install git+https://github.com/saezlab/corneto.git@dev cvxpy gurobipy
```

**Note:** Make sure you have a valid Gurobi license set up on your system. You can get a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

### 🆓 Installation with Open-Source Solvers

Prefer open-source? This installs CORNETO with:
- **SCIP** (via `pyscipopt`)
- **HIGHs** (via `highspy`)
- **CVXPY**

```bash
pip install git+https://github.com/saezlab/corneto.git@dev pyscipopt highspy cvxpy
```

These tools are effective and don't require any licensing setup.

---

### 🧪 Alternative Backend: PICOS

CORNETO also supports `picos`, another modeling layer for optimization. You can install CORNETO with `picos` and several solvers it supports:

```bash
pip install git+https://github.com/saezlab/corneto.git@dev picos swiglpk qics
```

This setup uses:
- **PICOS**: a high-level modeling tool
- **swiglpk**: Python bindings for GLPK, a lightweight solver
- **qics**: another solver supported by PICOS

---

## ✅ Verifying Installation

To check that CORNETO is installed and ready:

```python
import corneto

corneto.info()
```

This should print out CORNETO version info and configuration details.

---

## 🛠 Troubleshooting

If you run into installation problems:

1. ✅ Make sure you're using **Python 3.10 or higher**
2. 🧱 Check that you have at least **cvxpy** installed as a backend (this is the recommended one)
3. 🔑 For Gurobi users, verify that your **license is properly installed**

If you're still stuck, please open an issue on our [GitHub Issues](https://github.com/saezlab/corneto/issues) page—we're happy to help!
