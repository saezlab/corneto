<div align="center"> 
<img alt="corneto logo" src="docs/_static/logo/corneto-logo-512px.png" height="200"/>
<br>
<h3>CORNETO: Unified knowledge-driven network inference from omics data.</h3>

Maintained by [Saez-Rodriguez](https://saezlab.org) group.

<!-- badges: start -->
[![main](https://github.com/saezlab/corneto/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/saezlab/corneto/actions)
<!-- badges: end -->

</div>

---

CORNETO (Constrained Optimization for the Recovery of Networks from Omics) is a unified framework for multi-sample joint network inference, implemented in Python. It tackles common network inference problems in biology and extends them to support multiple samples or conditions simultaneously, enhancing network identification. The framework reformulates these problems using constrained optimization and mathematical programming, allowing them to be optimally solved with mathematical solvers. Additionally, it provides flexible modeling capabilities, enabling the exploration of hypotheses, modification, or development of new network inference problems through the use of modular constrained building blocks.

> **⚠️ Disclaimer**: This is an early preview of the CORNETO library. **CORNETO is under active development and has not yet reached a stable release for end users**. Contributions and feedback are not yet open until the first stable release. Please stay tuned for updates.

<div align="center"> 
<img alt="corneto logo" src="https://raw.githubusercontent.com/saezlab/corneto/refs/heads/main/docs/_static/corneto-fig-abstract.jpg" height="600"/>
</div>


## Installation

The library will be uploaded to pypi once the API is stable. Meanwhile, it can be installed by downloading the wheel file from the repository. It's recommended to use also conda to create a new environment, although it's not mandatory.

### Recommended setup

CORNETO does not include any backend nor solver by default to avoid issues with architectures for which some of the required binaries are not available. The recommended setup for CORNETO requires CVXPY and Gurobi:

```bash
pip install corneto cvxpy-base scipy gurobipy
```

Please note that **GUROBI is a commercial solver which offers free academic licenses**. If you have an academic email, this step is very easy to do in just few minutes: https://www.gurobi.com/features/academic-named-user-license/. You need to register GUROBI in your machine with the `grbgetkey` tool from GUROBI.

Alternatively, it is possible to use CORNETO with any free solver, such as HIGHS, included in Scipy. For this you don't need to install Gurobi. Please note that if `gurobipy` is installed but not registered with a valid license, CORNETO will choose it but the solver will fail due to license issues. If SCIPY is installed, when optimizing a problem, select SCIPY as the solver

```python
# P is a corneto problem
P.solve(solver="SCIPY")
```

> :warning: Please note that without any backend, you can't do much with CORNETO. There are two supported backends right now: [PICOS](https://picos-api.gitlab.io/picos/tutorial.html) and [CVXPY](https://www.cvxpy.org/). Both backends allow symbolic manipulation of expressions in matrix notation. 

## How to cite

The manuscript for this work is in preparation. In the meantime, if you are currently using CORNETO, please cite the repository:

```
@misc{rodriguezmier_corneto24,
  author       = {Rodriguez-Mier, Pablo and Garrido-Rodriguez, Martin and Gabor, Attila and Saez-Rodriguez, Julio},
  title        = {Unified knowledge-driven network inference from omics data},
  year         = {2024},
  howpublished = {\url{https://github.com/saezlab/corneto}},
  note         = {Manuscript in preparation}
}
```

## Acknowledgements

CORNETO is developed at the [Institute for Computational Biomedicine](https://saezlab.org) (Heidelberg University). The development of this project is supported by European Union's Horizon 2020 Programme under
PerMedCoE project ([permedcoe.eu](https://permedcoe.eu/)) agreement no. 951773.

<div align="left">
  <img src="https://raw.githubusercontent.com/saezlab/.github/main/profile/logos/saezlab.png" alt="Saez lab logo" height="64px" style="margin: 0 20px;">
  <img src="https://lcsb-biocore.github.io/COBREXA.jl/stable/assets/permedcoe.svg" alt="PerMedCoE logo" height="64px" style="margin: 0 20px;">
  <img src="https://yt3.googleusercontent.com/ytc/AIf8zZSHTQJs12aUZjHsVBpfFiRyrK6rbPwb-7VIxZQk=s176-c-k-c0x00ffffff-no-rj" alt="UKHD logo" height="64px" style="margin: 0 20px;">
</div>
