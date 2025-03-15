---
hide-toc: true
html_theme.sidebar_secondary.remove: true
---

<style>
  .bd-main .bd-content .bd-article-container {
    max-width: 100%;
  }
  .prev-next-footer {
    display: none;
  }

</style>

# CORNETO <span class="hidden-title-marker"></span>

<div style="text-align: center;"> 
  <img alt="corneto logo" src="_static/logo/corneto-logo-512px.png" height="200"/>
  <h2 style="margin: 1em 0 0.5em 0;">Unified knowledge-driven network inference framework</h2>

<!-- badges: start -->
[![GitHub stars](https://img.shields.io/github/stars/saezlab/corneto)](https://github.com/saezlab/corneto/stargazers)
[![main](https://github.com/saezlab/corneto/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/saezlab/corneto/actions)
<!-- badges: end -->

</div>

<div style="margin: 3em 0;">

```{gallery-grid}
:grid-columns: 1 2 2 4

- header: "**Installation** {octicon}`terminal;1.2em;pst-color-secondary`"
  content: "Get started with CORNETO by following our installation guide for different setups and requirements."
  website: install.html
- header: "**User guide** {octicon}`repo;1.2em;pst-color-secondary`"
  content: "Learn what CORNETO is and how you can use it to create and modify biological network problems."
  website: guide
- header: "**Examples & Tutorials** {octicon}`light-bulb;1.2em;pst-color-secondary`"
  content: "Examples using CORNETO to leverage the power of complex biological networks on real-world data."
  website: tutorials
- header: "**API Reference** {octicon}`stack;1.2em;pst-color-secondary`"
  content: "Python API reference for CORNETO."
  website: api
```

</div>

**Version**: {{version}}

----

## What is CORNETO?

CORNETO (Constraint-based Optimization for the Reconstruction of NETworks from Omics) is a package for unified biological network inference and contextualisation from prior knowledge, developed and maintained by the [Saez-Rodriguez Lab](https://saezlab.github.io/). The library is designed with minimal dependencies and is easily extendable, making it a powerful tool for both end-users and developers. 

<div style="margin: 2em 0;">
    <img src="_static/corneto-abstract.jpg" alt="Corneto Abstract" style="max-width: 768px; width: 100%; display: block; margin: 0 auto;">
</div>


To install CORNETO with open-source mathematical solvers (HIGHs and SCIP), use the following command:

```
# To install the development version, including the open-source solvers HIGHs and SCIP, use:
pip install git+https://github.com/saezlab/corneto.git@dev pyscipopt highspy

# If you have a license for Gurobi (free for academic use), you can install it with:
pip install gurobipy
```

## Acknowledgements

CORNETO is developed at the [Institute for Computational Biomedicine](https://saezlab.org) (Heidelberg University). The development of this project is supported by European Union's Horizon 2020 Programme under
PerMedCoE project ([permedcoe.eu](https://permedcoe.eu/)) agreement no. 951773 and [DECIDER](https://www.deciderproject.eu/) (965193).

<div style="margin: 1em 0;">
    <img src="_static/embl-ebi-logo.png" alt="EMBL-EBI logo" height="64px" style="height:64px; width:auto; margin-right: 5px;">
    <img src="_static/saezlab-logo.png" alt="Saez lab logo" height="64px" style="height:64px; width:auto; margin-right: 5px;">
    <img src="_static/ukhd-logo.jpg" alt="UKHD logo" height="64px" style="height:64px; width:auto; margin-right: 5px;"> 
    <img src="_static/permedcoe-eu-logo.png" alt="PerMedCoE logo" height="64px" style="height:64px; width:auto; margin-right: 5px;"> 
    <img src="_static/decider-eu-logo.png" alt="DECIDER logo" height="64px" style="height:64px; width:auto;">
</div>

```{toctree}
:hidden: true
:maxdepth: 3
:titlesonly: true

install
guide/index
tutorials/index
api/index
GitHub <https://github.com/saezlab/corneto>
```