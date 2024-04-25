# CORNETO <span class="hidden-title-marker"></span>

```{image} /_static/logo.png
:align: center
:width: 320px
```

<h3 style="text-align: center; margin-top: 20px;"> Unified Omics-Driven Framework for Network Inference </h3>

CORNETO (Constraint-based Optimization for the Reconstruction of NETworks from Omics) is a package for unified biological network inference and contextualisation from prior knowledge, developed and maintained by the [Saez-Rodriguez Lab](https://saezlab.github.io/) at Heidelberg University.

The library is designed with minimal dependencies and is easily extendable, making it a powerful tool for both end-users and developers. To install CORNETO with open-source mathematical solvers (HIGHs and SCIP), use the following command:

```
# To install the development version, use:
pip install git+https://github.com/saezlab/corneto.git@dev

# For the latest stable release, use:
pip install corneto
```

**Version**: {{version}}

```{gallery-grid}
:grid-columns: 1 2 3 3

- header: "{fas}`laptop;pst-color-primary` User guide"
  content: "[WIP] Learn what CORNETO is and how you can use it to create and modify biological network problems."
  website: guide
- header: "{fas}`lightbulb;pst-color-primary` Examples & Tutorials"
  content: "[WIP] Examples using CORNETO to leverage the power of complex biological networks on real-world data."
  website: tutorials
- header: "{fab}`python;pst-color-primary` API Reference"
  content: "[WIP] Python API reference for CORNETO."
  website: api
```

## Acknowledgements

CORNETO is developed at the [Institute for Computational Biomedicine](https://saezlab.org) (Heidelberg University). The development of this project is supported by European Union's Horizon 2020 Programme under
PerMedCoE project ([permedcoe.eu](https://permedcoe.eu/)) agreement no. 951773 and [DECIDER](https://www.deciderproject.eu/) (965193).

<img src="/_static/ukhd-logo.jpg" alt="UKHD logo" height="64px" style="height:64px; width:auto"> <img src="/_static/saezlab-logo.png" alt="Saez lab logo" height="64px" style="height:64px; width:auto"> <img src="/_static/permedcoe-eu-logo.png" alt="PerMedCoE logo" height="64px" style="height:64px; width:auto"> <img src="/_static/decider-eu-logo.png" alt="DECIDER logo" height="64px" style="height:64px; width:auto"> 



```{toctree}
:hidden: true
:maxdepth: 3
:titlesonly: true

guide/index
tutorials/index
api/index
GitHub <https://github.com/saezlab/corneto>
```