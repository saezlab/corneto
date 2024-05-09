# User guide

CORNETO (Constraint-based Optimization for the Reconstruction of NETworks from Omics) is a python package for unified biological network inference and contextualisation from prior knowledge, developed and maintained by the [Saez-Rodriguez Lab](https://saezlab.github.io/) at Heidelberg University.

CORNETO employs mathematical programming to translate network inference problems into exact, unified reformulations on top of Network Flows, and then uses mathematical solvers to find optimal solutions. It offers flexibility for various applications:

- Provides modular building blocks for diverse network inference problems.
- Reimplements methodologies such as Shortest Paths, Steiner Trees, Prize Collecting Steiner Forest, and more.
- Supports different types of prior knowledge, including protein-protein interactions, signalling networks, and genome-scale metabolic networks.
- Introduces novel methods for network inference and contextualisation.

The library is designed with minimal dependencies and is easily extendable, making it a powerful tool for both end-users and developers.


```{toctree}
:maxdepth: 3

intro/index
networks/index
metabolism/index
signaling/index
interactomics/index
interoperability/index
```
