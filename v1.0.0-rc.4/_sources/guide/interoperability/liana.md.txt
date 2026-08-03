# LIANA+: An all-in-one cell-cell communication framework

<center>
<img src="../../_static/lianaplus-abstract.png" alt="LIANA+" style="width: 600px; margin: 20px;"/>
</center>

[**LIANA+**](https://github.com/saezlab/liana-py) is a scalable framework for decoding coordinated inter- and intracellular signaling events from single- and multi-condition datasets in both single-cell and spatially-resolved data. It integrates and extends established methodologies and a rich knowledge base, and enables novel analyses using diverse molecular mediators, including those measured in multi-omics data.

One of LIANA+'s capabilities is to infer intracellular signaling networks driven by CCC events. CCC events can be thought of as upstream perturbants of intracellular signaling networks that lead to deregulations of downstream signaling events. These deregulations are expected to be associated with various conditions and diseases. Therefore, understanding intracellular signaling networks is critical to modeling cellular mechanisms. LIANA+ can infer intracellular signaling networks from CCC events by integrating CCC events with prior knowledge of intracellular signaling networks and then using CORNETO to build and solve the optimization problem of inferring the network.

For more information about LIANA+ and CORNETO interoperability, please see the tutorial ["Hypothesis-testing for CCC & Downstream Signalling Networks"](https://liana-py.readthedocs.io/en/latest/notebooks/targeted.html) available in LIANA+'s documentation
