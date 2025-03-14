r"""Methods (:mod:`corneto.methods`)
====================================

.. currentmodule:: corneto.methods

This module provides the implementations of the various methods used in CORNETO.
It is organized into several functional areas.

"""

# Import Carnival Methods
from corneto.methods.carnival import heuristic_carnival as fast_carnival
from corneto.methods.carnival import (
    runCARNIVAL_AcyclicFlow,
    runCARNIVAL_Flow_Acyclic,
    runCARNIVAL_Flow_Acyclic_Signal,
    runInverseCarnival,
    runVanillaCarnival,
)

# Import Shortest Path Methods
from corneto.methods.shortest_path import shortest_path, solve_shortest_path

# Import Signaling Methods
from corneto.methods.signaling import (
    create_flow_graph,
    default_sign_loss,
    expand_graph_for_flows,
    signflow_constraints,
)

# Legacy Aliases (for backward compatibility)
from corneto.methods.signaling import create_flow_graph as carnival_renet
from corneto.methods.signaling import default_sign_loss as carnival_loss
from corneto.methods.signaling import signflow_constraints as carnival_constraints
