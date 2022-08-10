# CORNETO

CORNETO (CORe NETwork Optimization) is a first step towards unification of ILP based problems (steady state analysis of networks) with the long term goal of unifying also signaling and constraint-based modeling of metabolism.

CORNETO will translate the specification of a high level problem into an (I)LP formulation through different backends (e.g Python-MIP, PICOS and CVXPY) which are in charge of implementing specific backends for different free/commercial solvers in a transparent way. Different methods like CARNIVAL, CellNopt-ILP, Phonemes, etc could be reimplemented in a simple way on top of MIOM, abstracting away all the low level details of ILP formulations.



