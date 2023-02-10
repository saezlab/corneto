from corneto.methods.signflow import signflow, create_flow_graph, Graph
from typing import Dict, List, Tuple, Union


def runVanillaCarnival(perturbations: Dict, measurements: Dict, priorKnowledgeNetwork: Union[List[Tuple], Graph], betaWeight: float = 0.2, solver=None, **kwargs):
    data = dict()
    for k, v in perturbations.items():
        data[k] = ('P', v)
    for k, v in measurements.items():
        data[k] = ('M', v)
    conditions = {'c0': data}
    if isinstance(priorKnowledgeNetwork, List):
        G = Graph.from_sif_tuples(priorKnowledgeNetwork)
    elif isinstance(priorKnowledgeNetwork, Graph):
        G = priorKnowledgeNetwork
    else:
        raise ValueError("Provide a list of sif tuples or a graph")
    Gf = create_flow_graph(G, conditions)
    P = signflow(Gf, conditions, l0_penalty_vertices=betaWeight)
    P.solve(solver=solver, **kwargs)
    return P, Gf
