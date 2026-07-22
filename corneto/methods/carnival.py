"""Documented CARNIVAL optimization methods."""

import numpy as np

import corneto as cn
from corneto.methods._carnival import CarnivalFlow, CarnivalILP

__all__ = ["CarnivalFlow", "CarnivalILP", "milp_carnival"]


def milp_carnival(
    G,
    perturbations,
    measurements,
    beta_weight: float = 0.2,
    max_dist=None,
    penalize="edges",  # nodes, edges, both
    use_perturbation_weights=False,
    interaction_graph_attribute="interaction",
    disable_acyclicity=False,
    backend=cn.DEFAULT_BACKEND,
):
    """Build the supported single-condition CARNIVAL ILP formulation.

    This simple formulation accepts one perturbation and measurement mapping. Use
    :class:`CarnivalILP` for the general multi-condition formulation, or
    :class:`CarnivalFlow` when a flow-based multi-condition model is required.

    NOTE: Since the method is decoupled from specific solvers, the default pool of
    solutions generated using CPLEX is not available.

    Args:
        G: The graph object representing the network.
        perturbations: Perturbations applied to vertices in the graph.
        measurements: Measured values for vertices in the graph.
        beta_weight: The weight for the regularization term in the objective function.
        max_dist: The maximum distance allowed for vertex positions in the graph.
        penalize: The type of regularization to apply ('nodes', 'edges', or 'both').
        use_perturbation_weights: Whether to weight perturbations in the objective.
        interaction_graph_attribute: Graph attribute containing interaction signs.
        disable_acyclicity: Whether to disable the acyclicity constraint.
        backend: The backend engine to use for the optimization.

    Returns:
        The optimization problem object.
    """
    max_dist = G.num_vertices if max_dist is None else max_dist

    # The problem uses 2*|V| + 2*|E| binary variables + |V| continuous variables
    V_act = backend.Variable("vertex_activated", shape=(len(G.V),), vartype=cn.VarType.BINARY)
    V_inh = backend.Variable("vertex_inhibited", shape=(len(G.V),), vartype=cn.VarType.BINARY)
    E_act = backend.Variable("edge_activating", shape=(len(G.E),), vartype=cn.VarType.BINARY)
    E_inh = backend.Variable("edge_inhibiting", shape=(len(G.E),), vartype=cn.VarType.BINARY)
    V_pos = backend.Variable(
        "vertex_position",
        shape=(len(G.V),),
        lb=0,
        ub=max_dist,
        vartype=cn.VarType.CONTINUOUS,
    )

    V_index = {v: i for i, v in enumerate(G.V)}

    P = backend.Problem()

    # A vertex can be activated or inhibited, but not both
    P += V_act + V_inh <= 1
    # An edge can activate or inhibit, but not both
    P += E_act + E_inh <= 1

    for i, (s, t) in enumerate(G.E):
        s = list(s)
        t = list(t)
        if len(s) == 0:
            continue
        if len(s) > 1:
            raise ValueError("Only one source vertex allowed")
        if len(t) > 1:
            raise ValueError("Only one target vertex allowed")
        s = s[0]
        t = t[0]
        # An edge can activate its downstream (target vertex) (E_act=1, E_inh=0),
        # inhibit it (E_act=0, E_inh=1), or do nothing (E_act=0, E_inh=0)
        si = V_index[s]
        ti = V_index[t]
        interaction = int(G.get_attr_edge(i).get(interaction_graph_attribute))
        # If edge interaction type is activatory, it can only activate the downstream
        # vertex if the source vertex is activated
        # NOTE: The 4 constraints can be merged by 2, but kept like this for clarity
        # This implements the basics of the sign consistency rules
        if interaction == 1:
            # Edge is activatory: E_act can only be 1 if V_act[source] is 1
            # edge (s->t) can activate t only if s is activated
            P += E_act[i] <= V_act[si]
            # edge (s->t) can inhibit t only if s is inhibited
            P += E_inh[i] <= V_inh[si]
        elif interaction == -1:
            # edge (s-|t) can activate t only if s is inhibited
            P += E_act[i] <= V_inh[si]
            # edge (s-|t) can inhibit t only if s is activated
            P += E_inh[i] <= V_act[si]
        else:
            raise ValueError(f"Invalid interaction value for edge {i}: {interaction}")

        # If the edge is selected, then we must respect the order of the vertices:
        # V_pos[target] - V_pos[source] >= 1
        # E.g., if a partial solution is A -> B -> C, and the ordering assigned is
        # A(0) -> B(1) -> C(2), we cannot select an edge C -> A since 2 > 0
        # The maximum numbering possible, starting with 0, cannot exceed the
        # number of vertices of the graph.
        # Note that there are two possible orderings: or target vertex is greater
        # than source (then edge can be selected), or less or equal to 0
        # (in which case the edge cannot be selected).
        # The acyclicity constraint is reduced to this simple constraint:
        # - if edge selected, then target vertex must be greater than source (diff >= 1)
        # If unselected, the vertex-position difference is unrestricted.
        # IMPORTANT: acyclicity can be disabled, but then self activatory loops that are
        # not downstream the perturbations can appear in the solution.
        if not disable_acyclicity:
            edge_selected = E_act[i] + E_inh[i]
            P += V_pos[ti] - V_pos[si] >= 1 - max_dist * (1 - edge_selected)

    # Now compute the value of each vertex, based on the incoming selected edges
    # NOTE: Here we force that a vertex can have at most 1 incoming edge but this
    # could be relaxed (e.g. allow many inputs and integrate signal).
    for v in G.V:
        in_edges_idx = [i for i, _ in G.in_edges(v)]
        i = V_index[v]
        perturbed_value = 0
        perturbed = v in perturbations
        if perturbed:
            perturbed_value = np.sign(perturbations[v])
        in_edges_selected = [E_act[i] + E_inh[i] for i in in_edges_idx]
        if len(in_edges_idx) > 0:
            P += sum(in_edges_selected) <= 1
        # And the value of the target vertex equals the value of the selected edge
        # If no edge is selected, then the value is 0]
        incoming_activating = sum(E_act[j] for j in in_edges_idx) if len(in_edges_idx) > 0 else 0
        incoming_inhibiting = sum(E_inh[j] for j in in_edges_idx) if len(in_edges_idx) > 0 else 0
        P += V_act[i] <= int(perturbed) + incoming_activating
        P += V_inh[i] <= int(perturbed) + incoming_inhibiting
        # If perturbed but value is 0, then perturbation can take any value,
        # otherwise it must be the same as the perturbation
        if perturbed_value > 0:
            P += V_act[i] == 1
            P += V_inh[i] == 0
        elif perturbed_value < 0:
            P += V_act[i] == 0
            P += V_inh[i] == 1

    # TODO: Remove, this is not required, as L1309-L1310 already exclude these vertices
    other_potential_inputs = set(v for v in G.V if len(set(G.in_edges(v))) == 0)
    other_potential_inputs = other_potential_inputs.difference(perturbations.keys())
    for v in other_potential_inputs:
        i = V_index[v]
        P += V_act[i] == 0
        P += V_inh[i] == 0

    data = measurements.copy()
    if use_perturbation_weights:
        data.update(perturbations)

    error_terms = []
    for k, v in data.items():
        i = V_index[k]
        prediction = V_act[i] - V_inh[i]  # -1, 0, 1
        sign = np.sign(v)
        if sign > 0:
            error_terms.append(np.abs(v) * (sign - prediction))
        elif sign < 0:
            error_terms.append(np.abs(v) * (prediction - sign))
    obj = sum(error_terms)
    reg = 0
    P.add_objectives(obj)
    if beta_weight > 0:
        if penalize == "nodes":
            reg = sum(V_act) + sum(V_inh)
        elif penalize == "edges":
            reg = sum(E_act) + sum(E_inh)
        elif penalize == "both":
            # This is the default implemented in CarnivalR,
            # although regularization by edges should be preferred
            reg = sum(V_act) + sum(V_inh) + sum(E_act) + sum(E_inh)
        P.add_objectives(reg, weights=beta_weight)

    # Finally, register some aliases for convenience
    P.register("vertex_values", V_act - V_inh)
    P.register("edge_values", E_act - E_inh)
    return P
