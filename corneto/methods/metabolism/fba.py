import warnings
from typing import Dict, Optional, Union

import numpy as np

from corneto import K
from corneto._graph import BaseGraph
from corneto.backend._base import (
    Backend,
    Direction,
    Indicator,
    NonZeroIndicator,
    ProblemDef,
    VarType,
)


class FBAProblem(ProblemDef):
    """A class representing a Flux Balance Analysis optimization problem.

    This class extends ProblemDef to implement Flux Balance Analysis (FBA),
    a mathematical method for calculating the flow of metabolites through a
    metabolic network.

    Args:
        graph (BaseGraph): The metabolic network graph.
        create_reaction_indicators (bool, optional): Whether to create binary indicators
            for reactions. Defaults to False.
        reaction_objective (Union[int, str], optional): The reaction to optimize.
            Can be reaction index or ID. Defaults to None.
        backend (Backend, optional): The optimization backend to use. Defaults to K.
    """

    def __init__(
        self,
        graph: BaseGraph,
        create_reaction_indicators: bool = False,
        reaction_objective: Optional[Union[int, str]] = None,
        backend: Optional["Backend"] = K,
    ) -> None:
        super().__init__(backend=backend, graph=graph, direction=Direction.MAX)
        self._graph = graph
        pf = fba_problem(
            graph,
            create_reaction_indicators=create_reaction_indicators,
            backend=backend,
        )
        self.merge(pf, inplace=True)
        # Set objective
        if reaction_objective is not None:
            if isinstance(reaction_objective, str):
                raise NotImplementedError()
                # reaction_objective = self._graph.get_edge_id(reaction_objective)
            self.add_objectives(self.symbols["_flow"][reaction_objective])

    def get_fluxes(self) -> np.ndarray:
        """Get the computed reaction fluxes.

        Returns:
            np.ndarray: Array of flux values for each reaction. Returns NaN for
                unsolved problems.
        """
        fluxes = self.symbols["_flow"].value
        if fluxes is None:
            return np.full(self._graph.ne, np.nan)
        return fluxes

    def get_fluxes_dict(self) -> Dict:
        """Get the computed reaction fluxes as a dictionary.

        Returns:
            Dict: Dictionary mapping reaction IDs to their flux values.
        """
        fluxes = self.get_fluxes()
        reaction_ids = [self._graph.get_attr_edge(i).get("id") for i in range(self._graph.ne)]
        return {r: fluxes[i] for i, r in enumerate(reaction_ids)}


def fba_problem(G, create_reaction_indicators=False, num_fluxes=1, eps=1e-4, backend=K):
    """Creates a basic FBA problem definition.

    Args:
        G (BaseGraph): The metabolic network graph.
        create_reaction_indicators (bool, optional): Whether to create binary indicators
            for reactions. Defaults to False.
        num_fluxes (int, optional): Number of flux variables per reaction. Defaults to 1.
        eps (float, optional): Tolerance for considering a flux as non-zero. Defaults to 1e-4.
        backend (Backend, optional): The optimization backend to use. Defaults to K.

    Returns:
        ProblemDef: The FBA problem definition.
    """
    lb, ub = [], []
    for i in range(G.ne):
        attr = G.get_attr_edge(i)
        lb.append(attr.default_lb)
        ub.append(attr.default_ub)
    P = backend.Flow(G, lb=np.array(lb), ub=np.array(ub), values=True, n_flows=num_fluxes)
    if create_reaction_indicators:
        P += NonZeroIndicator(tolerance=eps)
    return P


def multicondition_imat(
    model,
    w,
    alpha=1e-3,
    lambd=None,
    eps=1e-2,
    use_unblocked_flux_indicators=False,
    scale=False,
    backend=K,
):
    """Implements the multi-condition iMAT (integrative Metabolic Analysis Tool) algorithm.

    This function creates an optimization problem that integrates gene expression data
    with metabolic network analysis across multiple conditions.

    Args:
        model (BaseGraph): The metabolic network model.
        w (np.ndarray): Expression weights matrix. If 1D, treated as single condition.
            Positive values indicate desired active reactions, negative values indicate
            desired inactive reactions.
        alpha (float, optional): Network size regularization parameter. Defaults to 1e-3.
        lambd (float, optional): Deprecated parameter, use alpha instead. Defaults to None.
        eps (float, optional): Tolerance for considering a flux as non-zero. Defaults to 1e-2.
        use_unblocked_flux_indicators (bool, optional): Whether to use additional indicators
            for unblocked reactions. Defaults to False.
        scale (bool, optional): Whether to scale the weights and regularization term.
            Defaults to False.
        backend (Backend, optional): The optimization backend to use. Defaults to K.

    Returns:
        ProblemDef: The multi-condition iMAT problem definition.

    Raises:
        ValueError: If scale=True and alpha is not between 0 and 1.
    """
    if alpha is not None and lambd is not None:
        warnings.warn("Overriding alpha with `lambd`. Use `lambd` instead.")
        alpha = lambd
    if alpha is None and lambd is not None:
        alpha = lambd

    # Use or of blocked reactions (indicator=0)
    if len(w.shape) == 1:
        n_conditions = 1
        n_vars = len(w)
    else:
        n_vars = w.shape[0]
        n_conditions = w.shape[1]
    loss_w = 1
    if scale:
        if alpha < 0 or alpha > 1:
            raise ValueError("If scale=True, alpha has to be a number between 0 and 1 (inclusive)")
        # Scale the weights of each condition dividing by the total weight per condition
        # Scale in %, avoid very small error numbers (solver tolerances)
        w = (w / np.abs(w).sum(axis=0)) * 100  # error is in %
        loss_w = 1 - alpha

    # Loss is 100 * n_conditions
    # Scale reg. term, so is scale=True, alpha=0.01 means
    # that in the total loss, the size of the network only
    # contributes 1%. Scale n_edges so a network with all edges
    # has a penalty of 100 (%).

    P = fba_problem(model, create_reaction_indicators=True, num_fluxes=n_conditions, eps=eps)
    active = P.symbols["_flow_ineg"] + P.symbols["_flow_ipos"]
    if use_unblocked_flux_indicators and np.abs(alpha) > 0:
        P += Indicator()  # I = 0 <=> F = 0
        unblocked = P.symbols["_flow_i"]
        P += active <= unblocked
    else:
        unblocked = active
    for i in range(n_conditions):
        if n_conditions > 1:
            active_condition = active[:, i]
            weights = w[:, i]
        else:
            active_condition = active
            weights = w
        idx_pos = np.where(weights > 0)[0]
        idx_neg = np.where(weights < 0)[0]
        # print(len(idx_pos), len(idx_neg))
        if len(idx_pos) > 0:
            # errors of not selecting positive reactions
            obj_pos = weights[idx_pos] @ (1 - active_condition[idx_pos])
        else:
            obj_pos = 0
        if len(idx_neg) > 0:
            # errors of not selecting negative reactions
            obj_neg = np.abs(weights[idx_neg]) @ active_condition[idx_neg]
        else:
            obj_neg = 0
        loss = obj_pos + obj_neg  # minimize loss
        P.add_objectives(loss, weights=loss_w)

    if n_conditions > 1:
        total = sum(unblocked.T)
        # total = sum(active.T)
    else:
        total = unblocked
        # total = sum(active)

    # Add some useful expressions, not used for optimization
    # has_pos_flux = P.symbols["_flow"] >= eps * (1 - eps)
    # has_neg_flux = P.symbols["_flow"] <= -eps * (1 - eps)
    # has_flux = has_pos_flux + has_neg_flux
    # P.register("has_positive_flux", has_pos_flux)
    # P.register("has_negative_flux", has_neg_flux)
    # P.register("has_flux", has_flux)

    if np.abs(alpha) > 0:
        logic_or = backend.Variable(name="_or", shape=total.shape, vartype=VarType.BINARY)
        P += [
            # only 0 if total is 0
            logic_or >= total / n_conditions,
            logic_or <= total,
        ]
        # Scale the number of selected variables

        if scale:
            reg = (sum(logic_or) / n_vars) * 100
        else:
            reg = sum(logic_or)
        reg_w = alpha
        if scale:
            reg_w = alpha * n_conditions
        P.add_objectives(reg, weights=reg_w)
    return P
