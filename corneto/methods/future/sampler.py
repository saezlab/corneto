import logging
from typing import Any, Dict, List, Optional

import numpy as np

from corneto.backend._cvxpy_backend import CvxpyBackend

logger = logging.getLogger(__name__)


def sample_alternative_solutions(
    problem: Any,
    target_variable_name: str,
    percentage: float = 0.1,
    scale: float = 0.01,
    rel_opt_tol: float = 0.05,
    max_samples: int = 30,
    time_limit: int = 180,
    solver: str = "gurobi",
    display_backend_output: bool = False,
    verbose: bool = False,
    collect: Optional[Dict[str, str]] = None,
) -> Dict[str, List[np.ndarray]]:
    """Sample alternative solutions by perturbing the objective function.

    This function solves an optimization problem multiple times while randomly
    perturbing a provided target variable within the objective function. It then
    checks whether the resulting objective values are within a relative tolerance of
    the original solution. Any solution meeting the tolerance criteria is recorded
    based on the provided variables. By default, the variable being perturbed (target_variable)
    is always collected.

    Parameters
    ----------
    problem : CORNETO optimization problem
        The optimization problem to sample.
    target_variable_name : str
        The variable to perturb (in problem.expr).
    percentage : float, optional
        Percentage of elements to perturb, by default 0.1.
    scale : float, optional
        Scale parameter for normal distribution perturbation, by default 0.01.
    rel_opt_tol : float, optional
        Relative optimality tolerance, by default 0.05.
    max_samples : int, optional
        Maximum number of samples to generate, by default 30.
    time_limit : int, optional
        Time limit for the solver (in seconds), by default 180.
    solver : str, optional
        Optimization solver to be used (default is "gurobi").
    display_backend_output : bool, optional
        Whether to display the backend solver's output (for configuring its verbosity).
    verbose : bool, optional
        If True, progress messages will be printed directly to stdout.
    collect : dict, optional
        A dictionary where keys are names for the recorded variables and values are their
        corresponding expressions in the problem. If None, only the perturbed variable is collected (key "perturbed").

    Returns:
    -------
    Dict[str, List[np.ndarray]]
        A dictionary where each key corresponds to a collected variable name and the value is a list
        of the sampled values (one for each accepted sample).
    """
    # Validate inputs
    if not isinstance(problem.backend, CvxpyBackend):
        raise ValueError(
            "The sampler only works with the CvxpyBackend, but the provided problem uses a different backend."
        )
    if percentage <= 0 or percentage > 1:
        raise ValueError("Percentage must be between 0 and 1")

    if target_variable_name not in problem.expr:
        raise ValueError(f"Target variable '{target_variable_name}' not found in problem expressions")

    # Solve original problem and store objective values (except for the perturbation objective)
    verbosity: int = 1 if display_backend_output else 0
    problem.solve(solver=solver, verbosity=verbosity)
    orig_objectives: Dict[str, float] = {}
    for obj in problem.objectives:
        if obj.name == "perturbation":
            continue
        logger.debug("Original objective '%s': %s", obj.name, obj.value)
        if verbose:
            print(f"Original objective '{obj.name}': {obj.value}")
        orig_objectives[obj.name] = obj.value

    # Prepare perturbation: determine indices to be perturbed
    target_variable = problem.expr[target_variable_name]
    shape = target_variable.shape
    num_elements: int = shape[0]
    n_elements: int = max(int(num_elements * percentage), 1)  # Ensure at least one element is perturbed
    vec: np.ndarray = np.zeros(num_elements)

    # Create a parameter in the backend representing the perturbation
    pert = problem.backend.Parameter("perturbation", shape=vec.shape, value=vec)

    # Add the perturbation objective
    problem.add_objective(target_variable.T @ pert, name="perturbation")

    # Ensure the target variable is always collected
    if collect is None:
        collect = {"perturbed": target_variable_name}
    elif target_variable_name not in collect.values():
        collect = {**collect, "perturbed": target_variable_name}

    # Initialize dictionary to collect sampled values
    results: Dict[str, List[np.ndarray]] = {name: [] for name in collect.keys()}

    accepted_count: int = 0
    for i in range(max_samples):
        # Generate random perturbation vector
        c: np.ndarray = np.random.normal(scale=scale, size=n_elements)
        random_indices: np.ndarray = np.random.choice(vec.shape[0], size=n_elements, replace=False)
        vec[:] = 0  # Reset vector
        vec[random_indices] = c

        # Update the parameter value and solve the perturbed problem
        pert.value = vec
        solved_problem = problem.solve(
            solver=solver,
            warm_start=True,
            ignore_dpp=True,
            TimeLimit=time_limit,
            verbosity=verbosity,
        )

        log_message = f"Sample {i + 1}/{max_samples}: Objective value = {solved_problem.value:.4f}"
        logger.info(log_message)
        if verbose:
            print(log_message)

        # Check if solution is within tolerance based on objectives
        accept: bool = True
        for obj in problem.objectives:
            if obj.name == "perturbation":
                continue
            orig_val = orig_objectives.get(obj.name)
            if orig_val is None:
                warning_message = f"Original objective value for '{obj.name}' not found."
                logger.warning(warning_message)
                if verbose:
                    print(warning_message)
                continue
            relative_error = np.abs(obj.value - orig_val) / (np.abs(orig_val) + 1e-10)
            debug_message = f"Objective '{obj.name}': solved value {obj.value:.4f}, relative error {relative_error:.4f}"
            logger.debug(debug_message)
            if verbose:
                print(debug_message)
            if relative_error > rel_opt_tol:
                reject_message = (
                    f"Sample {i + 1} rejected: Objective '{obj.name}' relative error "
                    f"{relative_error:.4f} exceeds tolerance {rel_opt_tol:.4f}"
                )
                logger.debug(reject_message)
                if verbose:
                    print(reject_message)
                accept = False
                break

        # If the sample is acceptable, collect the requested variables
        if accept:
            accepted_count += 1
            for name, var_expr in collect.items():
                if verbose:
                    print(f"Collecting {name} (variable: {var_expr})")
                results[name].append(problem.expr[var_expr].value)
            if verbose:
                print(f"Sample {i + 1} accepted. Total accepted: {accepted_count}")

    if accepted_count == 0:
        logger.warning("No valid alternative solutions were found under the given constraints")
        if verbose:
            print("No valid alternative solutions were found. Consider adjusting parameters.")

    return results
