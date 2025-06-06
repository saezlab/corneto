import logging
import re
from collections.abc import Sequence
from typing import Dict, List, Union

import numpy as np

logger = logging.getLogger(__name__)


def sample_alternative_solutions(
    problem,
    variable_name: str,
    *,
    percentage: float = 0.10,
    scale: float = 0.05,
    rel_opt_tol: float = 0.10,
    max_samples: int = 30,
    perturbation_name: str = "perturbation",
    solver_kwargs: dict | None = None,
    rng: np.random.Generator | int | None = None,
    collect_vars: Sequence[str] | None = None,
    exclude_objectives_pattern: Union[str, None] = "regularization",
    verbose: int = 1,  # 0 = silent, 1 = summary, 2 = full detail
) -> Dict[str, np.ndarray]:
    """Generate alternative feasible solutions by perturbing one decision variable.

    This routine takes an optimization problem (with attributes `.expr`,
    `.solve()` and `.objectives`), identifies a target variable within it,
    and generates up to `max_samples` new feasible solutions by randomly
    perturbing a fraction of that variable's entries. Only those perturbations
    that keep all original, non-excluded objectives within a relative tolerance
    of the baseline are accepted.

    Args:
        problem: An optimization problem instance exposing:
            - `expr`: mapping of variable names to variable objects
            - `solve(...)`: method to solve the problem
            - `objectives`: list of objective objects (each with `.name` and `.value`)
        variable_name (str): Name of the variable in `problem.expr` to perturb.
        percentage (float, optional): Fraction of the variable's entries to
            perturb in each trial. Defaults to `0.10`.
        scale (float, optional): Standard deviation of the normal random noise.
            Defaults to `0.03`.
        rel_opt_tol (float, optional): Maximum allowed relative deviation of any
            original, non-excluded objective from its baseline value. Defaults to `0.05`.
        max_samples (int, optional): Maximum number of perturbation trials to
            attempt. Defaults to `30`.
        perturbation_name (str, optional): Name to assign to the added
            perturbation objective. Defaults to `"perturbation"`.
        solver_kwargs (dict | None, optional): Extra keyword arguments passed
            to `problem.solve()`. Defaults to `None`.
        rng (np.random.Generator | int | None, optional): Random number
            generator or seed for reproducibility. Defaults to `None`.
        collect_vars (Sequence[str] | None, optional): Names of the variables whose
            values you want returned.
            - If `None` (default), collect *every* variable in `problem.expr`.
            - If empty list `[]`, collect *none* (method runs but returns an empty dict).
            - Otherwise, collect only the named variables.
        exclude_objectives_pattern (str | None, optional): A regular-expression
            pattern. Objectives whose names match this pattern will be excluded
            from the relative-error tolerance check. If `None`, no objectives are
            excluded. Defaults to `"regularization"`.
        verbose (int, optional): Verbosity level:
            - `0`: silent
            - `1`: summary
            - `2`: full detail
            Defaults to `1`.

    Returns:
        dict[str, np.ndarray]:
            A mapping from each collected variable name to a NumPy array of shape
            `(n_samples, *variable.shape)`, where:
            - `n_samples ≥ 1` counts the incumbent plus every accepted perturbation
            - the remaining dimensions match the variable's own shape

            Example:
                out = sample_alternative_solutions(problem, "x", collect_vars=["x", "y"])
                x_stack = out["x"]        # shape (n_samples, *x.shape)
                incumbent_x = x_stack[0]  # first slice is always the baseline

    Raises:
        KeyError:
            If `variable_name` is not in `problem.expr`, or if any name in
            `collect_vars` is missing from `problem.expr`.
    """
    # Map verbosity to logging levels
    if verbose >= 2:
        log_level = logging.DEBUG
    elif verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logger.setLevel(log_level)

    if solver_kwargs is None:
        solver_kwargs = {}
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    if variable_name not in problem.expr:
        raise KeyError(f"Variable '{variable_name}' not found in problem.expr")

    if collect_vars is None:
        collect_vars = list(problem.expr.keys())
    else:
        missing = [v for v in collect_vars if v not in problem.expr]
        if missing:
            raise KeyError(f"Variables not found in problem.expr: {missing}")

    collected: Dict[str, List[np.ndarray]] = {v: [] for v in collect_vars}
    target_var = problem.expr[variable_name]

    # 1) original solve
    logger.debug("Solving original model …")
    problem.solve(**solver_kwargs, verbosity=0)
    baseline_obj = {o.name: float(o.value) for o in problem.objectives}
    logger.debug("Baseline objectives: " + ", ".join(f"{k}={v:.6g}" for k, v in baseline_obj.items()))

    for v in collect_vars:
        collected[v].append(np.asarray(problem.expr[v].value).copy())

    # 2) build perturbation parameter
    var_shape = tuple(int(s) for s in target_var.shape)
    total_elems = int(np.prod(var_shape))
    n_perturb = max(1, int(total_elems * percentage))

    noise_buf = np.zeros(var_shape, dtype=float)
    pert = problem.backend.Parameter(name=f"{perturbation_name}_param", shape=var_shape, value=noise_buf)
    problem.add_objective(
        (target_var.multiply(pert))
        .sum()
        .reshape(
            1,
        ),
        name=perturbation_name,
    )

    flat_buf = noise_buf.reshape(-1)
    n_accept = n_reject = 0

    # 3) sampling loop
    for trial in range(1, max_samples + 1):
        # 3a) new perturbation
        flat_buf.fill(0.0)
        idx = rng.choice(total_elems, n_perturb, replace=False)
        flat_buf[idx] = rng.normal(0.0, scale, n_perturb)
        pert.value = noise_buf

        # 3b) solve
        problem.solve(warm_start=True, **solver_kwargs, verbosity=0)

        # 3c) compute relative errors for each objective
        current_vals = {}
        all_relerrs = {}
        objectives_to_check_for_tolerance = {}

        for o in problem.objectives:
            if o.name == perturbation_name:
                continue

            # Ensure the objective was present in the baseline solve
            if o.name not in baseline_obj:
                logger.warning(
                    f"Objective '{o.name}' appeared after baseline solve and will be skipped for rel.err calculation."
                )
                continue

            val = float(o.value)
            current_vals[o.name] = val
            denom = max(abs(baseline_obj[o.name]), 1e-9)
            rel_err_val = abs(val - baseline_obj[o.name]) / denom
            all_relerrs[o.name] = rel_err_val

            # Decide if this objective's rel_err should be checked against rel_opt_tol
            excluded_by_pattern = False
            if exclude_objectives_pattern is not None:
                if re.search(exclude_objectives_pattern, o.name):
                    excluded_by_pattern = True

            if not excluded_by_pattern:
                objectives_to_check_for_tolerance[o.name] = rel_err_val
            else:
                if logger.isEnabledFor(logging.DEBUG):  # Log only if DEBUG is enabled
                    logger.debug(
                        f"Objective '{o.name}' rel.err={rel_err_val:.4f} "
                        f"not checked against tol={rel_opt_tol} due to exclusion pattern '{exclude_objectives_pattern}'."
                    )

        # Check tolerance only on non-excluded objectives
        violated = next(
            ((name, err) for name, err in objectives_to_check_for_tolerance.items() if err > rel_opt_tol),
            None,
        )

        # Log objective values and errors
        detail_msg_parts = []
        for name in sorted(current_vals.keys()):  # Sort for consistent log output
            # Ensure relerr exists; it might not if objective appeared mid-process and was skipped
            rel_err_str = f"{all_relerrs.get(name, float('nan')):.4f}"
            part = f"{name}: val={current_vals[name]:.6g}, rel.err={rel_err_str}"

            is_excluded_from_check_for_log = False
            if exclude_objectives_pattern is not None:
                if name in all_relerrs and re.search(
                    exclude_objectives_pattern, name
                ):  # Check name in all_relerrs to ensure it was processed
                    is_excluded_from_check_for_log = True

            if is_excluded_from_check_for_log:
                part += " (excluded from tol. check)"
            elif (
                name not in objectives_to_check_for_tolerance and name in all_relerrs
            ):  # Not excluded by pattern, but not in check list (e.g. new objective)
                part += " (not in tol. check)"

            detail_msg_parts.append(part)
        detail_msg = ", ".join(detail_msg_parts)

        if violated is None:
            for v_name in collect_vars:  # Use v_name to avoid conflict with collected value v
                collected[v_name].append(np.asarray(problem.expr[v_name].value).copy())
            n_accept += 1
            logger.info(f"[{trial}/{max_samples}] accepted (total accepted={n_accept}) -> {detail_msg}")
        else:
            n_reject += 1
            # More detailed rejection message
            violated_name, violated_err = violated
            logger.info(
                f"[{trial}/{max_samples}] rejected (tol={rel_opt_tol} violated by '{violated_name}' with rel.err={violated_err:.4f}) "
                f"(total rejected={n_reject}) -> {detail_msg}"
            )

    # 4) stack lists into arrays
    out: Dict[str, np.ndarray] = {}
    if collect_vars:  # Ensure collect_vars is not empty
        # Check if any variables were actually collected (n_accept > 0 or initial solution)
        # collected will always have the initial solution if collect_vars is not empty
        if any(collected.values()):
            out = {v_name: np.stack(values, axis=0) for v_name, values in collected.items() if values}
        else:  # Should not happen if initial solution is always added and collect_vars is not empty
            out = {v_name: np.array([]) for v_name in collect_vars}

    # Determine number of solutions returned
    # If out is empty (e.g. collect_vars=[]), then num_solutions is 0 for logging,
    # or handle based on collected dict before it's potentially emptied.
    num_solutions_returned = 0
    if collected and next(iter(collected)):  # Check if collected is not empty and has at least one key
        # Get the count from the first variable that has collected data
        # This assumes all collected variables will have the same number of samples
        first_collected_var_key = next(iter(collected))
        if collected[first_collected_var_key]:
            num_solutions_returned = len(collected[first_collected_var_key])

    logger.info(f"Done. accepted={n_accept}, rejected={n_reject}, solutions returned={num_solutions_returned}")
    return out
