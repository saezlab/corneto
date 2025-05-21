import logging
import numpy as np
from collections.abc import Sequence
from typing import Dict, List, Union

# Configure module-level logger; users can configure handlers in their application
logger = logging.getLogger(__name__)


def sample_alternative_solutions(
    problem,
    variable_name: str,
    *,
    percentage: float = 0.10,
    scale: float = 0.03,
    rel_opt_tol: float = 0.05,
    max_samples: int = 30,
    perturbation_name: str = "perturbation",
    solver_kwargs: dict | None = None,
    rng: np.random.Generator | int | None = None,
    collect_vars: Sequence[str] | None = None,
    verbose: int = 1,  # 0 = silent, 1 = summary, 2 = full detail
) -> Dict[str, np.ndarray]:
    """Sample alternative solutions by perturbing a chosen decision variable.

    This routine takes an optimization problem (with attributes .expr,
    .solve(), and .objectives), identifies a target variable within it,
    and generates up to max_samples new feasible solutions by randomly
    perturbing a fraction of that variable's entries.  Only those perturbations
    that keep all original objectives within a relative tolerance of the
    baseline are accepted.

    Args:
        problem: An optimization problem instance exposing
            - expr: a mapping of variable names to variable objects,
            - solve(...): method to solve the problem,
            - objectives: list of objective objects (each with .name and .value).
        variable_name (str): Name of the variable in problem.expr to perturb.
        percentage (float, optional): Fraction of the variable's entries to
            perturb in each trial (default 0.10).
        scale (float, optional): Standard deviation of the normal random noise
            (default 0.03).
        rel_opt_tol (float, optional): Maximum allowed relative deviation of
            any original objective from its baseline value (default 0.05).
        max_samples (int, optional): Maximum number of perturbation trials
            to attempt (default 30).
        perturbation_name (str, optional): Name to assign to the added
            perturbation objective (default `"perturbation").
        solver_kwargs (dict or None, optional): Extra keyword arguments passed
            to problem.solve() (default None).
        rng (np.random.Generator or int or None, optional): Random number
            generator or seed for reproducibility (default None).
        collect_vars (Sequence[str] or None, optional):
            Names of the variables whose values you want back.

            - `None (default) – collect **every** variable in problem.expr
            - `[] – collect **none** (method solves but returns an empty dict)
            - `["x", "y"] – collect only those named variables
        verbose (int, optional): Verbosity level:
            0 = silent, 1 = summary, 2 = full detail (default 1).

    Returns:
        dict:
            A dictionary that maps each collected variable name to a NumPy
            array with shape `(n_samples, *variable.shape) where

            * `n_samples ≥ 1 – it counts the incumbent plus every accepted
              perturbation;
            * the remaining dimensions match the variable’s own shape.

            Example::

                out = sample_alternative_solutions(problem, "x", collect_vars=["x", "y"])
                x_stack = out["x"]        # shape (n_samples, *x.shape)
                incumbent_x = x_stack[0]  # first slice is always the baseline

    Raises:
        KeyError:
            If `variable_name is not in problem.expr **or** if any name
            inside `collect_vars is missing from problem.expr.
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

    # ------------------ sanity checks ------------------
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

    # 1) original solve ---------------------------------
    logger.debug("Solving original model …")
    problem.solve(**solver_kwargs, verbosity=0)
    baseline_obj = {o.name: float(o.value) for o in problem.objectives}
    logger.debug(
        "Baseline objectives: "
        + ", ".join(f"{k}={v:.6g}" for k, v in baseline_obj.items())
    )

    for v in collect_vars:
        collected[v].append(np.asarray(problem.expr[v].value).copy())

    # 2) build perturbation parameter -------------------
    var_shape = tuple(int(s) for s in target_var.shape)
    total_elems = int(np.prod(var_shape))
    n_perturb = max(1, int(total_elems * percentage))

    noise_buf = np.zeros(var_shape, dtype=float)
    pert = problem.backend.Parameter(
        name=f"{perturbation_name}_param", shape=var_shape, value=noise_buf
    )
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
        relerrs = {}
        current_vals = {}
        for o in problem.objectives:
            if o.name == perturbation_name:
                continue
            val = float(o.value)
            current_vals[o.name] = val
            denom = max(abs(baseline_obj[o.name]), 1e-9)
            relerrs[o.name] = abs(val - baseline_obj[o.name]) / denom

        # check tolerance
        violated = next(
            ((name, err) for name, err in relerrs.items() if err > rel_opt_tol), None
        )

        # log objective values and errors
        detail_msg = ", ".join(
            f"{name}: val={current_vals[name]:.6g}, rel.err={relerrs[name]:.4f}"
            for name in current_vals
        )

        if violated is None:
            for v in collect_vars:
                collected[v].append(np.asarray(problem.expr[v].value).copy())
            n_accept += 1
            logger.info(
                f"[{trial}/{max_samples}] accepted (total accepted={n_accept}) -> {detail_msg}"
            )
        else:
            n_reject += 1
            logger.info(
                f"[{trial}/{max_samples}] rejected (tol={rel_opt_tol}) -> {detail_msg}"
            )

    # 4) stack lists into arrays ------------------------
    out: Dict[str, np.ndarray] = {
        v: np.stack(values, axis=0) for v, values in collected.items()
    }

    logger.info(
        f"Done. accepted={n_accept}, rejected={n_reject}, solutions returned="
        f"{out[next(iter(out))].shape[0]}"
    )
    return out
