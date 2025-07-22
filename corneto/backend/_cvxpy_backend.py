from numbers import Number
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np

from corneto._constants import *
from corneto._settings import LOGGER
from corneto.backend._base import (
    Backend,
    CExpression,
    CSymbol,
    ProblemDef,
    _get_unique_name,
)

try:
    import cvxpy as cp
except ImportError:
    cp = None  # type: ignore


class CvxpyExpression(CExpression):
    def __init__(self, expr: Any, symbols: Optional[Set["CSymbol"]] = None) -> None:
        super().__init__(expr, symbols)

    def _create_proxy_expr(self, expr: Any, symbols: Optional[Set["CSymbol"]] = None) -> "CvxpyExpression":
        return CvxpyExpression(expr, symbols)

    def _elementwise_mul(self, other: Any) -> Any:
        return cp.multiply(self._expr, other)

    def _norm(self, p: int = 2) -> Any:
        return cp.norm(self._expr, p=p)

    def _sum(self, axis: Optional[int] = None) -> Any:
        return cp.sum(self._expr, axis=axis)

    def _max(self, axis: Optional[int] = None) -> Any:
        return cp.max(self._expr, axis=axis)

    # def _abs(self) -> Any:
    #    return cp.abs(self._expr)

    def _hstack(self, other: Any) -> Any:
        return cp.hstack([self._expr, other])

    def _vstack(self, other: Any) -> Any:
        a = self._expr
        b = other
        # If vector, for vstack assume is a row vector
        if len(a.shape) == 1:
            a = cp.reshape(a, (1, a.shape[0]), order="F")
        if len(b.shape) == 1:
            b = cp.reshape(b, (1, b.shape[0]), order="F")
        return cp.vstack([a, b])

    def _reshape(self, shape: Tuple[int, ...]) -> Any:
        return cp.reshape(self._expr, shape, order="F")

    @property
    def value(self) -> np.ndarray:
        return self._expr.value


class CvxpySymbol(CSymbol, CvxpyExpression):
    def __init__(
        self,
        expr: Any,
        name: str,
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[Number, np.ndarray]] = None,
        ub: Optional[Union[Number, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
        variable: bool = True,
    ) -> None:
        super().__init__(expr, name, shape=shape, lb=lb, ub=ub, vartype=vartype, variable=variable)


class CvxpyBackend(Backend):
    def _load(self):
        import cvxpy

        return cvxpy

    def __str__(self) -> str:
        return "CVXPY"

    def available_solvers(self) -> List[str]:
        return cp.installed_solvers()

    def Constant(self, value: Any, name: Optional[str] = None) -> CSymbol:
        name = name or _get_unique_name(prefix="const")
        v = cp.Constant(value, name=name)
        return CvxpySymbol(v, name, shape=(), variable=False)

    def Variable(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[Number, np.ndarray]] = None,
        ub: Optional[Union[Number, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> CSymbol:
        if vartype == VarType.BINARY:
            lb, ub = None, None
        shape = shape or ()
        name = name or _get_unique_name()
        if vartype == VarType.INTEGER:
            v = cp.Variable(shape, name=name, integer=True)
        elif vartype == VarType.BINARY:
            v = cp.Variable(shape, name=name, boolean=True)
        else:
            v = cp.Variable(shape, name=name)
        return CvxpySymbol(v, name, shape=shape, lb=lb, ub=ub, vartype=vartype)

    def Parameter(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        value: Any = None,
    ) -> CSymbol:
        shape = shape or ()
        name = name or _get_unique_name()
        param = cp.Parameter(shape=shape, name=name, value=value)
        return CvxpySymbol(param, name, shape=shape, variable=False)

    def build(self, p: ProblemDef) -> Any:
        raise NotImplementedError()

    def _solve(
        self,
        p: ProblemDef,
        objective: Optional[CExpression] = None,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: int = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ) -> Any:
        o: Optional[Union[cp.Minimize, cp.Maximize]] = None
        if objective is not None:
            obj = objective.e if hasattr(objective, "_expr") else objective
            if p._direction == Direction.MIN:
                o = cp.Minimize(obj)
            elif p._direction == Direction.MAX:
                o = cp.Maximize(obj)
        if o is None:
            o = cp.Minimize(0)

        # Go through all the vars/params/etc to check bounds
        # and add the required ones.
        extras = []
        # Get variables
        # TODO: Check what happens if lb/ub are matrices
        for v in p.symbols.values():
            if v._provided_lb is not None:
                extras.append(v >= v.lb)
            if v._provided_ub is not None:
                extras.append(v <= v.ub)

        cstr = [c.e for c in extras + p.constraints]
        cvxpy_problem = cp.Problem(o, cstr)
        s = solver
        if solver:
            s = solver.upper()
        solvers = cp.installed_solvers()
        if s is not None and s not in solvers:
            raise ValueError(f"Solver {s} is not installed/supported, supported solvers are: {solvers}")
        # TODO: Implement parameter mapping for solvers and backends
        if max_seconds is not None:
            if s == "GUROBI":
                # https://www.gurobi.com/documentation/9.1/refman/parameters.html
                options["TimeLimit"] = max_seconds
            elif s == "COIN_OR_CBC":
                options["maximumSeconds"] = max_seconds
            elif s == "CPLEX":
                cfg = options.get("cplex_params", dict())
                cfg.update({"timelimit": max_seconds})
                options["cplex_params"] = cfg
            elif s == "GLPK_MI":
                LOGGER.warn("Timelimit for GLPK_MI is not supported")
            elif s == "SCIP":
                # https://www.scipopt.org/doc/html/PARAMETERS.php
                cfg = options.get("scip_params", dict())
                cfg.update({"limits/time": max_seconds})
                options["scip_params"] = cfg
            elif s == "MOSEK":
                # https://docs.mosek.com/latest/cxxfusion/parameters.html
                cfg = options.get("mosek_params", dict())
                cfg.update({"mioMaxTime": float(max_seconds)})
                options["mosek_params"] = cfg
            elif s == "SCIPY":
                cfg = options.get("scipy_options", dict())
                cfg.update({"time_limit": float(max_seconds), "disp": verbosity > 0})
                options["scipy_options"] = cfg
            elif s == "HIGHS":
                options["time_limit"] = int(max_seconds)
            else:
                # Warning that a mapping is not available, check backend documentation
                LOGGER.warn(f"""max_seconds parameter mapping for {s} not found.
                            Please refer to the backend documentation for more
                            information. For example, using CVXPY with GUROBI solver,
                            the parameter TimeLimit can be directly passed with
                            `problem.solve(solver='GUROBI', TimeLimit=max_seconds)`""")

        cvxpy_problem.solve(solver=s, verbose=verbosity > 0, warm_start=warm_start, **options)
        return cvxpy_problem
