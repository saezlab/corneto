from multiprocessing.sharedctypes import Value
from typing import Set, Any, List, Optional, Tuple, Union
from corneto.backend._base import (
    CtProxyExpression,
    CtProxySymbol,
    ProblemDef,
    Backend,
    _proxy,
)
import numpy as np
from corneto._constants import *
from corneto._settings import LOGGER


try:
    import cvxpy as cp
except ImportError:
    cp = None  # type: ignore


class CvxpyExpression(CtProxyExpression):
    def __init__(
        self, expr: Any, symbols: Optional[Set["CtProxySymbol"]] = None
    ) -> None:
        super().__init__(expr, symbols)

    def _create_proxy_expr(
        self, expr: Any, symbols: Optional[Set["CtProxySymbol"]] = None
    ) -> "CvxpyExpression":
        # TODO: Move to upper class
        # if symbols is not None:
        #    return CvxpyExpression(expr, self._proxy_symbols | symbols)
        # return CvxpyExpression(expr, self._proxy_symbols)
        return CvxpyExpression(expr, symbols)

    def _elementwise_mul(self, other: Any) -> Any:
        return cp.multiply(self._expr, other)

    @property
    def value(self) -> np.ndarray:
        return self._expr.value


class CvxpySymbol(CtProxySymbol, CvxpyExpression):
    def __init__(
        self,
        expr: Any,
        name: str,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> None:
        super().__init__(expr, name, lb, ub, vartype)


class CvxpyBackend(Backend):
    def _load(self):
        import cvxpy

        return cvxpy

    def __str__(self) -> str:
        return "CVXPY"

    def available_solvers(self) -> List[str]:
        return cp.installed_solvers()

    def Variable(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> CtProxySymbol:
        if not name:
            from uuid import uuid4

            name = hex(hash(uuid4()))
        if shape is None:
            shape = ()  # type: ignore
        if vartype == VarType.INTEGER:
            v = cp.Variable(shape, name=name, integer=True)
        elif vartype == VarType.BINARY:
            v = cp.Variable(shape, name=name, boolean=True)
        else:
            v = cp.Variable(shape, name=name)
        return CvxpySymbol(v, name, lb, ub, vartype)

    def _solve(
        self,
        p: ProblemDef,
        objective: Optional[CtProxyExpression] = None,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: int = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ) -> Any:
        o: Union[cp.Minimize, cp.Maximize]
        if objective is not None:
            obj = objective.e if hasattr(objective, "_expr") else objective
            if p._direction == Direction.MIN:
                o = cp.Minimize(obj)
            elif p._direction == Direction.MAX:
                o = cp.Maximize(obj)
        # Go through all the vars/params/etc to check bounds
        # and add the required ones.
        extras = []
        # Get variables
        for v in p.symbols.values():
            if v.lb is not None:
                extras.append(v >= v.lb)
            if v.ub is not None:
                extras.append(v <= v.ub)
        cstr = [c.e for c in extras + p.constraints]
        P = cp.Problem(o, cstr)
        s = solver
        if solver:
            s = solver.upper()
        solvers = cp.installed_solvers()
        if s is not None and s not in solvers:
            raise ValueError(
                f"Solver {s} is not installed/supported, supported solvers are: {solvers}"
            )
        # Translate max second depending on solver
        # TODO: Test that the translation of maxtime works for all solvers. Improve
        # parameter mapping.
        if max_seconds is not None:
            if s == "GUROBI":
                # https://www.gurobi.com/documentation/9.1/refman/parameters.html
                options["TimeLimit"] = max_seconds
            elif s == "COIN_OR_CBC":
                options["maximumSeconds"] = max_seconds
            elif s == "CPLEX":
                options["cplex_params"] = {"timelimit": max_seconds}
            elif s == "GLPK_MI":
                LOGGER.warn("Timelimit for GLPK_MI is not supported")
            elif s == "SCIP":
                # https://www.scipopt.org/doc/html/PARAMETERS.php
                options["scip_params"] = {"limits/time": max_seconds}
            elif s == "MOSEK":
                # https://docs.mosek.com/latest/cxxfusion/parameters.html
                options["mosek_params"] = {"mioMaxTime": max_seconds}
        P.solve(solver=s, verbose=verbosity > 0, warm_start=warm_start, **options)
        return P
