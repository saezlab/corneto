import numpy as np
from typing import Set, Any, List, Optional, Tuple, Union
from corneto._settings import _numpy_array
from corneto.backend._base import CExpression, CSymbol, Backend, ProblemDef
from corneto._constants import *

try:
    import picos as pc
except ImportError:
    pc = None


class PicosExpression(CExpression):
    def __init__(self, expr: Any, symbols: Optional[Set["CSymbol"]] = None) -> None:
        super().__init__(expr, symbols)

    def _create_proxy_expr(
        self, expr: Any, symbols: Optional[Set["CSymbol"]] = None
    ) -> "PicosExpression":
        # if symbols is not None:
        #    return PicosExpression(expr, self._proxy_symbols | symbols)
        # return PicosExpression(expr, self._proxy_symbols)
        return PicosExpression(expr, symbols)

    def _elementwise_mul(self, other: Any) -> Any:
        return self._expr ^ other

    def _norm(self, p: int = 2) -> CExpression:
        return pc.expressions.exp_norm.Norm(self._expr, p=p)
    
    def _sum(self, axis: Optional[int] = None) -> Any:
        return pc.expressions.sum(self._expr, axis=axis)

    @property
    def value(self) -> np.ndarray:
        return self._expr.value

    def __matmul__(self, other: Any) -> "CExpression":
        o = other
        if isinstance(other, CExpression):
            # Get internal PICOS expression to operate with
            o = other._expr
        return self._create(self._expr * o, [other])

    def __rmatmul__(self, other: Any) -> "CExpression":
        o = other
        if isinstance(other, CExpression):
            o = other._expr
        return self._create(o * self._expr, [other])


class PicosSymbol(CSymbol, PicosExpression):
    def __init__(
        self,
        expr: Any,
        name: str,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> None:
        super().__init__(expr, name, lb, ub, vartype)


class PicosBackend(Backend):
    def __init__(self, default_solver: Optional[str] = None) -> None:
        super().__init__(default_solver, _numpy_array)

    def _load(self):
        import picos

        return picos

    def __str__(self) -> str:
        return "PICOS"

    def available_solvers(self) -> List[str]:
        return pc.available_solvers()

    def build(self, p: ProblemDef) -> Any:
        raise NotImplementedError()

    def Variable(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> CSymbol:
        if shape is None:
            shape = ()  # type: ignore
        if name is None:
            name = ""
        if vartype == VarType.INTEGER:
            v = pc.IntegerVariable(name, shape, lower=lb, upper=ub)
        elif vartype == VarType.BINARY:
            v = pc.BinaryVariable(name, shape)
        else:
            v = pc.RealVariable(name, shape, lower=lb, upper=ub)
        return PicosSymbol(v, name, lb, ub)

    def _solve(
        self,
        p: ProblemDef,
        objective: Optional[CExpression] = None,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: int = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ):
        P = pc.Problem()
        for c in p.constraints:
            P += c.e
        if objective is not None:
            obj = objective.e if hasattr(objective, "_expr") else objective
            if p._direction == Direction.MIN:
                P.minimize = obj
            else:
                P.maximize = obj
        if warm_start:
            from corneto._settings import LOGGER

            LOGGER.warn("warm_start is not supported yet, ignored")
        P.solve(
            timelimit=max_seconds,
            solver=solver,
            verbosity=verbosity,
            hotstart=warm_start,
            **options,
        )
        return P
