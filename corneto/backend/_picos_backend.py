import numpy as np
from typing import Any, List, Optional, Tuple, Union
from corneto.backend._base import CtProxyExpression, CtProxySymbol, Backend, ProblemDef
from corneto._constants import *

try:
    import picos as pc
except ImportError:
    pc = None


class PicosExpression(CtProxyExpression):
    def __init__(self, expr, parent: Optional[CtProxyExpression] = None) -> None:
        super().__init__(expr, parent)

    def _create(self, expr: Any) -> "PicosExpression":
        return PicosExpression(expr, self)

    def _elementwise_mul(self, other: Any) -> Any:
        return self._expr ^ other

    @property
    def value(self) -> np.ndarray:
        return self._expr.value

    def __matmul__(self, other: Any) -> "CtProxyExpression":
        if isinstance(other, CtProxyExpression):
            other = other._expr
        return self._create(self._expr * other)

    def __rmatmul__(self, other: Any) -> "CtProxyExpression":
        if isinstance(other, CtProxyExpression):
            other = other._expr
        return self._create(other * self._expr)


class PicosSymbol(CtProxySymbol, PicosExpression):
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
    def _load(self):
        import picos

        picos

    def Variable(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> CtProxySymbol:
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
        objective: Optional[CtProxyExpression] = None,
        solver: Solver = Solver.GLPK_MI,
        max_seconds: int = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ):
        P = pc.Problem()
        for c in p.constraints:
            P += c.e
        if objective is not None:
            obj = objective.e if hasattr(objective, '_expr') else objective
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
