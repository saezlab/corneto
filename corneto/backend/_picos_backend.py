import numpy as np
from typing import Set, Any, List, Optional, Tuple, Union
from corneto.backend._base import CtProxyExpression, CtProxySymbol, Backend, ProblemDef
from corneto._constants import *

try:
    import picos as pc
except ImportError:
    pc = None


class PicosExpression(CtProxyExpression):
    def __init__(
        self, expr: Any, symbols: Optional[Set["CtProxySymbol"]] = None
    ) -> None:
        super().__init__(expr, symbols)

    def _create_proxy_expr(
        self, expr: Any, symbols: Optional[Set["CtProxySymbol"]] = None
    ) -> "PicosExpression":
        # if symbols is not None:
        #    return PicosExpression(expr, self._proxy_symbols | symbols)
        # return PicosExpression(expr, self._proxy_symbols)
        return PicosExpression(expr, symbols)

    def _elementwise_mul(self, other: Any) -> Any:
        return self._expr ^ other

    @property
    def value(self) -> np.ndarray:
        return self._expr.value

    def __matmul__(self, other: Any) -> "CtProxyExpression":
        o = other
        if isinstance(other, CtProxyExpression):
            # Get internal PICOS expression to operate with
            o = other._expr
        return self._create(self._expr * o, [other])

    def __rmatmul__(self, other: Any) -> "CtProxyExpression":
        o = other
        if isinstance(other, CtProxyExpression):
            o = other._expr
        return self._create(o * self._expr, [other])


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

        return picos

    def __str__(self) -> str:
        return "PICOS"

    def available_solvers(self) -> List[str]:
        return pc.available_solvers()

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
