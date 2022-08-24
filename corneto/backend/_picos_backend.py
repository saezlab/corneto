import warnings
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


class PicosProblemDef(ProblemDef):
    def __init__(
        self,
        symbols: List[CtProxySymbol],
        constraints: List[CtProxyExpression],
        objectives: Optional[List[CtProxyExpression]],
        weights: Optional[List[float]],
        direction: Direction = Direction.MIN,
    ) -> None:
        super().__init__(symbols, constraints, objectives, weights, direction)

    def _create(
        self,
        symbols: List[CtProxySymbol],
        constraints: List[CtProxyExpression],
        objectives: List[CtProxyExpression],
        weights: List[float],
        direction: Direction = Direction.MIN,
    ) -> "ProblemDef":
        return PicosProblemDef(symbols, constraints, objectives, weights, direction)

    def _build(self, obj: Optional[CtProxyExpression]) -> Any:
        P = pc.Problem()
        for c in self._constraints:
            P += c.e
        if obj is not None:
            if self._direction == Direction.MIN:
                P.minimize = obj.e
            else:
                P.maximize = obj.e
        return P

    def solve(
        self,
        solver: Solver = Solver.GLPK_MI,
        max_seconds: int = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ) -> Any:
        if warm_start:
            warnings.warn("warm_start is not supported yet, ignoring")
        P = self.build()
        P.solve(
            timelimit=max_seconds,
            solver=solver,
            verbosity=verbosity,
            hotstart=warm_start,
            **options,
        )
        return P


class PicosBackend(Backend):
    def load(self):
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

    def _create_problem(
        self,
        symbols: List[CtProxySymbol],
        constraints: List[CtProxyExpression],
        objectives: Optional[List[CtProxyExpression]] = None,
        weights: Optional[List[float]] = None,
        direction: Direction = Direction.MIN,
    ) -> ProblemDef:
        return PicosProblemDef(symbols, constraints, objectives, weights, direction)
