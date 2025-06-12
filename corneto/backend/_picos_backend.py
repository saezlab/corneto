from numbers import Number
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np

from corneto._constants import *
from corneto._settings import _numpy_array
from corneto.backend._base import (
    Backend,
    CExpression,
    CSymbol,
    ProblemDef,
    _get_unique_name,
)

try:
    import picos as pc
except ImportError:
    pc = None


def _get_shape(a: Any):
    if hasattr(a, "_csymbol_shape"):
        return a._csymbol_shape
    if hasattr(a, "_expr"):
        return a._expr._csymbol_shape if hasattr(a._expr, "_csymbol_shape") else a._expr.shape
    if hasattr(a, "shape"):
        return a.shape
    return ()


def _infer_shape(c: Any):
    shape = _get_shape(c)
    if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
        # This is the problematic case, as any transformation
        # using PICOS will result in a 2D array.
        if hasattr(c, "_proxy_symbols"):
            for s in c._proxy_symbols:
                # If there is some original symbol used in the expression
                # with ndim 1, and the current shape contains a dimension
                # with a 1, then we assume the original shape was 1D.
                s_shape = _get_shape(s)
                if len(s_shape) == 1:
                    if shape[0] == 1:
                        return (shape[1],)
                    return (shape[0],)
    return shape


class PicosExpression(CExpression):
    def __init__(self, expr: Any, symbols: Optional[Set["CSymbol"]] = None) -> None:
        super().__init__(expr, symbols)

    @property
    def shape(self) -> Tuple[int, ...]:
        return _infer_shape(self)

    def _create_proxy_expr(self, expr: Any, symbols: Optional[Set["CSymbol"]] = None) -> "PicosExpression":
        return PicosExpression(expr, symbols)

    def _elementwise_mul(self, other: Any) -> Any:
        return self._expr ^ other

    def _norm(self, p: int = 2) -> CExpression:
        return pc.Norm(self._expr, p=p)

    def _sum(self, axis: Optional[int] = None) -> Any:
        return pc.sum(self._expr, axis=axis)

    def _max(self, axis: Optional[int] = None) -> Any:
        raise NotImplementedError()

    def _abs(self) -> Any:
        raise NotImplementedError()

    def _hstack(self, other: Any) -> Any:
        a = self._expr
        b = other
        # PICOS assumes vectors are column vectors.
        # For hstack, if 1 dim, we assume is a row vector.
        a_shape = _infer_shape(self)
        b_shape = _infer_shape(other)
        # If both are 1-dim vecs, concatenate on columns
        if len(a_shape) == 1 and len(b_shape) == 1:
            a = a.reshaped((1, a_shape[0]))
            b = b.reshaped((1, b_shape[0]))
        return a & b

    def _vstack(self, other: Any) -> Any:
        a = self._expr
        b = other
        # PICOS assumes vectors are column vectors.
        # We need to keep track of the original dim.
        a_shape = _infer_shape(self)
        b_shape = _infer_shape(other)
        if len(a_shape) == 1:
            a = a.reshaped((1, a_shape[0]))
        if len(b_shape) == 1:
            b = b.reshaped((1, b_shape[0]))
        return a // b

    def _reshape(self, shape: Tuple[int, ...]) -> Any:
        return self._expr.reshaped(shape)

    @property
    def value(self) -> np.ndarray:
        return np.array(self._expr.value)

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
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[Number, np.ndarray]] = None,
        ub: Optional[Union[Number, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
        variable: bool = True,
    ) -> None:
        super().__init__(expr, name, shape=shape, lb=lb, ub=ub, vartype=vartype, variable=variable)

    @CSymbol.value.setter
    def value(self, value: Any) -> None:
        param = pc.Constant(self._name, value=value, shape=self._shape)
        self._expr.__dict__.clear()
        self._expr.__dict__.update(param.__dict__)
        if hasattr(self._expr, "__slots__"):
            for slot in self.__slots__:
                setattr(self, slot, getattr(param, slot, None))
        # Warn about limitations of this approach
        from corneto._settings import LOGGER

        LOGGER.warn(
            "PICOS Parameters are immutable, changing the value of a parameter "
            "is only partially supported, but changing it after after the "
            "problem is solved will not have any effect."
        )


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

    def Constant(self, value: Any, name: Optional[str] = None) -> CSymbol:
        name = name or _get_unique_name(prefix="const")
        v = pc.Constant(name, value=value)
        return PicosSymbol(v, name, shape=(), variable=False)

    def Variable(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[Number, np.ndarray]] = None,
        ub: Optional[Union[Number, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
        variable: bool = True,
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
        return PicosSymbol(v, name, shape=shape, lb=lb, ub=ub, vartype=vartype, variable=variable)

    def Parameter(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        value: Any = None,
    ) -> CSymbol:
        shape = shape or ()
        value = value or 0
        name = name or _get_unique_name()
        param = pc.Constant(name, value=value, shape=shape)
        return PicosSymbol(param, name, shape=shape, variable=False)

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
        # Use default solver tolerances
        P.options["*_tol"] = None
        # P.options["primals"] = False
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
