import abc
import numbers
import warnings
from copy import copy as shallow_copy
from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from corneto._constants import *
from corneto._decorators import _delegate
from corneto._graph import BaseGraph
from corneto._settings import LOGGER, _get_matrix_builder
from corneto.utils import Attributes


def _eq_shape(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        if len(a.shape) == 1 and len(b.shape) == 2:
            return a.shape[0] == b.shape[0] and b.shape[1] == 1
        if len(a.shape) == 2 and len(b.shape) == 1:
            return a.shape[0] == b.shape[0] and a.shape[1] == 1
    return a.shape == b.shape


def _identical_columns(array):
    # Get the first column as a reference column
    ref_column = array[:, 0]

    # Compare all columns to the reference column
    # np.all will check if all elements in the result are True along axis 0 (down the rows)
    are_columns_identical = np.all(array == ref_column[:, np.newaxis], axis=0)

    # np.all on the result checks if all columns are identical to the reference column
    return np.all(are_columns_identical)


def _get_unique_name(prefix: str = "_var") -> str:
    from uuid import uuid4

    return prefix + hex(hash(uuid4()))


class CExpression(abc.ABC):
    # Arithmetic operator overloading with Numpy
    # See: https://www.cvxpy.org/_modules/cvxpy/expressions/expression.html#Expression
    __array_priority__ = 100

    def __init__(self, expr: Any, symbols: Optional[Set["CSymbol"]] = None) -> None:
        super().__init__()
        self._expr = expr
        self._proxy_symbols: Set["CSymbol"] = set()
        self._name = ""
        if symbols:
            self._proxy_symbols.update(symbols)

    def is_symbol(self) -> bool:
        return False

    def _create(self, expr: Any, atoms: Iterable) -> "CExpression":
        symbols = {s for s in atoms if isinstance(s, CSymbol)}
        if isinstance(self, CSymbol):
            symbols.add(self)
        if isinstance(self, CExpression):
            symbols.update(self._proxy_symbols)
        if isinstance(expr, CSymbol):
            symbols.add(expr)
        if isinstance(expr, CExpression):
            symbols.update(expr._proxy_symbols)
        # Ask to create a CVXPY/PICOS/.. expression
        return self._create_proxy_expr(expr, symbols)

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def _create_proxy_expr(self, expr: Any, symbols: Optional[Set["CSymbol"]] = None) -> "CExpression":
        pass

    @property
    @abc.abstractmethod
    def value(self) -> Union[Number, np.ndarray]:
        pass

    @property
    def e(self) -> Any:
        return self._expr

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._expr.shape

    def __hash__(self) -> int:
        return self._expr.__hash__()

    def apply(self, fun, *args, **kwargs) -> "CExpression":
        return self._create(fun(self._expr, *args, **kwargs), {})

    @property
    def T(self):
        return self._create(self._expr.T, {})

    @abc.abstractmethod
    def _elementwise_mul(self, other: Any) -> Any:
        pass

    @_delegate
    def multiply(self, other: Any) -> "CExpression":
        return self._elementwise_mul(other)

    @abc.abstractmethod
    def _hstack(self, other: "CExpression") -> Any:
        pass

    @_delegate
    def hstack(self, other: "CExpression") -> "CExpression":
        return self._hstack(other)

    @abc.abstractmethod
    def _vstack(self, other: "CExpression") -> Any:
        pass

    @_delegate
    def vstack(self, other: "CExpression") -> "CExpression":
        return self._vstack(other)

    @abc.abstractmethod
    def _reshape(self, shape: Tuple[int, ...]) -> "CExpression":
        pass

    @_delegate(override=True)
    def reshape(self, shape: Union[int, Tuple[int, ...]]) -> "CExpression":
        this_shape = self.shape
        num_elements = 1
        for dim in this_shape:
            num_elements *= dim

        # Convert single int shape to tuple
        if isinstance(shape, int):
            shape = (shape,)

        # Validate the input shape
        if shape.count(-1) > 1:
            raise ValueError("Only one dimension can be -1")
        if any(dim < -1 for dim in shape):
            raise ValueError("Invalid shape: dimensions must be positive or -1")

        # Handle the case where shape is (-1,) or -1 to flatten the array
        if shape == (-1,):
            return self._reshape((num_elements,))

        # General case: if -1 is present, calculate the corresponding dimension
        if -1 in shape:
            new_shape = []
            unknown_index = shape.index(-1)
            known_size = 1

            for i, dim in enumerate(shape):
                if i != unknown_index:
                    known_size *= dim
                new_shape.append(dim)

            # Check that total elements match
            if num_elements % known_size != 0:
                raise ValueError("The total size of the new array must be unchanged")

            new_shape[unknown_index] = num_elements // known_size
            return self._reshape(tuple(new_shape))

        # Check total size is ok
        new_num_elements = 1
        for dim in shape:
            new_num_elements *= dim

        if new_num_elements != num_elements:
            raise ValueError("The total size of the new array must be unchanged")

        return self._reshape(shape)

    @abc.abstractmethod
    def _norm(self, p: int = 2) -> Any:
        pass

    @_delegate
    def norm(self, p: int = 2) -> "CExpression":
        return self._norm(p=p)

    @abc.abstractmethod
    def _sum(self, axis: Optional[int] = None) -> Any:
        pass

    @_delegate
    def sum(self, axis: Optional[int] = None) -> "CExpression":
        return self._sum(axis=axis)

    @abc.abstractmethod
    def _max(self, axis: Optional[int] = None) -> Any:
        pass

    @_delegate
    def max(self, axis: Optional[int] = None) -> "CExpression":
        return self._max(axis=axis)

    # @abc.abstractmethod
    # def _abs(self) -> Any:
    #    pass

    # @_delegate
    # def abs(self) -> "CExpression":
    #    return self._abs()

    # These delegated methods are invoked directly in the backend
    # and wrapped thanks to the _delegate decorator. If a new
    # backend has a different behavior, provide an abstract method
    # as in the previous cases.

    @_delegate
    def __getitem__(self, item) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __abs__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __pow__(self, power: float) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rpow__(self, base: float) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __add__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __radd__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __sub__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rsub__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __mul__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __matmul__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __truediv__(self, other: "CExpression") -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __div__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rdiv__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rtruediv__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rmul__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rmatmul__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __neg__(self) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __eq__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __le__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __lt__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __ge__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __gt__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    def __str__(self) -> str:
        if self._name:
            return f"{self._name}: {self._expr.__str__()}"
        return self._expr.__str__()

    def __repr__(self) -> str:
        if self._name:
            return f"{self._name}: {self._expr.__repr__()}"
        return self._expr.__repr__()

    # TODO: add functions along axis: https://www.cvxpy.org/tutorial/functions/index.html


class CSymbol(CExpression):
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
        """Create a symbol.

        This method defines an optimization symbol with optional bounds, type, and
        additional graph-related properties. Symbols can be variables, parameters
        or constants.

        Args:
            name (Optional[str]): The name of the symbol. Defaults to None.
            expr (Any): The expression of the symbol.
            shape (Optional[Tuple[int, ...]]): The shape of the symbol as a tuple. Defaults to None.
            lb (Optional[Union[float, np.ndarray]]): The lower bound of the symbol.
                Can be a scalar or an array. Defaults to None.
            ub (Optional[Union[float, np.ndarray]]): The upper bound of the symbol.
                Can be a scalar or an array. Defaults to None.
            vartype (VarType): The type of the symbol (e.g., continuous, integer).
                Defaults to VarType.CONTINUOUS.
            variable (bool): Whether the symbol is a variable or not. Defaults to True.

        Returns:
            CSymbol: The created symbol, to be used in further expressions or constraints.
        """
        super().__init__(expr)
        lb_r: Optional[np.ndarray] = None
        ub_r: Optional[np.ndarray] = None
        self._provided_lb = lb
        self._provided_ub = ub
        setattr(expr, "_csymbol_shape", shape)

        if shape is None:
            shape = ()  # type: ignore
        self._shape = shape
        self._is_variable = variable

        if lb is None:
            if vartype == VarType.CONTINUOUS:
                lb_r = np.full(expr.shape, -np.inf)
                self._lb_ignore = True
            elif vartype == VarType.INTEGER:
                lb_r = np.full(expr.shape, np.iinfo(int).min)
                self._lb_ignore = True
            elif vartype == VarType.BINARY:
                lb_r = np.zeros(expr.shape)
        if ub is None:
            if vartype == VarType.CONTINUOUS:
                ub_r = np.full(expr.shape, np.inf)
                self._ub_ignore = True
            elif vartype == VarType.INTEGER:
                ub_r = np.full(expr.shape, np.iinfo(int).max)
                self._ub_ignore = True
            elif vartype == VarType.BINARY:
                ub_r = np.ones(expr.shape)
        if isinstance(lb, np.ndarray):
            # TODO: change the way we handle this
            if not _eq_shape(lb, expr):
                raise ValueError(f"Shape of lb is {lb.shape}, whereas symbol has a shape of {expr.shape}")
            lb_r = lb
        elif isinstance(lb, numbers.Number):
            lb_r = np.full(expr.shape, lb)
        else:
            if lb is not None:
                raise ValueError(f"lb has an invalid type ({type(lb)}). It must be a number or numpy array")
        if isinstance(ub, np.ndarray):
            if not _eq_shape(ub, expr):
                raise ValueError(f"Shape of ub is {ub.shape}, whereas symbol has a shape of {expr.shape}")
            ub_r = ub
        elif isinstance(ub, numbers.Number):
            ub_r = np.full(expr.shape, ub)
        else:
            if ub is not None:
                raise ValueError(f"ub has an invalid type ({type(ub)}). It must be a number or numpy array")
        self._lb = lb_r
        self._ub = ub_r
        self._name = name
        self._vartype = vartype

    def is_symbol(self) -> bool:
        return True

    @property
    def lb(self) -> np.ndarray:
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        return self._ub

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def value(self) -> Any:
        return self._expr.value

    @value.setter
    def value(self, value: Any) -> None:
        self._expr.value = value

    @property
    def is_variable(self) -> bool:
        return self._is_variable


class ProblemDef:
    def __init__(
        self,
        backend: Optional["Backend"] = None,
        constraints: Optional[List[CExpression]] = None,
        objectives: Optional[List[CExpression]] = None,
        expressions: Optional[Dict[str, CExpression]] = None,
        weights: Optional[List[float]] = None,
        direction: Direction = Direction.MIN,
        graph: Optional[BaseGraph] = None,  # TODO: decouple this from backend
    ) -> None:
        if objectives is None:
            objectives = []
        if weights is None:
            weights = [1.0] * len(objectives)
        else:
            if len(weights) != len(objectives):
                raise ValueError(
                    f"The number of weights ({len(weights)}) should match the number of objectives ({len(objectives)})"
                )
        self._backend = backend
        self._constraints = constraints if constraints else []
        self._objectives = objectives if objectives else []
        self._weights = weights if weights else []
        self._direction = direction
        # Registered expressions with names
        self._expressions = dict()
        if expressions is not None:
            self._expressions.update(expressions)
        # Create a new class of problems on top of a graph
        # where edges/nodes have associated optimization variables
        # TODO: check which use cases are using _graph
        if graph is not None:
            # Warn that this is deprecated
            warnings.warn("The graph parameter is deprecated.", DeprecationWarning)
            self._graph = graph

    @property
    def symbols(self) -> Dict[str, CSymbol]:
        # show deprecated warning:
        # warnings.warn("Use ProblemDef.expressions instead.", DeprecationWarning)
        return {s.name: s for s in Backend.get_symbols(self._constraints + self._objectives)}

    @property
    def expressions(self) -> Attributes:
        attr = Attributes()
        sym = self.symbols
        attr.update(self.symbols)
        attr.update({k: v for k, v in self._expressions.items() if k not in sym})
        return attr

    @property
    def expr(self) -> Attributes:
        return self.expressions

    @property
    def backend(self) -> Optional["Backend"]:
        return self._backend

    def get_symbol(self, name) -> CSymbol:
        return self.symbols[name]

    def get_symbols(self, *args) -> List[CSymbol]:
        return [self.get_symbol(n) for n in args]

    def add_suffix(self, suffix: str, inplace: bool = False) -> "ProblemDef":
        o = self
        if not inplace:
            o = self.copy()
        obs = set()
        for e in self._constraints + self._objectives:
            s = getattr(e, "_proxy_symbols", {})
            for x in s:
                # TODO: x.rename(...) should be a symbol specific thing
                if hasattr(x, "_name"):
                    # ad-hoc symbol renaming
                    if hasattr(x.e, "_symbStr"):  # PICOS, move to rename
                        x.e._symbStr = x.e._symbStr + suffix
                    if hasattr(x.e, "_name"):
                        x.e._name = x.e._name + suffix
                    if x not in obs:
                        x._name = x._name + suffix
                        obs.add(x)
        expr = {k + suffix: v for k, v in self._expressions.items()}
        o._expressions = expr
        return o

    @property
    def constraints(self) -> List[CExpression]:
        return self._constraints

    @property
    def objectives(self) -> List[CExpression]:
        return self._objectives

    @property
    def weights(self) -> List[float]:
        return self._weights

    @property
    def direction(self) -> Direction:
        return self._direction

    def copy(self) -> "ProblemDef":
        return shallow_copy(self)

    def _add(self, other: Any, inplace: bool = False):
        if isinstance(other, ProblemDef):
            return self.merge(other, inplace=inplace)
        elif isinstance(other, CSymbol):
            LOGGER.warn(f"Ignoring request to add symbol {other} (not required)")
            return self
        elif isinstance(other, CExpression) and not isinstance(other, CSymbol):
            # TODO: Check if the expression is a constraint!
            return self.add_constraints([other], inplace=inplace)
        elif isinstance(other, Iterable):
            o = self
            if not inplace:
                o = self.copy()
            for e in other:
                if isinstance(e, CSymbol):
                    # o.add_symbols([e], inplace=True)
                    pass
                elif isinstance(e, CExpression) and not isinstance(e, CSymbol):
                    o.add_constraints([e], inplace=True)
                else:
                    raise ValueError(f"Unsupported type {type(e)}")
            return o
        else:
            raise ValueError(f"Cannot add {type(other)} to ProblemDef")

    def __add__(self, other: Any) -> "ProblemDef":
        return self._add(other, inplace=False)

    def __iadd__(self, other: Any) -> "ProblemDef":
        return self._add(other, inplace=True)

    def solve(
        self,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: Optional[int] = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ) -> Any:
        if self._backend is None:
            raise ValueError("No backend assigned.")
        if solver is not None:
            avail_solvers = self._backend.available_solvers()
            # We need to match solver with the available solvers
            # being case-insensitive
            solver = next((s for s in avail_solvers if s.lower() == solver.lower()), None)
        backend_problem = self._backend.solve(
            self,
            solver=solver,
            max_seconds=max_seconds,
            warm_start=warm_start,
            verbosity=verbosity,
            **options,
        )
        # Extract summary info (backend specific)
        return backend_problem

    def merge(self, other: "ProblemDef", inplace=False) -> "ProblemDef":
        # TODO: If the other is empty (or instance of grammar?) build the problem before merging
        if isinstance(other, ProblemDef) and hasattr(other, "_build_problem"):  # TODO Change by isinstance
            f = getattr(other, "_build_problem")
            other = f(self)
        b = self._backend if not None else other._backend
        if not b:
            raise ValueError("Problems have no backend associated.")
        if self._backend and other._backend and (self._backend != other._backend):
            raise ValueError("The two problems have different instantiations of the backend.")
        if inplace:
            self.add_constraints(other._constraints, inplace=True)
            self.add_objectives(other._objectives, other._weights, inplace=True)
            self.add_expressions(other._expressions, inplace=True)
            return self
        c = self._constraints + other._constraints
        e = self._expressions.copy()
        e.update(other._expressions)
        w = self._weights + other._weights
        o = self._objectives + other._objectives
        # TODO: Subclasses of ProblemDef not supported
        return self.__class__(b, c, o, e, w)

    def register(self, name: str, expr: CExpression, inplace: bool = True) -> "ProblemDef":
        if name in self._expressions:
            raise ValueError(f"Expression with name {name} already exists")
        if name in self.symbols:
            raise ValueError(f"Symbol with name {name} already exists")
        return self.add_expressions({name: expr}, inplace=inplace)

    def add_constraints(
        self,
        constraints: Union[CExpression, List[CExpression]],
        inplace: bool = True,
    ) -> "ProblemDef":
        if not isinstance(constraints, list):
            constraints = [constraints]
        for c in constraints:
            if isinstance(c, CSymbol):
                raise ValueError(f"The variable {c.name} was added as a constraint")
        if inplace:
            self._constraints.extend(constraints)
            return self
        # TODO: generalize for subclasses of ProblemDef?
        return ProblemDef(
            self._backend,
            self._constraints + constraints,
            self._objectives,
            self._expressions,
            self._weights,
        )

    def add_objectives(
        self,
        objectives: Union[CExpression, List[CExpression]],
        weights: Union[float, List[float]] = 1.0,
        inplace: bool = True,
        names: Optional[Union[str, List[str]]] = None,
    ) -> "ProblemDef":
        if not isinstance(objectives, list):
            objectives = [objectives]
        if not isinstance(weights, list):
            weights = [weights] * len(objectives)
        if len(weights) != len(objectives):
            raise ValueError("Number of weights must match number of objectives")
        if names is not None:
            if isinstance(names, str):
                names = [names]
            if len(names) != len(objectives):
                raise ValueError("Number of names must match number of objectives")
            for i, o in enumerate(objectives):
                o._name = names[i]
        if inplace:
            self._objectives.extend(objectives)
            self._weights.extend(weights)
            return self
        return ProblemDef(
            self._backend,
            self._constraints,
            self._objectives + objectives,
            self._expressions,
            self._weights + weights,
        )

    def add_objective(
        self,
        objective: CExpression,
        weight: float = 1.0,
        inplace: bool = True,
        name: Optional[str] = None,
    ) -> "ProblemDef":
        return self.add_objectives([objective], [weight], inplace=inplace, names=name)

    def add_expressions(
        self,
        expressions: Dict[str, CExpression],
        inplace: bool = True,
    ) -> "ProblemDef":
        if inplace:
            self._expressions.update(expressions)
            return self
        e = dict()
        e.update(self._expressions)
        e.update(expressions)
        return ProblemDef(
            self._backend,
            self._constraints,
            self._objectives,
            e,
            self._weights,
        )


class ProblemBuilder(ProblemDef):
    def __init__(self) -> None:
        super().__init__(None, None, None, None, None, Direction.MIN)

    def _build_problem(self, other: ProblemDef) -> ProblemDef:
        raise NotImplementedError()

    def merge(self, other: ProblemDef, inplace=False) -> ProblemDef:
        return other.merge(self._build_problem(other), inplace)


class Backend(abc.ABC):
    def __init__(
        self,
        default_solver: Optional[str] = None,
        sparse_class: Callable = _get_matrix_builder(),
    ) -> None:
        self._default_solver = default_solver
        self._sparse = sparse_class

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception as e:
            LOGGER.debug(str(e))
            return False

    def version(self) -> str:
        return self._load().__version__

    @staticmethod
    def get_symbols(expressions: Iterable[CExpression]) -> Set[CSymbol]:
        symbols: Set[CSymbol] = set()
        for e in expressions:
            s: Set[CSymbol] = getattr(e, "_proxy_symbols", {})  # type: ignore
            if isinstance(e, CSymbol):
                s.add(e)
            symbols.update(s)
        return symbols

    @abc.abstractmethod
    def _load(self) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def available_solvers(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def Constant(self, value: Any, name: Optional[str] = None) -> CSymbol:
        raise NotImplementedError()

    @abc.abstractmethod
    def Variable(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> CSymbol:
        """Create a variable for optimization.

        This method defines an optimization variable with optional bounds, type, and
        additional graph-related properties.

        Args:
            name (Optional[str]): The name of the variable. Defaults to None.
            shape (Optional[Tuple[int, ...]]): The shape of the variable as a tuple. Defaults to None.
            lb (Optional[Union[float, np.ndarray]]): The lower bound of the variable.
                Can be a scalar or an array. Defaults to None.
            ub (Optional[Union[float, np.ndarray]]): The upper bound of the variable.
                Can be a scalar or an array. Defaults to None.
            vartype (VarType): The type of the variable (e.g., continuous, integer).
                Defaults to VarType.CONTINUOUS.

        Returns:
            CSymbol: The created variable symbol, to be used in further expressions or constraints.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def Parameter(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        value: Any = None,
    ) -> CSymbol:
        raise NotImplementedError()

    def Problem(
        self,
        constraints: Optional[Union[CExpression, List[CExpression]]] = None,
        objectives: Optional[Union[CExpression, List[CExpression]]] = None,
        expressions: Optional[Dict[str, CExpression]] = None,
        weights: Optional[Union[float, List[float]]] = None,
        direction: Direction = Direction.MIN,
    ) -> ProblemDef:
        if isinstance(constraints, CExpression):
            constraints = [constraints]
        if constraints is None:
            constraints = []
        if isinstance(objectives, CExpression):
            objectives = [objectives]
        if isinstance(weights, float):
            weights = [weights]
        elif isinstance(weights, numbers.Number):
            weights = [float(weights)]
        return ProblemDef(self, constraints, objectives, expressions, weights, direction)

    @abc.abstractmethod
    def build(self, p: ProblemDef) -> Any:
        raise NotImplementedError()

    def solve(
        self,
        p: ProblemDef,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: Optional[int] = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ):
        if solver is None:
            if self._default_solver is None:
                from corneto.backend import DEFAULT_SOLVER

                solver = DEFAULT_SOLVER
            else:
                solver = self._default_solver

        o: Optional[CExpression]
        if p.objectives is not None and len(p.objectives) > 1:
            if len(p.weights) != len(p.objectives):
                raise ValueError("Number of weights must match number of objectives")
            # auto-convert to a weighted sum
            # TODO: support the use of parameters as weights. Comment line below
            # for future version
            # ov = self.vstack(p.objectives)
            o = sum(p.weights[i] * p.objectives[i] for i in range(len(p.objectives)))
        else:
            o = p.weights[0] * p.objectives[0] if p.objectives and p.weights[0] != 0 else None
        return self._solve(
            p,
            objective=o,
            solver=solver,
            max_seconds=max_seconds,
            warm_start=warm_start,
            verbosity=verbosity,
            **options,
        )

    @abc.abstractmethod
    def _solve(
        self,
        p: ProblemDef,
        objective: Optional[CExpression] = None,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: Optional[int] = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ):
        raise NotImplementedError()

    def Flow(
        self,
        g: BaseGraph,
        lb: Optional[Union[float, List, np.ndarray]] = 0,
        ub: Optional[Union[float, List, np.ndarray]] = DEFAULT_UB,
        n_flows: int = 1,
        values: bool = False,
        shared_bounds: bool = False,
        varname: str = VAR_FLOW,
        create_nonzero_indicators: bool = False,
        alias_flow: str = EXPR_NAME_FLOW,
        alias_flow_ipos: str = EXPR_NAME_FLOW_IPOS,
        alias_flow_ineg: str = EXPR_NAME_FLOW_INEG,
        alias_nonzero_flow: str = EXPR_NAME_FLOW_NZI,
        indicator_tolerance: float = 1e-4,
    ) -> ProblemDef:
        shape: Tuple = (g.num_edges,)
        if isinstance(lb, list):
            lb = np.array(lb)
        if isinstance(ub, list):
            ub = np.array(ub)
        if n_flows > 1:
            shape = (g.num_edges, n_flows)
            # If lb/ub are vectors, duplicate for each flow
            if isinstance(lb, (int, float)):
                lb = np.ones(shape) * lb
            if isinstance(ub, (int, float)):
                ub = np.ones(shape) * ub
            if isinstance(lb, np.ndarray) and len(lb.shape) == 1:
                lb = np.tile(lb, (n_flows, 1)).T
            if isinstance(ub, np.ndarray) and len(ub.shape) == 1:
                ub = np.tile(ub, (n_flows, 1)).T
        F = self.Variable(name=varname, shape=shape, lb=lb, ub=ub)
        A = self._sparse(g.get_vertex_incidence_matrix_as_lists(values=values))
        P = self.Problem(A @ F == 0)
        if shared_bounds and n_flows > 1:
            # check num dims of lb
            if len(shape) > 1 and shape[1] > 1 and not _identical_columns(lb):
                raise ValueError("shared_bounds=True cannot be used when lower bounds are not identical across flows")
            if len(shape) > 1 and shape[1] > 1 and not _identical_columns(ub):
                raise ValueError("shared_bounds=True cannot be used when upper bounds are not identical across flows")
            S = F.sum(axis=1)
            P += S <= ub[:, 0]
            P += S >= lb[:, 0]
        if create_nonzero_indicators:
            P += NonZeroIndicator(tolerance=indicator_tolerance)
            Ip = P.get_symbol(varname + "_ipos")
            In = P.get_symbol(varname + "_ineg")
            P.register(alias_flow_ipos, Ip)
            P.register(alias_flow_ineg, In)
            P.register(alias_nonzero_flow, Ip + In)
        P.register(alias_flow, F)
        return P

    def Acyclic0(
        self,
        g: BaseGraph,
        P: ProblemDef,
        indicator_positive_var_name: Optional[str] = None,
        indicator_negative_var_name: Optional[str] = None,
        acyclic_var_name: str = VAR_DAG,
        max_parents: Optional[Union[int, Dict[Any, int]]] = None,
        vertex_lb_dist: Optional[List[Dict[Any, int]]] = None,
        vertex_ub_dist: Optional[List[Dict[Any, int]]] = None,
    ) -> ProblemDef:
        """Create Acyclicity Constraint.

        This function creates acyclicity constraints, ensuring that the selected edges
        form an acyclic graph, meaning there are no cycles on the given property.
        Acyclicity can be applied, for example, over flow constraints or signal properties.

        Parameters
        ----------
        g : BaseGraph
            The graph that defines the problem.
        P : ProblemDef
            The problem definition.
        indicator_positive_var_name : str
            The name of the indicator variable, i.e., which edges are selected.
            Default is EXPR_NAME_FLOW_IPOS.
        indicator_negative_var_name : str, optional
            The name of the indicator variable for negative flows. Default is None.
            If a negative flow appears, the source and target nodes of the edge are reversed.
            For example, A->B with positive flow implies order(B) > order(A), with negative
            flow it implies order(A) > order(B).
        acyclic_var_name : str, optional
            The name of the acyclic variable. Default is VAR_DAG.
        max_parents : Optional[Union[int, Dict[Any, int]]], optional
            The maximum number of parents per node. If an integer is provided, the maximum
            number of parents is the same for all nodes. If a dictionary is provided, the
            maximum number of parents can be different for each node. Default is None.

        Returns:
        -------
        ProblemDef
            The problem definition with acyclic constraints.

        Raises:
        ------
        NotImplementedError
            If hyperedges are used.
        """
        for s, t in g.E:
            if len(s) > 1 or len(t) > 1:
                raise NotImplementedError("Hyperedges not supported")
        if isinstance(max_parents, int):
            max_parents = {v: max_parents for v in g.vertices}
        Ip = In = None
        if indicator_positive_var_name is not None and indicator_negative_var_name is not None:
            Ip = P.expressions[indicator_positive_var_name]
            In = P.expressions[indicator_negative_var_name]
            I = Ip + In
        elif indicator_positive_var_name is not None:
            Ip = P.expressions[indicator_positive_var_name]
            I = Ip
        elif indicator_negative_var_name is not None:
            In = P.expressions[indicator_negative_var_name]
            I = In
        else:
            raise ValueError("At least one indicator variable name is required")

        # Limit the number of parents per node, if requested
        if max_parents is not None:
            # Get indexes of edges vi->vj for all vi
            for v, max in max_parents.items():
                edges_idx = [i for i, _ in g.in_edges(v)]
                if len(edges_idx) > 0:
                    # Sum selected parent edges
                    P += np.ones((len(edges_idx),)) @ I[edges_idx] <= max
        # detect the number of DAG layers to add
        if len(I.shape) == 1:
            n_samples = 1
        else:
            n_samples = I.shape[1]

        # Create a DAG layer num for each vertex
        L = self.Variable(acyclic_var_name, (g.num_vertices, n_samples), 0, g.num_vertices - 1)
        vix = {v: i for i, v in enumerate(g.vertices)}
        for i_sample in range(n_samples):
            if Ip is not None:
                if len(Ip.shape) == 1:
                    Ip_i_order = Ip
                else:
                    Ip_i_order = Ip[:, i_sample]
            if In is not None:
                if len(In.shape) == 1:
                    In_i_order = In
                else:
                    In_i_order = In[:, i_sample]

            if Ip is not None:
                # Get edges s->t that can have a positive flow
                # check if Ip has ub field
                if hasattr(Ip, "ub"):
                    e_pos = [(i, g.get_edge(i)) for i in np.flatnonzero(Ip.ub > 0)]
                    e_ix = np.array([i for i, (s, t) in e_pos if len(s) > 0 and len(t) > 0])
                else:
                    e_ix = np.array([i for i, (s, t) in enumerate(g.E) if len(s) > 0 and len(t) > 0])
                edges = [g.get_edge(i) for i in e_ix]
                # Get the index of the source / target vertices of the edge
                s_idx = np.array([vix[list(s)[0]] for (s, _) in edges])
                t_idx = np.array([vix[list(t)[0]] for (_, t) in edges])
                # The layer position in a DAG of the target vertex of the edge
                # has to be greater than the source vertex, otherwise Ip (pos flow) has to be 0
                if len(e_ix) > 0:
                    P += L[t_idx, i_sample] - L[s_idx, i_sample] >= Ip_i_order[e_ix] + (1 - g.num_vertices) * (
                        1 - Ip_i_order[e_ix]
                    )
                    P += L[t_idx, i_sample] - L[s_idx, i_sample] <= g.num_vertices - 1
            if In is not None:
                # NOTE: Negative flows eq. to reversed directed edge
                # Get edges s->t that can have a positive flow
                if hasattr(In, "lb"):
                    e_neg = [(i, g.get_edge(i)) for i in np.flatnonzero(In.lb < 0)]
                    e_ix = np.array([i for i, (s, t) in e_neg if len(s) > 0 and len(t) > 0])
                else:
                    e_ix = np.array([i for i, (s, t) in enumerate(g.E) if len(s) > 0 and len(t) > 0])
                edges = [g.get_edge(i) for i in e_ix]
                # Get the index of the source / target vertices of the edge
                s_idx = np.array([vix[list(s)[0]] for (s, _) in edges])
                t_idx = np.array([vix[list(t)[0]] for (_, t) in edges])
                if len(e_ix) > 0:
                    P += L[s_idx, i_sample] - L[t_idx, i_sample] >= In_i_order[e_ix] + (1 - g.num_vertices) * (
                        1 - In_i_order[e_ix]
                    )
                    P += L[s_idx, i_sample] - L[t_idx, i_sample] <= g.num_vertices - 1
            # TODO: Raise error if hypergraph
        return P

    def Acyclic(
        self,
        g: BaseGraph,
        P: ProblemDef,
        indicator_positive_var_name: Optional[str] = None,
        indicator_negative_var_name: Optional[str] = None,
        acyclic_var_name: str = VAR_DAG,
        max_parents: Optional[Union[int, Dict[Any, int]]] = None,
        vertex_lb_dist: Optional[List[Dict[Any, int]]] = None,
        vertex_ub_dist: Optional[List[Dict[Any, int]]] = None,
    ) -> ProblemDef:
        """Create Acyclicity Constraint.

        This function creates acyclicity constraints, ensuring that the selected edges
        form an acyclic graph, meaning there are no cycles on the given property.
        Acyclicity can be applied, for example, over flow constraints or signal properties.

        Parameters
        ----------
        g : BaseGraph
            The graph that defines the problem.
        P : ProblemDef
            The problem definition.
        indicator_positive_var_name : str
            The name of the indicator variable, i.e., which edges are selected.
            Default is EXPR_NAME_FLOW_IPOS.
        indicator_negative_var_name : str, optional
            The name of the indicator variable for negative flows. Default is None.
            If a negative flow appears, the source and target nodes of the edge are reversed.
            For example, A->B with positive flow implies order(B) > order(A), with negative
            flow it implies order(A) > order(B).
        acyclic_var_name : str, optional
            The name of the acyclic variable. Default is VAR_DAG.
        max_parents : Optional[Union[int, Dict[Any, int]]], optional
            The maximum number of parents per node. If an integer is provided, the maximum
            number of parents is the same for all nodes. If a dictionary is provided, the
            maximum number of parents can be different for each node. Default is None.
        vertex_lb_dist : Optional[List[Dict[Any, int]]], optional
            A list (one entry per experiment) of dictionaries that assign a lower bound
            (minimum layer/distance) for each vertex.
        vertex_ub_dist : Optional[List[Dict[Any, int]]], optional
            A list (one entry per experiment) of dictionaries that assign an upper bound
            (maximum layer/distance) for each vertex.

        Returns:
        -------
        ProblemDef
            The problem definition with acyclic constraints.

        Raises:
        ------
        NotImplementedError
            If hyperedges are used.
        """
        # Check that hyperedges are not used
        for s, t in g.E:
            if len(s) > 1 or len(t) > 1:
                raise NotImplementedError("Hyperedges not supported")

        # Process max_parents argument: if an int is provided, convert it to a dict
        if isinstance(max_parents, int):
            max_parents = {v: max_parents for v in g.vertices}

        Ip = In = None
        if indicator_positive_var_name is not None and indicator_negative_var_name is not None:
            Ip = P.expressions[indicator_positive_var_name]
            In = P.expressions[indicator_negative_var_name]
            I = Ip + In
        elif indicator_positive_var_name is not None:
            Ip = P.expressions[indicator_positive_var_name]
            I = Ip
        elif indicator_negative_var_name is not None:
            In = P.expressions[indicator_negative_var_name]
            I = In
        else:
            raise ValueError("At least one indicator variable name is required")

        # Limit the number of parents per node, if requested
        if max_parents is not None:
            for v, max_val in max_parents.items():
                edges_idx = [i for i, _ in g.in_edges(v)]
                if edges_idx:
                    # Sum selected parent edges
                    P += np.ones((len(edges_idx),)) @ I[edges_idx] <= max_val

        # Determine number of samples (if I is 1D, assume 1 sample)
        if len(I.shape) == 1:
            n_samples = 1
        else:
            n_samples = I.shape[1]

        # Create a DAG layer variable for each vertex, one per sample.
        L = self.Variable(acyclic_var_name, (g.num_vertices, n_samples), 0, g.num_vertices - 1)
        vix = {v: i for i, v in enumerate(g.vertices)}

        # If bounds lists are provided, ensure their length matches the number of samples.
        if vertex_lb_dist is not None and len(vertex_lb_dist) != n_samples:
            raise ValueError("Length of vertex_lb_dist must match number of samples")
        if vertex_ub_dist is not None and len(vertex_ub_dist) != n_samples:
            raise ValueError("Length of vertex_ub_dist must match number of samples")

        # Loop over samples to add constraints based on indicator variables and the bounds.
        for i_sample in range(n_samples):
            if Ip is not None:
                Ip_i_order = Ip if len(Ip.shape) == 1 else Ip[:, i_sample]
            if In is not None:
                In_i_order = In if len(In.shape) == 1 else In[:, i_sample]

            if Ip is not None:
                # Get edges that can have a positive flow.
                if hasattr(Ip, "ub"):
                    ub = Ip.ub if len(Ip.shape) == 1 else Ip.ub[:, i_sample]
                    e_pos = [(i, g.get_edge(i)) for i in np.flatnonzero(ub > 0)]
                    e_ix = np.array([i for i, (s, t) in e_pos if s and t])
                else:
                    e_ix = np.array([i for i, (s, t) in enumerate(g.E) if s and t])
                edges = [g.get_edge(i) for i in e_ix]
                s_idx = np.array([vix[list(s)[0]] for (s, _) in edges])
                t_idx = np.array([vix[list(t)[0]] for (_, t) in edges])
                if len(e_ix) > 0:
                    P += L[t_idx, i_sample] - L[s_idx, i_sample] >= Ip_i_order[e_ix] + (1 - g.num_vertices) * (
                        1 - Ip_i_order[e_ix]
                    )
                    P += L[t_idx, i_sample] - L[s_idx, i_sample] <= g.num_vertices - 1

            if In is not None:
                # Negative flows are handled as reversed directed edges.
                if hasattr(In, "lb"):
                    lb = In.lb if len(In.shape) == 1 else In.lb[:, i_sample]
                    e_neg = [(i, g.get_edge(i)) for i in np.flatnonzero(lb < 0)]
                    e_ix = np.array([i for i, (s, t) in e_neg if s and t])
                else:
                    e_ix = np.array([i for i, (s, t) in enumerate(g.E) if s and t])
                edges = [g.get_edge(i) for i in e_ix]
                s_idx = np.array([vix[list(s)[0]] for (s, _) in edges])
                t_idx = np.array([vix[list(t)[0]] for (_, t) in edges])
                if len(e_ix) > 0:
                    P += L[s_idx, i_sample] - L[t_idx, i_sample] >= In_i_order[e_ix] + (1 - g.num_vertices) * (
                        1 - In_i_order[e_ix]
                    )
                    P += L[s_idx, i_sample] - L[t_idx, i_sample] <= g.num_vertices - 1

            # --- New: Add vertex lower and upper bound constraints ---
            if vertex_lb_dist is not None:
                list_vix = []
                list_dist = []
                for v in g.V:
                    idx = vix[v]
                    if v in vertex_lb_dist[i_sample]:
                        list_vix.append(idx)
                        list_dist.append(vertex_lb_dist[i_sample][v])
                if list_vix:
                    P += L[np.array(list_vix), i_sample] >= np.array(list_dist)
            if vertex_ub_dist is not None:
                list_vix = []
                list_dist = []
                for v in g.V:
                    idx = vix[v]
                    if v in vertex_ub_dist[i_sample]:
                        list_vix.append(idx)
                        list_dist.append(vertex_ub_dist[i_sample][v])
                if list_vix:
                    P += L[np.array(list_vix), i_sample] <= np.array(list_dist)

            # TODO: Raise error if hypergraph is used
        return P

    def AcyclicFlow(
        self,
        g: BaseGraph,
        lb: Union[float, np.ndarray] = 0,
        ub: Union[float, np.ndarray] = DEFAULT_UB,
        values: bool = False,
        max_parents: Optional[Union[int, Dict[Any, int]]] = None,
        vertex_lb_dist: Optional[np.ndarray] = None,
        varname: str = VAR_FLOW,
        alias_flow: str = EXPR_NAME_FLOW,
        alias_flow_ipos: str = EXPR_NAME_FLOW_IPOS,
        alias_flow_ineg: str = EXPR_NAME_FLOW_INEG,
        alias_nonzero_flow: str = EXPR_NAME_FLOW_NZI,
        indicator_tolerance: float = 1e-4,
    ) -> ProblemDef:
        if not varname:
            varname = VAR_FLOW
        if isinstance(lb, list):
            lb = np.array(lb)
        if isinstance(ub, list):
            ub = np.array(ub)
        if not isinstance(lb, np.ndarray):
            lb = np.array([lb] * g.num_edges)
        if not isinstance(ub, np.ndarray):
            ub = np.array([ub] * g.num_edges)
        for s, t in g.E:
            if len(s) > 1 or len(t) > 1:
                raise NotImplementedError("Hyperedges not supported")
        if isinstance(max_parents, int):
            max_parents = {v: max_parents for v in g.vertices}
        P = self.Flow(
            g,
            lb,
            ub,
            values=values,
            varname=varname,
            alias_flow=alias_flow,
            alias_flow_ipos=alias_flow_ipos,
            alias_flow_ineg=alias_flow_ineg,
            alias_nonzero_flow=alias_nonzero_flow,
            indicator_tolerance=indicator_tolerance,
            create_nonzero_indicators=True,
        )
        # TODO: recover easily the created indicators!
        # TODO: Optionally provide a lower bound of pos for each vertex
        Ip = P.get_symbol(varname + "_ipos") if any(ub > 0) else None
        In = P.get_symbol(varname + "_ineg") if any(lb < 0) else None
        if Ip is not None and In is not None:
            I = Ip + In
        elif Ip is not None:
            I = Ip
        elif In is not None:
            I = In
        else:
            raise ValueError()
        # Limit the number of parents per node, if requested
        if max_parents is not None:
            # Get indexes of edges vi->vj for all vi
            for v, max in max_parents.items():
                edges_idx = [i for i, _ in g.in_edges(v)]
                if len(edges_idx) > 0:
                    # Sum selected parent edges
                    P += np.ones((len(edges_idx),)) @ I[edges_idx] <= max
        # Create a DAG layer num for each vertex
        L = self.Variable("_dag_layer_pos", (g.num_vertices,), 0, g.num_vertices - 1)
        vix = {v: i for i, v in enumerate(g.vertices)}
        # These constraints are not compatible with hyperedges
        if np.any(ub > 0):
            # Get edges s->t that can have a positive flow
            e_pos = [(i, g.get_edge(i)) for i in np.flatnonzero(ub > 0)]
            e_ix = np.array([i for i, (s, t) in e_pos if len(s) > 0 and len(t) > 0])
            edges = [g.get_edge(i) for i in e_ix]
            # Get the index of the source / target vertices of the edge
            s_idx = np.array([vix[next(iter(s))] for (s, _) in edges])
            t_idx = np.array([vix[next(iter(t))] for (_, t) in edges])
            # The layer position in a DAG of the target vertex of the edge
            # has to be greater than the source vertex, otherwise Ip (pos flow) has to be 0
            if len(e_ix) > 0:
                P += L[t_idx] - L[s_idx] >= Ip[e_ix] + (1 - g.num_vertices) * (1 - Ip[e_ix])
                P += L[t_idx] - L[s_idx] <= g.num_vertices - 1
        if np.any(lb < 0):
            # NOTE: Negative flows eq. to reversed directed edge
            # Get edges s->t that can have a positive flow
            e_neg = [(i, g.get_edge(i)) for i in np.flatnonzero(lb < 0)]
            # Check if vertex not empty! print(e_neg)
            e_ix = np.array([i for i, (s, t) in e_neg if len(s) > 0 and len(t) > 0])
            edges = [g.get_edge(i) for i in e_ix]
            # Get the index of the source / target vertices of the edge
            s_idx = np.array([vix[next(iter(s))] for (s, _) in edges])
            t_idx = np.array([vix[next(iter(t))] for (_, t) in edges])
            if len(e_ix) > 0:
                P += L[s_idx] - L[t_idx] >= In[e_ix] + (1 - g.num_vertices) * (1 - In[e_ix])
                P += L[s_idx] - L[t_idx] <= g.num_vertices - 1
        # TODO: Raise error if hypergraph
        return P

    def Indicator(
        self,
        V: CSymbol,
        indexes: Optional[Union[Tuple, List, np.ndarray]] = None,
        suffix: str = "_i",
        name: Optional[str] = None,
    ) -> ProblemDef:
        # If I = 0 => V = 0, if I = 1, LB <= V <= UB (including 0)
        c = []
        S = V
        ub = V.ub
        lb = V.lb
        if indexes:
            S = V[indexes]
            ub = V.ub[indexes]
            lb = V.lb[indexes]
        if name is None:
            name = V.name + suffix
        # TODO: Add option to create shared indicators for n_flows > 1, so if
        # I_i = 0 => V_1i, ..., V_ni = 0, if I_i = 1, LB <= V_1i, ..., V_ni <= UB
        I = self.Variable(name, S.shape, 0, 1, vartype=VarType.BINARY)
        blocked = np.isclose(ub, 0) & np.isclose(lb, 0)
        if np.sum(blocked) > 0:
            # indexing compatible with CVXPY and PICOS
            idx = np.where(blocked)[0]
            c += [I[idx] == 0, S[idx] == 0]
        # Add constraint: lb * I <= V <= ub * I
        if V._provided_lb is None or V._provided_ub is None:
            raise ValueError(f"The continuous variable {V.name} is unbounded, indicators cannot be created.")
        c += [S >= I.multiply(lb), S <= I.multiply(ub)]
        return self.Problem(c)

    def NonZeroIndicator(
        self,
        V: CSymbol,
        *args,  # new positional indices for multi-dimensional indexing
        indexes: Optional[Union[int, slice, Tuple, List, np.ndarray]] = None,
        suffix_pos: str = "_ipos",
        suffix_neg: str = "_ineg",
        tolerance: float = 1e-3,
    ) -> ProblemDef:
        # Ensure the variable is bounded
        if V._provided_lb is None or V._provided_ub is None:
            raise ValueError(f"The continuous variable {V.name} is unbounded, indicators cannot be created.")

        # Avoid ambiguity: don't allow both positional indices and the 'indexes' keyword
        if args and indexes is not None:
            raise ValueError("Provide either positional indices or the 'indexes' keyword, not both.")

        # If args is not none, we need to check if it is a tuple and more than
        # one dimension was provided.
        if isinstance(args, tuple) and len(args) > 1:
            diff_len_shape = len(args) - len(V.shape)
            if diff_len_shape > 0:
                # If the last dimension is not 0, raise an error
                if args[-1] != 0:
                    raise ValueError(f"Cannot use {len(args)} positional indices for a variable of shape {V.shape}")
                else:
                    # We ignore the last dimension
                    args = args[:-1]

        # Determine which indexing to use
        idx = args if args else indexes

        # If an index is provided, use it to slice the variable and its bounds
        if idx is not None:
            S = V[idx]
            lb = V.lb[idx]
            ub = V.ub[idx]
        else:
            S = V
            lb = V.lb
            ub = V.ub

        c = []
        I_pos = self.Variable(V.name + suffix_pos, S.shape, 0, 1, vartype=VarType.BINARY)
        I_neg = self.Variable(V.name + suffix_neg, S.shape, 0, 1, vartype=VarType.BINARY)
        I = I_pos + I_neg
        c += [I <= 1]  # Ensure mutual exclusivity

        # Disable infeasible binary indicators based on bounds
        if np.sum(ub <= 0) > 0:
            c += [I_pos[np.where(ub <= 0)[0]] == 0]
        if np.sum(lb >= 0) > 0:
            c += [I_neg[np.where(lb >= 0)[0]] == 0]

        # Add constraints to enforce variable behavior depending on the indicator activation:
        # If I_pos = 1 and I_neg = 0: V >= tol AND V <= ub
        # If I_pos = 0 and I_neg = 1: V >= lb AND V <= -tol
        # If I_pos = 0 and I_neg = 0: V >= 0 AND V <= 0
        c += [
            S >= I_neg.multiply(lb) + I_pos * tolerance,
            S <= I_pos.multiply(ub) - I_neg * tolerance,
        ]

        return self.Problem(c)

    # TODO: Remove function
    def Indicators(
        self,
        V: CSymbol,
        tolerance=1e-3,
        positive=True,
        negative=True,
        suffix_pos="_ipos",
        suffix_neg="_ineg",
    ) -> ProblemDef:
        # SHow a deprecated
        warnings.warn(
            "The Indicators method is deprecated, use Indicator instead",
            DeprecationWarning,
        )

        # Get upper/lower bounds for flow variables
        constraints = []
        if not (positive or negative):
            raise ValueError("At least one of positive or negative must be True.")
        if positive:
            I_pos = self.Variable(V.name + suffix_pos, V.shape, 0, 1, vartype=VarType.BINARY)
            if np.sum(V.ub <= 0) > 0:
                constraints.append(I_pos[np.where(V.ub <= 0)[0]] == 0)
        if negative:
            I_neg = self.Variable(V.name + suffix_neg, V.shape, 0, 1, vartype=VarType.BINARY)
            if np.sum(V.lb >= 0) > 0:
                constraints.append(I_neg[np.where(V.lb >= 0)[0]] == 0)
        if positive and negative:
            constraints.append(I_pos + I_neg <= 1)

        # lower bound constraints: F >= F_lb * I_neg + eps * I_pos
        # TODO: Use better constraints to avoid precision errors
        # by multiplying tolerance * indicator
        I_LBN = I_neg.multiply(V.lb) if negative else V.lb
        I_LBP = tolerance * I_pos if positive else 0
        LB = I_LBN + I_LBP
        constraints.append(V >= LB)
        # upper bound constraints: F <= F_ub * I_pos - eps * I_neg
        I_UBN = I_pos.multiply(V.ub) if positive else V.ub
        I_UBP = tolerance * I_neg if negative else 0
        UB = I_UBN - I_UBP
        constraints.append(V <= UB)
        return self.Problem(constraints)

    def Xor(self, x: CExpression, y: CExpression, varname="_xor"):
        # Deprecated
        warnings.warn("The Xor method is deprecated, use linear_xor instead", DeprecationWarning)
        if isinstance(x, CSymbol) and x._vartype != VarType.BINARY:
            raise ValueError(f"Variable x has type {x._vartype} instead of BINARY")
        if isinstance(y, CSymbol) and y._vartype != VarType.BINARY:
            raise ValueError(f"Variable x has type {y._vartype} instead of BINARY")
        if x.shape != y.shape:
            raise ValueError(f"Shape of x ({x.shape}) is different from y ({y.shape})")
        # Create a new binary variable to compute xor(x,y)
        xor = self.Variable(varname, x.shape, 0, 1, vartype=VarType.BINARY)
        return self.Problem([xor >= x - y, xor >= y - x, xor <= x + y, xor <= 2 - x - y])

    def linear_or(
        self,
        x: CExpression,
        axis: Optional[int] = None,
        varname="or",
        ignore_type=False,
    ) -> ProblemDef:
        # Check if the variable has a vartype and is binary
        if hasattr(x, "_vartype") and x._vartype != VarType.BINARY and not ignore_type:
            raise ValueError(f"Variable x has type {x._vartype} instead of BINARY")
        else:
            for s in x._proxy_symbols:
                if s._vartype != VarType.BINARY:
                    # Show warning only
                    LOGGER.warn(f"Variable {s.name} has type {s._vartype}, expression is assumed to be binary")
                    break

        Z = x.sum(axis=axis)
        Z_norm = Z / x.shape[axis]  # between 0-1
        # Create a new binary variable to compute linearized or
        Or = self.Variable(varname, Z.shape, 0, 1, vartype=VarType.BINARY)
        return self.Problem([Or >= Z_norm, Or <= Z])

    def linear_and(self, x: CExpression, axis: Optional[int] = None, varname="and") -> ProblemDef:
        # Check if the variable is binary, otherwise throw an error
        if hasattr(x, "_vartype") and x._vartype != VarType.BINARY:
            raise ValueError(f"Variable x has type {x._vartype} instead of BINARY")
        else:
            for s in x._proxy_symbols:
                if s._vartype != VarType.BINARY:
                    # Show warning only
                    LOGGER.warn(f"Variable {s.name} has type {s._vartype}, expression is assumed to be binary")
                    break
        Z = x.sum(axis=axis)
        N = x.shape[axis]
        Z_norm = Z / N
        And = self.Variable(varname, Z.shape, 0, 1, vartype=VarType.BINARY)
        return self.Problem([And <= Z_norm, And >= Z - N + 1])

    def linear_xor(
        self,
        x: CExpression,
        axis: Optional[int] = None,
        varname="xor",
        ignore_type=False,
    ) -> ProblemDef:
        # Check if the variable is binary, otherwise throw an error
        if hasattr(x, "_vartype") and x._vartype != VarType.BINARY and not ignore_type:
            raise ValueError(f"Variable x has type {x._vartype} instead of BINARY")
        else:
            for s in x._proxy_symbols:
                if s._vartype != VarType.BINARY:
                    # Show warning only
                    LOGGER.warn(f"Variable {s.name} has type {s._vartype}, expression is assumed to be binary")
                    break
        # Sum the binary variables along the specified axis
        Z = x.sum(axis=axis)
        # Create a new binary variable to represent the XOR result
        Xor = self.Variable(varname, Z.shape, 0, 1, vartype=VarType.BINARY)
        # Introduce an integer variable to model the floor division
        K = self.Variable(varname + "_k", Z.shape, 0, None, vartype=VarType.INTEGER)
        # Add the constraint that Z - 2*K - Xor == 0
        constraints = [Z - 2 * K - Xor == 0]
        return self.Problem(constraints)

    def vstack(self, arg_list: Iterable[CExpression]) -> CExpression:
        v = None
        for a in arg_list:
            if v is None:
                v = a
            else:
                v = v.vstack(a)
        return v

    def hstack(self, arg_list: Iterable[CExpression]) -> CExpression:
        h = None
        for a in arg_list:
            if h is None:
                h = a
            else:
                h = h.hstack(a)
        return h

    def zero_function(self) -> CExpression:
        return self.Constant(0).sum()


class NoBackend(Backend):
    def __init__(self) -> None:
        self._error = (
            "No backend found. You can install one of the "
            "supported backends by `pip install cvxpy` or `pip install picos`."
        )

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, name):
        """Intercept any attribute or method call that isn't already defined in the Backend class
        and raise a NotImplementedError.
        """
        if hasattr(super(), name):
            return super().__getattr__(name)
        raise NotImplementedError(self._error)


def _find_continuous_var(p):
    # Search for continous vars
    cvars = [k for k, v in p.symbols.items() if v._vartype == VarType.CONTINUOUS]
    if len(cvars) == 0:
        raise ValueError("No available continuous vars for creating indicator vars")
    if len(cvars) == 1:
        LOGGER.debug(f"No variable provided, creating indicators for {cvars[0]}")
        return cvars[0]
    else:
        raise ValueError(f"There are {len(cvars)} continous vars, but no var_name is provided.")


class Indicator(ProblemBuilder):
    def __init__(
        self,
        name: Optional[str] = None,
        var_name: Optional[str] = None,
        indexes: Optional[Union[Tuple, List, np.ndarray]] = None,
        suffix: str = "_i",
    ) -> None:
        super().__init__()
        self.var_name = var_name
        self._suffix = suffix
        self._indexes = indexes
        self._name = name

    def _build_problem(self, other: ProblemDef):
        if other._backend is None:
            raise ValueError("Cannot combine problems without a main backend")
        if self.var_name is None:
            self.var_name = _find_continuous_var(other)
        return other._backend.Indicator(
            other.get_symbol(self.var_name),
            suffix=self._suffix,
            indexes=self._indexes,
            name=self._name,
        )


class NonZeroIndicator(ProblemBuilder):
    def __init__(
        self,
        var_name: Optional[str] = None,
        indexes: Optional[Union[Tuple, List, np.ndarray]] = None,
        tolerance=1e-4,
        suffix_pos: str = "_ipos",
        suffix_neg: str = "_ineg",
    ) -> None:
        super().__init__()
        self.var_name = var_name
        self._tolerance = tolerance
        self._suffix_pos = suffix_pos
        self._suffix_neg = suffix_neg
        self._indexes = indexes

    def _build_problem(self, other: ProblemDef):
        if other._backend is None:
            raise ValueError("Cannot combine empty grammars")
        if self.var_name is None:
            self.var_name = _find_continuous_var(other)
        return other._backend.NonZeroIndicator(
            other.get_symbol(self.var_name),
            suffix_pos=self._suffix_pos,
            suffix_neg=self._suffix_neg,
            tolerance=self._tolerance,
            indexes=self._indexes,
        )


class Indicators(ProblemBuilder):
    def __init__(
        self,
        var_name: Optional[str] = None,
        tolerance: float = 1e-3,
        positive: bool = True,
        negative: bool = False,
    ) -> None:
        # Probably there is some missing component which is not a problemdef
        super().__init__()
        self.var_name = var_name
        self._tol = tolerance
        self._pos = positive
        self._neg = negative

    def _build_problem(self, other: ProblemDef):
        if other._backend is None:
            raise ValueError("Cannot combine empty grammars")
        if self.var_name is None:
            self.var_name = _find_continuous_var(other)
        return other._backend.Indicators(
            other.get_symbol(self.var_name),
            tolerance=self._tol,
            positive=self._pos,
            negative=self._neg,
        )


class HammingLoss(ProblemBuilder):
    def __init__(
        self,
        reference: np.ndarray,
        y: Union[str, CExpression],
        penalty: float = 1.0,
    ) -> None:
        super().__init__()
        self.ref = reference
        self.y = y
        self.penalty = penalty

    def _build_problem(self, other: ProblemDef) -> ProblemDef:
        x = abs(self.ref)
        y = other.get_symbol(self.y) if isinstance(self.y, str) else self.y
        idx_one = np.where(x == 1)[0]
        idx_zero = np.where(x == 0)[0]
        P = ProblemDef()
        diff_zeros = y[idx_zero] - x[idx_zero]
        diff_ones = x[idx_one] - y[idx_one]
        hamming_dist = np.ones(diff_zeros.shape) @ diff_zeros + np.ones(diff_ones.shape) @ diff_ones
        P.add_objectives(hamming_dist, weights=self.penalty, inplace=True)  # type: ignore
        return P
