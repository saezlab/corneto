import abc
import numbers
from copy import copy as shallow_copy
from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from corneto._constants import (
    DEFAULT_UB,
    EXPR_NAME_FLOW,
    EXPR_NAME_FLOW_INEG,
    EXPR_NAME_FLOW_IPOS,
    EXPR_NAME_FLOW_NZI,
    VAR_DAG,
    VAR_FLOW,
    Direction,
    Solver,
    VarType,
)
from corneto._decorators import _delegate
from corneto._settings import LOGGER, _get_matrix_builder
from corneto.graph import BaseGraph
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


def _sparse_vector_replication(vector_size: int, repetitions: int):
    """Return a sparse map from a vector to repeated Fortran-order columns."""
    from scipy.sparse import csr_matrix

    if vector_size <= 0 or repetitions <= 0:
        return csr_matrix((vector_size * repetitions, vector_size))
    rows = np.arange(vector_size * repetitions, dtype=int)
    columns = np.tile(np.arange(vector_size, dtype=int), repetitions)
    return csr_matrix(
        (np.ones(rows.size, dtype=float), (rows, columns)),
        shape=(vector_size * repetitions, vector_size),
    )


def _sparse_vector_entry_repetition(vector_size: int, repetitions: int):
    """Return a sparse map that repeats each vector entry consecutively."""
    from scipy.sparse import csr_matrix

    if vector_size <= 0 or repetitions <= 0:
        return csr_matrix((vector_size * repetitions, vector_size))
    rows = np.arange(vector_size * repetitions, dtype=int)
    columns = np.repeat(np.arange(vector_size, dtype=int), repetitions)
    return csr_matrix(
        (np.ones(rows.size, dtype=float), (rows, columns)),
        shape=(vector_size * repetitions, vector_size),
    )


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

        # Constants already carry their exact value in ``expr``. Creating
        # synthetic +/-infinity bounds for them would allocate dense arrays
        # with the full expression shape, defeating sparse constants.
        if not variable and lb is None and ub is None:
            self._lb = None
            self._ub = None
            self._name = name
            self._vartype = vartype
            return

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

    @property
    def symbols(self) -> Dict[str, CSymbol]:
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
        force_matrix: bool = False,
    ) -> ProblemDef:
        shape: Tuple = (g.num_edges,)
        if isinstance(lb, list):
            lb = np.array(lb)
        if isinstance(ub, list):
            ub = np.array(ub)
        if n_flows > 1 or force_matrix:
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
        A = self.Constant(self._sparse(g.get_vertex_incidence_matrix_as_lists(values=values)))
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
            P += NonZeroIndicator(var_name=varname, tolerance=indicator_tolerance)
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
            indicator = Ip + In
        elif indicator_positive_var_name is not None:
            Ip = P.expressions[indicator_positive_var_name]
            indicator = Ip
        elif indicator_negative_var_name is not None:
            In = P.expressions[indicator_negative_var_name]
            indicator = In
        else:
            raise ValueError("At least one indicator variable name is required")

        # Limit the number of parents per node, if requested
        if max_parents is not None:
            # Get indexes of edges vi->vj for all vi
            for v, max in max_parents.items():
                edges_idx = [i for i, _ in g.in_edges(v)]
                if len(edges_idx) > 0:
                    # Sum selected parent edges
                    P += np.ones((len(edges_idx),)) @ indicator[edges_idx] <= max
        # detect the number of DAG layers to add
        if len(indicator.shape) == 1:
            n_samples = 1
        else:
            n_samples = indicator.shape[1]

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
                s_idx = np.array([vix[next(iter(s))] for (s, _) in edges])
                t_idx = np.array([vix[next(iter(t))] for (_, t) in edges])
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
                if hasattr(In, "ub"):
                    e_neg = [(i, g.get_edge(i)) for i in np.flatnonzero(In.ub > 0)]
                    e_ix = np.array([i for i, (s, t) in e_neg if len(s) > 0 and len(t) > 0])
                else:
                    e_ix = np.array([i for i, (s, t) in enumerate(g.E) if len(s) > 0 and len(t) > 0])
                edges = [g.get_edge(i) for i in e_ix]
                # Get the index of the source / target vertices of the edge
                s_idx = np.array([vix[next(iter(s))] for (s, _) in edges])
                t_idx = np.array([vix[next(iter(t))] for (_, t) in edges])
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
            indicator = Ip + In
        elif indicator_positive_var_name is not None:
            Ip = P.expressions[indicator_positive_var_name]
            indicator = Ip
        elif indicator_negative_var_name is not None:
            In = P.expressions[indicator_negative_var_name]
            indicator = In
        else:
            raise ValueError("At least one indicator variable name is required")

        # Limit the number of parents per node, if requested. A negative
        # selection traverses an edge in reverse, so it contributes a parent
        # at the edge's source rather than at its target.
        if max_parents is not None:
            from scipy.sparse import csr_matrix

            parent_vertices = list(max_parents)
            parent_vertex_index = {vertex: index for index, vertex in enumerate(parent_vertices)}
            positive_rows = []
            positive_columns = []
            negative_rows = []
            negative_columns = []
            for edge_index, (source, target) in enumerate(g.E):
                if target:
                    target_vertex = next(iter(target))
                    if target_vertex in parent_vertex_index:
                        positive_rows.append(parent_vertex_index[target_vertex])
                        positive_columns.append(edge_index)
                if source:
                    source_vertex = next(iter(source))
                    if source_vertex in parent_vertex_index:
                        negative_rows.append(parent_vertex_index[source_vertex])
                        negative_columns.append(edge_index)

            parent_count = None
            if Ip is not None and positive_rows:
                positive_selector = csr_matrix(
                    (
                        np.ones(len(positive_rows), dtype=float),
                        (positive_rows, positive_columns),
                    ),
                    shape=(len(parent_vertices), g.num_edges),
                )
                parent_count = self.Constant(positive_selector) @ Ip
            if In is not None and negative_rows:
                negative_selector = csr_matrix(
                    (
                        np.ones(len(negative_rows), dtype=float),
                        (negative_rows, negative_columns),
                    ),
                    shape=(len(parent_vertices), g.num_edges),
                )
                negative_parent_count = self.Constant(negative_selector) @ In
                parent_count = negative_parent_count if parent_count is None else parent_count + negative_parent_count
            if parent_count is not None:
                parent_count = parent_count.reshape((len(parent_vertices), 1))
                max_values = np.asarray([max_parents[vertex] for vertex in parent_vertices], dtype=float).reshape(
                    (-1, 1)
                )
                P += parent_count <= max_values

        # Determine number of samples (if the indicator is 1D, assume 1 sample)
        if len(indicator.shape) == 1:
            n_samples = 1
        else:
            n_samples = indicator.shape[1]

        # Create a DAG layer variable for each vertex, one per sample.
        L = self.Variable(acyclic_var_name, (g.num_vertices, n_samples), 0, g.num_vertices - 1)
        vix = {v: i for i, v in enumerate(g.vertices)}

        # If bounds lists are provided, ensure their length matches the number of samples.
        if vertex_lb_dist is not None and len(vertex_lb_dist) != n_samples:
            raise ValueError("Length of vertex_lb_dist must match number of samples")
        if vertex_ub_dist is not None and len(vertex_ub_dist) != n_samples:
            raise ValueError("Length of vertex_ub_dist must match number of samples")

        edge_indexes = np.array([i for i, (source, target) in enumerate(g.E) if source and target], dtype=int)
        if edge_indexes.size:
            from scipy.sparse import csr_matrix

            edges = [g.get_edge(i) for i in edge_indexes]
            source_indexes = np.array([vix[next(iter(source))] for source, _ in edges])
            target_indexes = np.array([vix[next(iter(target))] for _, target in edges])
            num_internal_edges = len(edge_indexes)
            edge_rows = np.arange(num_internal_edges)
            edge_selector = csr_matrix(
                (
                    np.ones(num_internal_edges),
                    (edge_rows, edge_indexes),
                ),
                shape=(num_internal_edges, g.num_edges),
            )
            layer_difference_matrix = csr_matrix(
                (
                    np.concatenate((-np.ones(num_internal_edges), np.ones(num_internal_edges))),
                    (
                        np.concatenate((edge_rows, edge_rows)),
                        np.concatenate((source_indexes, target_indexes)),
                    ),
                ),
                shape=(num_internal_edges, g.num_vertices),
            )
            edge_selector = self.Constant(edge_selector)
            layer_difference_matrix = self.Constant(layer_difference_matrix)

            if Ip is not None:
                positive = Ip.reshape((g.num_edges, 1)) if len(Ip.shape) == 1 else Ip
                selected = edge_selector @ positive
                layer_difference = layer_difference_matrix @ L
                P += layer_difference >= selected + (1 - g.num_vertices) * (1 - selected)
                P += layer_difference <= g.num_vertices - 1

            if In is not None:
                negative = In.reshape((g.num_edges, 1)) if len(In.shape) == 1 else In
                selected = edge_selector @ negative
                layer_difference = -(layer_difference_matrix @ L)
                P += layer_difference >= selected + (1 - g.num_vertices) * (1 - selected)
                P += layer_difference <= g.num_vertices - 1

        if vertex_lb_dist is not None:
            layer_lb = np.zeros((g.num_vertices, n_samples))
            for sample_index, bounds in enumerate(vertex_lb_dist):
                for vertex, distance in bounds.items():
                    layer_lb[vix[vertex], sample_index] = distance
            P += L >= layer_lb
        if vertex_ub_dist is not None:
            layer_ub = np.full((g.num_vertices, n_samples), g.num_vertices - 1)
            for sample_index, bounds in enumerate(vertex_ub_dist):
                for vertex, distance in bounds.items():
                    layer_ub[vix[vertex], sample_index] = distance
            P += L <= layer_ub
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
            indicator = Ip + In
        elif Ip is not None:
            indicator = Ip
        elif In is not None:
            indicator = In
        else:
            raise ValueError()
        # Limit the number of parents per node, if requested
        if max_parents is not None:
            # Get indexes of edges vi->vj for all vi
            for v, max in max_parents.items():
                edges_idx = [i for i, _ in g.in_edges(v)]
                if len(edges_idx) > 0:
                    # Sum selected parent edges
                    P += np.ones((len(edges_idx),)) @ indicator[edges_idx] <= max
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
        indicator = self.Variable(name, S.shape, 0, 1, vartype=VarType.BINARY)
        blocked = np.isclose(ub, 0) & np.isclose(lb, 0)
        if np.sum(blocked) > 0:
            if blocked.ndim == 0:
                c += [indicator == 0, S == 0]
            elif blocked.ndim == 1:
                idx = np.where(blocked)[0]
                c += [indicator[idx] == 0, S[idx] == 0]
            elif blocked.ndim == 2:
                for col in range(blocked.shape[1]):
                    idx = np.where(blocked[:, col])[0]
                    if len(idx) > 0:
                        c += [indicator[idx, col] == 0, S[idx, col] == 0]
            else:
                raise ValueError(f"Unsupported indicator bounds dimensionality: {blocked.ndim}")
        # Add constraint: lb * I <= V <= ub * I
        if V._provided_lb is None or V._provided_ub is None:
            raise ValueError(f"The continuous variable {V.name} is unbounded, indicators cannot be created.")
        c += [S >= indicator.multiply(lb), S <= indicator.multiply(ub)]
        return self.Problem(c)

    def ExactSupport(
        self,
        V: CSymbol,
        *,
        selected: Optional[CExpression] = None,
        indexes: Optional[Union[int, slice, Tuple, List, np.ndarray]] = None,
        epsilon: Union[float, List[float], np.ndarray] = 1.0,
        nonnegative: bool = False,
        name: Optional[str] = None,
        positive_name: Optional[str] = None,
        negative_name: Optional[str] = None,
    ) -> ProblemDef:
        """Link a bounded value to binary structural support with a gap.

        ``selected == 0`` forces the value to zero. ``selected == 1`` forces
        its magnitude to be at least ``epsilon``. For nonnegative values this
        uses one binary per entry. Signed values use mutually exclusive
        positive and negative binaries and expose their sum as ``selected``.

        ``epsilon`` may be a scalar or an array explicitly broadcastable to
        the selected value shape. "Exact" refers to this mathematical
        minimum-magnitude gap; it is not a solver-tolerance guarantee, and
        values smaller than the requested gap may still be returned by a
        numerically inaccurate solve and should be checked by the caller.

        An existing binary selector symbol can be supplied to avoid
        introducing a redundant binary variable, which is useful when a
        method already has a shared structural edge-selection variable. An
        arbitrary selector expression is linked to an auxiliary binary
        selector, so fractional expression values make the model infeasible
        instead of relaxing the support constraints.
        """
        if isinstance(epsilon, (bool, np.bool_)):
            raise TypeError("epsilon must be a finite positive number or array.")
        epsilon_input = np.asarray(epsilon)
        if epsilon_input.dtype.kind in {"b", "c", "U", "S"}:
            raise TypeError("epsilon must be a finite positive number or array.")
        if epsilon_input.dtype.kind == "O" and any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) for value in epsilon_input.flat
        ):
            raise TypeError("epsilon must be a finite positive number or array.")
        try:
            epsilon_input = np.asarray(epsilon, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("epsilon must be a finite positive number or array.") from error

        if V._provided_lb is None or V._provided_ub is None:
            raise ValueError(f"The continuous variable {V.name} is unbounded, exact support cannot be created.")

        S = V if indexes is None else V[indexes]
        lb = np.asarray(V.lb if indexes is None else V.lb[indexes], dtype=float)
        ub = np.asarray(V.ub if indexes is None else V.ub[indexes], dtype=float)
        if np.shape(lb) != S.shape or np.shape(ub) != S.shape:
            expected_size = int(np.prod(S.shape))
            if lb.size == expected_size:
                lb = lb.reshape(S.shape)
            else:
                lb = np.broadcast_to(lb, S.shape)
            if ub.size == expected_size:
                ub = ub.reshape(S.shape)
            else:
                ub = np.broadcast_to(ub, S.shape)
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise ValueError("ExactSupport requires finite lower and upper bounds.")
        if np.any(lb > ub):
            raise ValueError("ExactSupport requires lower bounds not to exceed upper bounds.")
        try:
            epsilon_values = np.broadcast_to(epsilon_input, S.shape)
        except ValueError as error:
            raise ValueError(
                f"epsilon with shape {epsilon_input.shape} cannot be broadcast to selected value shape {S.shape}."
            ) from error
        if not np.all(np.isfinite(epsilon_values)) or np.any(epsilon_values <= 0):
            raise ValueError("epsilon must be a finite positive number or array.")

        support_name = name or f"{V.name}_support"
        constraints = []
        selector = None
        if selected is not None:
            if not isinstance(selected, CExpression):
                raise TypeError("selected must be a binary-valued CExpression.")
            if selected.shape != S.shape:
                raise ValueError(f"selected has shape {selected.shape}; expected {S.shape}.")
            if isinstance(selected, CSymbol):
                if selected._vartype != VarType.BINARY:
                    raise TypeError("selected must be a binary-valued selector.")
                selector = selected
            else:
                selector = self.Variable(
                    f"{support_name}_selector",
                    S.shape,
                    0,
                    1,
                    vartype=VarType.BINARY,
                )
                constraints.append(selector == selected)

        if nonnegative:
            if np.any(lb < 0):
                raise ValueError("ExactSupport(..., nonnegative=True) requires nonnegative lower bounds.")
            if selected is None:
                selector = self.Variable(support_name, S.shape, 0, 1, vartype=VarType.BINARY)

            possible = np.asarray(ub >= epsilon_values, dtype=float)
            if not np.all(possible):
                constraints.append(selector.multiply(1 - possible) == 0)
            constraints += [S >= selector.multiply(epsilon_values), S <= selector.multiply(ub)]
            problem = self.Problem(constraints)
            if selector.name != support_name:
                problem.register(support_name, selector)
            return problem

        if positive_name is None:
            positive_name = f"{support_name}_positive"
        if negative_name is None:
            negative_name = f"{support_name}_negative"
        positive = self.Variable(positive_name, S.shape, 0, 1, vartype=VarType.BINARY)
        negative = self.Variable(negative_name, S.shape, 0, 1, vartype=VarType.BINARY)
        support = positive + negative
        if selector is not None:
            constraints.append(support == selector)
            support = selector

        positive_possible = np.asarray(ub >= epsilon_values, dtype=float)
        negative_possible = np.asarray(lb <= -epsilon_values, dtype=float)
        if not np.all(positive_possible):
            constraints.append(positive.multiply(1 - positive_possible) == 0)
        if not np.all(negative_possible):
            constraints.append(negative.multiply(1 - negative_possible) == 0)
        constraints += [
            positive + negative <= 1,
            S >= negative.multiply(lb) + positive.multiply(epsilon_values),
            S <= positive.multiply(ub) - negative.multiply(epsilon_values),
        ]
        problem = self.Problem(constraints)
        if support.name != support_name:
            problem.register(support_name, support)
        return problem

    def SelectedFlow(
        self,
        g: BaseGraph,
        *,
        lb: Optional[Union[float, List, np.ndarray]] = 0,
        ub: Optional[Union[float, List, np.ndarray]] = DEFAULT_UB,
        n_flows: int = 1,
        edge_indices: Optional[Iterable[int]] = None,
        flow_blocks: Optional[Iterable[Tuple[Iterable[int], Union[slice, Iterable[int]]]]] = None,
        selector_groups: Optional[Iterable[int]] = None,
        epsilon: float = 1.0,
        exact_support: bool = True,
        selected: Optional[CExpression] = None,
        acyclic_graph: Optional[BaseGraph] = None,
        max_parents: Optional[Union[int, Dict[Any, int]]] = None,
        flow_name: str = EXPR_NAME_FLOW,
        selected_by_flow_name: str = "selected_by_flow",
        selected_by_group_name: Optional[str] = None,
        selected_any_name: Optional[str] = "selected_any",
        dag_name: str = VAR_DAG,
    ) -> ProblemDef:
        """Create bounded flows with exact per-flow and shared support.

        Flow conservation is imposed on ``g``. Exact support binaries are
        created only for ``edge_indices`` so boundary edges do not consume
        unnecessary integer variables. Rows whose selected flow bounds are
        nonnegative use one binary per flow; rows that permit negative flow
        automatically use mutually exclusive positive and negative binaries.
        Set ``exact_support=False`` to use one bounded indicator per entry:
        nonzero flow still implies selection, but a selected entry may carry
        zero flow. This smaller formulation is suitable when a minimizing
        objective makes such false selections unattractive and flow direction
        is not otherwise required.

        ``flow_blocks`` can map multiple rectangular regions of the flow
        matrix to the same logical edge ordering. ``selector_groups`` can then
        map block columns to shared selector columns. This supports layouts
        such as forward and reversed edge blocks without indicators on the
        unused off-diagonal blocks. Set ``selected_any_name=None`` when a union
        across selector columns is not needed.

        The union across flows can be linked to an existing shared selector.
        With signed acyclic flows, direction-specific unions are used so a
        negative flow orders the edge in reverse. No redundant structural-union
        binary is created in that case.
        """
        if not isinstance(n_flows, int) or isinstance(n_flows, bool) or n_flows <= 0:
            raise ValueError("n_flows must be a positive integer.")
        if not isinstance(exact_support, bool):
            raise TypeError("exact_support must be a boolean.")
        if flow_blocks is not None and edge_indices is not None:
            raise ValueError("Provide either edge_indices or flow_blocks, not both.")

        problem = self.Flow(
            g,
            lb=lb,
            ub=ub,
            n_flows=n_flows,
            shared_bounds=False,
            alias_flow=flow_name,
            force_matrix=True,
        )
        flow = problem.expr[flow_name]

        if flow_blocks is None:
            rows = range(g.num_edges) if edge_indices is None else edge_indices
            flow_blocks = [(rows, slice(0, n_flows))]

        normalized_blocks = []
        used_flow_columns = set()
        num_selected_edges = None
        for block_index, (block_rows, block_columns) in enumerate(flow_blocks):
            rows = np.asarray(list(block_rows), dtype=int)
            if rows.ndim != 1:
                raise ValueError(f"flow_blocks[{block_index}] edge indices must be one-dimensional.")
            if np.any(rows < 0) or np.any(rows >= g.num_edges):
                raise ValueError(f"flow_blocks[{block_index}] contains an edge outside the flow graph.")
            if num_selected_edges is None:
                num_selected_edges = len(rows)
            elif len(rows) != num_selected_edges:
                raise ValueError("Every flow block must map the same number of logical edges.")

            if isinstance(block_columns, slice):
                start, stop, step = block_columns.indices(n_flows)
                columns = np.arange(start, stop, step, dtype=int)
            else:
                columns = np.asarray(list(block_columns), dtype=int)
            if columns.ndim != 1 or not columns.size:
                raise ValueError(f"flow_blocks[{block_index}] flow columns must be a non-empty one-dimensional set.")
            if np.any(columns < 0) or np.any(columns >= n_flows):
                raise ValueError(f"flow_blocks[{block_index}] contains a flow column outside [0, {n_flows}).")
            if columns.size > 1 and np.any(np.diff(columns) != 1):
                raise ValueError("Flow columns within each block must be contiguous and increasing.")
            duplicate_columns = used_flow_columns.intersection(columns.tolist())
            if duplicate_columns:
                raise ValueError("Flow blocks must use disjoint flow columns.")
            used_flow_columns.update(columns.tolist())
            normalized_blocks.append((rows, slice(int(columns[0]), int(columns[-1]) + 1), len(columns)))

        if not normalized_blocks or num_selected_edges is None or num_selected_edges == 0:
            raise ValueError("SelectedFlow requires at least one mapped edge.")
        if selected is not None and selected.shape != (num_selected_edges,):
            raise ValueError(f"selected has shape {selected.shape}; expected {(num_selected_edges,)}.")

        def combine_rows(parts):
            """Stack row groups and restore the caller's edge order."""
            if len(parts) == 1:
                return parts[0][1]
            positions = np.concatenate([part_positions for part_positions, _ in parts])
            stacked = self.vstack([expression for _, expression in parts])
            return stacked[np.argsort(positions), :]

        selection_blocks = []
        positive_blocks = []
        negative_blocks = []
        has_signed_rows = False
        multiple_blocks = len(normalized_blocks) > 1
        for block_index, (block_rows, block_columns, _) in enumerate(normalized_blocks):
            block_flow = flow[block_rows, block_columns]
            selected_lb = np.asarray(flow.lb[block_rows, block_columns], dtype=float)
            if selected_lb.shape != block_flow.shape:
                selected_lb = np.broadcast_to(selected_lb, block_flow.shape)
            signed_rows = np.any(selected_lb < 0, axis=1)
            nonnegative_positions = np.flatnonzero(~signed_rows)
            signed_positions = np.flatnonzero(signed_rows)
            has_signed_rows |= exact_support and bool(signed_positions.size)

            block_base_name = (
                f"{selected_by_flow_name}_block_{block_index}" if multiple_blocks else selected_by_flow_name
            )
            selection_parts = []
            positive_parts = []
            negative_parts = []

            if not exact_support:
                if signed_positions.size and acyclic_graph is not None:
                    raise ValueError("Signed acyclic flows require exact_support=True.")
                problem += self.Indicator(
                    flow,
                    indexes=(block_rows, block_columns),
                    name=block_base_name,
                )
                support = problem.expr[block_base_name]
                selection_blocks.append(support)
                positive_blocks.append(support)
                negative_blocks.append(self.Constant(np.zeros(support.shape)))
                continue

            if nonnegative_positions.size:
                nonnegative_edges = block_rows[nonnegative_positions]
                support_name = block_base_name if not signed_positions.size else f"{block_base_name}_nonnegative"
                problem += self.ExactSupport(
                    flow,
                    indexes=(nonnegative_edges, block_columns),
                    epsilon=epsilon,
                    nonnegative=True,
                    name=support_name,
                )
                support = problem.expr[support_name]
                selection_parts.append((nonnegative_positions, support))
                positive_parts.append((nonnegative_positions, support))
                negative_parts.append(
                    (
                        nonnegative_positions,
                        self.Constant(np.zeros(support.shape)),
                    )
                )

            if signed_positions.size:
                signed_edges = block_rows[signed_positions]
                support_name = block_base_name if not nonnegative_positions.size else f"{block_base_name}_signed"
                positive_name = f"{support_name}_positive"
                negative_name = f"{support_name}_negative"
                problem += self.ExactSupport(
                    flow,
                    indexes=(signed_edges, block_columns),
                    epsilon=epsilon,
                    name=support_name,
                    positive_name=positive_name,
                    negative_name=negative_name,
                )
                selection_parts.append((signed_positions, problem.expr[support_name]))
                positive_parts.append((signed_positions, problem.expr[positive_name]))
                negative_parts.append((signed_positions, problem.expr[negative_name]))

            selection_blocks.append(combine_rows(selection_parts))
            positive_blocks.append(combine_rows(positive_parts))
            negative_blocks.append(combine_rows(negative_parts))

        selected_by_flow = selection_blocks[0] if len(selection_blocks) == 1 else self.hstack(selection_blocks)
        if selected_by_flow_name not in problem.expressions:
            problem.register(selected_by_flow_name, selected_by_flow)

        positive_by_flow = negative_by_flow = None
        if has_signed_rows:
            positive_by_flow_name = f"{selected_by_flow_name}_positive"
            negative_by_flow_name = f"{selected_by_flow_name}_negative"
            positive_by_flow = positive_blocks[0] if len(positive_blocks) == 1 else self.hstack(positive_blocks)
            negative_by_flow = negative_blocks[0] if len(negative_blocks) == 1 else self.hstack(negative_blocks)
            if positive_by_flow_name not in problem.expressions:
                problem.register(positive_by_flow_name, positive_by_flow)
            if negative_by_flow_name not in problem.expressions:
                problem.register(negative_by_flow_name, negative_by_flow)

        num_mapped_flows = selected_by_flow.shape[1]
        if selector_groups is None:
            selector_groups = np.arange(num_mapped_flows, dtype=int)
        else:
            selector_groups = np.asarray(list(selector_groups), dtype=int)
            if selector_groups.shape != (num_mapped_flows,):
                raise ValueError(f"selector_groups has shape {selector_groups.shape}; expected {(num_mapped_flows,)}.")
            if np.any(selector_groups < 0):
                raise ValueError("selector_groups must contain nonnegative integers.")
            unique_groups = np.unique(selector_groups)
            if not np.array_equal(unique_groups, np.arange(len(unique_groups))):
                raise ValueError("selector_groups must use consecutive group indexes starting at zero.")

        num_selector_groups = int(np.max(selector_groups)) + 1
        identity_groups = np.array_equal(selector_groups, np.arange(num_mapped_flows))
        if identity_groups:
            selected_by_group = selected_by_flow
        else:
            group_name = selected_by_group_name or f"{selected_by_flow_name}_grouped"
            selected_by_group = self.Variable(
                group_name,
                (num_selected_edges, num_selector_groups),
                vartype=VarType.BINARY,
            )
            from scipy.sparse import csr_matrix

            group_matrix = csr_matrix(
                (
                    np.ones(num_mapped_flows, dtype=float),
                    (np.arange(num_mapped_flows), selector_groups),
                ),
                shape=(num_mapped_flows, num_selector_groups),
            )
            expanded_groups = selected_by_group @ group_matrix.T
            problem += selected_by_flow <= expanded_groups
            problem += selected_by_group <= selected_by_flow @ group_matrix
        if selected_by_group_name is not None and selected_by_group_name not in problem.expressions:
            problem.register(selected_by_group_name, selected_by_group)

        if selected_any_name is None:
            if selected is not None:
                raise ValueError("selected requires selected_any_name.")
            if acyclic_graph is not None:
                raise ValueError("acyclic_graph requires selected_any_name.")
            return problem

        def union_across_flows(values, name):
            num_value_columns = values.shape[1]
            if num_value_columns == 1:
                union = values[:, 0]
            else:
                union = self.Variable(
                    name,
                    (num_selected_edges,),
                    vartype=VarType.BINARY,
                )
                union_matrix = self.Constant(_sparse_vector_replication(num_selected_edges, num_value_columns)) @ union
                union_matrix = union_matrix.reshape((num_selected_edges, num_value_columns))
                problem.add_constraints(values <= union_matrix)
                problem.add_constraints(union <= values.sum(axis=1))
            if name not in problem.expressions:
                problem.register(name, union)
            return union

        signed_acyclic = has_signed_rows and acyclic_graph is not None
        positive_any = negative_any = None
        if signed_acyclic:
            positive_any_name = f"{selected_any_name}_positive"
            negative_any_name = f"{selected_any_name}_negative"
            positive_any = union_across_flows(positive_by_flow, positive_any_name)
            negative_any = union_across_flows(negative_by_flow, negative_any_name)
            # A shared DAG cannot use the same structural edge in both
            # directions, even when the directions occur in different flows.
            problem += positive_any + negative_any <= 1
            directional_union = positive_any + negative_any
            if selected is None:
                selected_any = directional_union
            else:
                selected_any = selected
                problem += selected_any == directional_union
        else:
            if selected is None:
                selected_any = (
                    selected_by_group[:, 0]
                    if num_selector_groups == 1
                    else self.Variable(
                        selected_any_name,
                        (num_selected_edges,),
                        vartype=VarType.BINARY,
                    )
                )
            else:
                selected_any = selected

            if num_selector_groups == 1:
                problem += selected_any == selected_by_group[:, 0]
            else:
                selected_matrix = (
                    self.Constant(_sparse_vector_replication(num_selected_edges, num_selector_groups)) @ selected_any
                )
                selected_matrix = selected_matrix.reshape((num_selected_edges, num_selector_groups))
                problem += selected_by_group <= selected_matrix
                problem += selected_any <= selected_by_group.sum(axis=1)

        if selected_any_name not in problem.expressions:
            problem.register(selected_any_name, selected_any)

        if acyclic_graph is not None:
            if acyclic_graph.num_edges != num_selected_edges:
                raise ValueError("acyclic_graph must have one edge for each selected flow edge.")
            if signed_acyclic:
                self.Acyclic(
                    acyclic_graph,
                    problem,
                    indicator_positive_var_name=f"{selected_any_name}_positive",
                    indicator_negative_var_name=f"{selected_any_name}_negative",
                    acyclic_var_name=dag_name,
                    max_parents=max_parents,
                )
            else:
                indicator_name = f"_{selected_any_name}_acyclic"
                problem.register(indicator_name, selected_any)
                self.Acyclic(
                    acyclic_graph,
                    problem,
                    indicator_positive_var_name=indicator_name,
                    acyclic_var_name=dag_name,
                    max_parents=max_parents,
                )
        return problem

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
        indicator = I_pos + I_neg
        c += [indicator <= 1]  # Ensure mutual exclusivity

        # Fix impossible directions with whole-array masks. Besides tightening
        # the model, this avoids relying on solver feasibility tolerances when
        # the nonzero tolerance is small. The matrix form keeps compilation
        # independent of the number of flow columns.
        c += [I_pos.multiply(np.asarray(ub <= 0, dtype=float)) == 0]
        c += [I_neg.multiply(np.asarray(lb >= 0, dtype=float)) == 0]

        # Add constraints to enforce variable behavior depending on the indicator activation:
        # If I_pos = 1 and I_neg = 0: V >= tol AND V <= ub
        # If I_pos = 0 and I_neg = 1: V >= lb AND V <= -tol
        # If I_pos = 0 and I_neg = 0: V >= 0 AND V <= 0
        c += [
            S >= I_neg.multiply(lb) + I_pos * tolerance,
            S <= I_pos.multiply(ub) - I_neg * tolerance,
        ]

        return self.Problem(c)

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
    cvars = [k for k, v in p.symbols.items() if v.is_variable and v._vartype == VarType.CONTINUOUS]
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
