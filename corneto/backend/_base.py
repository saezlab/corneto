import abc
from turtle import pos
import numpy as np
import numbers
from typing import Set, Any, Dict, Iterable, Optional, Tuple, Union, List
#import corneto as cnt
from corneto._constants import *
#from corneto._core import ReNet
from corneto._decorators import _proxy
from corneto._settings import LOGGER
from corneto._core import Graph


def _eq_shape(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        if len(a.shape) == 1 and len(b.shape) == 2:
            return a.shape[0] == b.shape[0] and b.shape[1] == 1
        if len(a.shape) == 2 and len(b.shape) == 1:
            return a.shape[0] == b.shape[0] and a.shape[1] == 1
    return a.shape == b.shape


class CtProxyExpression(abc.ABC):
    # Arithmetic operator overloading with Numpy
    # See: https://www.cvxpy.org/_modules/cvxpy/expressions/expression.html#Expression
    __array_priority__ = 100

    def __init__(self, expr: Any, symbols: Optional[Set["CtProxySymbol"]] = None) -> None:
        super().__init__()
        self._expr = expr
        self._proxy_symbols: Set["CtProxySymbol"] = set()
        if symbols:
            self._proxy_symbols.update(symbols)

    def is_symbol(self) -> bool:
        return False

    def _create(self, expr: Any, atoms: Iterable) -> "CtProxyExpression":
        symbols = {s for s in atoms if isinstance(s, CtProxySymbol)}
        if isinstance(self, CtProxySymbol):
            symbols.add(self)
        if isinstance(self, CtProxyExpression):
            symbols.update(self._proxy_symbols)
        if isinstance(expr, CtProxySymbol):
            symbols.add(expr)
        if isinstance(expr, CtProxyExpression):
            symbols.update(expr._proxy_symbols)
        # Ask to create a CVXPY/PICOS/.. expression
        return self._create_proxy_expr(expr, symbols)

    @abc.abstractmethod
    def _create_proxy_expr(self, expr: Any, symbols: Optional[Set["CtProxySymbol"]] = None) -> "CtProxyExpression":
        pass

    @property
    @abc.abstractmethod
    def value(self) -> np.ndarray:
        pass

    @property
    def e(self) -> Any:
        return self._expr

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._expr.shape

    def __hash__(self) -> int:
        return self._expr.__hash__()

    def apply(self, fun, *args, **kwargs) -> "CtProxyExpression":
        # TODO: remove the collection of symbols here
        # symbols: Set[CtProxyExpression] = {s for s in args if isinstance(s, CtProxySymbol)}
        return self._create(fun(self._expr, *args, **kwargs), {})

    @property
    def T(self):
        return self._create(self._expr.T, {})

    @abc.abstractmethod
    def _elementwise_mul(self, other: Any) -> Any:
        pass

    @_proxy
    def multiply(self, other: Any) -> "CtProxyExpression":
        return self._elementwise_mul(other)

    @_proxy
    def __getitem__(self, item) -> "CtProxyExpression":
        pass

    @_proxy
    def __abs__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __pow__(self, power: float) -> "CtProxyExpression":
        pass

    @_proxy
    def __rpow__(self, base: float) -> "CtProxyExpression":
        pass

    @_proxy
    def __add__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __radd__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __sub__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __rsub__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __mul__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __matmul__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __truediv__(self, other: "CtProxyExpression") -> "CtProxyExpression":
        pass

    @_proxy
    def __div__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __rdiv__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __rtruediv__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __rmul__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __rmatmul__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __neg__(self) -> "CtProxyExpression":
        pass

    @_proxy
    def __eq__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __le__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __lt__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __ge__(self, other: Any) -> "CtProxyExpression":
        pass

    @_proxy
    def __gt__(self, other: Any) -> "CtProxyExpression":
        pass

    def __str__(self) -> str:
        return self._expr.__str__()

    def __repr__(self) -> str:
        return self._expr.__repr__()


class CtProxySymbol(CtProxyExpression):
    def __init__(
        self,
        expr: Any,
        name: str,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> None:
        super().__init__(expr)
        lb_r: np.ndarray
        ub_r: np.ndarray

        if lb is None:
            if vartype == VarType.CONTINUOUS:
                lb_r = np.full(expr.shape, -np.inf)
            elif vartype == VarType.INTEGER:
                lb_r = np.full(expr.shape, np.iinfo(int).min)
            elif vartype == VarType.BINARY:
                lb_r = np.zeros(expr.shape)
        if ub is None:
            if vartype == VarType.CONTINUOUS:
                ub_r = np.full(expr.shape, np.inf)
            elif vartype == VarType.INTEGER:
                ub_r = np.full(expr.shape, np.iinfo(int).max)
            elif vartype == VarType.BINARY:
                ub_r = np.ones(expr.shape)
        if isinstance(lb, np.ndarray):
            if not _eq_shape(lb, expr):
                raise ValueError(
                    f"Shape of lb is {lb.shape}, whereas symbol has a shape of {expr.shape}"
                )
            lb_r = lb
        elif isinstance(lb, numbers.Number):
            lb_r = np.full(expr.shape, lb)
        else:
            if lb is not None:
                raise ValueError(
                    f"lb has an invalid type ({type(lb)}). It must be a number or numpy array"
                )
        if isinstance(ub, np.ndarray):
            if not _eq_shape(ub, expr):
                raise ValueError(
                    f"Shape of ub is {ub.shape}, whereas symbol has a shape of {expr.shape}"
                )
            ub_r = ub
        elif isinstance(ub, numbers.Number):
            ub_r = np.full(expr.shape, ub)
        else:
            if ub is not None:
                raise ValueError(
                    f"ub has an invalid type ({type(ub)}). It must be a number or numpy array"
                )
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


class ProblemDef:
    def __init__(
        self,
        backend: Optional["Backend"] = None,
        constraints: Optional[List[CtProxyExpression]] = None,
        objectives: Optional[List[CtProxyExpression]] = None,
        weights: Optional[List[float]] = None,
        direction: Direction = Direction.MIN,
        graph: Graph = None
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
        self._index: Dict[str, CtProxySymbol] = dict()
        self._weights = weights if weights else []
        self._direction = direction
        # Create a new class of problems on top of a graph
        # where edges/nodes have associated optimization variables
        self._graph = graph

    @property
    def symbols(self) -> Dict[str, CtProxySymbol]:
        return {s.name: s for s in Backend.get_symbols(self._constraints + self._objectives)}

    def get_symbol(self, name) -> CtProxySymbol:
        return self.symbols[name]

    def get_symbols(self, *args) -> List[CtProxySymbol]:
        return [self.get_symbol(n) for n in args]

    @property
    def constraints(self) -> List[CtProxyExpression]:
        return self._constraints

    @property
    def objectives(self) -> List[CtProxyExpression]:
        return self._objectives

    @property
    def weights(self) -> List[float]:
        return self._weights

    @property
    def direction(self) -> Direction:
        return self._direction

    def copy(self) -> "ProblemDef":
        return ProblemDef(
            self._backend,
            self._constraints,
            self._objectives,
            self._weights,
            self._direction,
            self._graph
        )

    def _add(self, other: Any, inplace: bool = False):
        if isinstance(other, ProblemDef):
            return self.merge(other, inplace=inplace)
        elif isinstance(other, CtProxySymbol):
            LOGGER.warn(f"Ignoring request to add symbol {other} (not required)")
            return self
        elif isinstance(other, CtProxyExpression) and not isinstance(other, CtProxySymbol):
            return self.add_constraints([other], inplace=inplace)
        elif isinstance(other, Iterable):
            o = self
            if not inplace:
                o = self.copy()
            for e in other:
                if isinstance(e, CtProxySymbol):
                    #o.add_symbols([e], inplace=True)
                    pass
                elif isinstance(e, CtProxyExpression) and not isinstance(e, CtProxySymbol):
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
        max_seconds: int = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ) -> Any:
        if self._backend is None:
            raise ValueError("No backend assigned.")
        return self._backend.solve(
            self,
            solver=solver,
            max_seconds=max_seconds,
            warm_start=warm_start,
            verbosity=verbosity,
            **options,
        )

    def merge(self, other: "ProblemDef", inplace=False) -> "ProblemDef":
        # TODO: If the other is empty (or instance of grammar?) build the problem before merging
        if isinstance(other, ProblemDef) and hasattr(other, "_build_problem"): # TODO Change by isinstance
            f = getattr(other, "_build_problem")
            other = f(self)
        b = self._backend if not None else other._backend
        if not b:
            raise ValueError("Problems have no backend associated.")
        if self._backend and other._backend and (self._backend != other._backend):
            raise ValueError(
                "The two problems have different instantiations of the backend."
            )
        if inplace:
            self.add_constraints(other._constraints, inplace=True)
            self.add_objectives(other._objectives, other._weights, inplace=True)
            return self
        c = self._constraints + other._constraints
        w = self._weights + other._weights
        o = self._objectives + other._objectives
        # TODO: manage the case of _graph in problem
        return ProblemDef(b, c, o, w)

    def add_constraints(
        self,
        constraints: Union[CtProxyExpression, List[CtProxyExpression]],
        inplace: bool = True,
    ) -> "ProblemDef":
        # TODO: Auto-load symbols from expressions
        if not isinstance(constraints, list):
            constraints = [constraints]
        for c in constraints:
            if isinstance(c, CtProxySymbol):
                raise ValueError(f"The variable {c.name} was added as a constraint")
        if inplace:
            self._constraints.extend(constraints)
            return self
        return ProblemDef(
            self._backend,
            self._constraints + constraints,
            self._objectives,
            self._weights,
        )



    def add_objectives(
        self,
        objectives: Union[CtProxyExpression, List[CtProxyExpression]],
        weights: Union[float, List[float]] = 1.0,
        inplace: bool = True,
    ) -> "ProblemDef":
        if not isinstance(objectives, list):
            objectives = [objectives]
        if not isinstance(weights, list):
            weights = [weights] * len(objectives)
        if len(weights) != len(objectives):
            raise ValueError("Number of weights must match number of objectives")
        if inplace:
            self._objectives.extend(objectives)
            self._weights.extend(weights)
            return self
        return ProblemDef(
            self._backend,
            self._constraints,
            self._objectives + objectives,
            self._weights + weights,
        )


class Grammar(ProblemDef):
    def __init__(self) -> None:
        super().__init__(None, None, None, None, Direction.MIN)

    def _build_problem(self, other: ProblemDef) -> ProblemDef:
        raise NotImplementedError()

    def merge(self, other: ProblemDef, inplace=False) -> ProblemDef:
        return other.merge(self._build_problem(other), inplace)


class Backend(abc.ABC):
    def __init__(self) -> None:
        pass

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception as e:
            return False

    def version(self) -> str:
        return self._load().__version__

    @staticmethod
    def get_symbols(expressions: Iterable[CtProxyExpression]) -> Set[CtProxySymbol]:
        symbols: Set[CtProxySymbol] = set()
        for e in expressions:
            s: Set[CtProxySymbol]  = getattr(e, '_proxy_symbols', {}) # type: ignore
            if isinstance(e, CtProxySymbol):
                s.add(e)
            symbols.update(s)
        return symbols

    @abc.abstractmethod
    def _load(self) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def available_solvers() -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def Variable(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> CtProxySymbol:
        pass

    def Problem(
        self,
        constraints: Optional[List[CtProxyExpression]] = None,
        objectives: Optional[Union[CtProxyExpression, List[CtProxyExpression]]] = None,
        weights: Optional[Union[float, List[float]]] = None,
        direction: Direction = Direction.MIN,
    ) -> ProblemDef:
        
        if isinstance(constraints, CtProxyExpression):
            constraints = [constraints]
        if constraints is None:
            constraints = []
        if isinstance(objectives, CtProxyExpression):
            objectives = [objectives]
        if isinstance(weights, float):
            weights = [weights]
        elif isinstance(weights, numbers.Number):
            weights = [float(weights)]
        return ProblemDef(self, constraints, objectives, weights, direction)

    def solve(
        self,
        p: ProblemDef,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: int = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ):
        o: Optional[CtProxyExpression]
        if p.objectives is not None and len(p.objectives) > 1:
            if len(p.weights) != len(p.objectives):
                raise ValueError("Number of weights must match number of objectives")
            # auto-convert to a weighted sum
            # type: ignore
            o = sum(
                p.weights[i] * p.objectives[i] if p.weights[i] != 0.0 else 0.0  # type: ignore
                for i in range(len(p.objectives))
            )
        else:
            o = p.objectives[0] if p.objectives and p.weights[0] != 0 else None
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
        objective: Optional[CtProxyExpression] = None,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: int = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ):
        raise NotImplementedError()

    def Constant(self) -> CtProxySymbol:
        raise NotImplementedError()

    def Parameter(self) -> CtProxySymbol:
        raise NotImplementedError()

    def Flow(
        self,
        g: Graph,
        lb: Union[float, np.ndarray] = 0,
        ub: Union[float, np.ndarray] = 10,
        values=False,
        acyclic: bool = False,
        varname: Optional[str] = VAR_FLOW,
    ) -> ProblemDef:
        V = self.Variable(varname, (g.num_edges,), lb, ub)
        S = g.vertex_incidence_matrix(values=values)
        C = S @ V == 0
        if acyclic:
            # TODO
            # Run BFS from source nodes for lower bound
            # Create Layer variables and restrict flow to be from L+1
            raise NotImplementedError()
        p = self.Problem(C)
        return p

    def Indicators(
        self,
        V: CtProxySymbol,
        tolerance=1e-3,
        positive=True,
        negative=True,
        absolute=True,
        suffix_pos="_ipos",
        suffix_neg="_ineg",
        suffix_abs="_iabs"
    ) -> ProblemDef:
        # Get upper/lower bounds for flow variables
        constraints = []
        if not (positive or negative):
            raise ValueError("At least one of positive or negative must be True.")
        if positive:
            I_pos = self.Variable(V.name + suffix_pos, V.shape, 0, 1, VarType.BINARY)
            #variables.append(I_pos)
            if sum(V.ub <= 0) > 0:
                constraints.append(I_pos[np.where(V.ub <= 0)[0]] == 0)
        if negative:
            I_neg = self.Variable(V.name + suffix_neg, V.shape, 0, 1, VarType.BINARY)
            #variables.append(I_neg)
            if sum(V.lb >= 0) > 0:
                constraints.append(I_neg[np.where(V.lb >= 0)[0]] == 0)
        if absolute:
            I_abs = self.Variable(V.name + suffix_abs, V.shape, 0, 1, VarType.BINARY)
            #variables.append(I_abs)
            constraints.append(I_abs == I_pos + I_neg)

        if positive and negative:
            constraints.append(I_pos + I_neg <= 1)

        # lower bound constraints: F >= F_lb * I_neg + eps * I_pos
        I_LBN = I_neg.multiply(V.lb) if negative else V.lb
        I_LBP = tolerance * I_pos if positive else 0
        LB = I_LBN + I_LBP
        constraints.append(V >= LB)
        # upper bound constraints: F <= F_ub * I_pos - eps * I_neg
        I_UBN = I_pos.multiply(V.ub) if positive else V.ub
        I_UBP = tolerance * I_neg if negative else 0
        UB = I_UBN - I_UBP
        constraints.append(V <= UB)
        # return variables, constraints
        #return self.Problem(variables, constraints)
        return self.Problem(constraints)


class Indicators(Grammar):
    def __init__(
        self,
        var_name: Optional[str] = None,
        tolerance: float = 1e-3,
        positive: bool = True,
        negative: bool = False,
        absolute: bool = False,
    ) -> None:
        # Probably there is some missing component which is not a problemdef
        super().__init__()
        self.var_name = var_name
        self._tol = tolerance
        self._pos = positive
        self._neg = negative
        self._abs = absolute

    def _build_problem(self, other: ProblemDef):
        if other._backend is None:
            raise ValueError("Cannot combine empty grammars")
        if self.var_name is None:
            # Search for continous vars
            cvars = [
                k for k, v in other.symbols.items() if v._vartype == VarType.CONTINUOUS
            ]
            if len(cvars) == 0:
                raise ValueError(
                    "No available continuous vars for creating indicator vars"
                )
            if len(cvars) == 1:
                self.var_name = cvars[0]
                LOGGER.debug(f"No variable provided, creating indicators for {cvars[0]}")
            else:
                raise ValueError(
                    f"There are {len(cvars)} continous vars, but no var_name is provided."
                )
        return other._backend.Indicators(
            other.get_symbol(self.var_name),
            tolerance=self._tol,
            positive=self._pos,
            negative=self._neg,
            absolute=self._abs,
        )


class HammingLoss(Grammar):
    def __init__(
        self,
        reference: np.ndarray,
        y: Union[str, CtProxyExpression],
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
        #P.add_objectives((sum(diff_zeros) + sum(diff_ones)) * self.penalty, inplace=True)  # type: ignore
        P.add_objectives(hamming_dist, weights=self.penalty, inplace=True)  # type: ignore
    
        return P


"""
class DAG(Grammar):
    def __init__(self, reactions=None) -> None:
        super().__init__()

    def _build_problem(self, other: ProblemDef) -> ProblemDef:
        # Create a position variable per node
        bck = other._backend
        if bck is None:
            raise ValueError("Cannot combine empty grammars")
        # Do for the entire network or for a subset of edges
        L = bck.Variable(varname, shape)
        L[target] - L[source] >= E_ind + (1 - N) * (1 - E_ind)
"""
